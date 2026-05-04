#!/usr/bin/env python3
"""
CRUCIBLE Hyperparameter / Configuration Sweep Harness

Runs multiple `engine.run` invocations across a Cartesian product of:
  - models             (--models  m1 m2 ...)
  - prompt-modes       (--prompt-modes balanced_competitive hard_max tournament)
  - seeds              (--seeds 1 2 3)
  - temperatures       (--temperatures 0.7 1.0 1.3)
  - rounds             (--rounds-list 10 25)
  - reflection on/off  (--reflection-modes on off)
  - asymmetric priming (--asymmetric-suffix-files <path>)

Each combination launches a separate `engine.run` subprocess. Runs are dispatched
in parallel up to --max-parallel; the script waits until all complete before exiting.

Cross-provider parallelism is safe (different rate-limit pools); same-provider
parallelism may hit rate limits depending on tier — set --max-parallel-per-provider
to limit concurrency per backend (default 2).

Usage examples:
  # Single-model temperature sweep
  python scripts/sweep.py --models claude-sonnet-4-6 --temperatures 0.7 1.0 1.3 \
      --seeds 1 --rounds-list 25 --prompt-modes hard_max

  # Multi-model 3-seed sweep on OpenRouter
  python scripts/sweep.py \
      --models nousresearch/hermes-4-70b deepseek/deepseek-chat-v3.1 \
      --seeds 1 2 3 --rounds-list 25 --prompt-modes hard_max

  # Asymmetric priming ablation
  python scripts/sweep.py --models claude-sonnet-4-6 --rounds-list 25 \
      --prompt-modes hard_max --seeds 1 \
      --asymmetric-suffix-files priming/opponent_defects.txt
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import shlex
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class SweepConfig:
    """One concrete experiment to run."""
    model: str
    prompt_mode: str
    seed: int
    rounds: int
    turns: int
    temperature: float | None
    top_p: float | None
    max_tokens: int | None
    reflection: bool
    asym_suffix_a_path: str | None
    asym_suffix_b_path: str | None

    @property
    def label(self) -> str:
        bits = [
            self.model.split("/")[-1],
            self.prompt_mode,
            f"r{self.rounds}",
            f"t{self.turns}",
            f"s{self.seed}",
        ]
        if self.temperature is not None:
            bits.append(f"T{self.temperature}")
        if self.top_p is not None:
            bits.append(f"P{self.top_p}")
        if not self.reflection:
            bits.append("noreflect")
        if self.asym_suffix_a_path or self.asym_suffix_b_path:
            bits.append("asym")
        return "_".join(bits)

    @property
    def provider(self) -> str:
        if self.model.startswith("claude"):
            return "anthropic"
        if "/" in self.model:
            return "openrouter"
        if self.model.startswith(("gpt-", "o1", "o3", "o4")):
            return "openai"
        if self.model.startswith("deepseek"):
            return "openai"
        return "google"

    def to_argv(self) -> list[str]:
        argv = [
            sys.executable, "-u", "-m", "engine.run",
            "--rounds", str(self.rounds),
            "--turns", str(self.turns),
            "--seed", str(self.seed),
            "--prompt-mode", self.prompt_mode,
        ]
        if self.temperature is not None:
            argv += ["--temperature", str(self.temperature)]
        if self.top_p is not None:
            argv += ["--top-p", str(self.top_p)]
        if self.max_tokens is not None:
            argv += ["--max-tokens", str(self.max_tokens)]
        if not self.reflection:
            argv += ["--no-reflection"]
        if self.asym_suffix_a_path:
            argv += ["--system-suffix-a-file", self.asym_suffix_a_path]
        if self.asym_suffix_b_path:
            argv += ["--system-suffix-b-file", self.asym_suffix_b_path]
        return argv


def _enumerate(args) -> list[SweepConfig]:
    """Cartesian product of every grid axis."""
    temps = args.temperatures or [None]
    top_ps = args.top_ps or [None]
    max_tokens = args.max_tokens_list or [None]
    reflection_modes = [m == "on" for m in (args.reflection_modes or ["on"])]
    asym_paths_a = args.asymmetric_suffix_files_a or [None]
    asym_paths_b = args.asymmetric_suffix_files_b or [None]
    if asym_paths_a == [None] and asym_paths_b == [None] and args.asymmetric_suffix_files:
        # If --asymmetric-suffix-files is given but per-agent isn't,
        # apply the file to agent A only (most common ablation: prime A, leave B baseline).
        asym_paths_a = args.asymmetric_suffix_files

    configs: list[SweepConfig] = []
    for (model, mode, seed, rounds, t, tp, mt, refl, sa, sb) in itertools.product(
        args.models, args.prompt_modes, args.seeds, args.rounds_list,
        temps, top_ps, max_tokens, reflection_modes, asym_paths_a, asym_paths_b,
    ):
        configs.append(SweepConfig(
            model=model, prompt_mode=mode, seed=seed, rounds=rounds, turns=args.turns,
            temperature=t, top_p=tp, max_tokens=mt, reflection=refl,
            asym_suffix_a_path=sa, asym_suffix_b_path=sb,
        ))
    return configs


@dataclass
class ProviderSemaphore:
    """Tracks how many runs are currently active per provider."""
    cap: int
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def can_admit(self, provider: str) -> bool:
        return self.counts[provider] < self.cap

    def acquire(self, provider: str) -> None:
        self.counts[provider] += 1

    def release(self, provider: str) -> None:
        self.counts[provider] = max(0, self.counts[provider] - 1)


async def _run_one(cfg: SweepConfig, log_dir: Path) -> tuple[SweepConfig, int, str]:
    log_path = log_dir / f"{cfg.label}.log"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CRUCIBLE_MODEL"] = cfg.model
    env["PYTHONUNBUFFERED"] = "1"
    proc = await asyncio.create_subprocess_exec(
        *cfg.to_argv(),
        cwd=str(REPO_ROOT),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    started = time.time()
    print(f"  [start] {cfg.label}  ({cfg.provider})", flush=True)
    with open(log_path, "wb") as logf:
        async for line in proc.stdout:
            logf.write(line)
            text = line.decode("utf-8", errors="replace").rstrip()
            # Only surface significant events to stdout to keep the parent log readable.
            if any(tag in text for tag in ("STEAL", "[ERROR]", "[ABORT]", "[TIMEOUT]",
                                            "[COMPLETE]", "[PARTIAL]", "Cooperation Rate:",
                                            "Deception Index:", "Spend:", "preflight")):
                print(f"  [{cfg.label}] {text}", flush=True)
    rc = await proc.wait()
    elapsed = time.time() - started
    status = "OK" if rc == 0 else f"EXIT {rc}"
    print(f"  [{status}] {cfg.label} after {elapsed:.0f}s  log={log_path.name}", flush=True)
    return cfg, rc, str(log_path)


async def _scheduler(configs: list[SweepConfig], max_parallel: int,
                      max_per_provider: int, log_dir: Path) -> list[tuple[SweepConfig, int, str]]:
    pending = deque(configs)
    active: dict[asyncio.Task, SweepConfig] = {}
    sem = ProviderSemaphore(cap=max_per_provider)
    results: list[tuple[SweepConfig, int, str]] = []

    while pending or active:
        # Try to start as many as fits within both caps.
        i = 0
        while i < len(pending):
            if len(active) >= max_parallel:
                break
            cfg = pending[i]
            if not sem.can_admit(cfg.provider):
                i += 1
                continue
            pending.remove(cfg)
            sem.acquire(cfg.provider)
            t = asyncio.create_task(_run_one(cfg, log_dir))
            active[t] = cfg

        if not active:
            # No tasks could start; brief pause to avoid spin (shouldn't happen unless every
            # remaining provider is at cap — which means we just need to wait for one to free).
            await asyncio.sleep(0.5)
            continue

        done, _ = await asyncio.wait(active.keys(), return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            cfg = active.pop(t)
            sem.release(cfg.provider)
            try:
                results.append(t.result())
            except Exception as e:
                results.append((cfg, -1, f"exception: {e}"))
    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CRUCIBLE configuration sweep harness")
    p.add_argument("--models", nargs="+", required=True,
                   help="One or more model strings (e.g. claude-sonnet-4-6 nousresearch/hermes-4-70b)")
    p.add_argument("--prompt-modes", nargs="+", default=["hard_max"],
                   choices=["balanced_competitive", "hard_max", "tournament", "legacy"])
    p.add_argument("--seeds", nargs="+", type=int, default=[1])
    p.add_argument("--rounds-list", nargs="+", type=int, default=[25],
                   help="Sweep over multiple game lengths")
    p.add_argument("--turns", type=int, default=3, help="Conversation turns per round (single value)")
    p.add_argument("--temperatures", nargs="*", type=float, default=None)
    p.add_argument("--top-ps", nargs="*", type=float, default=None)
    p.add_argument("--max-tokens-list", nargs="*", type=int, default=None)
    p.add_argument("--reflection-modes", nargs="*", default=["on"], choices=["on", "off"])
    p.add_argument("--asymmetric-suffix-files", nargs="*", default=None,
                   help="Treat as agent-A suffixes only (B remains baseline). For per-agent control "
                        "use --asymmetric-suffix-files-a / --asymmetric-suffix-files-b.")
    p.add_argument("--asymmetric-suffix-files-a", nargs="*", default=None)
    p.add_argument("--asymmetric-suffix-files-b", nargs="*", default=None)
    p.add_argument("--max-parallel", type=int, default=4,
                   help="Total concurrent runs across all providers")
    p.add_argument("--max-parallel-per-provider", type=int, default=2,
                   help="Concurrent runs per provider (rate-limit safety)")
    p.add_argument("--log-dir", default="data/sweep_logs")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned grid without executing")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    configs = _enumerate(args)

    print(f"Sweep: {len(configs)} configs across "
          f"{len(set(c.provider for c in configs))} providers, "
          f"max_parallel={args.max_parallel}, max_per_provider={args.max_parallel_per_provider}")
    if args.dry_run:
        for c in configs:
            print(f"  {c.label}  argv={shlex.join(c.to_argv())}")
        return

    log_dir = REPO_ROOT / args.log_dir
    started_at = time.time()
    results = asyncio.run(_scheduler(configs, args.max_parallel, args.max_parallel_per_provider, log_dir))
    elapsed = time.time() - started_at

    ok = [r for r in results if r[1] == 0]
    bad = [r for r in results if r[1] != 0]
    print(f"\nSweep complete in {elapsed:.0f}s. {len(ok)} OK, {len(bad)} non-zero exit.")
    if bad:
        for cfg, rc, info in bad:
            print(f"  FAIL {cfg.label}: rc={rc} log={info}")
    print(f"\nUse: python scripts/compare_runs.py --by-seed   to see results across the sweep.")


if __name__ == "__main__":
    main()
