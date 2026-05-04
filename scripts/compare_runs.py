#!/usr/bin/env python3
"""
Cross-model run comparison report.

Walks data/runs/*_metrics.json, extracts experiment metadata + derived stats, and
emits a comparison table grouped by (model_a, model_b, prompt_mode), aggregated
across seeds. Handles both the legacy single-`model` schema and the post-refactor
`model_a` / `model_b` schema for matchups.

Usage:
  python scripts/compare_runs.py
  python scripts/compare_runs.py --model claude
  python scripts/compare_runs.py --prompt-mode hard_max --by-seed
  python scripts/compare_runs.py --json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RUNS_DIR = os.path.join(REPO_ROOT, "data", "runs")


@dataclass
class RunRecord:
    path: str
    model_a: str
    model_b: str
    provider_a: str
    provider_b: str
    prompt_mode: str
    seed: Optional[int]
    rounds_total: int
    rounds_completed: int
    is_partial: bool
    enable_reflection: bool
    cooperation_rate: float
    mutual_destruction_rate: float
    deception_index: float
    first_betrayal_round: Optional[int]
    n_betrayals: int
    n_mutual_destructions: int
    ambiguous_parses: int
    timestamp: str

    @property
    def matchup_label(self) -> str:
        if self.model_a == self.model_b:
            return self.model_a
        return f"{self.model_a} vs {self.model_b}"


def _read_metrics(path: str) -> Optional[RunRecord]:
    """Parse one metrics file. Returns None if the file is malformed or missing required fields."""
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return None

    exp = data.get("_experiment") or {}
    rounds = data.get("rounds") or []

    # Schema bridge: post-refactor runs have model_a/model_b/provider_a/provider_b.
    # Legacy runs only have `model` and assume symmetric play.
    legacy_model = exp.get("model", "unknown")
    model_a = exp.get("model_a", legacy_model)
    model_b = exp.get("model_b", legacy_model)
    provider_a = exp.get("provider_a", "")
    provider_b = exp.get("provider_b", "")

    # Derived: first betrayal round, totals
    first_betrayal: Optional[int] = None
    n_betrayals = 0
    n_md = 0
    for r in rounds:
        if r.get("a_betrayed") or r.get("b_betrayed"):
            n_betrayals += 1
            if first_betrayal is None:
                first_betrayal = r.get("round_number")
        if r.get("mutual_destruction"):
            n_md += 1

    return RunRecord(
        path=path,
        model_a=model_a,
        model_b=model_b,
        provider_a=provider_a,
        provider_b=provider_b,
        prompt_mode=exp.get("prompt_mode", "unknown"),
        seed=exp.get("seed"),
        rounds_total=int(exp.get("rounds", len(rounds))),
        rounds_completed=int(exp.get("completed_rounds", len(rounds))),
        is_partial=bool(data.get("_partial", False)),
        enable_reflection=bool(exp.get("enable_reflection", True)),
        cooperation_rate=float(data.get("cooperation_rate", 0.0)),
        mutual_destruction_rate=float(data.get("mutual_destruction_rate", 0.0)),
        deception_index=float(data.get("deception_index", 0.0)),
        first_betrayal_round=first_betrayal,
        n_betrayals=n_betrayals,
        n_mutual_destructions=n_md,
        ambiguous_parses=int(exp.get("ambiguous_parses", 0)),
        timestamp=str(exp.get("timestamp", "")),
    )


def _load_runs(runs_dir: str, model_filter: Optional[str], prompt_mode_filter: Optional[str]) -> list[RunRecord]:
    paths = sorted(glob.glob(os.path.join(runs_dir, "*_metrics.json")))
    records: list[RunRecord] = []
    for p in paths:
        r = _read_metrics(p)
        if r is None:
            continue
        if model_filter and model_filter not in r.model_a and model_filter not in r.model_b:
            continue
        if prompt_mode_filter and r.prompt_mode != prompt_mode_filter:
            continue
        records.append(r)
    return records


@dataclass
class GroupAgg:
    matchup: str
    prompt_mode: str
    rounds_total: int
    n_runs: int
    coop_rates: list[float] = field(default_factory=list)
    md_rates: list[float] = field(default_factory=list)
    deception_indices: list[float] = field(default_factory=list)
    first_betrayals: list[int] = field(default_factory=list)
    betrayal_counts: list[int] = field(default_factory=list)
    rounds_completed: list[int] = field(default_factory=list)
    seeds: list[Any] = field(default_factory=list)
    partials: int = 0


def _summary_stat(xs: list[float]) -> str:
    """Format mean (min..max) when n>1, single value otherwise."""
    if not xs:
        return "—"
    if len(xs) == 1:
        return f"{xs[0]:.3f}"
    return f"{statistics.mean(xs):.3f} ({min(xs):.3f}..{max(xs):.3f})"


def _summary_int(xs: list[int]) -> str:
    if not xs:
        return "—"
    if len(xs) == 1:
        return str(xs[0])
    return f"{statistics.mean(xs):.1f} ({min(xs)}..{max(xs)})"


def _aggregate(records: list[RunRecord]) -> list[GroupAgg]:
    groups: dict[tuple[str, str, int], GroupAgg] = {}
    for r in records:
        key = (r.matchup_label, r.prompt_mode, r.rounds_total)
        if key not in groups:
            groups[key] = GroupAgg(
                matchup=r.matchup_label,
                prompt_mode=r.prompt_mode,
                rounds_total=r.rounds_total,
                n_runs=0,
            )
        g = groups[key]
        g.n_runs += 1
        g.coop_rates.append(r.cooperation_rate)
        g.md_rates.append(r.mutual_destruction_rate)
        g.deception_indices.append(r.deception_index)
        if r.first_betrayal_round is not None:
            g.first_betrayals.append(r.first_betrayal_round)
        g.betrayal_counts.append(r.n_betrayals)
        g.rounds_completed.append(r.rounds_completed)
        g.seeds.append(r.seed if r.seed is not None else "—")
        if r.is_partial:
            g.partials += 1
    return list(groups.values())


def _print_table(rows: list[list[str]], headers: list[str]) -> None:
    """Print a fixed-width text table."""
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(x)) for x in col) for col in cols]
    sep = "  ".join("-" * w for w in widths)
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print(sep)
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))


def _per_run_view(records: list[RunRecord]) -> None:
    headers = [
        "Matchup", "Prompt", "Seed", "Refl", "Rounds",
        "Coop%", "MD%", "DI", "1st-betray", "Betrayals", "Status",
    ]
    rows = []
    for r in sorted(records, key=lambda x: (x.matchup_label, x.prompt_mode, x.seed or 0, x.timestamp)):
        rows.append([
            r.matchup_label,
            r.prompt_mode,
            str(r.seed) if r.seed is not None else "—",
            "ON" if r.enable_reflection else "OFF",
            f"{r.rounds_completed}/{r.rounds_total}",
            f"{r.cooperation_rate * 100:.0f}",
            f"{r.mutual_destruction_rate * 100:.0f}",
            f"{r.deception_index:.1f}",
            str(r.first_betrayal_round) if r.first_betrayal_round is not None else "never",
            str(r.n_betrayals),
            "PARTIAL" if r.is_partial else "ok",
        ])
    _print_table(rows, headers)


def _aggregate_view(groups: list[GroupAgg]) -> None:
    headers = [
        "Matchup", "Prompt", "Rounds", "n",
        "Coop% mean (range)", "MD% mean (range)", "DI mean (range)",
        "1st-betray mean (range)", "Partials",
    ]
    rows = []
    for g in sorted(groups, key=lambda x: (x.matchup, x.prompt_mode, x.rounds_total)):
        coop_pct = [c * 100 for c in g.coop_rates]
        md_pct = [m * 100 for m in g.md_rates]
        rows.append([
            g.matchup,
            g.prompt_mode,
            str(g.rounds_total),
            str(g.n_runs),
            _summary_stat(coop_pct),
            _summary_stat(md_pct),
            _summary_stat(g.deception_indices),
            _summary_int(g.first_betrayals) if g.first_betrayals else "never",
            f"{g.partials}/{g.n_runs}" if g.partials else "0",
        ])
    _print_table(rows, headers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-model comparison report for CRUCIBLE runs")
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR, help=f"Default: {DEFAULT_RUNS_DIR}")
    parser.add_argument("--model", default=None, help="Filter to runs whose model_a or model_b contains this substring")
    parser.add_argument("--prompt-mode", default=None, help="Filter to a specific prompt mode")
    parser.add_argument("--by-seed", action="store_true", help="Show one row per run instead of aggregating across seeds")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table")
    args = parser.parse_args()

    records = _load_runs(args.runs_dir, args.model, args.prompt_mode)
    if not records:
        print(f"No runs found in {args.runs_dir}", file=sys.stderr)
        if args.model or args.prompt_mode:
            print("(filters applied)", file=sys.stderr)
        sys.exit(1)

    if args.json:
        if args.by_seed:
            payload = [r.__dict__ for r in records]
        else:
            payload = [g.__dict__ for g in _aggregate(records)]
        print(json.dumps(payload, indent=2, default=str))
        return

    print(f"Found {len(records)} run(s) in {args.runs_dir}")
    print()
    if args.by_seed:
        _per_run_view(records)
    else:
        _aggregate_view(_aggregate(records))


if __name__ == "__main__":
    main()
