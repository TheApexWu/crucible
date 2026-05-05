"""
CRUCIBLE Runner (Hardened)

Execute a full game and save results with experiment metadata.
Usage: python -m engine.run [--rounds 25] [--turns 2] [--seed 42]

IMPORTANT: Run ONE seed at a time. NEVER run parallel on the same provider.
Wait 5 minutes between runs on Tier 1 accounts.

Safety features:
- Preflight health check (1 attempt, fail fast -- no retry loop)
- Per-round checkpointing (crash at round N = you keep rounds 1 to N-1)
- Graceful shutdown on Ctrl+C / SIGTERM (saves partial results)
- Per-round timeout (kills hung API calls, default 180s)
- Rate limit safety: max 5 retries, 60s min wait, 180s total cap
- Output verification (alerts on 0-byte or missing output)
"""

import asyncio
import argparse
import json
import os
import signal
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.game import (
    run_round,
    model_name,
    model_name_b,
    get_prompt_bundle,
    GameState,
    get_agent_configs,
    preflight,
)
from engine.metrics import compute_all_metrics
from engine import spend as _spend


# Graceful shutdown flag
_shutdown_requested = False


def _request_shutdown(signum, frame):
    global _shutdown_requested
    print(f"\n[SHUTDOWN] Signal {signum} received. Finishing current round then saving...")
    _shutdown_requested = True


def on_round_complete(round_state):
    """Callback for each completed round."""
    r = round_state
    print(
        f"Round {r.round_number}: "
        f"A={r.agent_a_choice.upper()} B={r.agent_b_choice.upper()} "
        f"| A=${r.agent_a_total} B=${r.agent_b_total}"
    )


def _save_results(game_state, metrics, experiment_meta, run_tag, partial=False):
    """Save game and metrics to disk. Works for both complete and partial runs."""
    os.makedirs("data/runs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    tag = f"{run_tag}_PARTIAL" if partial else run_tag

    game_data = json.loads(game_state.model_dump_json())
    game_data["_experiment"] = experiment_meta
    game_data["_partial"] = partial

    metrics_data = json.loads(metrics.model_dump_json()) if metrics else {"_error": "no metrics computed"}
    metrics_data["_experiment"] = experiment_meta
    metrics_data["_partial"] = partial

    game_path = f"data/runs/{tag}_game.json"
    metrics_path = f"data/runs/{tag}_metrics.json"

    with open(game_path, "w") as f:
        json.dump(game_data, f, indent=2)
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    # Only overwrite latest if this is a complete run
    if not partial:
        with open("data/latest_game.json", "w") as f:
            json.dump(game_data, f, indent=2)
        with open("data/latest_metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=2)

    status = "PARTIAL" if partial else "COMPLETE"
    print(f"\n[{status}] Results saved to {game_path}")
    return game_path, metrics_path


def _save_checkpoint(game_state, experiment_meta, run_tag, round_number):
    """Save full game_state checkpoint after each round, supporting --resume.

    Stores the complete GameState (rounds + memory + totals) plus experiment_meta.
    A timed-out or crashed run can be resumed via:
        python -m engine.run --resume <run_tag>
    which loads this file, replays game_state, and continues from round N+1 with
    the original prompts + per-agent configs.
    """
    os.makedirs("data/checkpoints", exist_ok=True)
    checkpoint_path = f"data/checkpoints/{run_tag}.json"
    payload = {
        "schema_version": 2,
        "round": round_number,
        "rounds_completed": len(game_state.rounds),
        "_experiment": experiment_meta,
        "game_state": json.loads(game_state.model_dump_json()),
    }
    # Atomic write: tmp file + rename, so a crash mid-write can't corrupt the checkpoint.
    tmp = checkpoint_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, checkpoint_path)


def _load_checkpoint(run_tag):
    """Load the most recent checkpoint for `run_tag`. Returns (game_state, experiment_meta, last_round)."""
    checkpoint_path = f"data/checkpoints/{run_tag}.json"
    if not os.path.exists(checkpoint_path):
        return None, None, None
    with open(checkpoint_path) as f:
        payload = json.load(f)
    if payload.get("schema_version") != 2:
        # Old checkpoint format (just summary stats, no game_state) — can't resume from this.
        return None, None, None
    game_state = GameState.model_validate(payload["game_state"])
    return game_state, payload.get("_experiment"), payload.get("round", 0)


async def main():
    global _shutdown_requested

    parser = argparse.ArgumentParser(description="Run CRUCIBLE (Hardened)")
    parser.add_argument("--rounds", type=int, default=25, help="Number of rounds (default: 25)")
    parser.add_argument("--turns", type=int, default=2, help="Conversation turns per round")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (logged, not enforced on LLM)")
    parser.add_argument(
        "--prompt-mode",
        type=str,
        choices=["balanced_competitive", "hard_max", "tournament", "legacy"],
        default=None,
        help="Prompt strategy mode",
    )
    parser.add_argument(
        "--psychology-block",
        type=str,
        choices=["on", "off"],
        default=None,
        help="Include compact human psychology summary",
    )
    parser.add_argument(
        "--deception-policy",
        type=str,
        choices=["explicit", "implicit", "discourage"],
        default=None,
        help="How directly prompts allow strategic bluffing/deception",
    )
    parser.add_argument(
        "--no-reflection",
        action="store_true",
        default=False,
        help="Disable private reflection phase (for ablation studies)",
    )
    parser.add_argument(
        "--round-timeout",
        type=int,
        default=180,
        help="Timeout per round in seconds (default: 180)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (sets CRUCIBLE_TEMPERATURE for the run)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p / nucleus sampling (sets CRUCIBLE_TOP_P for the run)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens per generation (sets CRUCIBLE_MAX_TOKENS, default 1024)",
    )
    parser.add_argument(
        "--system-suffix-a",
        type=str,
        default="",
        help="Optional priming text appended to agent A's system prompt only (asymmetric priming).",
    )
    parser.add_argument(
        "--system-suffix-b",
        type=str,
        default="",
        help="Optional priming text appended to agent B's system prompt only (asymmetric priming).",
    )
    parser.add_argument(
        "--system-suffix-a-file",
        type=str,
        default=None,
        help="Path to a file whose contents become agent A's priming suffix. Overrides --system-suffix-a.",
    )
    parser.add_argument(
        "--system-suffix-b-file",
        type=str,
        default=None,
        help="Path to a file whose contents become agent B's priming suffix. Overrides --system-suffix-b.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume from a saved checkpoint (give the run_tag, e.g. "
            "'claude-sonnet-4-6_balanced_competitive_s1_20260505_154233'). "
            "Loads data/checkpoints/<tag>.json, replays the saved game_state, and "
            "continues from the round immediately after the last completed one. "
            "All other flags (--rounds, --turns, --prompt-mode, etc.) are taken "
            "from the checkpoint's experiment_meta to preserve experimental integrity."
        ),
    )
    args = parser.parse_args()
    # Resolve file-based suffixes (more readable for multi-line priming text)
    if args.system_suffix_a_file:
        with open(args.system_suffix_a_file) as _f:
            args.system_suffix_a = _f.read()
    if args.system_suffix_b_file:
        with open(args.system_suffix_b_file) as _f:
            args.system_suffix_b = _f.read()

    # Hyperparameter overrides go through env vars so the engine reads them per-call.
    if args.temperature is not None:
        os.environ["CRUCIBLE_TEMPERATURE"] = str(args.temperature)
    if args.top_p is not None:
        os.environ["CRUCIBLE_TOP_P"] = str(args.top_p)
    if args.max_tokens is not None:
        os.environ["CRUCIBLE_MAX_TOKENS"] = str(args.max_tokens)

    prompt_mode = args.prompt_mode or os.environ.get("CRUCIBLE_PROMPT_MODE", "balanced_competitive")
    psychology_block = args.psychology_block or os.environ.get("CRUCIBLE_PSYCHOLOGY_BLOCK", "on")
    deception_policy = args.deception_policy or os.environ.get("CRUCIBLE_DECEPTION_POLICY", "explicit")
    enable_reflection = not args.no_reflection

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)

    agents = get_agent_configs()
    matchup = agents["A"]["model"] != agents["B"]["model"]
    model_summary = (
        f"A={agents['A']['model']} ({agents['A']['provider']}) "
        f"B={agents['B']['model']} ({agents['B']['provider']})"
    )

    print(f"Running CRUCIBLE: {args.rounds} rounds, {args.turns} conversation turns each")
    print(f"Prompt mode: {prompt_mode} | Psychology: {psychology_block} | Deception: {deception_policy}")
    print(f"Reflection: {'ON' if enable_reflection else 'OFF (ablation)'} | Seed: {args.seed}")
    print(f"Models: {model_summary}")
    print(f"Round timeout: {args.round_timeout}s | Ctrl+C for graceful shutdown")
    print("=" * 60)

    # === PREFLIGHT (sync, one round-trip per unique provider/model) ===
    ok, messages = preflight()
    for m in messages:
        print(m)
    if not ok:
        sys.exit(1)

    # === BUILD EXPERIMENT META (or restore from checkpoint when --resume) ===
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    def _safe(s: str) -> str: return s.replace("/", "_")

    if args.resume:
        # Resume mode: load the checkpoint and reuse its experiment_meta + run_tag exactly.
        # Don't override any flags — use the original config so the resumed run is
        # statistically equivalent to one that hadn't crashed.
        run_tag = args.resume
        ck_state, ck_meta, ck_round = _load_checkpoint(run_tag)
        if ck_state is None:
            print(f"[ABORT] No resumable checkpoint found at data/checkpoints/{run_tag}.json")
            print("(Either the file doesn't exist or it was written under the old schema.)")
            sys.exit(1)
        if ck_meta is None:
            print(f"[ABORT] Checkpoint {run_tag} has no experiment_meta — cannot resume safely.")
            sys.exit(1)
        # Restore game_state and experiment_meta from the checkpoint.
        game_state_resumed = ck_state
        experiment_meta = dict(ck_meta)
        experiment_meta.setdefault("resume_history", []).append({
            "resumed_at": timestamp,
            "from_round": ck_round,
        })
        # Re-derive args from meta so the rest of the function reads consistent values.
        args.rounds = experiment_meta.get("rounds", args.rounds)
        args.turns = experiment_meta.get("conversation_turns", args.turns)
        args.seed = experiment_meta.get("seed", args.seed)
        prompt_mode = experiment_meta.get("prompt_mode", prompt_mode)
        psychology_block = experiment_meta.get("psychology_block", psychology_block)
        deception_policy = experiment_meta.get("deception_policy", deception_policy)
        enable_reflection = experiment_meta.get("enable_reflection", enable_reflection)
        # Sampling params — reapply to env so engine.game uses them.
        if experiment_meta.get("temperature") is not None:
            os.environ["CRUCIBLE_TEMPERATURE"] = str(experiment_meta["temperature"])
        if experiment_meta.get("top_p") is not None:
            os.environ["CRUCIBLE_TOP_P"] = str(experiment_meta["top_p"])
        if experiment_meta.get("max_tokens") is not None:
            os.environ["CRUCIBLE_MAX_TOKENS"] = str(experiment_meta["max_tokens"])
        # Asymmetric priming — reapply.
        args.system_suffix_a = experiment_meta.get("system_suffix_a") or ""
        args.system_suffix_b = experiment_meta.get("system_suffix_b") or ""
        starting_round = ck_round + 1
        print(f"[resume] {run_tag} — checkpoint at round {ck_round}, "
              f"continuing from round {starting_round}/{args.rounds}")
    else:
        # Include hyperparameter / asymmetric markers in run_tag so concurrent runs of the same
        # model/prompt/seed don't collide on identical timestamps. Earlier bug: launching H (T=0.7)
        # and I (T=1.3) at the same wall-clock second produced identical tags, and I's save
        # overwrote H's. Now T= and P= and asym tags differentiate.
        asym_tag = "_asym" if (args.system_suffix_a or args.system_suffix_b) else ""
        temp_tag = f"_T{args.temperature}" if args.temperature is not None else ""
        top_p_tag = f"_P{args.top_p}" if args.top_p is not None else ""
        refl_tag = "_norefl" if not enable_reflection else ""
        extras = f"{asym_tag}{temp_tag}{top_p_tag}{refl_tag}"
        if matchup:
            run_tag = f"{_safe(model_name)}_vs_{_safe(model_name_b)}_{prompt_mode}{extras}_s{args.seed or 'none'}_{timestamp}"
        else:
            run_tag = f"{_safe(model_name)}_{prompt_mode}{extras}_s{args.seed or 'none'}_{timestamp}"
        experiment_meta = {
            "timestamp": timestamp,
            "model": model_name,            # backward-compat: A's model
            "model_a": agents["A"]["model"],
            "provider_a": agents["A"]["provider"],
            "model_b": agents["B"]["model"],
            "provider_b": agents["B"]["provider"],
            "matchup": matchup,
            "seed": args.seed,
            "rounds": args.rounds,
            "conversation_turns": args.turns,
            "prompt_mode": prompt_mode,
            "psychology_block": psychology_block,
            "deception_policy": deception_policy,
            "enable_reflection": enable_reflection,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "asymmetric_priming": bool(args.system_suffix_a or args.system_suffix_b),
            "system_suffix_a": args.system_suffix_a or None,
            "system_suffix_b": args.system_suffix_b or None,
        }
        starting_round = 1
        game_state_resumed = None

    # Begin spend tracking for this run.
    _spend.start_run(run_tag)

    # === GET PROMPTS ===
    prompts = get_prompt_bundle(
        prompt_mode=prompt_mode,
        psychology_block=psychology_block,
        deception_policy=deception_policy,
    )

    # === GAME LOOP WITH PER-ROUND CHECKPOINTING ===
    game_state = game_state_resumed if game_state_resumed is not None else GameState()
    completed_rounds = len(game_state.rounds)
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3

    for round_n in range(starting_round, args.rounds + 1):
        if _shutdown_requested:
            print(f"\n[SHUTDOWN] Stopping after round {round_n - 1}.")
            break

        try:
            base_system = prompts["system_prompt"]
            sys_a = (base_system + ("\n\n" + args.system_suffix_a if args.system_suffix_a else ""))
            sys_b = (base_system + ("\n\n" + args.system_suffix_b if args.system_suffix_b else ""))
            await asyncio.wait_for(
                run_round(
                    game_state=game_state,
                    round_number=round_n,
                    total_rounds=args.rounds,
                    conversation_turns=args.turns,
                    on_update=on_round_complete,
                    system_prompt=base_system,
                    system_prompt_a=sys_a if args.system_suffix_a else "",
                    system_prompt_b=sys_b if args.system_suffix_b else "",
                    game_prompt=prompts["game_prompt"],
                    choice_prompt=prompts["choice_prompt"],
                    reflection_prompt=prompts["reflection_prompt"],
                    enable_reflection=enable_reflection,
                ),
                timeout=args.round_timeout,
            )
            completed_rounds = round_n
            consecutive_failures = 0
            _save_checkpoint(game_state, experiment_meta, run_tag, round_n)

        except asyncio.TimeoutError:
            consecutive_failures += 1
            print(f"\n[TIMEOUT] Round {round_n} exceeded {args.round_timeout}s (failure {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"[ABORT] {MAX_CONSECUTIVE_FAILURES} consecutive timeouts. Saving partial results.")
                break

        except Exception as e:
            consecutive_failures += 1
            print(f"\n[ERROR] Round {round_n}: {e} (failure {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"[ABORT] {MAX_CONSECUTIVE_FAILURES} consecutive errors. Saving partial results.")
                break

    # === COMPUTE METRICS ===
    is_partial = completed_rounds < args.rounds
    metrics = None

    if completed_rounds > 0:
        print(f"\nComputing metrics for {completed_rounds} rounds...")
        embedder = None
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading embedding model for language drift...")
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            print("WARNING: sentence-transformers not installed, language drift = 0.0")
        except Exception as e:
            print(f"WARNING: embedder failed: {e}")

        metrics = compute_all_metrics(game_state, embedder=embedder)

        # Update experiment meta with final stats
        ambiguous_count = sum(
            1 for r in game_state.rounds
            if r.agent_a_ambiguous_parse or r.agent_b_ambiguous_parse
        )
        # Close spend tracking and pull totals into experiment_meta.
        spend_record = _spend.end_run()
        spend_summary = {
            "calls": spend_record.calls if spend_record else 0,
            "input_tokens": spend_record.input_tokens if spend_record else 0,
            "output_tokens": spend_record.output_tokens if spend_record else 0,
            "cache_read_input_tokens": spend_record.cache_read_input_tokens if spend_record else 0,
            "cost_usd": spend_record.cost_usd if spend_record else 0.0,
        }
        experiment_meta.update({
            "completed_rounds": completed_rounds,
            "ambiguous_parses": ambiguous_count,
            "cooperation_rate": metrics.cooperation_rate,
            "mutual_destruction_rate": metrics.mutual_destruction_rate,
            "deception_index": metrics.deception_index,
            "spend": spend_summary,
        })

        # === SAVE ===
        game_path, metrics_path = _save_results(
            game_state, metrics, experiment_meta, run_tag, partial=is_partial
        )

        # === VERIFY OUTPUT ===
        game_size = os.path.getsize(game_path)
        metrics_size = os.path.getsize(metrics_path)
        if game_size < 100:
            print(f"[WARN] Game file suspiciously small: {game_size} bytes")
        if metrics_size < 100:
            print(f"[WARN] Metrics file suspiciously small: {metrics_size} bytes")

        # === SUMMARY ===
        print(f"\n{'='*60}")
        print(f"{'PARTIAL RUN' if is_partial else 'COMPLETE'}: {completed_rounds}/{args.rounds} rounds")
        print(f"Deception Index: {metrics.deception_index:.1f}")
        print(f"Cooperation Rate: {metrics.cooperation_rate:.1%}")
        print(f"Mutual Destruction Rate: {metrics.mutual_destruction_rate:.1%}")
        if ambiguous_count:
            print(f"Ambiguous parses: {ambiguous_count}")
        if spend_summary["calls"]:
            print(
                f"Spend: ${spend_summary['cost_usd']:.4f} across {spend_summary['calls']} calls "
                f"(in={spend_summary['input_tokens']} out={spend_summary['output_tokens']})"
            )
        print(f"Results: {game_path}")

        # Cleanup checkpoint on successful complete run
        checkpoint_path = f"data/checkpoints/{run_tag}.json"
        if not is_partial and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
    else:
        # Still flush any spend incurred during preflight / aborted attempts.
        _spend.end_run()
        print("\n[ABORT] No rounds completed. No output saved.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
