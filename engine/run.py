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

from engine.game import run_round, model_name, get_prompt_bundle, GameState
from engine.metrics import compute_all_metrics


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
    """Save lightweight checkpoint after each round."""
    os.makedirs("data/checkpoints", exist_ok=True)
    checkpoint_path = f"data/checkpoints/{run_tag}.json"
    checkpoint = {
        "round": round_number,
        "agent_a_total": game_state.agent_a_total,
        "agent_b_total": game_state.agent_b_total,
        "rounds_completed": len(game_state.rounds),
        "_experiment": experiment_meta,
    }
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f)


async def main():
    global _shutdown_requested

    parser = argparse.ArgumentParser(description="Run CRUCIBLE (Hardened)")
    parser.add_argument("--rounds", type=int, default=25, help="Number of rounds (default: 25)")
    parser.add_argument("--turns", type=int, default=2, help="Conversation turns per round")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (logged, not enforced on LLM)")
    parser.add_argument(
        "--prompt-mode",
        type=str,
        choices=["balanced_competitive", "hard_max", "legacy"],
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
    args = parser.parse_args()

    prompt_mode = args.prompt_mode or os.environ.get("CRUCIBLE_PROMPT_MODE", "balanced_competitive")
    psychology_block = args.psychology_block or os.environ.get("CRUCIBLE_PSYCHOLOGY_BLOCK", "on")
    deception_policy = args.deception_policy or os.environ.get("CRUCIBLE_DECEPTION_POLICY", "explicit")
    enable_reflection = not args.no_reflection

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)

    print(f"Running CRUCIBLE: {args.rounds} rounds, {args.turns} conversation turns each")
    print(f"Prompt mode: {prompt_mode} | Psychology: {psychology_block} | Deception: {deception_policy}")
    print(f"Reflection: {'ON' if enable_reflection else 'OFF (ablation)'} | Model: {model_name} | Seed: {args.seed}")
    print(f"Round timeout: {args.round_timeout}s | Ctrl+C for graceful shutdown")
    print("=" * 60)

    # === PREFLIGHT (synchronous, no retries, no event loop tricks) ===
    print(f"[preflight] Testing {model_name}...", end=" ", flush=True)
    try:
        from engine.game import _provider, _get_openai_client, _get_anthropic_client, _get_gemini_client
        import httpx

        if _provider == "google":
            client = _get_gemini_client()
            # Sync call via httpx
            resp = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
                json={"contents": [{"parts": [{"text": "Say OK."}]}]},
                params={"key": os.environ["GEMINI_API_KEY"]},
                timeout=30,
            )
            if resp.status_code == 200:
                print(f"OK (google, {resp.status_code})")
            else:
                print(f"FAILED (google, {resp.status_code}: {resp.text[:200]})")
                sys.exit(1)
        elif _provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            resp = client.messages.create(
                model=model_name, max_tokens=10,
                messages=[{"role": "user", "content": "Say OK."}],
            )
            print(f"OK ({len(resp.content[0].text)} chars)")
        elif _provider == "openai":
            import openai
            if model_name.startswith("deepseek"):
                client = openai.OpenAI(
                    api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url="https://api.deepseek.com",
                )
            else:
                client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            resp = client.chat.completions.create(
                model=model_name, max_completion_tokens=10,
                messages=[{"role": "user", "content": "Say OK."}],
            )
            print(f"OK ({len(resp.choices[0].message.content)} chars)")
        else:
            print(f"FAILED (unknown provider: {_provider})")
            sys.exit(1)
    except Exception as e:
        print(f"FAILED ({e})")
        sys.exit(1)

    # === BUILD EXPERIMENT META ===
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_tag = f"{model_name}_{prompt_mode}_s{args.seed or 'none'}_{timestamp}"
    experiment_meta = {
        "timestamp": timestamp,
        "model": model_name,
        "seed": args.seed,
        "rounds": args.rounds,
        "conversation_turns": args.turns,
        "prompt_mode": prompt_mode,
        "psychology_block": psychology_block,
        "deception_policy": deception_policy,
        "enable_reflection": enable_reflection,
    }

    # === GET PROMPTS ===
    prompts = get_prompt_bundle(
        prompt_mode=prompt_mode,
        psychology_block=psychology_block,
        deception_policy=deception_policy,
    )

    # === GAME LOOP WITH PER-ROUND CHECKPOINTING ===
    game_state = GameState()
    completed_rounds = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3

    for round_n in range(1, args.rounds + 1):
        if _shutdown_requested:
            print(f"\n[SHUTDOWN] Stopping after round {round_n - 1}.")
            break

        try:
            await asyncio.wait_for(
                run_round(
                    game_state=game_state,
                    round_number=round_n,
                    total_rounds=args.rounds,
                    conversation_turns=args.turns,
                    on_update=on_round_complete,
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
        experiment_meta.update({
            "completed_rounds": completed_rounds,
            "ambiguous_parses": ambiguous_count,
            "cooperation_rate": metrics.cooperation_rate,
            "mutual_destruction_rate": metrics.mutual_destruction_rate,
            "deception_index": metrics.deception_index,
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
        print(f"Results: {game_path}")

        # Cleanup checkpoint on successful complete run
        checkpoint_path = f"data/checkpoints/{run_tag}.json"
        if not is_partial and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
    else:
        print("\n[ABORT] No rounds completed. No output saved.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
