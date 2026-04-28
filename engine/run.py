"""
CRUCIBLE Runner

Execute a full game and save results with experiment metadata.
Usage: python -m engine.run [--rounds 100] [--turns 3] [--seed 42]
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.game import run_game, model_name
from engine.metrics import compute_all_metrics
from engine.instrumentation import init_all, bt_log_round


def on_round_complete(round_state):
    """Callback for each completed round."""
    r = round_state
    print(
        f"Round {r.round_number}: "
        f"A={r.agent_a_choice.upper()} B={r.agent_b_choice.upper()} "
        f"| A=${r.agent_a_total} B=${r.agent_b_total}"
    )


async def main():
    parser = argparse.ArgumentParser(description="Run CRUCIBLE")
    parser.add_argument("--rounds", type=int, default=100, help="Number of rounds")
    parser.add_argument("--turns", type=int, default=2, help="Conversation turns per round")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (logged, not enforced on LLM)")
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
    args = parser.parse_args()

    prompt_mode = args.prompt_mode or os.environ.get("CRUCIBLE_PROMPT_MODE", "balanced_competitive")
    psychology_block = args.psychology_block or os.environ.get("CRUCIBLE_PSYCHOLOGY_BLOCK", "on")
    deception_policy = args.deception_policy or os.environ.get("CRUCIBLE_DECEPTION_POLICY", "explicit")
    enable_reflection = not args.no_reflection

    # Initialize Datadog LLM Observability
    init_all()

    print(f"Running CRUCIBLE: {args.rounds} rounds, {args.turns} conversation turns each")
    print(f"Prompt mode: {prompt_mode} | Psychology block: {psychology_block} | Deception policy: {deception_policy}")
    print(f"Reflection: {'on' if enable_reflection else 'OFF (ablation)'} | Model: {model_name} | Seed: {args.seed}")
    print("=" * 60)

    game_state = await run_game(
        total_rounds=args.rounds,
        conversation_turns=args.turns,
        on_update=on_round_complete,
        prompt_mode=prompt_mode,
        psychology_block=psychology_block,
        deception_policy=deception_policy,
        enable_reflection=enable_reflection,
    )

    # Compute metrics (with language drift embeddings)
    print("\nComputing metrics...")
    embedder = None
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model for language drift...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedding model loaded.")
    except ImportError:
        print("WARNING: sentence-transformers not installed, language drift will be 0.0")
    except Exception as e:
        print(f"WARNING: embedder failed to load: {e}")

    metrics = compute_all_metrics(game_state, embedder=embedder)

    # Count ambiguous parses for the summary
    ambiguous_count = sum(
        1 for r in game_state.rounds
        if r.agent_a_ambiguous_parse or r.agent_b_ambiguous_parse
    )

    # Build experiment metadata
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
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
        "ambiguous_parses": ambiguous_count,
        "cooperation_rate": metrics.cooperation_rate,
        "mutual_destruction_rate": metrics.mutual_destruction_rate,
        "deception_index": metrics.deception_index,
    }

    # Save results: both latest (for quick access) and timestamped (for reproducibility)
    os.makedirs("data/runs", exist_ok=True)
    run_tag = f"{model_name}_{prompt_mode}_s{args.seed or 'none'}_{timestamp}"

    game_data = json.loads(game_state.model_dump_json())
    game_data["_experiment"] = experiment_meta

    metrics_data = json.loads(metrics.model_dump_json())
    metrics_data["_experiment"] = experiment_meta

    # Timestamped copies
    with open(f"data/runs/{run_tag}_game.json", "w") as f:
        json.dump(game_data, f, indent=2)
    with open(f"data/runs/{run_tag}_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)

    # Latest symlinks (overwritten each run)
    os.makedirs("data", exist_ok=True)
    with open("data/latest_game.json", "w") as f:
        json.dump(game_data, f, indent=2)
    with open("data/latest_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)

    # Log rounds to Braintrust
    for r, m in zip(game_state.rounds, metrics.rounds):
        bt_log_round(r, m)

    print(f"\nDeception Index: {metrics.deception_index:.1f}")
    print(f"Cooperation Rate: {metrics.cooperation_rate:.1%}")
    print(f"Mutual Destruction Rate: {metrics.mutual_destruction_rate:.1%}")
    if ambiguous_count:
        print(f"Ambiguous parses: {ambiguous_count} (check logs above)")
    print(f"\nResults saved to data/runs/{run_tag}_*.json")


if __name__ == "__main__":
    asyncio.run(main())
