"""
CRUCIBLE Runner

Execute a full game and save results.
Usage: python -m engine.run [--rounds 100] [--turns 3]
"""

import asyncio
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.game import run_game
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
    parser.add_argument("--turns", type=int, default=3, help="Conversation turns per round")
    args = parser.parse_args()

    # Initialize Datadog LLM Observability
    init_all()

    print(f"Running CRUCIBLE: {args.rounds} rounds, {args.turns} conversation turns each")
    print("=" * 60)

    game_state = await run_game(
        total_rounds=args.rounds,
        conversation_turns=args.turns,
        on_update=on_round_complete,
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

    # Save results
    os.makedirs("data", exist_ok=True)
    with open("data/latest_game.json", "w") as f:
        f.write(game_state.model_dump_json(indent=2))
    with open("data/latest_metrics.json", "w") as f:
        f.write(metrics.model_dump_json(indent=2))

    # Log rounds to Braintrust
    for r, m in zip(game_state.rounds, metrics.rounds):
        bt_log_round(r, m)

    print(f"\nDeception Index: {metrics.deception_index:.1f}")
    print(f"Cooperation Rate: {metrics.cooperation_rate:.1%}")
    print(f"Mutual Destruction Rate: {metrics.mutual_destruction_rate:.1%}")
    print(f"\nResults saved to data/")


if __name__ == "__main__":
    asyncio.run(main())
