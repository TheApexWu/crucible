"""
Compare behavior between legacy and balanced competitive prompt modes.

Usage:
  venv/bin/python scripts/compare_prompt_modes.py --rounds 25 --turns 2
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = DATA_DIR / "compare_prompt_modes"


def _run_mode(mode: str, rounds: int, turns: int) -> dict:
    cmd = [
        "venv/bin/python",
        "-m",
        "engine.run",
        "--rounds",
        str(rounds),
        "--turns",
        str(turns),
        "--prompt-mode",
        mode,
    ]
    print(f"Running mode={mode}...")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    with open(DATA_DIR / "latest_metrics.json") as f:
        metrics = json.load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DATA_DIR / "latest_game.json", OUT_DIR / f"{mode}_game.json")
    shutil.copyfile(DATA_DIR / "latest_metrics.json", OUT_DIR / f"{mode}_metrics.json")

    return metrics


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _avg_betrayal_when_opp_split(metrics: dict) -> float:
    agents = (metrics.get("strategy") or {}).get("agents") or []
    if not agents:
        return 0.0
    values = [a.get("betrayal_rate_when_opponent_split", 0.0) for a in agents]
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare legacy vs balanced competitive prompt modes")
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--turns", type=int, default=2)
    args = parser.parse_args()

    legacy = _run_mode("legacy", args.rounds, args.turns)
    balanced = _run_mode("balanced_competitive", args.rounds, args.turns)

    legacy_coop = float(legacy.get("cooperation_rate", 0.0))
    balanced_coop = float(balanced.get("cooperation_rate", 0.0))
    legacy_md = float(legacy.get("mutual_destruction_rate", 0.0))
    balanced_md = float(balanced.get("mutual_destruction_rate", 0.0))
    legacy_di = float(legacy.get("deception_index", 0.0))
    balanced_di = float(balanced.get("deception_index", 0.0))
    legacy_betr = _avg_betrayal_when_opp_split(legacy)
    balanced_betr = _avg_betrayal_when_opp_split(balanced)

    print("\n=== Prompt Mode Comparison ===")
    print(f"Cooperation rate: legacy={_pct(legacy_coop)} | balanced={_pct(balanced_coop)} | delta={_pct(balanced_coop - legacy_coop)}")
    print(f"Mutual destruction: legacy={_pct(legacy_md)} | balanced={_pct(balanced_md)} | delta={_pct(balanced_md - legacy_md)}")
    print(f"Deception index: legacy={legacy_di:.2f} | balanced={balanced_di:.2f} | delta={balanced_di - legacy_di:.2f}")
    print(f"Avg betrayal-vs-split: legacy={legacy_betr:.3f} | balanced={balanced_betr:.3f} | delta={balanced_betr - legacy_betr:.3f}")
    print(f"Saved per-mode artifacts under {OUT_DIR}")


if __name__ == "__main__":
    main()
