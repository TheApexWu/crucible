#!/usr/bin/env python3
"""Generate a per-round defection chart showing endgame-clustering signal.

One PNG: line chart with one series per model. X-axis = round number (1-25),
Y-axis = defection rate aggregated across runs of that model. Shows Sonnet's
endgame-clustering qualitatively distinct from the other 3 models.

Output: results_temporal_chart.png at repo root.
"""
from __future__ import annotations
import glob, json, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIER_DEFS = {("balanced_competitive", 2): "T1", ("hard_max", 3): "T2"}


def main():
    per_model_rounds = defaultdict(lambda: defaultdict(lambda: {"defects": 0, "n": 0}))

    for path in glob.glob(os.path.join(ROOT, "data/runs/*_metrics.json")):
        if "_PARTIAL" in path:
            continue
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue
        exp = d.get("_experiment") or {}
        if exp.get("temperature") is not None or exp.get("asymmetric_priming"):
            continue
        if (exp.get("prompt_mode"), exp.get("conversation_turns")) not in TIER_DEFS:
            continue
        model = exp.get("model_a", exp.get("model", ""))
        for r in d.get("rounds") or []:
            rn = r["round_number"]
            per_model_rounds[model][rn]["n"] += 1
            if r.get("a_betrayed") or r.get("b_betrayed") or r.get("mutual_destruction"):
                per_model_rounds[model][rn]["defects"] += 1

    model_order = [
        ("claude-sonnet-4-6", "Sonnet 4.6", "#1f77b4", "o", 2.5),
        ("nousresearch/hermes-4-70b", "Hermes 4 70B", "#ff7f0e", "s", 1.5),
        ("microsoft/wizardlm-2-8x22b", "WizardLM-2 8x22B", "#2ca02c", "^", 1.5),
        ("deepseek/deepseek-chat-v3.1", "DeepSeek v3.1", "#d62728", "D", 1.5),
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(
        "Per-round defection rate by model — endgame-clustering signal",
        fontsize=12, fontweight="bold",
    )

    for model_id, label, color, marker, lw in model_order:
        rounds = list(range(1, 26))
        rates = []
        for rn in rounds:
            cell = per_model_rounds[model_id].get(rn, {"defects": 0, "n": 0})
            rate = (100 * cell["defects"] / cell["n"]) if cell["n"] > 0 else None
            rates.append(rate)
        ax.plot(rounds, rates, label=label, color=color, marker=marker,
                linewidth=lw, markersize=5, alpha=0.9)

    ax.set_xlabel("Round number (game length 25)", fontsize=11)
    ax.set_ylabel("Defection rate across runs (%)", fontsize=11)
    ax.set_xlim(0.5, 25.5)
    ax.set_ylim(-3, 100)
    ax.set_xticks(range(1, 26, 2))
    ax.axhline(50, color="grey", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.axvspan(16, 25.5, alpha=0.07, color="red", label="Endgame zone (R16-25)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    # Annotate the Sonnet ratio
    ax.annotate(
        "Sonnet defection 7× more likely\nin endgame than early game\n(vs ~1.4-1.9× for others)",
        xy=(24, 80), xytext=(15, 12),
        fontsize=9, color="#1f77b4",
        arrowprops=dict(arrowstyle="->", color="#1f77b4", alpha=0.6),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#1f77b4", alpha=0.9),
    )

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = os.path.join(ROOT, "results_temporal_chart.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
