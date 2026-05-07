#!/usr/bin/env python3
"""Generate the matchup chart for RESULTS.md finding 8.

Bar chart with 4 matchup cells (Sonnet vs {Hermes,DeepSeek} x refl on/off)
plus a horizontal reference line for Sonnet's single-model T2 baseline.

Output: results_matchup_chart.png at repo root.
"""
from __future__ import annotations
import glob, json, math, os, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}


def load_t2_runs():
    """T2 hard_max + 3 turns runs only (matchup design)."""
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data/runs/*_metrics.json"))):
        if "_PARTIAL" in path: continue
        try:
            with open(path) as f: d = json.load(f)
        except Exception: continue
        exp = d.get("_experiment") or {}
        if exp.get("prompt_mode") != "hard_max": continue
        if exp.get("conversation_turns") != 3: continue
        if exp.get("temperature") is not None: continue
        if exp.get("asymmetric_priming"): continue
        ma = exp.get("model_a", exp.get("model", ""))
        mb = exp.get("model_b", ma)
        refl = "OFF" if not exp.get("enable_reflection", True) else "ON"
        out.append({
            "model_a": ma, "model_b": mb,
            "matchup": ma != mb,
            "refl": refl,
            "coop": d.get("cooperation_rate", 0) * 100,
        })
    return out


def cell_ci(vals: list[float]) -> tuple[float, float, float]:
    n = len(vals)
    if n == 0: return (0, 0, 0)
    if n == 1: return (vals[0], 0, 0)
    m = statistics.mean(vals)
    sd = statistics.stdev(vals)
    t = T_CRIT.get(n - 1, 2.0)
    half = t * sd / math.sqrt(n)
    return m, max(0, m - half), min(100, m + half)


def main():
    runs = load_t2_runs()

    # Single-model Sonnet T2 baselines (reference)
    s_off = [r["coop"] for r in runs if not r["matchup"] and r["model_a"] == "claude-sonnet-4-6" and r["refl"] == "OFF"]
    s_on  = [r["coop"] for r in runs if not r["matchup"] and r["model_a"] == "claude-sonnet-4-6" and r["refl"] == "ON"]

    # Matchup cells
    cells = [
        ("Sonnet vs\nHermes 4 70B\nrefl-OFF",
         [r["coop"] for r in runs if r["matchup"] and r["model_b"].endswith("hermes-4-70b") and r["refl"] == "OFF"],
         "#ff7f0e"),
        ("Sonnet vs\nHermes 4 70B\nrefl-ON",
         [r["coop"] for r in runs if r["matchup"] and r["model_b"].endswith("hermes-4-70b") and r["refl"] == "ON"],
         "#ff7f0e"),
        ("Sonnet vs\nDeepSeek v3.1\nrefl-OFF",
         [r["coop"] for r in runs if r["matchup"] and r["model_b"].endswith("chat-v3.1") and r["refl"] == "OFF"],
         "#d62728"),
        ("Sonnet vs\nDeepSeek v3.1\nrefl-ON",
         [r["coop"] for r in runs if r["matchup"] and r["model_b"].endswith("chat-v3.1") and r["refl"] == "ON"],
         "#d62728"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.suptitle(
        "Cross-model matchup cooperation rate (Tier 2 hard_max + 3 turns, n=3 per cell)",
        fontsize=12, fontweight="bold",
    )

    x = list(range(len(cells)))
    means, lows, highs, ns, colors = [], [], [], [], []
    for label, vals, color in cells:
        m, lo, hi = cell_ci(vals)
        means.append(m)
        lows.append(m - lo)
        highs.append(hi - m)
        ns.append(len(vals))
        colors.append(color)

    bars = ax.bar(x, means, yerr=[lows, highs], capsize=8,
                   color=colors, alpha=0.85, edgecolor="black", linewidth=0.7)
    for i, (bar, n) in enumerate(zip(bars, ns)):
        ax.text(bar.get_x() + bar.get_width()/2, means[i] + 2,
                f"n={n}", ha="center", fontsize=9)
        ax.text(bar.get_x() + bar.get_width()/2, means[i] / 2,
                f"{means[i]:.1f}%", ha="center", fontsize=11, fontweight="bold",
                color="white")

    # Sonnet single-model baselines (reference lines)
    if s_off:
        m_off = statistics.mean(s_off)
        ax.axhline(m_off, color="#1f77b4", linestyle="--", linewidth=1.3, alpha=0.8)
        ax.text(3.6, m_off + 1, f"Sonnet self-play OFF\n{m_off:.1f}% (n={len(s_off)})",
                fontsize=8, color="#1f77b4", ha="right", va="bottom")
    if s_on:
        m_on = statistics.mean(s_on)
        ax.axhline(m_on, color="#1f77b4", linestyle=":", linewidth=1.3, alpha=0.8)
        ax.text(3.6, m_on + 1, f"Sonnet self-play ON\n{m_on:.1f}% (n={len(s_on)})",
                fontsize=8, color="#1f77b4", ha="right", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cells], fontsize=9)
    ax.set_ylabel("Cooperation rate (%, mean ± 95% CI)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = os.path.join(ROOT, "results_matchup_chart.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
