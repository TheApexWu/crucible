#!/usr/bin/env python3
"""Generate the headline results chart for RESULTS.md.

One PNG: 4-panel bar chart showing cooperation% per (model × reflection × tier)
with 95% CI error bars from the sweep data.

Reads the saved metrics JSONs in data/runs/, groups by (model, prompt_mode, turns,
refl), computes mean + SD + 95% CI per cell, draws bars with error bars.

Output: results_main_chart.png at repo root (committed).
"""
from __future__ import annotations
import glob, json, math, os, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def load_runs():
    out = []
    for path in glob.glob(os.path.join(ROOT, "data/runs/*_metrics.json")):
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue
        exp = d.get("_experiment") or {}
        out.append({
            "model": exp.get("model_a", exp.get("model", "")),
            "prompt": exp.get("prompt_mode", ""),
            "turns": exp.get("conversation_turns", 0),
            "rounds": exp.get("rounds", 0),
            "seed": exp.get("seed"),
            "refl": "OFF" if not exp.get("enable_reflection", True) else "ON",
            "coop": d.get("cooperation_rate", 0) * 100,
            "temp": exp.get("temperature"),
            "asym": exp.get("asymmetric_priming", False),
            "partial": d.get("_partial", False),
        })
    return out


def cell_stats(coops):
    n = len(coops)
    if n == 0:
        return None
    if n == 1:
        return {"mean": coops[0], "n": 1, "ci_low": coops[0], "ci_high": coops[0]}
    mean = statistics.mean(coops)
    sd = statistics.stdev(coops)
    se = sd / math.sqrt(n)
    tc = T_CRIT.get(n - 1, 2.0)
    ci = tc * se
    return {"mean": mean, "n": n, "sd": sd, "ci_low": mean - ci, "ci_high": mean + ci}


def filter_vanilla(runs, prompt, turns):
    """Vanilla = no temperature override and no asymmetric priming."""
    return [
        r for r in runs
        if r["prompt"] == prompt and r["turns"] == turns and r["rounds"] == 25
        and r["temp"] is None and not r["asym"]
    ]


def main():
    runs = load_runs()

    tier1 = filter_vanilla(runs, "balanced_competitive", 2)
    tier2 = filter_vanilla(runs, "hard_max", 3)

    # Models we want to plot. Order matters for the visual story.
    model_order = [
        "claude-sonnet-4-6",
        "nousresearch/hermes-4-70b",
        "microsoft/wizardlm-2-8x22b",
        "deepseek/deepseek-chat-v3.1",
        "gemini-2.5-flash",
    ]
    model_label = {
        "claude-sonnet-4-6": "Sonnet 4.6",
        "nousresearch/hermes-4-70b": "Hermes 4 70B",
        "microsoft/wizardlm-2-8x22b": "WizardLM-2 8x22B",
        "deepseek/deepseek-chat-v3.1": "DeepSeek v3.1",
        "gemini-2.5-flash": "Gemini 2.5 (prior)",
    }

    def collect(tier_runs, refl):
        per_model = {}
        for r in tier_runs:
            if r["refl"] != refl:
                continue
            per_model.setdefault(r["model"], []).append(r["coop"])
        return {m: cell_stats(coops) for m, coops in per_model.items()}

    t1_off = collect(tier1, "OFF")
    t1_on = collect(tier1, "ON")
    t2_off = collect(tier2, "OFF")
    t2_on = collect(tier2, "ON")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(
        "CRUCIBLE Tier 1 (replication) vs Tier 2 (hard_max + 3 turns) — "
        "cooperation rate by model × reflection (n=3 per cell, 95% CI bars)",
        fontsize=12, fontweight="bold",
    )

    bar_w = 0.35
    color_off = "#2c7fb8"   # blue for refl OFF
    color_on = "#d95f02"    # orange for refl ON

    for ax, (off_dict, on_dict, tier_label) in [
        (axes[0], (t1_off, t1_on, "Tier 1 — balanced_competitive · 2 turns")),
        (axes[1], (t2_off, t2_on, "Tier 2 — hard_max · 3 turns")),
    ]:
        models_present = [m for m in model_order if m in off_dict or m in on_dict]
        x = list(range(len(models_present)))

        off_means, off_errs, on_means, on_errs = [], [], [], []
        off_ns, on_ns = [], []
        for m in models_present:
            o = off_dict.get(m)
            n = on_dict.get(m)
            off_means.append(o["mean"] if o else 0)
            off_errs.append((o["mean"] - o["ci_low"]) if (o and o["n"] >= 2) else 0)
            off_ns.append(o["n"] if o else 0)
            on_means.append(n["mean"] if n else 0)
            on_errs.append((n["mean"] - n["ci_low"]) if (n and n["n"] >= 2) else 0)
            on_ns.append(n["n"] if n else 0)

        # Clip error bars at the [0, 100] cooperation-rate bound — CIs that exceed this are
        # formally invalid (t-distribution assumes unbounded data). The clip preserves
        # signal direction while keeping the plot honest about the bound.
        def clip_err(means, errs):
            err_lo = [min(e, m) for m, e in zip(means, errs)]
            err_hi = [min(e, 100 - m) for m, e in zip(means, errs)]
            return [err_lo, err_hi]

        bars_off = ax.bar([xi - bar_w/2 for xi in x], off_means, bar_w,
                          yerr=clip_err(off_means, off_errs), label="Reflection OFF",
                          color=color_off, capsize=4, edgecolor="black", linewidth=0.5)
        bars_on = ax.bar([xi + bar_w/2 for xi in x], on_means, bar_w,
                         yerr=clip_err(on_means, on_errs), label="Reflection ON",
                         color=color_on, capsize=4, edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([model_label.get(m, m) for m in models_present], rotation=20, ha="right")
        ax.set_ylim(0, 110)
        ax.set_ylabel("Cooperation rate (%)") if ax == axes[0] else None
        ax.set_title(tier_label, fontsize=11)
        ax.axhline(50, color="grey", linewidth=0.5, linestyle=":", alpha=0.6)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)

        # Annotate n above each bar
        for bars, ns in [(bars_off, off_ns), (bars_on, on_ns)]:
            for b, n in zip(bars, ns):
                if n > 0:
                    ax.text(b.get_x() + b.get_width() / 2,
                            b.get_height() + 2,
                            f"n={n}", ha="center", va="bottom", fontsize=8, color="dimgray")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = os.path.join(ROOT, "results_main_chart.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
