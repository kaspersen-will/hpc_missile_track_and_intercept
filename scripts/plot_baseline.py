#!/usr/bin/env python3
"""Plot the baseline sweep produced by slurm/baseline_omp.sh.

usage: python3 scripts/plot_baseline.py results/baseline_<jobid>.csv

Writes into results/:
    <stem>_speedup.png    speedup vs threads (median of repeats) + ideal line
    <stem>_stages.png     per-stage time vs threads (where the time goes)
    <stem>_cycle.png      mean / max update-cycle latency vs the 100 ms deadline
and prints a markdown summary table (paste into slides/report).
Only needs numpy + matplotlib.
"""
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

# Validated categorical slots 1-4 (blue, orange, aqua, yellow); ink/surface.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURF = "#1f1f1e", "#5f5e5a", "#e4e3df", "#fcfcfb"
STAGES = [("t_predict", "Predict"), ("t_weight", "Weight"),
          ("t_resample", "Resample + roughen"), ("t_impact", "Impact ensemble")]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
})


def load(path):
    rows = defaultdict(lambda: defaultdict(list))  # label -> threads -> [row]
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["label"]][int(r["threads"])].append(r)
    return rows


def med(rs, key):
    return st.median(float(r[key]) for r in rs)


def style_threads_axis(ax, threads):
    ax.set_xscale("log", base=2)
    ax.set_xticks(threads)
    ax.set_xticklabels([str(t) for t in threads])
    ax.set_xlabel("OpenMP threads")


def plain_log_y(ax):
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def legend_right(ax):
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)


def main():
    path = Path(sys.argv[1])
    data = load(path)
    out = path.with_suffix("")
    strong = [l for l in ("strong_1e5", "strong_1e6") if l in data]

    # ---- speedup ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    allt = sorted({t for l in strong for t in data[l]})
    ax.plot(allt, allt, ls="--", lw=1.2, color=INK2, label="Ideal")
    for i, l in enumerate(strong):
        th = sorted(data[l])
        base = med(data[l][1], "t_total")
        sp = [base / med(data[l][t], "t_total") for t in th]
        ax.plot(th, sp, lw=2, marker="o", ms=7, color=C[i],
                mec=SURF, mew=2, label=l.replace("strong_", "N = "))
        ax.annotate(f"{sp[-1]:.1f}x", (th[-1], sp[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK)
    style_threads_axis(ax, allt)
    ax.set_yscale("log", base=2)
    plain_log_y(ax)
    ax.set_ylabel("Speedup vs 1 thread")
    ax.set_title("Strong scaling, OpenMP, one node", loc="left")
    legend_right(ax)
    fig.tight_layout(); fig.savefig(f"{out}_speedup.png", dpi=160)

    # ---- stage breakdown (one panel per problem size) ----------------------
    fig, axes = plt.subplots(1, len(strong), figsize=(6.0 * len(strong), 4.2),
                             squeeze=False)
    for ax, l in zip(axes[0], strong):
        th = sorted(data[l])
        for i, (k, name) in enumerate(STAGES):
            ax.plot(th, [med(data[l][t], k) for t in th], lw=2, marker="o",
                    ms=7, color=C[i], mec=SURF, mew=2, label=name)
        style_threads_axis(ax, th)
        ax.set_yscale("log")
        plain_log_y(ax)
        ax.set_ylabel("Time per run (s)")
        ax.set_title(f"Where the time goes, {l.replace('strong_', 'N = ')}",
                     loc="left")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(f"{out}_stages.png", dpi=160)

    # ---- update-cycle latency vs deadline ---------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, l in enumerate(strong):
        th = sorted(data[l])
        lab = l.replace("strong_", "N = ")
        ax.plot(th, [med(data[l][t], "cycle_mean_ms") for t in th], lw=2,
                marker="o", ms=7, color=C[i], mec=SURF, mew=2, label=f"{lab} mean")
        ax.plot(th, [med(data[l][t], "cycle_max_ms") for t in th], lw=2,
                ls=":", marker="s", ms=7, color=C[i], mec=SURF, mew=2,
                label=f"{lab} worst")
    ax.axhline(100, color=INK, lw=1)
    ax.annotate("100 ms deadline", (allt[-1], 100), xytext=(0, 5),
                textcoords="offset points", ha="right", color=INK)
    style_threads_axis(ax, allt)
    ax.set_yscale("log")
    plain_log_y(ax)
    ax.set_ylabel("Update cycle (ms)")
    ax.set_title("Track update latency", loc="left")
    legend_right(ax)
    fig.tight_layout(); fig.savefig(f"{out}_cycle.png", dpi=160)

    # ---- markdown table ---------------------------------------------------
    print("| run | threads | time (s) | speedup | efficiency | resample share "
          "| cycle mean/max (ms) | in deadline | pos RMSE (m) | impact err / spread (m) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for l in data:
        base = med(data[l][min(data[l])], "t_total")
        for t in sorted(data[l]):
            rs = data[l][t]
            tt = med(rs, "t_total")
            if l.startswith("strong"):
                sp = base / tt
                sp_s, eff_s = f"{sp:.2f}", f"{sp / t:.0%}"
            else:
                sp_s = eff_s = "-"
            print(f"| {l} | {t} | {tt:.3f} | {sp_s} | {eff_s} | "
                  f"{med(rs, 't_resample') / tt:.0%} | "
                  f"{med(rs, 'cycle_mean_ms'):.1f} / {med(rs, 'cycle_max_ms'):.1f} | "
                  f"{med(rs, 'deadline_frac'):.0%} | {med(rs, 'rmse_pos_m'):.1f} | "
                  f"{med(rs, 'impact_err_m'):.0f} / {med(rs, 'impact_spread_m'):.0f} |")
    print(f"\nplots: {out}_speedup.png, {out}_stages.png, {out}_cycle.png",
          file=sys.stderr)


if __name__ == "__main__":
    main()
