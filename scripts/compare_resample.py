#!/usr/bin/env python3
"""Before/after table + plots for slurm/opt_resample.sh.

usage: python3 scripts/compare_resample.py results/opt_resample_<jobid>.csv

Labels are <mode>_<size> (serial_1e6, parallel_1e6, ...). Speedups of BOTH
modes are relative to the serial-mode 1-thread time of the same size, so the
"after" curve shows the real gain over the original baseline.

Writes results/<stem>_speedup.png and results/<stem>_resample.png, prints a
markdown table plus an Amdahl fit of the resample stage, T(p) = s + r/p.
"""
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#1f1f1e", "#5f5e5a", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
})
MODES = [("serial", "Serial resample (before)", ORANGE),
         ("parallel", "Parallel prefix-sum resample (after)", BLUE)]


def load(path):
    d = defaultdict(lambda: defaultdict(list))  # (mode, size) -> threads -> rows
    with open(path) as f:
        for r in csv.DictReader(f):
            mode, size = r["label"].split("_", 1)
            d[(mode, size)][int(r["threads"])].append(r)
    return d


def med(rows, key):
    return st.median(float(r[key]) for r in rows)


def amdahl_fit(series):
    """Least squares T = s + r/p over all thread counts."""
    xs = [1.0 / p for p in series]; ys = list(series.values())
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    r = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    return my - r * mx, r


def main(path):
    d = load(path)
    sizes = sorted({s for _, s in d}, key=lambda s: float(s.replace("e", "e+")))
    out = Path(path).with_suffix("")

    print("| N | threads | before (s) | after (s) | gain | speedup before | speedup after "
          "| resample before / after (s) | max cycle before / after (ms) | in deadline before / after "
          "| RMSE before / after (m) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for size in sizes:
        s, p = d.get(("serial", size), {}), d.get(("parallel", size), {})
        base = med(s[1], "t_total") if 1 in s else None
        for t in sorted(set(s) & set(p)):
            tb, ta = med(s[t], "t_total"), med(p[t], "t_total")
            sp = (lambda x: f"{base / x:.1f}x") if base else (lambda x: "-")
            print(f"| {size} | {t} | {tb:.3f} | {ta:.3f} | {tb / ta:.2f}x | {sp(tb)} | {sp(ta)} "
                  f"| {med(s[t], 't_resample'):.3f} / {med(p[t], 't_resample'):.3f} "
                  f"| {med(s[t], 'cycle_max_ms'):.1f} / {med(p[t], 'cycle_max_ms'):.1f} "
                  f"| {100 * med(s[t], 'deadline_frac'):.0f}% / {100 * med(p[t], 'deadline_frac'):.0f}% "
                  f"| {med(s[t], 'rmse_pos_m'):.1f} / {med(p[t], 'rmse_pos_m'):.1f} |")

    print("\nAmdahl fit of the resample stage, T(p) = s + r/p (median times):")
    for size in sizes:
        for mode, _, _ in MODES:
            ser = {t: med(rs, "t_resample") for t, rs in d.get((mode, size), {}).items()}
            if len(ser) >= 3 and 1 in ser:
                s_, r_ = amdahl_fit(ser)
                print(f"  {mode:8s} {size}: serial part s = {s_:.3f} s, parallel part r = {r_:.3f} s")

    plotted = [z for z in sizes if 1 in d.get(("serial", z), {})]
    fig, axes = plt.subplots(1, len(plotted), figsize=(5 * len(plotted), 4), squeeze=False)
    for ax, size in zip(axes[0], plotted):
        base = med(d[("serial", size)][1], "t_total")
        ts = sorted(d[("serial", size)])
        ax.plot(ts, ts, ls="--", color=INK2, lw=1, label="Ideal")
        for mode, name, col in MODES:
            ser = d.get((mode, size), {})
            th = sorted(ser)
            ax.plot(th, [base / med(ser[t], "t_total") for t in th], "o-", color=col, label=name)
        ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
        ax.set_xticks(ts); ax.set_xticklabels(map(str, ts))
        ax.set_yticks(ts); ax.set_yticklabels(map(str, ts))
        ax.set_xlabel("OpenMP threads"); ax.set_ylabel("Speedup vs 1-thread serial baseline")
        ax.set_title(f"N = {size} particles", loc="left", color=INK)
    axes[0][0].legend(loc="upper left")
    fig.tight_layout(); fig.savefig(f"{out}_speedup.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, len(plotted), figsize=(5 * len(plotted), 4), squeeze=False)
    for ax, size in zip(axes[0], plotted):
        for mode, name, col in MODES:
            ser = d.get((mode, size), {})
            th = sorted(ser)
            ax.plot(th, [med(ser[t], "t_resample") for t in th], "o-", color=col, label=name)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(th); ax.set_xticklabels(map(str, th))
        ax.set_xlabel("OpenMP threads"); ax.set_ylabel("Resample + roughen time (s)")
        ax.set_title(f"N = {size} particles", loc="left", color=INK)
    axes[0][0].legend(loc="lower left")
    fig.tight_layout(); fig.savefig(f"{out}_resample.png", dpi=150); plt.close(fig)
    print(f"\nwrote {out}_speedup.png, {out}_resample.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
