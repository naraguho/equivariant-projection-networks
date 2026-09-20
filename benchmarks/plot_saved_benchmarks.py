#!/usr/bin/env python3
"""Reproduce ED-vs-ML correlation and Holstein collapse plots from real runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def holstein_validation(root: Path, output: Path):
    ed = pd.read_csv(root / "holstein/ed_raw_q_correlation_vs_step.csv")
    ml = pd.read_csv(root / "holstein/ml_epoch0040_correlation.csv")
    steps = sorted(set(ed.step) & set(ml.step))
    selected = np.linspace(0, len(steps) - 1, min(6, len(steps)), dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True, sharey=True)
    for ax, index in zip(axes.flat, selected):
        step = steps[index]
        a, b = ed[ed.step == step], ml[ml.step == step]
        ax.plot(a.integer_radius, a.C_mean, "o-", ms=3, label="ED")
        ml_radius = "radius" if "radius" in b else "integer_radius"
        ml_value = "C_mean" if "C_mean" in b else "correlation_mean"
        ax.plot(b[ml_radius], b[ml_value], "s--", ms=3, label="ML")
        ax.set_title(f"step {step}")
        ax.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False)
    fig.supxlabel("distance r"); fig.supylabel("raw C(r)")
    fig.tight_layout(); fig.savefig(output / "holstein_ed_vs_ml_correlation.png", dpi=180)


def holstein_collapse(root: Path, output: Path):
    data = np.loadtxt(root / "holstein/collapse_long.txt")
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for time in np.unique(data[:, 0]):
        rows = data[data[:, 0] == time]
        ax.plot(rows[:, 3], rows[:, 6], marker="o", ms=3, label=f"t={time:g}")
    ax.set(xlabel="r/L(t)", ylabel="C(r,t)/C(0,t)", xlim=(0, 5))
    ax.grid(alpha=0.2); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output / "holstein_correlation_collapse.png", dpi=180)


def fk_validation(root: Path, output: Path):
    data = pd.read_csv(root / "fk/correlation_comparison_epoch0300.csv")
    chosen = [s for s in (0, 20, 40, 60, 80, 100) if s in set(data.sweep)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True, sharey=True)
    for ax, sweep in zip(axes.flat, chosen):
        rows = data[data.sweep == sweep]
        ax.plot(rows.radius, rows.ED_C, "o-", ms=3, label="ED-kMC")
        ax.plot(rows.radius, rows.ML_C, "s--", ms=3, label="EPN-kMC")
        ax.set_title(f"sweep {sweep}"); ax.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False)
    fig.supxlabel("distance r"); fig.supylabel("raw connected C(r)")
    fig.tight_layout(); fig.savefig(output / "fk_ed_vs_ml_correlation.png", dpi=180)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/full/manuscript_benchmarks"))
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    holstein_validation(args.data, args.output)
    holstein_collapse(args.data, args.output)
    fk_validation(args.data, args.output)
    print(f"wrote benchmark figures to {args.output}")


if __name__ == "__main__":
    main()

