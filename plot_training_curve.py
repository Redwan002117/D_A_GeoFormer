"""Plots train/val loss and per-class F1 from a train.py --log-csv output."""

from __future__ import annotations

import argparse
import csv

import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-csv", type=str, default="training_log.csv")
    p.add_argument("--out", type=str, default="training_curve.png")
    args = p.parse_args()

    rows = []
    with open(args.log_csv, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})

    epochs = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, [r["train_loss"] for r in rows], label="train", color="#1E7FA0", lw=2)
    axes[0].plot(epochs, [r["val_loss"] for r in rows], label="val", color="#B24B1E", lw=2)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("Tversky loss")
    axes[0].set_title("Loss (synthetic pipeline-validation run)")
    axes[0].legend(); axes[0].grid(alpha=0.25)

    class_names = ["background", "building", "road", "flooded"]
    colors = ["#B7C2B9", "#0B4A5C", "#1E7FA0", "#B24B1E"]
    for name, color in zip(class_names, colors):
        axes[1].plot(epochs, [r[f"val_f1_{name}"] for r in rows], label=name, color=color, lw=2)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("F1")
    axes[1].set_title("Per-class validation F1")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(); axes[1].grid(alpha=0.25)

    fig.suptitle("Dual-Axis GeoFormer -- pipeline-validation training run (synthetic data)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
