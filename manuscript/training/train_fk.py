#!/usr/bin/env python3
"""Train the manuscript FK EPN on real ED-kMC free-energy differences."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from epn import DirectionalEPN
from epn.data import (
    FK_TRAIN_TRAJECTORIES,
    FK_VALIDATION_TRAJECTORIES,
    fk_local_columns,
    load_fk,
)

TARGETS = ["deltaF_px", "deltaF_mx", "deltaF_py", "deltaF_my"]
MASKS = ["mask_px", "mask_mx", "mask_py", "mask_my"]


def dataset(frame, local_columns, target_mean, target_std):
    x = torch.tensor(frame[local_columns].to_numpy(np.float32))
    y = torch.tensor((frame[TARGETS].to_numpy(np.float32) - target_mean) / target_std)
    mask = torch.tensor(frame[MASKS].to_numpy(bool))
    return TensorDataset(x, y, mask)


def masked_mse(prediction, target, mask):
    return ((prediction - target).square() * mask).sum() / mask.sum()


@torch.no_grad()
def evaluate(model, loader, device):
    numerator = denominator = 0.0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        numerator += float(((model(x) - y).square() * mask).sum())
        denominator += float(mask.sum())
    return numerator / denominator


@torch.no_grad()
def save_validation_scatter(
    model, loader, device, target_mean, target_std, output, max_points=100_000
):
    """Save ML predictions against ED labels for every legal validation move."""
    model.eval()
    actual, predicted = [], []
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        prediction = model(x)
        actual.append((y[mask] * target_std + target_mean).cpu().numpy())
        predicted.append(
            (prediction[mask] * target_std + target_mean).cpu().numpy()
        )

    actual = np.concatenate(actual)
    predicted = np.concatenate(predicted)
    if len(actual) > max_points:
        keep = np.linspace(0, len(actual) - 1, max_points, dtype=int)
        actual, predicted = actual[keep], predicted[keep]

    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    padding = 0.03 * (high - low or 1.0)
    limits = (low - padding, high + padding)
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(actual, predicted, s=5, alpha=0.25, rasterized=True)
    ax.plot(limits, limits, "k--", lw=1.2, label="$y=x$")
    ax.set(
        xlabel="ED $\\Delta F$",
        ylabel="ML $\\Delta F$",
        xlim=limits,
        ylim=limits,
        aspect="equal",
        title=f"FK validation (RMSE={rmse:.3g})",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/sample/fk_real_sample.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/fk"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    torch.manual_seed(12345)
    frame = load_fk(args.data)
    local_columns, coordinates = fk_local_columns(frame.columns)
    if len(coordinates) != 317:
        raise ValueError(f"Expected 317 radius-10 inputs, found {len(coordinates)}")

    train = frame[frame.trajectory.isin(FK_TRAIN_TRAJECTORIES)].copy()
    validation = frame[frame.trajectory.isin(FK_VALIDATION_TRAJECTORIES)].copy()
    if train.empty or validation.empty:
        raise ValueError("Dataset must contain both training and validation trajectories")

    valid_training_targets = train[TARGETS].to_numpy()[train[MASKS].to_numpy(bool)]
    target_mean = float(valid_training_targets.mean())
    target_std = float(valid_training_targets.std())

    train_loader = DataLoader(
        dataset(train, local_columns, target_mean, target_std),
        batch_size=args.batch_size, shuffle=True,
    )
    validation_loader = DataLoader(
        dataset(validation, local_columns, target_mean, target_std),
        batch_size=args.batch_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirectionalEPN(coordinates).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(x), y, mask)
            loss.backward()
            optimizer.step()
        train_mse = evaluate(model, train_loader, device)
        validation_mse = evaluate(model, validation_loader, device)
        history.append((epoch, train_mse, validation_mse))
        print(f"epoch {epoch:4d} train MSE={train_mse:.6e} validation MSE={validation_mse:.6e}")

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history, columns=["epoch", "train_normalized_mse", "validation_normalized_mse"]).to_csv(
        args.output / "history.csv", index=False
    )
    save_validation_scatter(
        model, validation_loader, device, target_mean, target_std,
        args.output / "validation_ed_vs_ml.png",
    )
    torch.save({
        "model_state_dict": model.state_dict(),
        "coordinates": coordinates,
        "target_mean": target_mean,
        "target_std": target_std,
    }, args.output / "model.pt")


if __name__ == "__main__":
    main()
