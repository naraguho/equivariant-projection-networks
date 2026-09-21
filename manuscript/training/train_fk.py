#!/usr/bin/env python3
"""Train the manuscript FK EPN on real ED-kMC free-energy differences."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
    torch.save({
        "model_state_dict": model.state_dict(),
        "coordinates": coordinates,
        "target_mean": target_mean,
        "target_std": target_std,
    }, args.output / "model.pt")


if __name__ == "__main__":
    main()

