#!/usr/bin/env python3
"""Train the conservative Holstein IPN on real ED force labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from epn.data import (
    HOLSTEIN_TRAIN_RUNS,
    HOLSTEIN_VALIDATION_RUNS,
    holstein_indices,
    load_holstein,
)
from epn.holstein import HolsteinEnergyModel


def make_loader(q, force, indices, batch_size, shuffle):
    tensors = TensorDataset(
        torch.tensor(np.asarray(q[indices]), dtype=torch.float32),
        torch.tensor(np.asarray(force[indices]), dtype=torch.float32),
    )
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, device, force_rms):
    numerator = count = 0
    model.eval()
    for q, target in loader:
        q = q.to(device).reshape(-1, 30, 30)
        target = target.to(device).reshape_as(q)
        with torch.enable_grad():
            prediction = model.force(q)
        numerator += float((prediction - target).square().sum())
        count += target.numel()
    return numerator / count / force_rms**2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/sample"))
    parser.add_argument("--output", type=Path, default=Path("outputs/holstein"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(42)
    q, force, metadata = load_holstein(args.data_dir)
    train_index = holstein_indices(metadata, HOLSTEIN_TRAIN_RUNS)
    validation_index = holstein_indices(metadata, HOLSTEIN_VALIDATION_RUNS)
    if not len(train_index) or not len(validation_index):
        # The compact sample intentionally contains only a few real runs.
        runs = np.unique(metadata[:, 0])
        train_index = holstein_indices(metadata, runs[:-1])
        validation_index = holstein_indices(metadata, runs[-1:])

    q_train = np.asarray(q[train_index], dtype=np.float64)
    f_train = np.asarray(force[train_index], dtype=np.float64)
    q_mean, q_std = float(q_train.mean()), float(q_train.std())
    force_rms = float(np.sqrt(np.mean(f_train**2)))
    train_loader = make_loader(q, force, train_index, args.batch_size, True)
    validation_loader = make_loader(q, force, validation_index, args.batch_size, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HolsteinEnergyModel(q_mean, q_std, force_rms).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for q_batch, target in train_loader:
            q_batch = q_batch.to(device).reshape(-1, 30, 30)
            target = target.to(device).reshape_as(q_batch)
            optimizer.zero_grad(set_to_none=True)
            prediction = model.force(q_batch, create_graph=True)
            loss = ((prediction - target) / force_rms).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
        scheduler.step()
        train_mse = evaluate(model, train_loader, device, force_rms)
        validation_mse = evaluate(model, validation_loader, device, force_rms)
        history.append((epoch, train_mse, validation_mse))
        print(f"epoch {epoch:4d} train MSE={train_mse:.6e} validation MSE={validation_mse:.6e}")

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history, columns=["epoch", "train_normalized_mse", "validation_normalized_mse"]).to_csv(
        args.output / "history.csv", index=False
    )
    torch.save({
        "model_state_dict": model.state_dict(), "q_mean": q_mean,
        "q_std": q_std, "force_rms": force_rms,
    }, args.output / "model.pt")


if __name__ == "__main__":
    main()

