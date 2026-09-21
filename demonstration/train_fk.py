#!/usr/bin/env python3
"""Pedagogical FK training with real ED labels and a y=x validation plot."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "fk_tiny_real.csv.gz"
OUTPUT = HERE / "output"

TARGETS = ["deltaF_px", "deltaF_mx", "deltaF_py", "deltaF_my"]
MASKS = ["mask_px", "mask_mx", "mask_py", "mask_my"]
DIRECTION_COORDINATES = ((1, 0), (-1, 0), (0, 1), (0, -1))

# D4 = {I, S, R, RS, R^2, R^2S, R^3, R^3S}.
# Each tuple is (name, number of 90-degree rotations, reflect first?).
GROUP_ACTIONS = (
    ("I", 0, False), ("S", 0, True),
    ("R", 1, False), ("RS", 1, True),
    ("R^2", 2, False), ("R^2S", 2, True),
    ("R^3", 3, False), ("R^3S", 3, True),
)


def transform_coordinate(x, y, rotations, reflected):
    """Apply S first, followed by ``rotations`` copies of R."""
    if reflected:                 # S(x,y) = (x,-y)
        y = -y
    for _ in range(rotations):    # R(x,y) = (-y,x)
        x, y = -y, x
    return x, y


def inverse_transform_coordinate(x, y, rotations, reflected):
    """Undo R^k S by undoing the rotation first and reflection second."""
    for _ in range(rotations):    # R^(-1)(x,y) = (y,-x)
        x, y = y, -x
    if reflected:                 # S^(-1) = S
        y = -y
    return x, y


def permutation(coordinates, rotations, reflected):
    """Return indices that rotate/reflect scalar values on named sites."""
    index = {coordinate: i for i, coordinate in enumerate(coordinates)}
    result = []
    for target_coordinate in coordinates:
        # The value appearing at target r after transformation came from
        # source g^(-1)r before transformation.
        source_coordinate = inverse_transform_coordinate(
            *target_coordinate, rotations, reflected
        )
        result.append(index[source_coordinate])
    return torch.tensor(result, dtype=torch.long)


def local_columns_and_coordinates(columns):
    """Read coordinates such as nf_dx-2_dy+3 directly from CSV headers."""
    pattern = re.compile(r"nf_dx([+-]\d+)_dy([+-]\d+)$")
    parsed = []
    for column in columns:
        match = pattern.fullmatch(column)
        if match:
            parsed.append((column, (int(match.group(1)), int(match.group(2)))))
    return [item[0] for item in parsed], tuple(item[1] for item in parsed)


class FKEquivariantMLP(nn.Module):
    """Ordinary MLP projected onto four-direction D4 equivariance."""

    def __init__(self, input_coordinates):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(len(input_coordinates), 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 4),
        )
        self.input_permutations = [
            permutation(input_coordinates, rotations, reflected)
            for _, rotations, reflected in GROUP_ACTIONS
        ]
        self.output_permutations = [
            permutation(DIRECTION_COORDINATES, rotations, reflected)
            for _, rotations, reflected in GROUP_ACTIONS
        ]

    def forward(self, x):
        aligned_predictions = []
        for input_permutation, output_permutation in zip(
            self.input_permutations, self.output_permutations, strict=True
        ):
            # Transform the scalar input sites, evaluate the ordinary MLP,
            # then return its four outputs to the original directional frame.
            transformed_x = x[:, input_permutation.to(x.device)]
            transformed_y = self.mlp(transformed_x)
            inverse_output = torch.argsort(output_permutation).to(x.device)
            aligned_predictions.append(transformed_y[:, inverse_output])

        # P[f](x) = (1/8) sum_g D_out(g)^(-1) f(D_in(g)x).
        return torch.stack(aligned_predictions).mean(dim=0)


def make_dataset(frame, local_columns, target_mean, target_std):
    x = torch.tensor(frame[local_columns].to_numpy(np.float32))
    y = torch.tensor(
        (frame[TARGETS].to_numpy(np.float32) - target_mean) / target_std
    )
    legal_move = torch.tensor(frame[MASKS].to_numpy(bool))
    return TensorDataset(x, y, legal_move)


def masked_mse(prediction, target, legal_move):
    """Use only physically allowed hopping directions in the loss."""
    squared_error = (prediction - target).square()
    return squared_error[legal_move].mean()


@torch.no_grad()
def validation_predictions(model, loader, device, target_mean, target_std):
    """Return denormalized ED and ML values for legal validation moves."""
    model.eval()
    ed_values, ml_values = [], []
    for x, y, legal_move in loader:
        x, y, legal_move = x.to(device), y.to(device), legal_move.to(device)
        prediction = model(x)
        ed_values.append((y[legal_move] * target_std + target_mean).cpu())
        ml_values.append(
            (prediction[legal_move] * target_std + target_mean).cpu()
        )
    return torch.cat(ed_values).numpy(), torch.cat(ml_values).numpy()


def save_y_equals_x(ed, ml, output):
    """Save the only validation figure used in this demonstration."""
    low, high = float(min(ed.min(), ml.min())), float(max(ed.max(), ml.max()))
    padding = 0.03 * (high - low or 1.0)
    limits = (low - padding, high + padding)
    rmse = float(np.sqrt(np.mean((ml - ed) ** 2)))

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(ed, ml, s=10, alpha=0.35)
    ax.plot(limits, limits, "k--", lw=1.2, label="$y=x$")
    ax.set(
        xlabel="ED $\\Delta F$", ylabel="ML $\\Delta F$",
        xlim=limits, ylim=limits, aspect="equal",
        title=f"FK held-out validation (RMSE={rmse:.3g})",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(7)
    frame = pd.read_csv(DATA)
    local_columns, input_coordinates = local_columns_and_coordinates(frame.columns)
    train = frame[frame.trajectory == 1].copy()
    validation = frame[frame.trajectory == 9].copy()

    # Normalize using legal training labels only. Validation information never
    # enters the normalization or optimization.
    legal_training_labels = train[TARGETS].to_numpy(np.float32)[
        train[MASKS].to_numpy(bool)
    ]
    target_mean = float(legal_training_labels.mean())
    target_std = float(legal_training_labels.std())

    train_loader = DataLoader(
        make_dataset(train, local_columns, target_mean, target_std),
        batch_size=64, shuffle=True,
    )
    validation_loader = DataLoader(
        make_dataset(validation, local_columns, target_mean, target_std),
        batch_size=128,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FKEquivariantMLP(input_coordinates).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)

    print(f"device={device}  inputs={len(local_columns)}")
    print(f"trainable parameters={parameter_count:,}")
    print(f"training rows={len(train)}  validation rows={len(validation)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y, legal_move in train_loader:
            x, y, legal_move = x.to(device), y.to(device), legal_move.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(x), y, legal_move)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"epoch {epoch:2d}  normalized training MSE={np.mean(losses):.6f}")

    ed, ml = validation_predictions(
        model, validation_loader, device, target_mean, target_std
    )
    OUTPUT.mkdir(exist_ok=True)
    figure = OUTPUT / "fk_validation_y_equals_x.png"
    save_y_equals_x(ed, ml, figure)
    print(f"saved {figure}")


if __name__ == "__main__":
    main()
