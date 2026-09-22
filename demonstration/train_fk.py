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

# Keep the four outputs in the same visual order used by equivariant.py.
DIRECTION_NAMES = ("up", "right", "down", "left")
TARGETS = ["deltaF_py", "deltaF_px", "deltaF_my", "deltaF_mx"]
MASKS = ["mask_py", "mask_px", "mask_my", "mask_mx"]

# D4 = {I, S, R, RS, R^2, R^2S, R^3, R^3S}.
# Each tuple is (name, number of 90-degree rotations, reflect first?).
GROUP_ACTIONS = (
    ("I", 0, False), ("S", 0, True),
    ("R", 1, False), ("RS", 1, True),
    ("R^2", 2, False), ("R^2S", 2, True),
    ("R^3", 3, False), ("R^3S", 3, True),
)


def transform_patch(patch, rotations, reflected):
    """Rotate/reflect a visible square patch, exactly as in equivariant.py."""
    if reflected:
        # S: exchange the top and bottom rows (reflection across the x axis).
        patch = torch.flip(patch, dims=(-2,))

    # R^k: rotate the patch counterclockwise k times.
    return torch.rot90(patch, k=rotations, dims=(-2, -1))


def reflect_directions(y):
    """Reflect outputs ordered as (up, right, down, left)."""
    return y[:, [2, 1, 0, 3]]


def inverse_transform_directions(y, rotations, reflected):
    """Return four predictions from the transformed frame to the original."""
    # The input action is g=R^k S: reflect first, then rotate.  Consequently,
    # g^(-1)=S R^(-k): undo the rotation first, then undo the reflection.
    y = torch.roll(y, shifts=rotations, dims=-1)
    if reflected:
        y = reflect_directions(y)
    return y


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

        # The CSV stores only the 317 sites inside a circular cutoff.  Find the
        # surrounding square and remember where those circular sites belong.
        radius = max(max(abs(x), abs(y)) for x, y in input_coordinates)
        self.patch_side = 2 * radius + 1
        center = radius
        rows = [center - y for x, y in input_coordinates]
        columns = [center + x for x, y in input_coordinates]
        self.register_buffer("circle_rows", torch.tensor(rows, dtype=torch.long))
        self.register_buffer(
            "circle_columns", torch.tensor(columns, dtype=torch.long)
        )

        self.mlp = nn.Sequential(
            nn.Linear(len(input_coordinates), 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 4),
        )

    def put_circle_in_square_patch(self, x):
        """Place 317 flat circular values at their visible 2D positions."""
        patch = x.new_zeros((x.shape[0], self.patch_side, self.patch_side))
        patch[:, self.circle_rows, self.circle_columns] = x
        return patch

    def read_circle_from_square_patch(self, patch):
        """Read the same 317 circular sites back into the original CSV order."""
        return patch[:, self.circle_rows, self.circle_columns]

    def forward(self, x):
        # First turn the hard-to-read flat input back into a visible 2D patch.
        circular_patch = self.put_circle_in_square_patch(x)
        aligned_predictions = []

        for _, rotations, reflected in GROUP_ACTIONS:
            # This is now the same transparent flip/rot90 operation used in
            # equivariant.py.  Zeros outside the circular cutoff remain zeros.
            transformed_patch = transform_patch(
                circular_patch, rotations, reflected
            )

            # Give only the 317 physical circular sites to the ordinary MLP.
            transformed_x = self.read_circle_from_square_patch(
                transformed_patch
            )
            raw_prediction = self.mlp(transformed_x)

            # The MLP outputs are (up, right, down, left) in the transformed
            # frame.  Undo that frame change before averaging predictions.
            aligned_prediction = inverse_transform_directions(
                raw_prediction, rotations, reflected
            )
            aligned_predictions.append(aligned_prediction)

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


def save_and_show_y_equals_x(ed, ml, output):
    """Save and display the only validation figure in this demonstration."""
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
    plt.show()
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
    save_and_show_y_equals_x(ed, ml, figure)
    print(f"saved {figure}")


if __name__ == "__main__":
    main()
