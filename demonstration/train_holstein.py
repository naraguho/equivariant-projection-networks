#!/usr/bin/env python3
"""Pedagogical Holstein energy/force training with real ED labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "holstein_tiny_real.npz"
OUTPUT = HERE / "output"
LATTICE_SIZE = 30
PATCH_SIZE = 5

# D4 = {I, S, R, RS, R^2, R^2S, R^3, R^3S}.
GROUP_ACTIONS = (
    ("I", 0, False), ("S", 0, True),
    ("R", 1, False), ("RS", 1, True),
    ("R^2", 2, False), ("R^2S", 2, True),
    ("R^3", 3, False), ("R^3S", 3, True),
)


def periodic_patches(q):
    """Extract one periodic 5x5 scalar patch around every lattice site."""
    half = PATCH_SIZE // 2
    padded = F.pad(q[:, None], (half, half, half, half), mode="circular")
    patches = padded.unfold(2, PATCH_SIZE, 1).unfold(3, PATCH_SIZE, 1)
    # Shape: (batch, Ly, Lx, 5, 5).
    return patches.squeeze(1)


def transform_patches(patches, rotations, reflected):
    """Apply one D4 action to the last two dimensions of every patch."""
    if reflected:
        patches = torch.flip(patches, dims=(-2,))
    return torch.rot90(patches, k=rotations, dims=(-2, -1))


class HolsteinEnergyModel(nn.Module):
    """Shared D4-invariant local energies summed into one total energy."""

    def __init__(self, q_mean, q_std, force_rms):
        super().__init__()
        self.local_mlp = nn.Sequential(
            nn.Linear(PATCH_SIZE**2, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(),
            # Force labels cannot determine a constant energy offset.
            nn.Linear(32, 1, bias=False),
        )
        self.register_buffer("q_mean", torch.tensor(q_mean))
        self.register_buffer("q_std", torch.tensor(q_std))
        self.register_buffer("energy_scale", torch.tensor(q_std * force_rms))

    def total_energy(self, q):
        """Return E_ML(Q)=sum_i epsilon_i(Q) for each full lattice."""
        patches = periodic_patches(q)
        local_predictions = []

        for _, rotations, reflected in GROUP_ACTIONS:
            transformed = transform_patches(patches, rotations, reflected)
            normalized = (transformed - self.q_mean) / self.q_std
            flat = normalized.reshape(-1, PATCH_SIZE**2)
            local_predictions.append(self.local_mlp(flat).squeeze(-1))

        # Average the eight predictions to make each local energy invariant.
        local_energy = torch.stack(local_predictions).mean(dim=0)
        local_energy = local_energy.reshape(q.shape[0], LATTICE_SIZE, LATTICE_SIZE)

        # The same local network is shared over every site. Their sum is the
        # extensive total ML energy for one 30x30 configuration.
        return self.energy_scale * local_energy.sum(dim=(1, 2))


def force_from_total_energy(model, q, create_graph):
    """Compute the conservative force F_i=-dE_ML/dQ_i explicitly."""
    q_for_derivative = q.detach().requires_grad_(True)
    total_energy = model.total_energy(q_for_derivative)
    force = -torch.autograd.grad(
        total_energy.sum(), q_for_derivative, create_graph=create_graph
    )[0]
    return force


def validation_predictions(model, loader, device):
    model.eval()
    ed_values, ml_values = [], []
    for q, ed_force in loader:
        q = q.to(device).reshape(-1, LATTICE_SIZE, LATTICE_SIZE)
        ed_force = ed_force.to(device).reshape_as(q)
        with torch.enable_grad():
            ml_force = force_from_total_energy(model, q, create_graph=False)
        ed_values.append(ed_force.detach().cpu().reshape(-1))
        ml_values.append(ml_force.detach().cpu().reshape(-1))
    return torch.cat(ed_values).numpy(), torch.cat(ml_values).numpy()


def save_y_equals_x(ed, ml, output):
    """Save the only validation figure used in this demonstration."""
    low, high = float(min(ed.min(), ml.min())), float(max(ed.max(), ml.max()))
    padding = 0.03 * (high - low or 1.0)
    limits = (low - padding, high + padding)
    rmse = float(np.sqrt(np.mean((ml - ed) ** 2)))

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(ed, ml, s=10, alpha=0.3)
    ax.plot(limits, limits, "k--", lw=1.2, label="$y=x$")
    ax.set(
        xlabel="ED force", ylabel="ML force",
        xlim=limits, ylim=limits, aspect="equal",
        title=f"Holstein held-out validation (RMSE={rmse:.3g})",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(7)
    data = np.load(DATA)
    q = torch.tensor(data["Q"], dtype=torch.float32)
    force = torch.tensor(data["force"], dtype=torch.float32)
    metadata = data["metadata_run_step"]

    # Runs are kept separate: run 1 trains the model, while run 10 is held out.
    train_index = np.flatnonzero(metadata[:, 0] == 1)
    validation_index = np.flatnonzero(metadata[:, 0] == 10)
    q_train, force_train = q[train_index], force[train_index]

    q_mean = float(q_train.mean())
    q_std = float(q_train.std(unbiased=False))
    force_rms = float(torch.sqrt(torch.mean(force_train.square())))

    train_loader = DataLoader(
        TensorDataset(q[train_index], force[train_index]),
        batch_size=1, shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(q[validation_index], force[validation_index]),
        batch_size=1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HolsteinEnergyModel(q_mean, q_std, force_rms).to(device)
    # The tiny three-snapshot demonstration uses a larger learning rate than
    # the full production run so that ten epochs visibly improve the model.
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-6)

    print(f"device={device}  patch={PATCH_SIZE}x{PATCH_SIZE}")
    print(f"training snapshots={len(train_index)}  validation snapshots={len(validation_index)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for q_batch, ed_force in train_loader:
            q_batch = q_batch.to(device).reshape(-1, LATTICE_SIZE, LATTICE_SIZE)
            ed_force = ed_force.to(device).reshape_as(q_batch)
            optimizer.zero_grad(set_to_none=True)

            # Training remains conservative because the predicted force is
            # always obtained by differentiating the learned total energy.
            ml_force = force_from_total_energy(model, q_batch, create_graph=True)
            loss = ((ml_force - ed_force) / force_rms).square().mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"epoch {epoch:2d}  normalized training MSE={np.mean(losses):.6f}")

    ed, ml = validation_predictions(model, validation_loader, device)
    OUTPUT.mkdir(exist_ok=True)
    figure = OUTPUT / "holstein_validation_y_equals_x.png"
    save_y_equals_x(ed, ml, figure)
    print(f"saved {figure}")


if __name__ == "__main__":
    main()
