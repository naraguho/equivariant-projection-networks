"""Holstein local-energy model acting on complete periodic lattices."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .models import InvariantEnergyModel


def periodic_patches(q: Tensor, patch_size: int = 5) -> Tensor:
    """Return all site-centered patches as ``(batch, sites, patch_size^2)``."""
    half = patch_size // 2
    padded = F.pad(q[:, None], (half, half, half, half), mode="circular")
    patches = padded.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    return patches.squeeze(1).reshape(q.shape[0], -1, patch_size**2)


class HolsteinEnergyModel(nn.Module):
    """Sum shared D4-invariant local energies over a periodic lattice."""

    def __init__(
        self,
        q_mean: float,
        q_std: float,
        force_rms: float,
        hidden_dims: Sequence[int] = (512, 256, 128),
        patch_size: int = 5,
    ) -> None:
        super().__init__()
        from .d4 import square_offsets

        self.patch_size = patch_size
        self.local_energy = InvariantEnergyModel(
            square_offsets(patch_size), hidden_dims=hidden_dims
        )
        self.register_buffer("q_mean", torch.tensor(float(q_mean)))
        self.register_buffer("q_std", torch.tensor(float(q_std)))
        # This scale makes normalized energy derivatives naturally order one.
        self.register_buffer("energy_scale", torch.tensor(float(q_std * force_rms)))

    def forward(self, q: Tensor) -> Tensor:
        patches = periodic_patches(q, self.patch_size)
        normalized = (patches - self.q_mean) / self.q_std
        local = self.local_energy(normalized.reshape(-1, self.patch_size**2))
        return self.energy_scale * local.reshape(q.shape[0], -1).sum(dim=1)

    def force(self, q: Tensor, create_graph: bool = False) -> Tensor:
        q_leaf = q.detach().requires_grad_(True)
        energy = self(q_leaf)
        return -torch.autograd.grad(
            energy.sum(), q_leaf, create_graph=create_graph, retain_graph=create_graph
        )[0]

