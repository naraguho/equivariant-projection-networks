"""Readable invariant- and equivariant-projection models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .d4 import D4_MATRICES, DIRECTION_VECTORS
from .projection import (
    equivariant_average,
    input_permutations,
    invariant_average,
    output_permutations,
)


def make_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    final_bias: bool = True,
) -> nn.Sequential:
    """Construct the ordinary SiLU MLP used inside the projection."""
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, width), nn.SiLU()))
        previous = width
    layers.append(nn.Linear(previous, output_dim, bias=final_bias))
    return nn.Sequential(*layers)


class DirectionalEPN(nn.Module):
    """D4-equivariant predictor for four directional scalar outputs."""

    def __init__(
        self,
        input_coordinates: Sequence[tuple[int, int]],
        hidden_dims: Sequence[int] = (512, 512, 256, 128),
    ) -> None:
        super().__init__()
        self.input_coordinates = tuple(input_coordinates)
        self.backbone = make_mlp(len(self.input_coordinates), hidden_dims, 4)
        self._input_perms = input_permutations(self.input_coordinates)
        self._output_perms = output_permutations(DIRECTION_VECTORS)

    def forward(self, x: Tensor) -> Tensor:
        return equivariant_average(
            self.backbone, x, self._input_perms, self._output_perms
        )


class InvariantEnergyModel(nn.Module):
    """D4-invariant scalar local-energy model."""

    def __init__(
        self,
        input_coordinates: Sequence[tuple[int, int]],
        hidden_dims: Sequence[int] = (512, 256, 128),
    ) -> None:
        super().__init__()
        self.input_coordinates = tuple(input_coordinates)
        # A constant local-energy offset cannot be learned from force labels.
        self.backbone = make_mlp(
            len(self.input_coordinates), hidden_dims, 1, final_bias=False
        )
        self._input_perms = input_permutations(self.input_coordinates)

    def forward(self, x: Tensor) -> Tensor:
        return invariant_average(self.backbone, x, self._input_perms).squeeze(-1)

    def force(self, x: Tensor, create_graph: bool = False) -> Tensor:
        """Return ``-dE/dx`` for a batch of local energy inputs."""
        if not x.requires_grad:
            x = x.requires_grad_(True)
        energy = self(x).sum()
        gradient = torch.autograd.grad(energy, x, create_graph=create_graph)[0]
        return -gradient
