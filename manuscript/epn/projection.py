"""Permutation construction and finite-group averaging utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from .d4 import D4_MATRICES, transform_coordinate


def representation_permutations(
    coordinates: Sequence[tuple[int, int]],
) -> tuple[Tensor, ...]:
    """Build permutations for the D4 action on coordinate-indexed vectors.

    If ``x`` stores values indexed by coordinates, ``x[..., permutation]``
    represents the actively transformed field

        (D(g) x)(r) = x(g^{-1} r).
    """
    coordinates = tuple(coordinates)
    index = {coordinate: i for i, coordinate in enumerate(coordinates)}
    if len(index) != len(coordinates):
        raise ValueError("coordinates must be unique")

    permutations = []
    for matrix in D4_MATRICES:
        inverse = matrix.T  # all D4 matrices are orthogonal
        source_coordinates = [
            transform_coordinate(coordinate, inverse)
            for coordinate in coordinates
        ]
        try:
            permutation = [index[c] for c in source_coordinates]
        except KeyError as error:
            raise ValueError("coordinates are not closed under D4") from error
        permutations.append(torch.tensor(permutation, dtype=torch.long))
    return tuple(permutations)


def input_permutations(
    coordinates: Sequence[tuple[int, int]],
) -> tuple[Tensor, ...]:
    """D4 permutations for local input variables."""
    return representation_permutations(coordinates)


def output_permutations(
    output_directions: Sequence[tuple[int, int]],
) -> tuple[Tensor, ...]:
    """D4 permutations for directional output components."""
    return representation_permutations(output_directions)


def inverse_permutation(permutation: Tensor) -> Tensor:
    """Return the inverse of a one-dimensional index permutation."""
    return torch.argsort(permutation)


def invariant_average(backbone: torch.nn.Module, x: Tensor, permutations) -> Tensor:
    """Average scalar backbone predictions over all transformed inputs."""
    predictions = [backbone(x[..., p.to(x.device)]) for p in permutations]
    return torch.stack(predictions, dim=0).mean(dim=0)


def equivariant_average(
    backbone: torch.nn.Module,
    x: Tensor,
    input_perms,
    output_perms,
) -> Tensor:
    """Project an arbitrary backbone onto an exactly equivariant map.

    This directly implements

        P[f](x) = |G|^{-1} sum_g D_Y(g)^{-1} f(D_X(g)x).
    """
    aligned_predictions = []
    for input_perm, output_perm in zip(input_perms, output_perms, strict=True):
        transformed_x = x[..., input_perm.to(x.device)]
        transformed_y = backbone(transformed_x)
        inverse = inverse_permutation(output_perm).to(x.device)
        aligned_predictions.append(transformed_y[..., inverse])
    return torch.stack(aligned_predictions, dim=0).mean(dim=0)

