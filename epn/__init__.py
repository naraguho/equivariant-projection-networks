"""Small, explicit building blocks for finite-group projection networks."""

from .d4 import D4_MATRICES, DIRECTION_VECTORS, circular_offsets, square_offsets
from .models import DirectionalEPN, InvariantEnergyModel, make_mlp
from .projection import input_permutations, output_permutations

__all__ = [
    "D4_MATRICES",
    "DIRECTION_VECTORS",
    "DirectionalEPN",
    "InvariantEnergyModel",
    "circular_offsets",
    "input_permutations",
    "make_mlp",
    "output_permutations",
    "square_offsets",
]

