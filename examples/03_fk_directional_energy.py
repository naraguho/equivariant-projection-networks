"""Minimal Falicov-Kimball EPN with the manuscript dimensions."""

import torch

from epn import DirectionalEPN, circular_offsets


torch.manual_seed(7)
coordinates = circular_offsets(radius=10)
assert len(coordinates) == 317

model = DirectionalEPN(
    input_coordinates=coordinates,
    hidden_dims=(512, 512, 256, 128),
)

# Each row is one binary local f-electron environment.
local_occupations = torch.randint(0, 2, (8, 317)).float()
delta_free_energy = model(local_occupations)

print("input shape :", tuple(local_occupations.shape))
print("output shape:", tuple(delta_free_energy.shape))
print("output order: (+x, -x, +y, -y)")

