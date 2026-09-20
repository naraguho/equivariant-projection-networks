"""Minimal Holstein invariant local energy and conservative force."""

import torch

from epn import InvariantEnergyModel, square_offsets


torch.manual_seed(7)
coordinates = square_offsets(size=5)
model = InvariantEnergyModel(
    input_coordinates=coordinates,
    hidden_dims=(512, 256, 128),
)

# Each row represents one 5x5 local displacement patch.
patches = torch.randn(8, 25, requires_grad=True)
local_energies = model(patches)
total_energy = local_energies.sum()
forces = -torch.autograd.grad(total_energy, patches)[0]

print("patch shape       :", tuple(patches.shape))
print("local-energy shape:", tuple(local_energies.shape))
print("force shape       :", tuple(forces.shape))
print("forces were obtained as -dE/dQ")

