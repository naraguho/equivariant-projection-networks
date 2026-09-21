"""A scalar output becomes exactly D4 invariant after projection."""

import torch

from epn import InvariantEnergyModel, input_permutations, square_offsets


torch.manual_seed(7)
coordinates = square_offsets(5)
model = InvariantEnergyModel(coordinates, hidden_dims=(32, 16))
x = torch.randn(4, 25)

reference = model(x)
for group_index, permutation in enumerate(input_permutations(coordinates)):
    transformed = model(x[:, permutation])
    error = (transformed - reference).abs().max().item()
    print(f"D4 element {group_index}: maximum invariance error = {error:.3e}")

