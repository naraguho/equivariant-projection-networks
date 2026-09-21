"""Four directional outputs transform exactly under D4."""

import torch

from epn import (
    DIRECTION_VECTORS,
    DirectionalEPN,
    circular_offsets,
    input_permutations,
    output_permutations,
)


torch.manual_seed(7)
coordinates = circular_offsets(3)
model = DirectionalEPN(coordinates, hidden_dims=(32, 16))
x = torch.randn(4, len(coordinates))

reference = model(x)
input_perms = input_permutations(coordinates)
output_perms = output_permutations(DIRECTION_VECTORS)

for group_index, (input_perm, output_perm) in enumerate(
    zip(input_perms, output_perms, strict=True)
):
    transformed_input_prediction = model(x[:, input_perm])
    transformed_reference = reference[:, output_perm]
    error = (transformed_input_prediction - transformed_reference).abs().max().item()
    print(f"D4 element {group_index}: maximum equivariance error = {error:.3e}")

