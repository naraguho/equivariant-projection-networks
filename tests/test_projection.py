import torch

from epn import (
    DIRECTION_VECTORS,
    DirectionalEPN,
    InvariantEnergyModel,
    circular_offsets,
    input_permutations,
    output_permutations,
    square_offsets,
)


def test_scalar_projection_is_invariant():
    torch.manual_seed(11)
    coordinates = square_offsets(5)
    model = InvariantEnergyModel(coordinates, hidden_dims=(16, 8)).double()
    x = torch.randn(3, 25, dtype=torch.double)
    reference = model(x)
    for permutation in input_permutations(coordinates):
        torch.testing.assert_close(model(x[:, permutation]), reference)


def test_directional_projection_is_equivariant():
    torch.manual_seed(11)
    coordinates = circular_offsets(3)
    model = DirectionalEPN(coordinates, hidden_dims=(16, 8)).double()
    x = torch.randn(3, len(coordinates), dtype=torch.double)
    reference = model(x)
    for input_perm, output_perm in zip(
        input_permutations(coordinates),
        output_permutations(DIRECTION_VECTORS),
        strict=True,
    ):
        torch.testing.assert_close(
            model(x[:, input_perm]), reference[:, output_perm]
        )


def test_invariant_energy_has_covariant_gradient():
    torch.manual_seed(11)
    coordinates = square_offsets(5)
    model = InvariantEnergyModel(coordinates, hidden_dims=(16, 8)).double()
    x = torch.randn(2, 25, dtype=torch.double, requires_grad=True)
    force = model.force(x)
    for permutation in input_permutations(coordinates):
        transformed_x = x.detach()[:, permutation].requires_grad_(True)
        transformed_force = model.force(transformed_x)
        torch.testing.assert_close(transformed_force, force[:, permutation])

