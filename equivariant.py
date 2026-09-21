"""Make four scalar directional outputs exactly D4 equivariant."""

import torch
from torch import nn


torch.set_default_dtype(torch.float64)
torch.manual_seed(7)


def d4_matrices():
    """Return the eight 2 x 2 matrices acting on square-lattice positions."""
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])  # 90 degrees
    reflection = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    identity = torch.eye(2)

    group = []
    power = identity
    for _ in range(4):
        # ``@`` means matrix multiplication. Here ``power @ reflection`` is
        # R^k S: S acts first and R^k acts second on a column coordinate.
        group.extend((power, power @ reflection))
        power = power @ rotation

    # Sanity check: the ordering produced above is
    # D4 = {I, S, R, RS, R^2, R^2S, R^3, R^3S}.
    assert len(group) == 8
    assert len({tuple(matrix.flatten().tolist()) for matrix in group}) == 8
    return group


GROUP_NAMES = ("I", "S", "R", "RS", "R^2", "R^2S", "R^3", "R^3S")
GROUP = d4_matrices()


def action_matrix(g, coordinates):
    """Return the permutation induced by g on the supplied lattice sites."""
    action = torch.zeros(len(coordinates), len(coordinates))
    coordinate_to_index = {coordinate: i for i, coordinate in enumerate(coordinates)}

    for source, coordinate in enumerate(coordinates):
        transformed = g @ torch.tensor(coordinate, dtype=g.dtype)
        target_coordinate = tuple(int(value.item()) for value in transformed)
        target = coordinate_to_index[target_coordinate]
        action[target, source] = 1.0

    return action


# The input is nine scalar values on a 3 x 3 patch. Coordinates are ordered
# from the top-left site to the bottom-right site.
PATCH_COORDINATES = tuple((col - 1, 1 - row) for row in range(3) for col in range(3))

# The output is four scalar predictions associated with directions. These are
# labels attached to lattice directions, not components of the input field.
DIRECTION_NAMES = ("up", "right", "down", "left")
DIRECTION_COORDINATES = ((0, 1), (1, 0), (0, -1), (-1, 0))

INPUT_ACTIONS = [action_matrix(g, PATCH_COORDINATES) for g in GROUP]
OUTPUT_ACTIONS = [action_matrix(g, DIRECTION_COORDINATES) for g in GROUP]
mlp = nn.Sequential(nn.Linear(9, 16), nn.Tanh(), nn.Linear(16, 4))


def transform_rows(x, action):
    """Apply an action matrix to samples stored as rows.

    If x has shape (batch, n) and action has shape (n, n), then
    ``x @ action.T`` has shape (batch, n).

    ``@`` means matrix multiplication and ``.T`` means transpose. A column
    transforms as x' = action x. Since PyTorch stores this batch as rows, the
    equivalent expression is x' = x @ action.T.
    """
    return x @ action.T


def equivariant_model(x):
    """Map (batch, 9) scalar patches to (batch, 4) directional scalars."""
    aligned_predictions = []
    for input_action, output_action in zip(
        INPUT_ACTIONS, OUTPUT_ACTIONS, strict=True
    ):
        transformed_x = transform_rows(x, input_action)
        raw_prediction = mlp(transformed_x)  # shape: (batch, 4)

        # output_action is a permutation matrix, so its inverse is its
        # transpose. In row storage, applying that inverse becomes
        # raw_prediction @ output_action.
        aligned_predictions.append(raw_prediction @ output_action)

    # stacked shape: (8, batch, 4); mean over the eight group elements.
    return torch.stack(aligned_predictions).mean(dim=0)


def print_group_actions(x):
    """Print the scalar input patch after every D4 transformation."""
    print("Input scalar patch:")
    print(x[0].reshape(3, 3))
    for name, action in zip(GROUP_NAMES, INPUT_ACTIONS, strict=True):
        transformed = transform_rows(x, action)
        print(f"\n{name} acting on the input:")
        print(transformed[0].reshape(3, 3))


# Nine distinguishable scalar site values make every spatial action visible.
x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
print_group_actions(x)

# Equivariance means P[f](h x) = h P[f](x). Here h permutes the four
# directional scalar outputs (up, right, down, left).
reference = equivariant_model(x)
print("\nDirectional output order:", DIRECTION_NAMES)
print("Output for the original input:", reference[0].detach())

errors = []
for name, input_action, output_action in zip(
    GROUP_NAMES, INPUT_ACTIONS, OUTPUT_ACTIONS, strict=True
):
    transformed_x = transform_rows(x, input_action)
    prediction_after_input_action = equivariant_model(transformed_x)
    expected_transformed_output = transform_rows(reference, output_action)
    error = (
        prediction_after_input_action - expected_transformed_output
    ).abs().max().item()
    errors.append(error)
    print(f"{name:4s}: |P[f](h x) - h P[f](x)| = {error:.3e}")

print(f"\nLargest equivariance error: {max(errors):.3e}")
assert max(errors) < 1e-12
