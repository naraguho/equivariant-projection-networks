"""Make four scalar directional outputs exactly D4 equivariant."""

import torch
from torch import nn


# Float64 makes the final numerical symmetry error easier to see.
torch.set_default_dtype(torch.float64)
# Fix the randomly initialized MLP so every run prints the same result.
torch.manual_seed(7)


def d4_matrices():
    """Return the eight 2 x 2 matrices acting on square-lattice positions."""
    # R sends the coordinate (x, y) to (-y, x): a 90-degree rotation.
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    # S sends (x, y) to (x, -y): reflection across the x axis.
    reflection = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    # I leaves every coordinate unchanged.
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
    """Turn a 2 x 2 coordinate action into a scalar-label permutation.

    If ``coordinates`` contains n sites, the returned matrix has shape (n, n).
    The 2 x 2 matrix ``g`` acts on each site's coordinate, while the resulting
    n x n matrix moves the scalar values attached to those sites.
    """
    # Start with an empty n x n matrix. It will contain one 1 per row/column.
    action = torch.zeros(len(coordinates), len(coordinates))

    # This dictionary converts a transformed coordinate back to its list index.
    # Example for the four outputs: (0, 1) -> index 0, meaning ``up``.
    coordinate_to_index = {coordinate: i for i, coordinate in enumerate(coordinates)}

    # Visit each source site or direction exactly once.
    for source, coordinate in enumerate(coordinates):
        # Convert the coordinate tuple to a length-2 tensor and apply the
        # ordinary 2 x 2 matrix multiplication (x', y') = g @ (x, y).
        transformed = g @ torch.tensor(coordinate, dtype=g.dtype)

        # Matrix entries are exact integers here; convert the tensor back to a
        # coordinate tuple and find which site/direction has that coordinate.
        target_coordinate = tuple(int(value.item()) for value in transformed)
        target = coordinate_to_index[target_coordinate]

        # Record that the scalar label at ``source`` moves to ``target``.
        action[target, source] = 1.0

    return action


# The input is nine scalar values on a 3 x 3 patch.
PATCH_SIDE = 3
PATCH_CENTER = PATCH_SIDE // 2  # 1 for a 3 x 3 patch

# Convert array indices to centered Cartesian coordinates:
# x = column - center and y = center - row. The minus sign in y is needed
# because array rows increase downward, while Cartesian y increases upward.
# The ordering remains top-left, top-center, ..., bottom-right.
PATCH_COORDINATES = tuple(
    (col - PATCH_CENTER, PATCH_CENTER - row)
    for row in range(PATCH_SIDE)
    for col in range(PATCH_SIDE)
)

# The output is four scalar predictions associated with directions. These are
# labels attached to lattice directions, not components of the input field.
DIRECTION_NAMES = ("up", "right", "down", "left")
DIRECTION_COORDINATES = ((0, 1), (1, 0), (0, -1), (-1, 0))

# Each 2 x 2 D4 matrix becomes a 9 x 9 permutation of scalar input sites.
INPUT_ACTIONS = [action_matrix(g, PATCH_COORDINATES) for g in GROUP]
# The same matrix becomes a 4 x 4 permutation of directional scalar outputs.
OUTPUT_ACTIONS = [action_matrix(g, DIRECTION_COORDINATES) for g in GROUP]

# Ordinary MLP: nine scalar inputs -> 16 hidden values -> four scalar outputs.
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
    # We will put all eight MLP predictions back into the original output
    # orientation before averaging them.
    aligned_predictions = []
    for input_action, output_action in zip(
        INPUT_ACTIONS, OUTPUT_ACTIONS, strict=True
    ):
        # Rotate/reflect the nine scalar input sites.
        transformed_x = transform_rows(x, input_action)
        # Evaluate the same ordinary MLP; no special equivariant layer is used.
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


# Store one 3 x 3 scalar patch as one flattened row with shape (1, 9).
# Distinct values make every rotation and reflection visible in the printout.
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
    # Apply the test group element h to the nine scalar input sites.
    transformed_x = transform_rows(x, input_action)
    # Left side of the equivariance identity: P[f](h x).
    prediction_after_input_action = equivariant_model(transformed_x)
    # Right side: h P[f](x), which permutes the four directional scalars.
    expected_transformed_output = transform_rows(reference, output_action)
    error = (
        prediction_after_input_action - expected_transformed_output
    ).abs().max().item()
    errors.append(error)
    print(f"{name:4s}: |P[f](h x) - h P[f](x)| = {error:.3e}")

print(f"\nLargest equivariance error: {max(errors):.3e}")
assert max(errors) < 1e-12
