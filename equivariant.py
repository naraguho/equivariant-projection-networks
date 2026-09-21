"""Make four scalar directional outputs exactly D4 equivariant."""

import torch
from torch import nn


torch.set_default_dtype(torch.float64)
torch.manual_seed(7)


# Each entry is (name, number of 90-degree counterclockwise rotations,
# reflect first?). This is D4 = {I, S, R, RS, R^2, R^2S, R^3, R^3S}.
GROUP_ACTIONS = (
    ("I", 0, False),
    ("S", 0, True),
    ("R", 1, False),
    ("RS", 1, True),
    ("R^2", 2, False),
    ("R^2S", 2, True),
    ("R^3", 3, False),
    ("R^3S", 3, True),
)

# The four MLP outputs are scalar labels attached to lattice directions.
DIRECTION_NAMES = ("up", "right", "down", "left")


def transform_patch(x, rotations, reflected):
    """Rotate/reflect scalar patches with shape (batch, 3, 3)."""
    if reflected:
        # S: exchange the top and bottom rows (reflection across the x axis).
        x = torch.flip(x, dims=(-2,))

    # R^k: rotate the two spatial dimensions counterclockwise k times.
    return torch.rot90(x, k=rotations, dims=(-2, -1))


def reflect_directions(y):
    """Reflect (up, right, down, left) across the x axis."""
    # Up and down exchange; right and left remain where they are.
    return y[:, [2, 1, 0, 3]]


def inverse_transform_directions(y, rotations, reflected):
    """Undo a D4 action on the four directional scalar outputs."""
    # The forward action is g = R^k S: reflect first, then rotate.
    # Therefore g^(-1) = (R^k S)^(-1) = S^(-1) R^(-k) = S R^(-k).
    # Operations are consequently undone in reverse order: first apply
    # R^(-k) with a positive roll, and then apply S if reflection was used.
    y = torch.roll(y, shifts=rotations, dims=-1)
    if reflected:
        y = reflect_directions(y)
    return y


# Ordinary MLP: nine scalar inputs -> 16 hidden values -> four scalar outputs.
mlp = nn.Sequential(nn.Linear(9, 16), nn.Tanh(), nn.Linear(16, 4))


def equivariant_model(x):
    """Map (batch, 3, 3) scalar patches to (batch, 4) directional scalars."""
    aligned_predictions = []

    for _, rotations, reflected in GROUP_ACTIONS:
        transformed = transform_patch(x, rotations, reflected)

        # Flatten only when passing the 3 x 3 scalar patch into the MLP.
        raw_prediction = mlp(transformed.flatten(start_dim=1))

        # Return this prediction to the original directional frame.
        aligned = inverse_transform_directions(
            raw_prediction, rotations, reflected
        )
        aligned_predictions.append(aligned)

    # Shape before averaging: (8, batch, 4). Average over the D4 actions.
    return torch.stack(aligned_predictions).mean(dim=0)


def print_group_actions(x):
    """Print one scalar patch after all eight D4 actions."""
    print("Input scalar patch:")
    print(x[0])

    for name, rotations, reflected in GROUP_ACTIONS:
        transformed = transform_patch(x, rotations, reflected)
        print(f"\n{name} acting on the input:")
        print(transformed[0])


# One batch member containing a 3 x 3 patch of scalar site values.
x = torch.tensor([[[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]]])
print_group_actions(x)

# Verify P[f](h x) = h P[f](x) for every h in D4.
reference = equivariant_model(x)
print("\nDirectional output order:", DIRECTION_NAMES)
print("Output for the original input:", reference[0].detach())


# The function below is not part of the equivariant neural network. It is used
# only to verify the identity P[f](h x) = h P[f](x).
def transform_directions(y, rotations, reflected):
    """Apply a D4 action to four directional scalars for the final test."""
    if reflected:
        y = reflect_directions(y)

    # One counterclockwise rotation changes [up, right, down, left] into
    # [right, down, left, up].
    return torch.roll(y, shifts=-rotations, dims=-1)


errors = []
for name, rotations, reflected in GROUP_ACTIONS:
    transformed_x = transform_patch(x, rotations, reflected)
    prediction_after_input_action = equivariant_model(transformed_x)
    expected_transformed_output = transform_directions(
        reference, rotations, reflected
    )
    error = (
        prediction_after_input_action - expected_transformed_output
    ).abs().max().item()
    errors.append(error)
    print(f"{name:4s}: |P[f](h x) - h P[f](x)| = {error:.3e}")

print(f"\nLargest equivariance error: {max(errors):.3e}")
assert max(errors) < 1e-12
