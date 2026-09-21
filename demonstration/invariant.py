"""Make a scalar-output MLP exactly D4 invariant by group averaging."""

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


def transform_patch(x, rotations, reflected):
    """Rotate/reflect scalar patches with shape (batch, 3, 3)."""
    if reflected:
        # S: exchange the top and bottom rows (reflection across the x axis).
        x = torch.flip(x, dims=(-2,))

    # R^k: rotate the two spatial dimensions counterclockwise k times.
    return torch.rot90(x, k=rotations, dims=(-2, -1))


# Ordinary MLP: nine scalar inputs -> 16 hidden values -> one scalar output.
mlp = nn.Sequential(nn.Linear(9, 16), nn.Tanh(), nn.Linear(16, 1))


def invariant_model(x):
    """Map (batch, 3, 3) scalar patches to (batch, 1) invariant scalars."""
    predictions = []

    for _, rotations, reflected in GROUP_ACTIONS:
        transformed = transform_patch(x, rotations, reflected)

        # Linear layers expect one vector per sample, so flatten 3 x 3 to 9.
        # This is the only flattening needed in the example.
        transformed = transformed.flatten(start_dim=1)
        predictions.append(mlp(transformed))

    # Shape before averaging: (8, batch, 1). Average over the D4 actions.
    return torch.stack(predictions).mean(dim=0)


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

# Verify P[f](h x) = P[f](x) for every h in D4.
reference = invariant_model(x)
print(f"\nInvariant scalar for the original input: {reference.item():.12f}")

errors = []
for name, rotations, reflected in GROUP_ACTIONS:
    transformed_x = transform_patch(x, rotations, reflected)
    error = (invariant_model(transformed_x) - reference).abs().max().item()
    errors.append(error)
    print(f"{name:4s}: |P[f](h x) - P[f](x)| = {error:.3e}")

print(f"\nLargest invariance error: {max(errors):.3e}")
assert max(errors) < 1e-12
