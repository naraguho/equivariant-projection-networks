"""Make a scalar-output MLP exactly D4 invariant by group averaging."""

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


def patch_action_matrix(g, side=3):
    """Convert a 2 x 2 spatial action into a permutation of scalar sites.

    ``g`` is only 2 x 2 because it acts on one coordinate (x, y). We apply it
    separately to every site, then record where each site's scalar value moves.
    The input contains ``side * side`` scalar values, not vector components.
    The returned matrix has shape (side^2, side^2) and only moves those scalar
    values between lattice sites.
    """
    # For side=3, center=1. Array position (row=1, col=1) is coordinate (0, 0).
    center = side // 2

    # This will become a 9 x 9 permutation matrix for a 3 x 3 patch.
    action = torch.zeros(side * side, side * side)

    # Visit each of the nine source sites once.
    for source_row in range(side):
        for source_col in range(side):
            # Convert array indices to a centered coordinate:
            # top-left -> (-1, 1), center -> (0, 0), bottom-right -> (1, -1).
            coordinate = torch.tensor(
                [source_col - center, center - source_row], dtype=g.dtype
            )

            # g is 2 x 2 and coordinate has length 2, so this ordinary matrix
            # multiplication returns the transformed coordinate (x', y').
            transformed_coordinate = g @ coordinate

            # Convert the transformed coordinate back to array indices.
            target_col = int(transformed_coordinate[0].item()) + center
            target_row = center - int(transformed_coordinate[1].item())

            # Flatten (row, col) into one index from 0 to 8:
            # [[0, 1, 2], [3, 4, 5], [6, 7, 8]].
            source = source_row * side + source_col
            target = target_row * side + target_col

            # Record: the scalar at ``source`` must move to ``target``.
            action[target, source] = 1.0

    # Applying x @ action.T now rearranges all nine scalar site values at once.
    return action


INPUT_ACTIONS = [patch_action_matrix(g) for g in GROUP]
mlp = nn.Sequential(nn.Linear(9, 16), nn.Tanh(), nn.Linear(16, 1))


def transform_scalar_patch(x, action):
    """Apply one D4 action to a batch of flattened scalar patches.

    Dimensions:
        x            : (batch, 9)
        action       : (9, 9)
        action.T     : (9, 9)
        returned x'  : (batch, 9)

    ``@`` is Python's matrix-multiplication operator. ``action.T`` is the
    transpose of ``action``. Mathematically, a column patch transforms as
    x' = action x. PyTorch stores each sample as a row, so the batched form is
    x' = x @ action.T. This permutes sites; it does not rotate scalar values.
    """
    return x @ action.T


def invariant_model(x):
    """Map (batch, 9) scalar patches to (batch, 1) invariant scalars."""
    predictions = [
        mlp(transform_scalar_patch(x, action)) for action in INPUT_ACTIONS
    ]
    # stacked shape: (8, batch, 1); mean over the eight group elements.
    return torch.stack(predictions).mean(dim=0)


def print_group_actions(x):
    """Print the original scalar patch after every D4 transformation."""
    print("Input scalar patch:")
    print(x[0].reshape(3, 3))
    for name, action in zip(GROUP_NAMES, INPUT_ACTIONS, strict=True):
        transformed = transform_scalar_patch(x, action)
        print(f"\n{name} acting on the input:")
        print(transformed[0].reshape(3, 3))


# Nine distinguishable scalar site values make every spatial action visible.
x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
print_group_actions(x)

# Invariance means P[f](h x) = P[f](x) for every h in D4.
reference = invariant_model(x)
print(f"\nInvariant scalar for the original input: {reference.item():.12f}")

errors = []
for name, action in zip(GROUP_NAMES, INPUT_ACTIONS, strict=True):
    transformed_x = transform_scalar_patch(x, action)
    transformed_prediction = invariant_model(transformed_x)
    error = (transformed_prediction - reference).abs().max().item()
    errors.append(error)
    print(f"{name:4s}: |P[f](h x) - P[f](x)| = {error:.3e}")

print(f"\nLargest invariance error: {max(errors):.3e}")
assert max(errors) < 1e-12
