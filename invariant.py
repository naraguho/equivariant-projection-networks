"""Make an ordinary scalar MLP exactly D4 invariant by averaging."""

import torch
from torch import nn


torch.set_default_dtype(torch.float64)
torch.manual_seed(7)


def d4_matrices():
    """The four rotations and four reflections of a square."""
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    reflection = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    identity = torch.eye(2)

    group = []
    power = identity
    for _ in range(4):
        group.extend((power, power @ reflection))
        power = power @ rotation
    return group


GROUP = d4_matrices()
mlp = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))


def invariant_model(x):
    """P[f](x) = (1/|G|) sum_g f(g x)."""
    predictions = [mlp(x @ g.T) for g in GROUP]
    return torch.stack(predictions).mean(dim=0)


# Two arbitrary input vectors. No training is needed for the symmetry test.
x = torch.tensor([[0.3, -0.8], [1.2, 0.4]])
reference = invariant_model(x)

errors = []
for index, h in enumerate(GROUP):
    error = (invariant_model(x @ h.T) - reference).abs().max().item()
    errors.append(error)
    print(f"D4 element {index}: invariance error = {error:.3e}")

print(f"\nlargest invariance error: {max(errors):.3e}")
assert max(errors) < 1e-12
