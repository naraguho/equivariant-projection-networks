"""Make an ordinary vector MLP exactly D4 equivariant by averaging."""

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
mlp = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))


def equivariant_model(x):
    """P[f](x) = (1/|G|) sum_g g^(-1) f(g x)."""
    aligned_predictions = []
    for g in GROUP:
        raw_prediction = mlp(x @ g.T)
        # Row-vector notation: multiplying by g applies g^{-1} to the output.
        aligned_predictions.append(raw_prediction @ g)
    return torch.stack(aligned_predictions).mean(dim=0)


# Two arbitrary input vectors. No training is needed for the symmetry test.
x = torch.tensor([[0.3, -0.8], [1.2, 0.4]])
reference = equivariant_model(x)

errors = []
for index, h in enumerate(GROUP):
    prediction_after_transform = equivariant_model(x @ h.T)
    transformed_reference = reference @ h.T
    error = (prediction_after_transform - transformed_reference).abs().max().item()
    errors.append(error)
    print(f"D4 element {index}: equivariance error = {error:.3e}")

print(f"\nlargest equivariance error: {max(errors):.3e}")
assert max(errors) < 1e-12
