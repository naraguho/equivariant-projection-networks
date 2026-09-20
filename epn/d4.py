"""The eight rotations and reflections of the square.

Coordinates are integer pairs ``(x, y)``.  Every matrix below maps a lattice
coordinate to another lattice coordinate.  Keeping the matrices explicit
makes the code easy to audit and avoids hidden conventions.
"""

from __future__ import annotations

import numpy as np


I = np.array([[1, 0], [0, 1]], dtype=int)
R = np.array([[0, -1], [1, 0]], dtype=int)  # 90-degree counterclockwise
M = np.array([[-1, 0], [0, 1]], dtype=int)  # reflection x -> -x

D4_MATRICES = tuple(
    [np.linalg.matrix_power(R, k).astype(int) for k in range(4)]
    + [(np.linalg.matrix_power(R, k) @ M).astype(int) for k in range(4)]
)

# Output ordering used for the Falicov-Kimball model.
DIRECTION_VECTORS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRECTION_NAMES = ("+x", "-x", "+y", "-y")


def square_offsets(size: int) -> tuple[tuple[int, int], ...]:
    """Return row-major offsets for an odd ``size x size`` local patch."""
    if size <= 0 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")
    radius = size // 2
    return tuple(
        (x, y)
        for y in range(-radius, radius + 1)
        for x in range(-radius, radius + 1)
    )


def circular_offsets(radius: int) -> tuple[tuple[int, int], ...]:
    """Return integer lattice sites inside ``x^2 + y^2 <= radius^2``."""
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    return tuple(
        (x, y)
        for y in range(-radius, radius + 1)
        for x in range(-radius, radius + 1)
        if x * x + y * y <= radius * radius
    )


def transform_coordinate(
    coordinate: tuple[int, int], matrix: np.ndarray
) -> tuple[int, int]:
    """Apply a D4 matrix to one integer coordinate."""
    transformed = matrix @ np.asarray(coordinate, dtype=int)
    return int(transformed[0]), int(transformed[1])

