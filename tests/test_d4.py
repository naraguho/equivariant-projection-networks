import numpy as np

from epn import D4_MATRICES, circular_offsets, square_offsets


def test_d4_has_eight_unique_orthogonal_matrices():
    flattened = {tuple(matrix.ravel()) for matrix in D4_MATRICES}
    assert len(flattened) == 8
    for matrix in D4_MATRICES:
        np.testing.assert_array_equal(matrix.T @ matrix, np.eye(2, dtype=int))
        assert round(np.linalg.det(matrix)) in (-1, 1)


def test_local_environment_sizes_match_manuscript():
    assert len(square_offsets(5)) == 25
    assert len(circular_offsets(10)) == 317


def test_d4_is_closed():
    elements = {tuple(matrix.ravel()) for matrix in D4_MATRICES}
    for left in D4_MATRICES:
        for right in D4_MATRICES:
            assert tuple((left @ right).ravel()) in elements

