"""Load the real manuscript datasets and reproduce their leakage-free splits."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


HOLSTEIN_TRAIN_RUNS = (1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 17, 18, 19, 20)
HOLSTEIN_VALIDATION_RUNS = (10, 16)
HOLSTEIN_TEST_RUNS = (8, 15)
FK_TRAIN_TRAJECTORIES = tuple(range(1, 9))
FK_VALIDATION_TRAJECTORIES = (9, 10)


def load_holstein(directory: Path):
    """Load full extracted files or the compact real-data sample."""
    directory = Path(directory)
    sample = directory / "holstein_real_sample.npz"
    if sample.is_file():
        data = np.load(sample)
        return data["Q"], data["force"], data["metadata_run_step"]
    return (
        np.load(directory / "input_Q_snapshots_steps20_to10000.npy", mmap_mode="r"),
        np.load(directory / "output_force_snapshots_steps20_to10000.npy", mmap_mode="r"),
        np.load(directory / "metadata_run_step.npy"),
    )


def holstein_indices(metadata: np.ndarray, runs) -> np.ndarray:
    return np.flatnonzero(np.isin(metadata[:, 0], runs))


_LOCAL_COLUMN = re.compile(r"nf_dx([+-]\d+)_dy([+-]\d+)$")


def fk_local_columns(columns) -> tuple[list[str], tuple[tuple[int, int], ...]]:
    parsed = []
    for column in columns:
        match = _LOCAL_COLUMN.fullmatch(column)
        if match:
            parsed.append((column, (int(match.group(1)), int(match.group(2)))))
    parsed.sort(key=lambda item: (item[1][1], item[1][0]))
    return [item[0] for item in parsed], tuple(item[1] for item in parsed)


def load_fk(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

