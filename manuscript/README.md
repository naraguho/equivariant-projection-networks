# Manuscript material

This directory contains the full FK and Holstein research workflow. It is not
needed for the two introductory examples in the repository root.

## Contents

- `epn/`: reusable $D_4$ transformations and projection models.
- `examples/`: manuscript-sized model demonstrations.
- `training/`: FK and Holstein training programs.
- `benchmarks/`: saved ED-versus-ML and correlation-collapse plots.
- `notebooks/`: the visual tutorial with rendered outputs.
- `scripts/`: dataset download and notebook-generation utilities.
- `tests/`: numerical symmetry tests.
- `DATASETS.md`: data provenance, splits, formats, and checksums.
- `MODELS.md`: manuscript network dimensions.

## Install and test

From this directory:

```bash
python -m pip install -e ".[dev]"
pytest
```

The full datasets and trained checkpoints remain in the GitHub release rather
than in the Git repository. See `DATASETS.md` for download instructions.
