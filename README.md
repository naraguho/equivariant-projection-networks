# Equivariant Projection Networks

Exact finite-group equivariance from ordinary neural networks.

This repository accompanies the manuscript **"Exact Equivariance from
Ordinary Neural Networks for Lattice Many-Body Dynamics."**  Its goal is to
show the central method with short, readable PyTorch code.  The implementation
favors explicit transformations and straightforward loops over highly
optimized, application-specific machinery.

## The main idea

Let a finite group `G` act on an input `x` through `D_X(g)` and on an output
through `D_Y(g)`.  An arbitrary neural network `f` can be projected onto an
exactly equivariant function:

```text
                         1
P[f](x) =             --------  sum  D_Y(g)^(-1) f(D_X(g) x).
                       | G |    g in G
```

The internal network does not need equivariant layers.  It can be an ordinary
MLP.  Exact symmetry follows from transforming the input, undoing the output
transformation, and averaging.

For an invariant scalar, `D_Y(g) = 1`, so the expression becomes ordinary
group averaging:

```python
predictions = [mlp(transform(x, g)) for g in group]
invariant_prediction = torch.stack(predictions).mean(dim=0)
```

For a directional output, each prediction is first returned to the original
coordinate frame:

```python
aligned = [inverse_output(g, mlp(transform(x, g))) for g in group]
equivariant_prediction = torch.stack(aligned).mean(dim=0)
```

The code in [`epn/projection.py`](epn/projection.py) is essentially these two
snippets plus explicit permutation handling.

## Two manuscript examples

### Falicov-Kimball directional free-energy changes

The input consists of 317 binary occupations in a circular neighborhood of
radius `R_c = 10`.  The four outputs are ordered as

```text
(Delta F_{+x}, Delta F_{-x}, Delta F_{+y}, Delta F_{-y}).
```

Rotations and reflections permute both the input sites and output directions.
The equivariant projection enforces the correct transformation exactly.  The
MLP architecture is

```text
317 -> 512 -> 512 -> 256 -> 128 -> 4
```

with SiLU activations between hidden layers.

### Holstein invariant local energy

The input is a `5 x 5` displacement patch.  The scalar local energy is averaged
over the eight elements of `D4`.  Summing local energies produces a total
machine-learning energy,

```text
E_ML(Q) = sum_i epsilon_i(Q),
```

and PyTorch automatic differentiation gives a conservative force,

```text
F_i = -d E_ML / d Q_i.
```

The local-energy MLP architecture is

```text
25 -> 512 -> 256 -> 128 -> 1
```

with SiLU activations between hidden layers.

## Installation

```bash
git clone https://github.com/naraguho/equivariant-projection-networks.git
cd equivariant-projection-networks
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run the examples

Start with the two smallest symmetry demonstrations:

```bash
python examples/01_invariant_scalar.py
python examples/02_equivariant_vector.py
```

They print the maximum symmetry error for every element of `D4`.  In double
precision, the tests verify these identities to numerical roundoff.

The manuscript-sized model demonstrations are:

```bash
python examples/03_fk_directional_energy.py
python examples/04_holstein_conservative_force.py
```

These scripts initialize the actual network dimensions but use small random
inputs, so no training dataset or checkpoint is required.

## Tests

```bash
pytest
```

The tests check:

1. the eight matrices form the group `D4`;
2. the manuscript neighborhood sizes are 25 and 317;
3. scalar projection is exactly invariant;
4. four-direction projection is exactly equivariant; and
5. differentiating an invariant energy produces a covariant force.

## Repository map

```text
epn/d4.py          explicit D4 matrices and lattice neighborhoods
epn/projection.py  invariant and equivariant group averages
epn/models.py      ordinary MLP backbones wrapped by the projection
examples/          progressively more physical demonstrations
tests/             numerical symmetry checks
```

## Scope

The repository includes small samples drawn from the real ED-derived datasets
in `data/sample/`.  They exercise the complete data-loading and training path.
The full datasets, saved validation rollouts, and trained checkpoints are
distributed as checksummed GitHub release assets:

```bash
python scripts/download_data.py --extract
```

Train on the full FK data with

```bash
python training/train_fk.py \
  --data data/full/fk_edkmc_rc10_full.csv.gz \
  --epochs 500
```

Train on the full Holstein data with

```bash
python training/train_holstein.py \
  --data-dir data/full/holstein_ed_forces_full \
  --epochs 100 --batch-size 24
```

Reproduce the saved ED-versus-ML correlation benchmarks and Holstein scaling
collapse with

```bash
python benchmarks/plot_saved_benchmarks.py
```

See [`DATASETS.md`](DATASETS.md) for provenance, physical parameters, split
definitions, file formats, and checksums.

The large production trajectories themselves remain on the HPC system; the
release contains their correlation observables needed for the manuscript
figures rather than hundreds of gigabytes of redundant snapshots.

## Citation

Citation information will be added when the manuscript record is public.

## License

MIT
