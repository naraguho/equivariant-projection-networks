# Pedagogical demonstrations

This folder contains the shortest complete path from symmetry averaging to
training on real manuscript data.

| File | Purpose |
|---|---|
| `invariant.py` | Scalar $D_4$ invariance on a visible $3\times3$ patch |
| `equivariant.py` | Four-direction $D_4$ equivariance on the same patch |
| `train_holstein.py` | Local energy, total energy, $-\partial E/\partial Q$, training, and held-out $y=x$ plot |
| `train_fk.py` | Four-direction free-energy model, masked loss, training, and held-out $y=x$ plot |

The training scripts are self-contained: their model, symmetry projection,
loss, and validation code remain in the same file. They do not call the more
modular implementation under `manuscript/`.

The bundled data are small deterministic subsets of the real ED-derived data:

- FK: 512 training environments from trajectory 1 and 256 validation
  environments from trajectory 9.
- Holstein: two complete $30\times30$ training snapshots from run 1 and one
  complete validation snapshot from run 10.

From the repository root, run both examples with:

```bash
bash run_training.sh
```

The default is 50 epochs, Holstein first and FK second. Each validation plot
is displayed on screen and also saved as:

```text
demonstration/output/holstein_validation_y_equals_x.png
demonstration/output/fk_validation_y_equals_x.png
```

Plot windows require a graphical Python session. On a headless SSH or batch
node, the PNG files are still produced and can be downloaded or opened later.

The Holstein MLP is `25 -> 64 -> 32 -> 1` with 3,776 trainable parameters.
The FK MLP is `317 -> 128 -> 64 -> 4` with 49,220 trainable parameters. The
eightfold symmetry projections reuse each MLP, so they add no parameters.
