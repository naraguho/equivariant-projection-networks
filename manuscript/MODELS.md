# Manuscript configurations

The compact examples in `examples/` use the same projection as the manuscript.
The manuscript-scale network dimensions are:

| Application | Local input | MLP |
|---|---:|---|
| Holstein invariant energy | 5 x 5 patch (25 values) | 25 -> 512 -> 256 -> 128 -> 1 |
| Falicov-Kimball directional energy | circular radius 10 (317 values) | 317 -> 512 -> 512 -> 256 -> 128 -> 4 |

Large training datasets, checkpoints, and production trajectories are not
stored in Git.  This keeps the repository focused on the symmetry construction
and makes every included example runnable on a laptop.

