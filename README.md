# PyBMD: A Python BMD package

A Python package for **bispectral mode decomposition** (BMD) and **cross-bispectral mode
decomposition** (CBMD).

Triadic interactions are the fundamental mechanism of energy transfer in fluid flows. BMD
detects quadratic phase coupling through the bispectrum and extracts the coherent structures
associated with it, distinguishing sum- from difference-interactions and producing interaction
maps that identify the regions of nonlinear coupling.

The architecture follows [PySPOD](https://github.com/MathEXLab/PySPOD): a `params`-dict-driven
`Base`/`Standard` class pair, an optional MPI communicator and disk-backed mode storage.

```
          f2 or l
             ^
     ________|
     |\      |\
     |  \  7 |  \
     | 6  \  | 8 /\
     |      \| / 1  \
 ----+-------+-------+-> f1 or k
      \  5 / |\      |
        \/ 4 |  \  2 |
          \  | 3  \  |
            \|______\|
             |
```
*Regions of the f1–f2 plane, selected with `params['regions']`. Triads are expressed as frequency
triplets `{f1, f2, f1+f2}`, or index triplets `(k, l, k+l)`.*

## Installation

```bash
pip install -e .              # core: numpy, scipy, pyyaml, matplotlib
pip install -e '.[mpi]'       # add mpi4py for parallel runs
pip install -e '.[io,test]'   # .mat reader (v7.3), and pytest
```

## Usage

```python
import numpy as np
from pybmd.bmd.standard import Standard
import pybmd.utils.weights as utils_weights

# data has time first and the variable index last: (nt, *spatial, n_variables).
# Spatial axes follow NumPy/matplotlib: (ny, nx) in 2-D, (nz, ny, nx) in 3-D.
# MATLAB/Fortran arrays are (nx, ny): transpose them first, e.g. u.transpose(0, 2, 1).
data = ...   # (nt, ny, nx, 2)

params = dict(
    n_dft=256,               # snapshots per block
    time_step=0.12,
    n_space_dims=2,
    n_variables=2,
    overlap=50,              # percent; or n_overlap=128 in snapshots
    regions=[1, 2],          # sum- and difference-interactions
    max_freq_idx=24,         # restrict to |k|, |l| <= 24
    solver='MengiOverton',
    savedir='bmd_results',
)
weights = utils_weights.trapz_2d(x, y, n_vars=2)   # x: (nx,), y: (ny,)
bmd = Standard(params=params, weights=weights).fit(data)

# the mode bispectrum, NaN outside the computed triads
L = bmd.L

# look a triad up by its index doublet, then load its two modes
i = bmd.triads.find(k=5, l=-2)
psi_sum, psi_prod = bmd.get_modes_at_triad(i)   # phi_{k+l}, phi_{k o l}
```

Set `params['constituent_modes'] = True` (`Standard` only) to also compute the two modes at the
triad's own constituent frequencies, `phi_k` and `phi_l`; `get_modes_at_triad` then returns four
modes, `(phi_{k+l}, phi_{k o l}, phi_k, phi_l)`, and `plot_triad_modes` draws two extra rows.

Plotting:

```python
from pybmd.bmd.postproc import plot_mode_bispectrum, plot_triad_modes
plot_mode_bispectrum(bmd.L, bmd.freq)
plot_triad_modes(bmd.get_modes_at_triad(i), k=5, l=-2, x=x, y=y)
```

Post-processing takes a fitted `Standard`/`Cross`, not a path. To revisit a result later, pickle
the fitted object (modes saved with `save_modes` are read back from `bmd.savedir_sim`, so keep
that directory):

```python
import pickle
from pybmd.bmd.postproc import top_triads, plot_mode_bispectrum, plot_triad_modes

with open('bmd.pkl', 'wb') as f:
    pickle.dump(bmd, f)
with open('bmd.pkl', 'rb') as f:
    bmd = pickle.load(f)

top = top_triads(bmd, n=5)
plot_mode_bispectrum(bmd.L, bmd.freq)
plot_triad_modes(bmd.get_modes_at_triad(int(top[0]['triad_idx'])),
                 int(top[0]['k']), int(top[0]['l']), x=x, y=y)
```

Running in parallel — the triad loop is distributed across ranks and results are identical to a
serial run:

```bash
mpirun -n 8 python my_script.py     # pass comm=MPI.COMM_WORLD to the constructor
```

Cross-BMD, for a quadratic term built from different variables:

```python
from pybmd.bmd.cross import Cross
# s_0 <- q_1 * r_2, with 0-based variable indices
cbmd = Cross(params=dict(params, state_idx=[0], qr_idx=[[1, 2]]),
             weights=utils_weights.trapz_2d(x, y, n_vars=None)).fit(data)
```

See [`examples/`](examples/) for the three worked cases, which mirror `example1.m`–`example3.m` of
the original MATLAB implementation, and for reproductions of Schmidt (2020)'s figures:
`example4_hypothesis_testing.py` (surrogate data, Figs. 4 and 5)
and `example5_cylinder_paper.py` (cylinder-wake mode bispectrum and modes, Figs. 7-9; see
`example5_cylinder_paper.md`).

## Parameters

**Required:** `n_dft`, `time_step`, `n_space_dims`, `n_variables`.

| Optional | Default | Meaning |
| --- | --- | --- |
| `overlap` | `50` | block overlap, in **percent** |
| `n_overlap` | — | block overlap in snapshots; takes precedence over `overlap` |
| `window` | `'hamming'` | `'hamming'` or `'hann'` |
| `regions` | `[1, 2]` | regions of the bispectrum to compute, in 1..8 |
| `max_freq_idx` | `None` | bound on `\|k\|` and `\|l\|`; default is Nyquist |
| `tol` | `1e-6` | Mengi–Overton solver tolerance |
| `n_it_max` | `500` | Mengi–Overton iteration cap |
| `dtype` | `'double'` | `'double'` or `'single'`: precision of the computation |
| `save_dtype` | `dtype` | `'double'` or `'single'`: precision of the arrays written to disk (`bispectrum.npz`'s `L`/`T`, `coeffs.npy`, `weights.npy`, `ltm_modes.npy`, `modes/`); the frequencies stay double |
| `save_modes` | `True` | write `modes/triad_idx_{i:08d}.npy` |
| `save_modes_top` | `None` | write the modes of only the strongest triads, ranked by `\|L\|` as `top_triads` ranks them (`k = 0` and `l = 0` excluded): an `int` is a count, a `float` in (0, 1] a fraction; `None` writes every triad |
| `store_modes` | `False` | also keep all modes in memory, exposed as `.modes` |
| `max_modes_gb` | `8.0` | refuse to run if `save_modes`/`store_modes` would keep more than this many GB of modes (on disk at `save_dtype`, in memory at `dtype`); `None` disables the check |
| `compute_energy_transfer` | `True` | fill the energy-transfer term `T` |
| `savedir` | `'bmd_results'` | results directory |

Results are written to `<savedir>/nfft{n_dft}_novlp{n_overlap}_nblks{n_blocks}/`, holding
`bispectrum.npz`, `triads.npz`, `coeffs.npy`, `weights.npy`, `ltm_modes.npy`,
`params_modes.yaml` and `modes/`. With `save_modes_top`, `modes/` holds only the selected
triads — still named by their global `triad_idx`, so the numbering is sparse — and
`modes/saved_triad_idx.npy` lists them.

`coeffs.npy` holds the maximisers of the numerical radius, one short vector per triad. Since the
modes are just `Q @ a`, they can be rebuilt from these without re-running the optimizer — which
is what makes it practical to run a large case with `save_modes=False` and decide afterwards
which triads are worth reconstructing. On the fitted object, `bmd.reconstruct_modes(data, i)`
recomputes only the DFT rows of triad(s) `i` and applies `coeffs`, giving the same modes `fit`
would have saved; `get_modes_at_triad(i, data=data)` / `get_modes_at_freqs(k, l, data=data)` fall
back to it for any triad without a mode file, so they can feed `plot_triad_modes` directly:

```python
plot_triad_modes(bmd.get_modes_at_freqs(k, l, data=data), k, l, x=x, y=y)
```

## Deviations from the reference implementation

The algorithm is ported from O. T. Schmidt's MATLAB `bmd.m` and `cbmd.m`. Four deliberate
departures, each of which changes results:

1. **`max_fov` uses the signed largest eigenvalue** of the Hermitian part, not the largest in
   modulus. The Mengi–Overton level set is defined by the signed `λ_max`; filtering the
   crossing angles by modulus discards valid ones, so the search terminates at a *local*
   maximum. Measured on a random 7×7 complex matrix: 3.4973 against a true 4.4346.
2. **The matrix is pre-scaled by a power of two** before the level-set search. The unit-circle
   test `|‖D‖ − 1| ≤ sqrt(eps)·‖A‖₁` is an absolute tolerance scaled by the norm, and the
   matrices BMD produces are small — `B` carries a `1/n_blocks` and the quadrature weights.
   Without rescaling, every crossing is rejected and the solver returns a local maximum;
   measured at `‖A‖₁ ~ 1e-6`, it returned 93.7 % of the true value. Scaling by a power of two
   is exact in binary floating point, so this only re-conditions the problem.
3. **The energy-transfer term `T` is computed**, and the solver uses no RNG, so results do not
   depend on how triads are distributed across MPI ranks.
4. **The level-set filter uses `sqrt(eps)·max(w, 1)`** rather than the reference's
   `sqrt(eps)·w`, which for the tiny levels of real BMD matrices rejects valid crossings. See
   [`pybmd/bmd/CLAUDE.md`](pybmd/bmd/CLAUDE.md) for the measurements, and for a fifth, cosmetic
   difference in how crossing angles are de-duplicated.

The numerical radius is always maximised with Mengi–Overton's globally convergent level-set
algorithm (the default of `bmd.m` since its 17-Aug-2023 revision), with the fixes above.

## Testing

```bash
pytest                            # everything, ~90 s (Octave cross-validation, one mpirun test)
pytest -m "not slow and not mpi"  # fast subset, ~30 s
```

The suite checks the numerical-radius solvers against brute force, reproduces Schmidt (2020)'s
hypothesis test on surrogate data, asserts bit-identical results between `mpirun -n 1` and `-n 2`,
and regresses `L`, `T`, the modes and CBMD against the original MATLAB implementation run live
under Octave on the cylinder-wake dataset (see [`tests/CLAUDE.md`](tests/CLAUDE.md)).

## References

The original MATLAB implementation: <https://github.com/olivertschmidt/bmd>

~~~bibtex
@article{schmidt2020bispectral,
  title   = {Bispectral mode decomposition of nonlinear flows},
  author  = {Schmidt, Oliver T.},
  journal = {Nonlinear Dynamics},
  volume  = {102},
  number  = {4},
  pages   = {2479--2501},
  year    = {2020},
  doi     = {10.1007/s11071-020-06037-z}
}
~~~

The architectural template:

~~~bibtex
@article{mengaldo2021pyspod,
  title   = {PySPOD: A {P}ython package for Spectral Proper Orthogonal Decomposition ({SPOD})},
  author  = {Mengaldo, Gianmarco and Maulik, Romit},
  journal = {Journal of Open Source Software},
  volume  = {6},
  number  = {60},
  pages   = {2862},
  year    = {2021},
  doi     = {10.21105/joss.02862}
}
~~~

## License

MIT — see [LICENSE](LICENSE). The cylinder-wake test fixture is subsampled from the dataset
distributed with the reference MATLAB implementation.
