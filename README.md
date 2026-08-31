# PyBMD: A Python BMD package

A Python package for **bispectral mode decomposition** (BMD) and **cross-bispectral mode
decomposition** (CBMD).

Triadic interactions are the fundamental mechanism of energy transfer in fluid flows. BMD
detects quadratic phase coupling through the bispectrum and extracts the coherent structures
associated with it, distinguishing sum- from difference-interactions and producing interaction
maps that identify the regions of nonlinear coupling.

The architecture follows [PySPOD](https://github.com/MathEXLab/PySPOD): a `params`-dict-driven
`Base`/`Standard` class pair, an optional MPI communicator, disk-backed mode storage, and a YAML
config reader.

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
pip install -e '.[io,test]'   # .mat/.nc readers, and pytest
```

## Usage

```python
import numpy as np
from pybmd.bmd.standard import Standard
import pybmd.utils.weights as utils_weights

# data has time first and the variable index last: (nt, *spatial, n_variables)
data = ...

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
weights = utils_weights.trapz_2d(x, y, n_vars=2)
bmd = Standard(params=params, weights=weights).fit(data)

# the mode bispectrum, NaN outside the computed triads
L = bmd.bispectrum

# look a triad up by its index doublet, then load its two modes
i = bmd.triads.find(k=5, l=-2)
psi_sum, psi_prod = bmd.get_modes_at_triad(i)   # phi_{k+l}, phi_{k o l}
```

Plotting:

```python
from pybmd.bmd.postproc import plot_mode_bispectrum, plot_triad_modes
plot_mode_bispectrum(bmd.L, bmd.freq)
plot_triad_modes(bmd.get_modes_at_triad(i), k=5, l=-2, x1=x[:, 0], x2=y[0, :])
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
the original MATLAB implementation.

## Parameters

**Required:** `n_dft`, `time_step`, `n_space_dims`, `n_variables`.

| Optional | Default | Meaning |
| --- | --- | --- |
| `overlap` | `50` | block overlap, in **percent** |
| `n_overlap` | — | block overlap in snapshots; takes precedence over `overlap` |
| `window` | `'hamming'` | `'hamming'`, `'hann'`, `'boxcar'`, or an array |
| `mean_type` | `'longtime'` | `'longtime'`, `'blockwise'`, `'zero'` |
| `regions` | `[1, 2]` | regions of the bispectrum to compute, in 1..8 |
| `max_freq_idx` | `None` | bound on `\|k\|` and `\|l\|`; default is Nyquist |
| `solver` | `'MengiOverton'` | also `'HeWatson'`, `'simpleIteration'` |
| `tol` | `1e-6` | solver tolerance |
| `n_it_max` | `500` | solver iteration cap |
| `dtype` | `'double'` | `'double'` or `'single'` |
| `save_modes` | `True` | write `modes/triad_idx_{i:08d}.npy` |
| `store_modes` | `False` | also keep all modes in memory, exposed as `.modes` |
| `max_modes_gb` | `8.0` | refuse to write more than this without an explicit raise |
| `compute_energy_transfer` | `True` | fill the energy-transfer term `T` |
| `savedir` | `'bmd_results'` | results directory |

Results are written to `<savedir>/nfft{n_dft}_novlp{n_overlap}_nblks{n_blocks}/`, holding
`bispectrum.npz`, `triads.npz`, `coeffs.npy`, `weights.npy`, `ltm_modes.npy`,
`params_modes.yaml` and `modes/`.

`coeffs.npy` holds the maximisers of the numerical radius, one short vector per triad. Since the
modes are just `Q @ a`, they can be rebuilt from these without re-running the optimizer — which
is what makes it practical to run a large case with `save_modes=False` and decide afterwards
which triads are worth reconstructing.

## Deviations from the reference implementation

The algorithm is ported from O. T. Schmidt's MATLAB `bmd.m` and `cbmd.m`. Three deliberate
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
3. **The energy-transfer term `T` is computed**, and the solvers use a deterministic start
   vector rather than a global RNG, so results do not depend on how triads are distributed
   across MPI ranks.

`solver='simpleIteration'` reproduces the reference's only solver. It is not globally
convergent — on random matrices it under-estimated the numerical radius in 14 of 40 cases, worst
case 62 % low — so `MengiOverton` is the default.

## Testing

```bash
pytest                       # everything
pytest -m "not slow"         # fast subset, ~35 s
```

The suite verifies the bispectrum against a **closed-form analytic result** — for an on-grid,
boxcar-windowed, block-random-phase signal, `L(k1,k2) = (a1 a2 a3 / 8) Σ w conj(φ3) φ1 φ2`
exactly — as well as conjugate symmetry, exact triad counts, CBMD reducing to BMD when the three
variables coincide, bit-identical results across MPI rank counts, and a regression against an
independent implementation on the cylinder-wake dataset.

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
