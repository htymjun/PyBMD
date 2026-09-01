# Octave Cross-Validation

PyBMD is cross-validated against the original MATLAB `bmd.m` and `cbmd.m`
reference implementation by running that code under Octave. The reference is
kept as the `refs/bmd` git submodule rather than vendored into this repository.

The executable regression tests live in `tests/test_octave_reference.py`; this
file records the method those tests are meant to preserve.

## Prerequisites

Run the reference tests with:

```bash
git submodule update --init
pytest tests/test_octave_reference.py -q
```

The tests skip locally when Octave or the submodule is missing. The dedicated
GitHub Actions job sets `PYBMD_REQUIRE_OCTAVE_REF=1`, which turns those missing
prerequisites into failures so the cross-validation job cannot pass vacuously.

## Fairness Rules

- The Octave reference is run unmodified for end-to-end comparisons.
- Tier A uses an instrumented copy in a temporary directory, never by editing
  `refs/bmd/*.m`. The copy only exposes `Q_hat` and each per-triad `B`.
- Inputs are cast to double precision before writing the `.mat` files.
- PyBMD's C-order spatial flattening is applied before the data reaches Octave.
  The weights are flattened in the same order, so MATLAB's column-major `(:)`
  never gets to choose a different spatial permutation.
- BMD data is passed as `(nt, nxv)` to Octave. CBMD data is passed as
  `(nt, n_variables, nx)`, matching `cbmd.m` while preserving PyBMD's variables
  last convention at the Python boundary.
- The same window, overlap in snapshots, `dt`, regions, `nfreq`, solver
  tolerance, and iteration cap are passed to both implementations.

## Test Tiers

Tier A checks DFT, blocking, weighting, and triad construction. It compares
PyBMD's `Q_hat` rows and every per-triad `B` matrix against the instrumented
reference with relative tolerance `1e-12`. This is the strictest and most
important language-boundary check.

Tier B checks mode assembly and the energy-transfer term `T` against the
unmodified reference, but only on triads whose solver result already agrees.
This intentionally isolates mode normalization and `T` from known numerical
radius solver differences.

Tier C checks the numerical-radius solver on identical `B` matrices. PyBMD's
Mengi-Overton result is compared against an independent angular scan with local
refinement with relative tolerance `5e-7`, and the reference result is required
not to exceed PyBMD's result beyond a `1e-8` relative guard.
On the full cylinder-wake fixture with `regions=[1, 2]` and `max_freq_idx=12`,
the measured reference deviations are pinned at 52/169 triads above 1% and
29/169 triads above 10%, always as under-estimates.

## Regenerating Figures

The report figures in `docs/figures/octave/` can be regenerated with:

```bash
python docs/build_octave_report.py
```

That script requires Octave and the populated `refs/bmd` submodule.
