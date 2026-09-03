# tests/

Testing strategy detail. See the root [`CLAUDE.md`](../CLAUDE.md) for the pytest commands and
[`pybmd/bmd/CLAUDE.md`](../pybmd/bmd/CLAUDE.md) for the solver deviations these tests guard.

The primary regression net does not depend on MATLAB and runs everywhere — `test_bmd_serial.py`,
`test_cbmd.py`, `test_bmd_mpi.py`, `test_io.py`:

- `test_bispectrum_matches_closed_form`: an on-grid, boxcar-windowed, non-overlapping signal with
  components at `k1`, `k2`, `k1+k2` carrying independent random phases per block, the third
  locked to the sum of the first two. Every `B` is then rank one, so for *every* triad
  `L = (a_k·a_l·a_{k+l}/8)·Σ w·conj(φ_{k+l})·φ_k·φ_l` (non-zero only for the 12 sum/difference
  views of that one physical triad — `(5,3)`, `(8,-3)`, `(8,-5)` in regions 1, 2 — and exactly
  zero elsewhere), `T` is the same sum without the weights, and the modes are the spatial
  patterns themselves. Matches to ~1e-12 relative. `expected_bispectrum` in that file is the
  oracle; extend it rather than hard-coding values.
- conjugate symmetry `L(-k,-l) = conj(L(k,l))`, `T(-k,-l) = T(k,l)` on random data, all 8 regions;
- exact triad counts: `(m+1)²` for regions {1,2}, `(m+1)(m+2)/2` for {1}, a brute-force count of
  `|k+l| < Nyquist` for all eight (625 / 325 / 2401 / **12223** at `n_dft=128` — not 12288);
- CBMD reducing to BMD triad-for-triad when q=r=s, and two identical CBMD states giving identical
  mode slices and `L = 2·L_BMD` — the guard for `Cross._unflatten_modes` (state-slowest flat axis);
- `test_bmd_mpi.py`: bit-identical `L`, `T`, `coeffs` and every mode file between `mpirun -n 1`
  and `-n 2` (marker `mpi`, self-skips without `mpirun`/`mpi4py`); its helper also checks
  `allreduce` on a big-endian buffer;
- `test_io.py`: the h5py (MATLAB v7.3) and scipy (v5) `.mat` paths return the same arrays.

Every bug-fix regression in these files (weights mutation, CBMD mode scrambling, `normalize_data`,
stale mode files, mean layout, `_pow2_scale` subnormals, `simple_iteration` tolerance, the h5py
transpose, the signed `T` plot, ...) was confirmed to fail on the code before its fix.
`tests/conftest.py` puts the checkout first on `sys.path`, so a non-editable install cannot
shadow the source.

Modes are defined only up to a unit-modulus phase — compare them with
`|<a,b>| / (‖a‖‖b‖) ≈ 1`, never elementwise.

Octave is available on this machine and can run `refs/bmd/bmd.m`/`cbmd.m` directly, so the
Deviations numbers in [`pybmd/bmd/CLAUDE.md`](../pybmd/bmd/CLAUDE.md) are now cross-checked against
the genuine MATLAB source rather than only a Python transcription of it. `refs/bmd` is a **git
submodule** pointing at `olivertschmidt/bmd` — the reference's research/non-commercial license
means it must stay a pointer rather than vendored code; run `git submodule update --init` to
populate it locally. `tests/test_octave_reference.py` (`pytest -m slow`, or `pytest
tests/test_octave_reference.py`) exercises this; it self-skips, not errors, when `octave-cli` is
absent or the submodule hasn't been initialized. One caveat: `bmd.m`'s Hamming window `hammwin`
is a file-local subfunction Octave cannot call from outside `bmd.m`, so
`test_default_window_matches_reference` evaluates the *transcribed formula* under Octave — it
checks NumPy against Octave arithmetic on `bmd.m:310`'s expression, not the file itself. `.github/workflows/octave_reference.yml` runs it
in CI (checks out the submodule, `apt-get install`s Octave). See
[`docs/octave_cross_validation.md`](../docs/octave_cross_validation.md) for the method (three
comparison tiers, isolating the DFT/blocking/weighting stage from the solver) and the full measured
tables, with figures.
