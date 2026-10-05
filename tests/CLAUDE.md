# tests/

Testing strategy detail. See the root [`CLAUDE.md`](../CLAUDE.md) for the pytest commands and
[`pybmd/bmd/CLAUDE.md`](../pybmd/bmd/CLAUDE.md) for the solver deviations these tests guard.

Four layers, from solver to paper:

- `tests/optimizers/` (one test function per file): `mengi_overton` against a brute-force
  angular scan and closed-form radii — signed `max_fov`, `_pow2_scale` (including subnormals),
  scale equivariance, iteration caps, determinism. These run in a few seconds and need nothing but NumPy/SciPy.
- `test_hypothesis.py`: Schmidt (2020)'s hypothesis test on the surrogate data of
  `examples/example4_hypothesis_testing.py` (whose `fit_case`/`surrogate_waves` it imports) — the
  resonant triad is detected at the right bin and scale, non-resonant and quartet cases stay
  flat, the peak survives unit-SNR noise, and the modes recover the travelling waves. Four fits
  are memoized across the module.
- `test_bmd_mpi.py` (marker `mpi`, self-skips without `mpirun`/`mpi4py`): bit-identical `L`, `T`,
  `coeffs` and every mode file between `mpirun -n 1` and `-n 2`, through `tests/mpi_fit.py`; its
  helper also checks `allreduce` on a big-endian buffer.
- `test_octave_reference.py` (marker `slow`): the reference `refs/bmd/bmd.m`/`cbmd.m` run live
  under Octave, in three tiers — A: `Q_hat` and every per-triad `B` from an instrumented copy,
  isolating the DFT/blocking/weighting stage from the solver; B: `L`, `T` and the modes end to
  end; C: the solver deviations, measured.
  `tests/octave/octave_ref.py` is the harness, `tests/octave/build_report.py` regenerates the
  figures of [`octave/octave_cross_validation.md`](octave/octave_cross_validation.md).
- `test_io_rejects_complex.py`: complex data is refused rather than cast to its real part.

`tests/conftest.py` puts the checkout first on `sys.path`, so a non-editable install cannot
shadow the source.

Modes are defined only up to a unit-modulus phase — compare them with
`|<a,b>| / (‖a‖‖b‖) ≈ 1`, never elementwise.

Octave is available on this machine and can run `refs/bmd/bmd.m`/`cbmd.m` directly, so the
Deviations numbers in [`pybmd/bmd/CLAUDE.md`](../pybmd/bmd/CLAUDE.md) are cross-checked against
the genuine MATLAB source rather than only a Python transcription of it. `refs/bmd` is a **git
submodule** pointing at `olivertschmidt/bmd` — the reference's research/non-commercial license
means it must stay a pointer rather than vendored code; run `git submodule update --init` to
populate it locally. `test_octave_reference.py` self-skips, not errors, when `octave-cli` is
absent or the submodule hasn't been initialized. One caveat: `bmd.m`'s Hamming window `hammwin`
is a file-local subfunction Octave cannot call from outside `bmd.m`, so
`test_default_window_matches_reference` evaluates the *transcribed formula* under Octave — it
checks NumPy against Octave arithmetic on `bmd.m:310`'s expression, not the file itself.
`.github/workflows/octave_reference.yml` runs it in CI (checks out the submodule, `apt-get
install`s Octave). See [`octave/octave_cross_validation.md`](octave/octave_cross_validation.md)
for the method and the full measured tables, with figures.
