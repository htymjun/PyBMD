# tests/

Testing strategy detail. See the root [`CLAUDE.md`](../CLAUDE.md) for the pytest commands and
[`pybmd/bmd/CLAUDE.md`](../pybmd/bmd/CLAUDE.md) for the solver deviations these tests guard.

The primary regression net does not depend on MATLAB and runs everywhere:
`test_bispectrum_matches_closed_form`: for an on-grid, boxcar-windowed, block-random-phase signal
the complex bispectrum is analytically `L = (a1·a2·a3/8)·Σ w·conj(φ3)·φ1·φ2`, with every other
triad exactly zero. It matches to ~5e-15. Supporting checks: conjugate symmetry
`L(-k,-l) = conj(L(k,l))`, exact triad counts (625/325/2401/12288 — closed-form, e.g. `(m+1)²` for
regions {1,2}), CBMD reducing to BMD exactly when q=r=s, and the cross-implementation regression
above (complex `L` to 1e-13, modes to 1.0 correlation).

Modes are defined only up to a unit-modulus phase — compare them with
`|<a,b>| / (‖a‖‖b‖) ≈ 1`, never elementwise.

Octave is available on this machine and can run `refs/bmd/bmd.m`/`cbmd.m` directly, so the
Deviations numbers in [`pybmd/bmd/CLAUDE.md`](../pybmd/bmd/CLAUDE.md) are now cross-checked against
the genuine MATLAB source rather than only a Python transcription of it. `refs/bmd` is a **git
submodule** pointing at `olivertschmidt/bmd` — the reference's research/non-commercial license
means it must stay a pointer rather than vendored code; run `git submodule update --init` to
populate it locally. `tests/test_octave_reference.py` (`pytest -m slow`, or `pytest
tests/test_octave_reference.py`) exercises this; it self-skips, not errors, when `octave-cli` is
absent or the submodule hasn't been initialized. `.github/workflows/octave_reference.yml` runs it
in CI (checks out the submodule, `apt-get install`s Octave). See
[`docs/octave_cross_validation.md`](../docs/octave_cross_validation.md) for the method (three
comparison tiers, isolating the DFT/blocking/weighting stage from the solver) and the full measured
tables, with figures.
