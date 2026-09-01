# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e '.[mpi,io,test]'      # editable install; extras: mpi, io (.mat/.nc), test, docs
git submodule update --init          # populate refs/bmd (the MATLAB reference), only needed
                                     # for tests/test_octave_reference.py -- see Testing strategy

pytest                                # full suite, ~60 s, 223 tests
pytest -m "not slow"                  # fast subset, ~45 s
pytest tests/optimizers -q            # one directory (numerical-radius solver tests,
                                      # one test function per file)
pytest tests/test_bmd_serial.py::test_bispectrum_matches_closed_form -q   # one test
pytest -k "conjugate or closed_form"  # by name

python -m pyflakes pybmd/ tests/ examples/   # only linter used; ignore the
                                             # "f-string is missing placeholders"
                                             # hits, they match the PySPOD print idiom
```

Set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` when running the MPI tests — they assert
**bit-identical** results across rank counts, and threaded BLAS reorders reductions.

Examples are run from the `examples/` directory (they load the fixture by relative path):
`cd examples && MPLBACKEND=Agg python example1_cylinder.py`.

## Architecture

Bispectral mode decomposition, ported from O. T. Schmidt's MATLAB `bmd.m`/`cbmd.m`, structured
like [PySPOD](https://github.com/MathEXLab/PySPOD)
for conventions. BMD finds triads `(k, l, k+l)` whose components are quadratically phase coupled,
and returns a mode bispectrum `L` plus two modes per triad.

### The Base/Standard/Cross split

`Base` owns everything: params, weights, mean, DFT blocking, the triad map, the solve loop, MPI,
and storage. **Subclasses override only three things**, so the algorithm lives in exactly one place:

| | `Standard` (BMD) | `Cross` (CBMD) |
| --- | --- | --- |
| `_triad_matrices(q_hat, i)` | returns `(Q3, Q1*Q2, weights)` | stacks `n_state` blocks; sums the `q*r` terms |
| `_compute_qhat` block shape | `(nx*nv,)` | `(nx, nv)` so `q_hat[f][:, v]` is contiguous |
| `define_weights` | `(*xshape, nv)` | overridden: `xshape`, **no variable axis** |

Everything after `B = Q_sum^H (Q_prod * w) / n_blocks` is shared in `Base._triad_loop`.

### Data flow

`fit()` → `_initialize` (dims, `n_blocks`, weights, mean, `Triads`, savedir, size guard) →
`_compute_qhat` → `_triad_loop` → `_store_and_save`.

`q_hat` is a **dict keyed by global frequency row**, holding only `triads.freq_needed` — the rows
some triad actually references. With `max_freq_idx` set that is a small fraction of `n_dft`.

### Invariants that are easy to break

- **C order everywhere. Never pass `order='F'`.** `L` and `T` are full reductions over space, so
  they are invariant to the flattening permutation; the *modes* are equivariant. Get flatten and
  unflatten out of step and `L` stays perfect while the modes come out scrambled — the failure the
  original port shipped with. The one real hazard is the weights: `define_weights` therefore
  **rejects a bare flat vector** and demands the full spatial shape.
- **The reduction accumulates into zeros, not NaN.** `NaN + SUM` poisons every rank. The reference's
  NaN-outside-the-triads semantics is restored *after* the `allreduce`, via `triads.mask`.
- **Determinism is a requirement, not a nicety.** The MPI tests assert bit-identical `L`, `T`,
  `coeffs` and modes across rank counts. `optimizers.py` contains no RNG at all — both solvers
  start from `default_start`'s deterministic angular scan, or an explicit `solver_z0` — so this
  is structural, not a convention to maintain. Triads are split round-robin because solver cost
  varies in bands across the `f1`-`f2` plane.
- **`T` carries no weight**, unlike `B`. That is deliberate and matches the reference — don't
  "fix" it.
- **`coeffs.npy` is the durable artifact.** Modes are just `Q @ a`, so a large case can run with
  `save_modes=False` and have any triad reconstructed later.

### Deviations from the MATLAB reference — do not revert these

Each fixes a silent wrong answer; all three are covered by regression tests. Measured end-to-end
on the 169 triads of the full cylinder-wake dataset (`regions=[1,2]`, `max_freq_idx=12`), run
*directly under Octave* against `refs/bmd/bmd.m` itself (see
[`docs/octave_cross_validation.md`](docs/octave_cross_validation.md)): `MengiOverton` matches a
brute-force scan of the numerical radius to ~5e-8, the genuine `refs/bmd.m` is off by >1% on 52 of
the 169 triads (>10% on 29), always an *under*-estimate, since `B = Q3^H (Q1∘Q2∘w)/n_blocks` is
tiny (median `‖B‖₁ ~ 5.2e-6` there). These figures were originally measured against a Python
transcription of `bmd.m` and now reproduce exactly against the real source.
Deviation 2 alone fixes 51/52; deviation 1 alone fixes none on this fixture — it only matters
once deviation 2 has rescaled the problem into a regime where a second, subtler mismatch shows up.
This is why the two must be applied together, not as alternatives.

1. `max_fov` returns the **signed** λ_max, not `max(abs(eig(H)))`. The Mengi–Overton level set is
   defined by signed λ_max; filtering crossing angles by modulus discards valid ones and the search
   stops at a local maximum (measured 3.4973 vs a true 4.4346 on a random 7×7).
2. `mengi_overton` **pre-scales by a power of two** (`_pow2_scale`). The unit-circle test
   `|‖D‖−1| ≤ sqrt(eps)·‖A‖₁` is absolute; real BMD matrices are tiny (`B` carries `1/n_blocks` and
   the weights), so without rescaling every crossing is rejected. At `‖A‖₁ ~ 1e-6` the solver
   returned 93.7 % of the true value. Power-of-two scaling is exact in binary FP.
3. Solvers use a deterministic start (`default_start`, a coarse angular scan) instead of a global
   RNG. `T` is actually computed.
4. `mengi_overton`'s level filter uses `sqrt(eps) * max(w, 1.0)`, not the reference's
   `sqrt(eps) * w` — undocumented until now. It only matters for `w < 1`, i.e. every real BMD
   case, and without it the `max(w,1.0)` clamp would make deviation 2 alone insufficient (see the
   `only pow2-prescale fix` row not being enough on its own for the last cylinder-wake triad).

Confirmed live under Octave, for both `bmd.m` and `cbmd.m` (see
[`docs/octave_cross_validation.md`](docs/octave_cross_validation.md)): the reference's actually
*reachable* solvers are `'MengiOverton'` and `'HeWatson'`. `'simpleIteration'` passes the option
validator but the inner `switch` has no matching case (`case {'simpleit'}` is what's there instead)
and errors with `'Unknown solver.'`; `'eig'` fails the same way; `'simpleit'` itself fails the
validator, one step earlier. So neither spelling of the power-iteration solver is reachable from a
real `bmd.m`/`cbmd.m` call — `solver='simpleIteration'` in `optimizers.py` reproduces the
*algorithm* (Watson's simple iteration, Algorithm 1 of the paper's appendix) for regression
purposes, not a path the reference itself can actually take; it is not globally convergent
(under-estimated in 14/40 random matrices, worst 62 % low).

**Only two solvers are ported: `MengiOverton` (default) and `simpleIteration`.** The paper's
appendix (`refs/Schmidt_2020_NODY_r2.tex`) actually prescribes He & Watson's nested algorithm —
`simpleIteration` is its inner loop (Algorithm 1) — but `refs/bmd.m`'s 17-Aug-2023 revision made
Mengi–Overton the standard solver because He–Watson is only locally convergent per restart and
needs a random start vector to escape local optima, which Mengi–Overton doesn't. `he_watson` and
`solver='MengiOvertonMATLAB'` (the `matlab_compat` bug-compatibility mode, deviations 1/2/4
reverted) were both removed from `optimizers.py`: neither is reachable from a real BMD run, and
both pulled in the module's only RNG usage (`_random_start`, `seed`). Octave can now run
`refs/bmd.m` directly (see above), so a `matlab_compat` cross-check no longer needs reconstructing
from a note — if `MengiOvertonMATLAB` is ever revisited, validate it the same way
`docs/octave_cross_validation.md` validates everything else, live, rather than re-adding a
permanent code path for it.

### Conventions that bite

- `overlap` is **percent**; `n_overlap` is **snapshots** and takes precedence.
- `regions` is **1-based** (they are labels on the published octant figure);
  `state_idx`/`qr_idx` are **0-based** Python indices. `Cross._validate_var_idx` catches a
  1-based index copied from MATLAB.
- Data is always `(nt, *xshape, n_variables)` — variables **last**, for both classes. MATLAB's
  `cbmd.m` puts them second; PyBMD does not.
- BMD always needs the full two-sided spectrum (difference-interactions use negative
  frequencies), so there is no `rfft` path and no `fullspectrum` option.
- `Triads.find(k, l)` is the supported way to reach a triad. `linear_idx` exists for MATLAB parity
  but is **C-order** where MATLAB's `sub2ind` is Fortran-order.

## Testing strategy

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
Deviations numbers above are now cross-checked against the genuine MATLAB source rather than only
a Python transcription of it. `refs/bmd` is a **git submodule** pointing at
`olivertschmidt/bmd` — the reference's research/non-commercial license means it must stay a
pointer rather than vendored code; run `git submodule update --init` to populate it locally.
`tests/test_octave_reference.py` (`pytest -m slow`, or `pytest tests/test_octave_reference.py`)
exercises this; it self-skips, not errors, when `octave-cli` is absent or the submodule hasn't been
initialized. `.github/workflows/octave_reference.yml` runs it in CI (checks out the submodule,
`apt-get install`s Octave). See
[`docs/octave_cross_validation.md`](docs/octave_cross_validation.md) for the method (three
comparison tiers, isolating the DFT/blocking/weighting stage from the solver) and the full measured
tables, with figures.
