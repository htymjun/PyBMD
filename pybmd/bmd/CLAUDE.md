# pybmd/bmd/

Guidance for the BMD/CBMD implementation itself (`base.py`, `standard.py`, `cross.py`,
`optimizers.py`, `postproc.py`, `utils.py`). See the root [`CLAUDE.md`](../../CLAUDE.md) for
commands and the project overview.

## The Base/Standard/Cross split

`Base` owns everything: params, weights, mean, DFT blocking, the triad map, the solve loop, MPI,
and storage. **Subclasses override only three things**, so the algorithm lives in exactly one place:

| | `Standard` (BMD) | `Cross` (CBMD) |
| --- | --- | --- |
| `_triad_matrices(q_hat, i)` | returns `(Q3, Q1*Q2, weights)` | stacks `n_state` blocks; sums the `q*r` terms |
| `_compute_qhat` block shape | `(nx*nv,)` | `(nx, nv)` so `q_hat[f][:, v]` is contiguous |
| `define_weights` | `(*xshape, nv)` | overridden: `xshape`, **no variable axis** |

Everything after `B = Q_sum^H (Q_prod * w) / n_blocks` is shared in `Base._triad_loop`.

## Data flow

`fit()` → `_initialize` (dims, `n_blocks`, weights, mean, `Triads`, savedir, size guard) →
`_compute_qhat` → `_triad_loop` → `_store_and_save`.

`q_hat` is a **dict keyed by global frequency row**, holding only `triads.freq_needed` — the rows
some triad actually references. With `max_freq_idx` set that is a small fraction of `n_dft`.

## Invariants that are easy to break

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

## Deviations from the MATLAB reference — do not revert these

Each fixes a silent wrong answer; all three are covered by regression tests. Measured end-to-end
on the 169 triads of the full cylinder-wake dataset (`regions=[1,2]`, `max_freq_idx=12`), run
*directly under Octave* against `refs/bmd/bmd.m` itself (see
[`docs/octave_cross_validation.md`](../../docs/octave_cross_validation.md)): `MengiOverton` matches a
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
[`docs/octave_cross_validation.md`](../../docs/octave_cross_validation.md)): the reference's actually
*reachable* solvers are `'MengiOverton'` and `'HeWatson'`. `'simpleIteration'` passes the option
validator but the inner `switch` has no matching case (`case {'simpleit'}` is what's there instead)
and errors with `'Unknown solver.'`; `'eig'` fails the same way; `'simpleit'` itself fails the
validator, one step earlier. So neither spelling of the power-iteration solver is reachable from a
real `bmd.m`/`cbmd.m` call — `solver='simpleIteration'` in `optimizers.py` reproduces the
*algorithm* (Watson's simple iteration, Algorithm 1 of the paper's appendix) for regression
purposes, not a path the reference itself can actually take; it is not globally convergent
(under-estimated in 14/40 random matrices, worst 62 % low).

**Three solvers are ported: `MengiOverton` (default), `MengiOvertonMATLAB`, and `simpleIteration`.**
The paper's appendix (`refs/Schmidt_2020_NODY_r2.tex`) actually prescribes He & Watson's nested
algorithm — `simpleIteration` is its inner loop (Algorithm 1) — but `refs/bmd.m`'s 17-Aug-2023
revision made Mengi–Overton the standard solver because He–Watson is only locally convergent per
restart and needs a random start vector to escape local optima, which Mengi–Overton doesn't.
`he_watson` itself is still not ported: on real BMD matrices it collapses to one Watson simple
iteration from a random start, because its own unit-circle test (`bmd.m:363`) has the same
absolute-tolerance bug as `MengiOverton`'s, so its outer level-set loop almost always exits on the
first pass. PyBMD has no use for a solver that needs an unseeded random start vector when
`simpleIteration` already reproduces the underlying algorithm deterministically via
`default_start` — confirmed live on the paper's own hypothesis-test triad case (`tests/test_hypothesis.py`'s
surrogate-data recipe, run through both implementations; see
[`docs/octave_cross_validation.md`](../../docs/octave_cross_validation.md)): `simpleIteration` agrees
with `MengiOverton` everywhere there (max relative deviation
4.4e-4, 0/780 triads above 1%), while `refs/bmd.m`'s `HeWatson` disagrees with both on up to
753/780 triads on the flat, non-resonant case — random-start non-convergence on a featureless
bispectrum, not a meaningful reference value to chase.

`MengiOvertonMATLAB` (`solver='MengiOvertonMATLAB'`, equivalently
`mengi_overton(..., matlab_compat=True)`) *is* ported, as an explicit opt-in that reverts
deviations 1/2/4 (not 3 — it stays RNG-free, so the determinism requirement above is unaffected)
and reproduces `refs/bmd/bmd.m`'s own `MengiOverton` instead. It exists only to reproduce a
specific published MATLAB result, never to analyse new data with, since it reproduces a confirmed
under-estimation bug. Measured live under Octave on the same 169-triad cylinder-wake fixture cited
above: max relative deviation from `refs/bmd.m` 3.8e-6 (median 9.6e-16), 0/169 triads above 1%,
where PyBMD's own `MengiOverton` differs from it by up to 45.6% (52/169 above 1%, as already
noted). The `only pow2-prescale fix` row cited above is measured on the smaller shipped fixture
(81 triads, median `‖B‖₁ ~ 2.7e-4`): reverting only `_pow2_scale` leaves a 28.1% max deviation
from `bmd.m` (vs 66.6% reverting nothing), confirming deviation 2 alone carries essentially the
whole gap, while reverting only the signed-λ or `max(w,1)` deviations changes PyBMD's own answer
by <1e-7; all three reverted together reach 4.9e-6. Fidelity degrades as `‖B‖₁` falls toward the
tolerance floor itself: on the paper's hypothesis-test surrogate, `MengiOvertonMATLAB` reproduces
`bmd.m`'s `MengiOverton` to 4.4e-15 at `‖B‖₁ ~ 2e-3` (SNR=1) but only within 50% on 9/780 triads
without noise (`‖B‖₁ ~ 3.5e-9`) — reproducing a branch decision taken exactly at a tolerance
boundary is inherently unstable across LAPACK builds and language boundaries, and is not a defect
to chase further. Validated live against Octave in
`tests/test_octave_reference.py::test_tier_c_matlab_compat_reproduces_reference`; see
[`docs/octave_cross_validation.md`](../../docs/octave_cross_validation.md) for the figures.

## Conventions that bite

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
