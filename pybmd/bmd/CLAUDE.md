# pybmd/bmd/

Guidance for the BMD/CBMD implementation itself (`base.py`, `standard.py`, `cross.py`,
`optimizers.py`, `postproc.py`, `utils.py`). See the root [`CLAUDE.md`](../../CLAUDE.md) for
commands and the project overview.

## The Base/Standard/Cross split

`Base` owns everything, including `fit()` itself: params, weights, mean, DFT blocking, the triad
map, the solve loop, MPI, and storage. **Subclasses override only the per-triad matrices and a few
shape hooks**, so the algorithm lives in exactly one place:

| | `Standard` (BMD) | `Cross` (CBMD) |
| --- | --- | --- |
| `_triad_matrices(q_hat, i)` | returns `(Q3, Q1*Q2, weights)` | stacks `n_state` blocks; sums the `q*r` terms |
| `_constituent_matrices(q_hat, i)` | returns `(Q1, Q2)` (only with `constituent_modes`) | not supported — rejected at construction |
| `_block_shape()` (one `q_hat` row) | `(nx*nv,)` | `(nx, nv)` so `q_hat[f][:, v]` is contiguous |
| `_mode_shape` (property) | `(*xshape, nv)` | `(*xshape, n_state)` |
| `_expected_weights_shape()` | `(*xshape, nv)` | `xshape`, **no variable axis** |
| `_post_initialize()` | no-op | tiles the weights over the states, prints `state_idx`/`qr_idx` |
| `_unflatten_modes(psi)` | plain reshape | state-slowest unflatten, see below |
| `_label` | `'BMD'` | `'CBMD'` (timing print only) |

Everything after `B = Q_sum^H (Q_prod * w) / n_blocks` is shared in `Base._triad_loop`.

## Data flow

`Base.fit()` → `_initialize` (dims, `n_blocks`, weights, mean, `Triads`, savedir — clearing stale
`modes/triad_idx_*.npy` from an earlier run into the same directory — size guard, then
`_post_initialize`) →
`_compute_qhat` → `_triad_loop` → `_store_and_save` (arrays first, `params_modes.yaml` last, so a
YAML failure cannot lose results).

`q_hat` is a **dict keyed by global frequency row**, holding only `triads.freq_needed` — the rows
some triad actually references. With `max_freq_idx` set that is a small fraction of `n_dft`.

## Invariants that are easy to break

- **C order everywhere. Never pass `order='F'`.** `L` and `T` are full reductions over space, so
  they are invariant to the flattening permutation; the *modes* are equivariant. Get flatten and
  unflatten out of step and `L` stays perfect while the modes come out scrambled — the failure the
  original port shipped with, and then shipped *again* for CBMD: `Cross._triad_matrices` stacks
  the states along the flat axis with the state **slowest** (`flat = j*nx + p`, matching
  `cbmd.m`'s `repmat`), so a C-order reshape straight into `(*xshape, n_state)` scrambles every
  `n_state > 1` mode. `_unflatten_modes` is the single place a flat mode becomes a field —
  `Cross` overrides it to unflatten as `(n_state, *xshape)` and move the state axis last;
  `tests/test_octave_reference.py`'s CBMD tiers compare the modes against `cbmd.m`.
  The other hazard is the weights: they are therefore checked against the full `(*xshape, nv)`
  shape and a bare flat vector (or the reference's variable-first layout) is **rejected**.
- **The reduction accumulates into zeros, not NaN.** `NaN + SUM` poisons every rank. The reference's
  NaN-outside-the-triads semantics is restored *after* the `allreduce`, via `triads.mask`.
- **Determinism is a requirement, not a nicety.** `tests/test_bmd_mpi.py` asserts bit-identical
  `L`, `T`, `coeffs` and modes between `mpirun -n 1` and `-n 2`. `optimizers.py` contains no RNG at all — Mengi–Overton
  needs no start vector — so this is structural, not a convention to maintain. Triads are split round-robin because solver cost
  varies in bands across the `f1`-`f2` plane.
- **`T` carries no weight**, unlike `B`. That is deliberate and matches the reference — don't
  "fix" it.
- **`coeffs.npy` is the durable artifact.** Modes are just `Q @ a`, so a large case can run with
  `save_modes=False` and have any triad reconstructed later — `reconstruct_modes(data, i)`
  recomputes the DFT rows of that triad (`_compute_qhat(needed)`) and applies `a` through
  `_modes_from_coeffs`, the same path `save_modes_top` uses, so the result is bit-identical to a
  saved mode. It needs the fitted object (blocking, window, `t_mean`, `coeffs`) and the same data;
  `get_modes_at_triad(..., data=)` falls back to it when no mode file exists.
- **`save_modes_top` writes the modes after the reduction, not in the triad loop**, because the
  ranking needs the complete `L`. Each rank rebuilds its own selected triads from `q_hat` and the
  reduced `coeffs` (`_save_top_modes`), so the files are bit-identical to a full `save_modes` run.
  The selection is `postproc.top_triads(self, n)` — keep the two in step. Files keep the global
  `triad_idx` (never renumber: `coeffs`, `find` and `L` index by it); `modes/saved_triad_idx.npy`
  lists them. An `int` is a count, a `float` in (0, 1] a fraction of the `k != 0`, `l != 0`
  triads, so `1` and `1.0` differ.
- **`store_modes` costs as much as `save_modes`, on every rank** (the full `(n_triads, n_comp,
  *mode_shape)` array plus its `allreduce` buffer, `n_comp` being 2 or 4 with
  `constituent_modes`); the `params['max_modes_gb']` guard (default `MAX_MODES_GB`, 8 GB; `None`
  disables it) in `base.py` covers both, sizing `save_modes` at `save_dtype` and `store_modes` at
  `dtype`.
- **`constituent_modes` is a PyBMD addition, not a reference feature.** `bmd.m` allocates
  `P = zeros(2,nTriads,nx)` and never forms a mode from `Q_hat_f1*a` or `Q_hat_f2*a` alone — only
  their product feeds `B` and `psi_prod`. Setting `params['constituent_modes'] = True` appends
  `phi_k = normalize(Q1 @ a)` and `phi_l = normalize(Q2 @ a)` as modes 2 and 3, using the triad's
  own expansion vector `a`; existing indices 0 (`phi_{k+l}`) and 1 (`phi_{k o l}`) are unchanged.
  Note `phi_{k o l} != phi_k * phi_l` pointwise — `(Q1*Q2) @ a != (Q1 @ a) * (Q2 @ a)` — so the two
  new modes are independent information, not a decomposition of the existing product mode.
  `Standard` only: `Cross` rejects the flag at construction, since CBMD's quadratic term sums
  `n_terms` `q*r` pairs and no single constituent mode is well defined for `n_terms > 1`.

## Deviations from the MATLAB reference — do not revert these

Each fixes a silent wrong answer; all three are covered by regression tests. Measured end-to-end
on the 169 triads of the full cylinder-wake dataset (`regions=[1,2]`, `max_freq_idx=12`), run
*directly under Octave* against `refs/bmd/bmd.m` itself (see
[`tests/octave/octave_cross_validation.md`](../../tests/octave/octave_cross_validation.md)): `MengiOverton` matches a
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
3. The solver is RNG-free. `T` is actually computed.
4. `mengi_overton`'s level filter uses `sqrt(eps) * max(w, 1.0)`, not the reference's
   `sqrt(eps) * w` — undocumented until now. It only matters for `w < 1`, i.e. every real BMD
   case, and without it the `max(w,1.0)` clamp would make deviation 2 alone insufficient (see the
   `only pow2-prescale fix` row not being enough on its own for the last cylinder-wake triad).
5. `mengi_overton` rounds the crossing angles to 10 decimals before `np.unique`, so near-duplicate
   crossings from the pencil collapse into one; the reference's plain `unique` keeps them, which
   only costs redundant midpoint tests. Cosmetic — it changes no value.

**Only `MengiOverton` is ported.** The paper's appendix (`refs/Schmidt_2020_NODY_r2.tex`)
prescribes He & Watson's nested algorithm, but `refs/bmd.m`'s 17-Aug-2023 revision made
Mengi–Overton the standard solver, since He–Watson is only locally convergent per restart and
needs an unseeded random start vector. Watson's simple iteration and a bug-compatible
`MengiOvertonMATLAB` were ported earlier and have been removed to keep the code small; the
ablation numbers above were measured with them. Under Octave the reference's only reachable
solvers are `'MengiOverton'` and `'HeWatson'` (`'simpleIteration'` passes the option validator
but errors with `'Unknown solver.'`), which `tests/test_octave_reference.py` still checks.

## Conventions that bite

- `overlap` is **percent**; `n_overlap` is **snapshots** and takes precedence.
- `regions` is **1-based** (they are labels on the published octant figure);
  `state_idx`/`qr_idx` are **0-based** Python indices. `Cross._validate_var_idx` catches a
  1-based index copied from MATLAB.
- Data is always `(nt, *xshape, n_variables)` — variables **last**, for both classes. MATLAB's
  `cbmd.m` puts them second; PyBMD does not.
- Spatial axes follow NumPy/matplotlib: `xshape = (ny, nx)` in 2-D, `(nz, ny, nx)` in 3-D, x
  **last**. `trapz_2d(x, y)`/`trapz_3d(x, y, z)` return that shape and `plot_triad_modes` contours
  `field` untransposed. MATLAB/Fortran arrays are `(nx, ny)` and must be transposed by the caller
  (`examples/data.py` and `example5` do). The decomposition itself never uses the meaning of an
  axis — only the plots and the weight constructors do — so `L` and the modes are unchanged by the
  convention. `tests/test_octave_reference.py` passes the *untransposed* fixture to both `bmd.m`
  and PyBMD, which is why its `trapz_2d` call takes the last-axis coordinate first.
- BMD always needs the full two-sided spectrum (difference-interactions use negative
  frequencies), so there is no `rfft` path and no `fullspectrum` option.
- `Triads.find(k, l)` is the supported way to reach a triad.
