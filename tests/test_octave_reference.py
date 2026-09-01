#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Cross-validation of PyBMD against the original MATLAB ``bmd.m``/``cbmd.m``,
run for real under Octave (see ``tests/octave/octave_ref.py``).

Every test here is ``@pytest.mark.slow`` and self-skips (not errors) when
``octave-cli`` is not on PATH or ``refs/bmd/`` -- a git submodule, populated
with ``git submodule update --init`` -- is absent; see
``octave_ref.require_octave`` / ``require_refs`` / ``require_full_dataset``.

Three tiers, so a disagreement is attributed to one pipeline stage rather
than to "PyBMD vs. MATLAB" as a whole:

- Tier A: the DFT/blocking/weighting stage, via an instrumented *copy* of
  the reference that also returns ``Q_hat`` and every per-triad ``B``
  (``refs/bmd/*.m`` itself is never modified -- see ``octave_ref.instrument``).
  Expected to agree to ~1e-13 relative.
- Tier B: end-to-end, against the unmodified reference.
- Tier C: the numerical-radius solver, head-to-head on identical ``B``
  matrices -- this is where the deviations documented in ``CLAUDE.md`` live.

See ``docs/octave_cross_validation.md`` for the full measured tables.
'''
import os
import sys

import numpy as np
import pytest

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../'))
sys.path.append(os.path.join(CFD, 'octave'))

from pybmd.bmd.standard import Standard
from pybmd.bmd.cross import Cross
from pybmd.bmd.optimizers import mengi_overton
import pybmd.utils.weights as utils_weights
import pybmd.bmd.utils as utils_bmd

import octave_ref as oref

pytestmark = pytest.mark.slow

DATA_PATH = os.path.join(CFD, 'data', 'wake_Re500_sub.npz')


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def small_data():
    '''The shipped, subsampled cylinder-wake fixture, cast to double.'''
    d = np.load(DATA_PATH)
    nt, n1, n2 = d['u'].shape
    return dict(u=d['u'].astype(np.float64), v=d['v'].astype(np.float64),
               dt=float(d['dt']), n1=n1, n2=n2)


def _small_bmd(small_data, tmp_path, **overrides):
    '''An initialized (not fit) Standard on the small fixture.'''
    params = dict(
        n_dft=64, time_step=small_data['dt'], n_space_dims=2, n_variables=1,
        n_overlap=32, regions=[1, 2], max_freq_idx=8, save_modes=False,
        tol=1e-6, n_it_max=500, savedir=str(tmp_path))
    params.update(overrides)
    x = small_data['u'][..., np.newaxis]
    w = utils_weights.uniform((small_data['n1'], small_data['n2']), 1, 1.0)
    bmd = Standard(params=params, weights=w)
    bmd._initialize(x)
    bmd._mode_shape = (*bmd._xshape, bmd._nv)   # normally set by fit()
    return bmd, x, w


def _pybmd_B(bmd, q_hat, i):
    '''The exact ``B`` PyBMD's own ``_triad_loop`` builds for triad ``i``.'''
    q_sum, q_prod, weights = bmd._triad_matrices(q_hat, i)
    return q_sum.conj().T @ (q_prod * weights) / bmd.n_blocks


def _octave_bmd(bmd, x, w, **kw):
    '''Run bmd.m/bmd_instr.m with parameters matched to an initialized bmd.'''
    kwargs = dict(window=bmd._window.ravel(), weight=w['weights'],
                 n_overlap=bmd.n_overlap, dt=bmd.dt, regions=bmd._regions,
                 max_freq_idx=bmd._max_freq_idx, tol=bmd._solver_tol,
                 n_it_max=bmd._solver_n_it_max)
    kwargs.update(kw)
    return oref.run('bmd', x, **kwargs)


# ---------------------------------------------------------------------------
# structure: frequency axis, triad count/order, idx convention
# ---------------------------------------------------------------------------

def test_structure_matches_reference(small_data, tmp_path):
    oref.require_octave()
    oref.require_refs()
    bmd, x, w = _small_bmd(small_data, tmp_path)
    out = _octave_bmd(bmd, x, w)
    t = bmd.triads

    assert out['f'].ravel().size == t.n_freq
    np.testing.assert_allclose(out['f'].ravel(), bmd.freq, rtol=0, atol=1e-12)

    idx_ref = out['idx'].ravel().astype(np.int64)
    assert idx_ref.size == t.n_triads

    # MATLAB's idx is sub2ind's *Fortran*-order linear index, 1-based;
    # recover the (row, col) pair and compare, rather than the linear value
    # itself, which Triads.linear_idx deliberately does not reproduce (it is
    # C-order -- see its docstring).
    i0 = (idx_ref - 1) % t.n_freq
    j0 = (idx_ref - 1) // t.n_freq
    np.testing.assert_array_equal(i0, t.f1_idx)
    np.testing.assert_array_equal(j0, t.f2_idx)

    # the reference's row-major (i,j) loop order matches Triads' own
    # row-major np.nonzero order, so triad i must line up on both sides
    # without any re-sorting -- exercised for real by every Tier A/C test
    # below, which index B_all/Q_hat by the same i.


def test_solver_reachability_matches_claude_md(small_data):
    '''
    Pins the corrected reachability note: 'MengiOverton' and 'HeWatson' both
    run (HeWatson non-deterministically, via `rand`); 'simpleIteration' and
    'eig' pass the option validator but the inner `switch` has no matching
    case; 'simpleit' fails the validator itself. All four failures raise the
    same 'Unknown solver.' message from two different call sites.
    '''
    oref.require_octave()
    oref.require_refs()
    for kind in ('bmd', 'cbmd'):
        for solver in ('MengiOverton', 'HeWatson'):
            ok, msg = oref.check_solver_reachable(kind, solver)
            assert ok, f'{kind}/{solver}: {msg}'
        for solver in ('simpleIteration', 'eig', 'simpleit'):
            ok, msg = oref.check_solver_reachable(kind, solver)
            assert not ok and 'Unknown solver' in msg, f'{kind}/{solver}: {msg}'


def test_default_window_matches_reference():
    '''``pybmd.bmd.utils.hamming_window`` against ``bmd.m``'s own ``hammwin``,
    evaluated live under Octave rather than compared by reading the source.'''
    oref.require_octave()
    ref = oref.octave_hamming_window(64)
    py = utils_bmd.hamming_window(64)
    np.testing.assert_allclose(py, ref, rtol=0, atol=1e-14)


# ---------------------------------------------------------------------------
# Tier A: DFT / blocking / weighting, independent of the solver
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('weight_kind', ['uniform', 'nonuniform'])
def test_tier_a_qhat_and_b_match(small_data, tmp_path, weight_kind):
    oref.require_octave()
    oref.require_refs()
    if weight_kind == 'uniform':
        w = utils_weights.uniform((small_data['n1'], small_data['n2']), 1, 1.0)
    else:
        x1 = np.linspace(0, 1, small_data['n1'])
        x2 = np.linspace(0, 2, small_data['n2'])
        w = utils_weights.trapz_2d(x1, x2, n_vars=1)

    x = small_data['u'][..., np.newaxis]
    params = dict(n_dft=64, time_step=small_data['dt'], n_space_dims=2,
                 n_variables=1, n_overlap=32, regions=[1, 2], max_freq_idx=8,
                 save_modes=False, tol=1e-6, n_it_max=500,
                 savedir=str(tmp_path))
    bmd = Standard(params=params, weights=w)
    bmd._initialize(x)
    t = bmd.triads

    out = _octave_bmd(bmd, x, w, instrumented=True)
    q_hat_ref, b_all_ref = out['Q_hat'], out['B_all']
    q_hat = bmd._compute_qhat(block_shape=(bmd._nxv,))

    for f in t.freq_needed:
        f = int(f)
        rel = (np.max(np.abs(q_hat[f] - q_hat_ref[f, :, :]))
              / np.max(np.abs(q_hat_ref[f, :, :])))
        assert rel < 1e-10, f'Q_hat row {f}: rel diff {rel:.3e}'

    max_rel = 0.0
    for i in range(bmd.n_triads):
        B = _pybmd_B(bmd, q_hat, i)
        B_ref = b_all_ref[:, :, i]
        rel = np.max(np.abs(B - B_ref)) / max(np.max(np.abs(B_ref)), 1e-300)
        max_rel = max(max_rel, rel)
    assert max_rel < 1e-10, f'max rel B diff: {max_rel:.3e}'


# ---------------------------------------------------------------------------
# Tier C: the numerical-radius solver, head-to-head on identical B
# ---------------------------------------------------------------------------

def test_tier_c_mengi_overton_matches_truth_and_dominates_reference(
        small_data, tmp_path):
    '''
    The crux of the cross-validation. On the *same* B matrices (Tier A
    already showed those match to ~1e-10):

    - PyBMD's mengi_overton matches a dense brute-force angular scan on
      every triad;
    - the reference's |L| never exceeds PyBMD's -- the one-sided invariant
      that holds regardless of solver noise, from CLAUDE.md's deviations 1/2:
      the reference under-estimates the numerical radius, never over.
    '''
    oref.require_octave()
    oref.require_refs()
    bmd, x, w = _small_bmd(small_data, tmp_path)
    t = bmd.triads
    out = _octave_bmd(bmd, x, w, instrumented=True)
    b_all_ref, L_ref = out['B_all'], out['L']

    max_rel_bf = 0.0
    n_over_1pct = n_over_10pct = 0
    for i in range(bmd.n_triads):
        B = b_all_ref[:, :, i]
        w_py, _ = mengi_overton(B, tol=1e-6, n_it_max=500)
        bf = oref.brute_force_radius(B, n_theta=8001)
        max_rel_bf = max(max_rel_bf,
                         abs(abs(w_py) - bf) / max(bf, 1e-300))

        r_ref = L_ref[t.f1_idx[i], t.f2_idx[i]]
        assert abs(r_ref) <= abs(w_py) * (1 + 1e-6), (
            f'triad {i} (k={t.k[i]}, l={t.l[i]}): reference {abs(r_ref):.6e} '
            f'exceeds PyBMD {abs(w_py):.6e}')
        if bf > 1e-9:
            rel = abs(abs(r_ref) - abs(w_py)) / abs(w_py)
            n_over_1pct += rel > 0.01
            n_over_10pct += rel > 0.10

    assert max_rel_bf < 1e-5, (
        f'PyBMD mengi_overton vs. brute force: max rel diff {max_rel_bf:.3e}')
    # not asserted exactly (data/config dependent): recorded here so a
    # regression in either solver is visible without re-deriving the numbers.
    print(f'\n[tier C, small fixture] {n_over_1pct}/{bmd.n_triads} triads '
         f'off by >1%, {n_over_10pct}/{bmd.n_triads} by >10%')


@pytest.mark.parametrize('regions,max_freq_idx,n_triads,n_over_1pct,n_over_10pct', [
    ([1, 2], 12, 169, 52, 29),
])
def test_tier_c_full_dataset_matches_measured_deviation(
        regions, max_freq_idx, n_triads, n_over_1pct, n_over_10pct, tmp_path):
    '''
    Reproduces the exact configuration CLAUDE.md's Deviations section cites
    (regions=[1,2], max_freq_idx=12) on the full cylinder-wake dataset, and
    pins the measured counts as a regression: 52/169 triads off by >1%,
    29/169 by >10%, always an under-estimate. See
    docs/octave_cross_validation.md for the full table this comes from.
    '''
    import scipy.io
    mat_path = oref.require_full_dataset()
    d = scipy.io.loadmat(mat_path)
    dt = float(d['dt'][0, 0])
    nt, n1, n2 = d['u'].shape
    x = d['u'].astype(np.float64)[..., np.newaxis]
    dV = float((d['x'][1, 0] - d['x'][0, 0]) * (d['y'][0, 1] - d['y'][0, 0]))

    params = dict(n_dft=256, time_step=dt, n_space_dims=2, n_variables=1,
                 n_overlap=128, regions=regions, max_freq_idx=max_freq_idx,
                 save_modes=False, tol=1e-6, n_it_max=500,
                 savedir=str(tmp_path))
    w = utils_weights.uniform((n1, n2), 1, dV)
    bmd = Standard(params=params, weights=w).fit(x)
    assert bmd.n_triads == n_triads
    t = bmd.triads

    out = oref.run('bmd', x, window=bmd._window.ravel(), weight=w['weights'],
                   n_overlap=128, dt=dt, regions=regions,
                   max_freq_idx=max_freq_idx, tol=1e-6, n_it_max=500,
                   timeout=280)
    L_ref = out['L']
    vals_py = np.abs(bmd.L[t.f1_idx, t.f2_idx])
    vals_ref = np.abs(L_ref[t.f1_idx, t.f2_idx])

    assert np.all(vals_ref <= vals_py * (1 + 1e-6)), (
        'reference exceeded PyBMD on at least one triad')
    rel = np.abs(vals_ref - vals_py) / np.maximum(vals_py, 1e-300)
    assert int(np.sum(rel > 0.01)) == n_over_1pct
    assert int(np.sum(rel > 0.10)) == n_over_10pct


# ---------------------------------------------------------------------------
# Tier B: modes and energy transfer, on triads where the solvers agree
# ---------------------------------------------------------------------------

def test_tier_b_modes_and_energy_transfer_agree(small_data, tmp_path):
    '''
    Isolates mode assembly/normalization and the energy-transfer term from
    the solver, by restricting the comparison to triads where the two
    solvers already agree on |L| to 1e-6.
    '''
    oref.require_octave()
    oref.require_refs()
    bmd, x, w = _small_bmd(small_data, tmp_path, store_modes=True)
    t = bmd.triads
    out = _octave_bmd(bmd, x, w)
    L_ref, T_ref, P_ref = out['L'], out['T'], out['P']

    q_hat = bmd._compute_qhat(block_shape=(bmd._nxv,))
    bmd._triad_loop(q_hat)

    vals_py = np.abs(bmd.L[t.f1_idx, t.f2_idx])
    vals_ref = np.abs(L_ref[t.f1_idx, t.f2_idx])
    agree = np.abs(vals_py - vals_ref) <= 1e-6 * np.maximum(vals_py, 1e-300)
    assert agree.sum() > 0.5 * t.n_triads, (
        'too few triads agree on |L| to isolate mode/T comparison')

    for i in np.flatnonzero(agree):
        i = int(i)
        assert bmd.T[t.f1_idx[i], t.f2_idx[i]] == pytest.approx(
            float(T_ref[t.f1_idx[i], t.f2_idx[i]]), abs=1e-6, rel=1e-4)
        for comp in (0, 1):
            a = bmd.modes[i, comp].ravel()
            b = np.asarray(P_ref[comp, i]).ravel()
            overlap = abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
            assert overlap == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# CBMD
# ---------------------------------------------------------------------------

def _small_cbmd(small_data, tmp_path, x3, **overrides):
    params = dict(
        n_dft=32, time_step=small_data['dt'], n_space_dims=2, n_variables=3,
        n_overlap=16, regions=[1, 2], max_freq_idx=6, save_modes=False,
        tol=1e-6, n_it_max=500, state_idx=[0], qr_idx=[[1, 2]],
        savedir=str(tmp_path))
    params.update(overrides)
    cb = Cross(params=params)
    cb._initialize(x3)
    cb._mode_shape = (*cb._xshape, cb.n_state)   # normally set by fit()
    cb._weights_tiled = np.tile(cb._weights, (cb.n_state, 1))
    return cb


def _octave_cbmd(cb, x3, **kw):
    w = np.ones((cb._xshape))
    kwargs = dict(window=cb._window.ravel(), weight=w,
                 n_overlap=cb.n_overlap, dt=cb.dt, regions=cb._regions,
                 max_freq_idx=cb._max_freq_idx, tol=cb._solver_tol,
                 n_it_max=cb._solver_n_it_max, s_idx=[1], qr_idx=[[2, 3]])
    kwargs.update(kw)
    return oref.run('cbmd', x3, **kwargs)


@pytest.mark.parametrize('case', ['reduces_to_bmd', 'genuinely_cross'])
def test_cbmd_tier_a_matches_reference(small_data, tmp_path, case):
    oref.require_octave()
    oref.require_refs()
    u, v = small_data['u'], small_data['v']
    x3 = np.stack([u, u, u], axis=-1) if case == 'reduces_to_bmd' \
        else np.stack([u, u, v], axis=-1)   # s=u, q=u, r=v: genuinely cross

    cb = _small_cbmd(small_data, tmp_path, x3)
    out = _octave_cbmd(cb, x3, instrumented=True)
    b_all_ref = out['B_all']
    q_hat = cb._compute_qhat(block_shape=(cb._nx, cb._nv))

    max_rel = 0.0
    for i in range(cb.n_triads):
        q_s, q_qr, weights = cb._triad_matrices(q_hat, i)
        B = q_s.conj().T @ (q_qr * weights) / cb.n_blocks
        B_ref = b_all_ref[:, :, i]
        rel = np.max(np.abs(B - B_ref)) / max(np.max(np.abs(B_ref)), 1e-300)
        max_rel = max(max_rel, rel)
    assert max_rel < 1e-10, f'max rel B diff: {max_rel:.3e}'


def test_cbmd_reduces_to_bmd_against_reference(small_data, tmp_path):
    '''cbmd.m with q=r=s=u reduces to bmd.m's own result -- the reference's
    own consistency check, reproduced under Octave (and already covered
    PyBMD-internally by test_cbmd.py::test_cbmd_reduces_to_bmd).'''
    oref.require_octave()
    oref.require_refs()
    u = small_data['u']
    x1 = u[..., np.newaxis]
    x3 = np.stack([u, u, u], axis=-1)

    bmd, _, w_bmd = _small_bmd(small_data, tmp_path / 'b', n_dft=32,
                               n_overlap=16, max_freq_idx=6)
    out_bmd = _octave_bmd(bmd, x1, w_bmd)

    cb = _small_cbmd(small_data, tmp_path / 'c', x3)
    out_cbmd = _octave_cbmd(cb, x3)

    t = bmd.triads
    L_bmd = out_bmd['L'][t.f1_idx, t.f2_idx]
    L_cbmd = out_cbmd['L'][t.f1_idx, t.f2_idx]
    np.testing.assert_allclose(L_bmd, L_cbmd, rtol=1e-9, atol=1e-14)
