#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
MATLAB-independent regression net for the standard BMD.

The closed-form case: an on-grid, boxcar-windowed signal whose three
components at ``k1``, ``k2`` and ``k1+k2`` carry independent random phases per
block, with the third phase locked to the sum of the first two. Blocks do not
overlap, so each block's DFT is exactly ``q_hat_k = (a_k/2) phi_k e^{i
theta_k}``, and for every triad ``B`` is the rank-one matrix
``(c/n_blocks) v v^H`` with ``c = (a_k a_l a_{k+l}/8) sum w conj(phi_{k+l})
phi_k phi_l``. Its numerical radius is ``|c|`` at ``z = v/|v|``, so ``L = c``
exactly, and ``T`` is the same sum without the weights. Every triad whose
three rows are not all populated has ``B = 0``.
'''
import copy
import os

import numpy as np
import pytest

from pybmd.bmd import utils as bmd_utils
from pybmd.bmd.postproc import load_results, plot_energy_transfer
from pybmd.bmd.postproc import plot_mode_bispectrum, plot_triad_modes
from pybmd.bmd.standard import Standard
from pybmd.utils.postproc import get_all_modes
import pybmd.utils.weights as utils_weights


XSHAPE = (5, 4)


def _params(savedir, **kwargs):
    params = dict(n_dft=16, time_step=0.5, n_space_dims=2, n_variables=1,
                  n_overlap=8, regions=[1, 2], max_freq_idx=4,
                  savedir=str(savedir))
    params.update(kwargs)
    return params


def _random_data(nt=200, xshape=XSHAPE, nv=1, seed=0):
    return np.random.default_rng(seed).standard_normal((nt, *xshape, nv))


def _correlation(a, b):
    '''Modes are defined up to a unit-modulus phase: compare by |<a,b>|.'''
    a, b = np.ravel(a), np.ravel(b)
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


# ---------------------------------------------------------------------------
# closed form
# ---------------------------------------------------------------------------

def coupled_signal(n_dft=32, n_blocks=12, xshape=XSHAPE, k1=3, k2=5,
                   amps=(1.0, 0.7, 0.4), seed=0):
    '''Block-random-phase triad ``(k1, k2, k1+k2)`` on the DFT grid.'''
    rng = np.random.default_rng(seed)
    ks = (k1, k2, k1 + k2)
    phis = {k: rng.standard_normal(xshape) for k in ks}
    n = np.arange(n_dft)
    blocks = []
    for _ in range(n_blocks):
        th1, th2 = rng.uniform(0, 2 * np.pi, 2)
        th = {k1: th1, k2: th2, k1 + k2: th1 + th2}
        blk = sum(a * np.cos(2 * np.pi * k * n / n_dft + th[k])[:, None, None]
                  * phis[k] for k, a in zip(ks, amps))
        blocks.append(blk)
    data = np.concatenate(blocks, axis=0)[..., np.newaxis]
    return data, {k: (a, phis[k]) for k, a in zip(ks, amps)}


def expected_bispectrum(triads, comps, w):
    '''``L`` and ``T`` of the closed form, on the ``(n_freq, n_freq)`` grid.'''
    def component(k):
        if k in comps:
            return comps[k]
        if -k in comps:                 # real data: q_hat(-k) = conj(q_hat(k))
            a, phi = comps[-k]
            return a, np.conj(phi)
        return None

    L = np.full((triads.n_freq, triads.n_freq), np.nan, dtype=complex)
    T = np.full((triads.n_freq, triads.n_freq), np.nan)
    for i in range(triads.n_triads):
        c1 = component(int(triads.k[i]))
        c2 = component(int(triads.l[i]))
        c3 = component(int(triads.kl[i]))
        val, transfer = 0j, 0.0
        if c1 and c2 and c3:
            amp = c1[0] * c2[0] * c3[0] / 8
            prod = np.conj(c3[1]) * c1[1] * c2[1]
            val = amp * np.sum(w * prod)
            transfer = amp * np.real(np.sum(prod))
        L[triads.f1_idx[i], triads.f2_idx[i]] = val
        T[triads.f1_idx[i], triads.f2_idx[i]] = transfer
    return L, T


def test_bispectrum_matches_closed_form(tmp_path):
    n_dft = 32
    data, comps = coupled_signal(n_dft=n_dft)
    x1 = np.cumsum(np.linspace(1.0, 2.0, XSHAPE[0]))     # non-uniform grid
    x2 = np.cumsum(np.linspace(1.0, 1.5, XSHAPE[1]))
    weights = utils_weights.trapz_2d(x1, x2, n_vars=1)
    bmd = Standard(
        params=_params(tmp_path, n_dft=n_dft, n_overlap=0, window='boxcar',
                       regions=[1, 2, 3, 4, 5, 6, 7, 8], max_freq_idx=10,
                       store_modes=True),
        weights=weights).fit(data)

    L_exp, T_exp = expected_bispectrum(bmd.triads, comps,
                                       weights['weights'][..., 0])
    # the three sum/difference views of the one physical triad are non-zero
    nonzero = np.abs(np.nan_to_num(L_exp)) > 0
    assert nonzero.sum() == 12
    np.testing.assert_allclose(bmd.L, L_exp, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(bmd.T, T_exp, rtol=1e-11, atol=1e-13)

    # and the modes are the spatial patterns of the components themselves
    i = bmd.triads.find(5, 3)
    psi_sum, psi_prod = bmd.get_modes_at_triad(i)
    assert _correlation(psi_sum, comps[8][1]) == pytest.approx(1.0, abs=1e-12)
    assert _correlation(psi_prod, comps[3][1] * comps[5][1]) == \
        pytest.approx(1.0, abs=1e-12)


def test_constituent_modes_match_closed_form(tmp_path):
    '''With ``constituent_modes``, the two extra modes are the spatial
    patterns of the triad's own frequency components ``k`` and ``l``.'''
    n_dft = 32
    data, comps = coupled_signal(n_dft=n_dft)
    weights = utils_weights.trapz_2d(
        np.cumsum(np.linspace(1.0, 2.0, XSHAPE[0])),
        np.cumsum(np.linspace(1.0, 1.5, XSHAPE[1])), n_vars=1)
    bmd = Standard(
        params=_params(tmp_path, n_dft=n_dft, n_overlap=0, window='boxcar',
                       regions=[1, 2, 3, 4, 5, 6, 7, 8], max_freq_idx=10,
                       store_modes=True, constituent_modes=True),
        weights=weights).fit(data)

    assert bmd.modes.shape[1] == 4
    i = bmd.triads.find(5, 3)
    psi_sum, psi_prod, psi_k, psi_l = bmd.get_modes_at_triad(i)
    assert _correlation(psi_sum, comps[8][1]) == pytest.approx(1.0, abs=1e-12)
    assert _correlation(psi_prod, comps[3][1] * comps[5][1]) == \
        pytest.approx(1.0, abs=1e-12)
    assert _correlation(psi_k, comps[5][1]) == pytest.approx(1.0, abs=1e-12)
    assert _correlation(psi_l, comps[3][1]) == pytest.approx(1.0, abs=1e-12)


def test_constituent_modes_are_opt_in(tmp_path):
    '''Default runs keep the 2-mode axis; the flag is required for 4.'''
    data = _random_data()
    default = Standard(params=_params(tmp_path / 'default',
                                      store_modes=True)).fit(data)
    assert default.modes.shape[1] == 2
    with_flag = Standard(params=_params(tmp_path / 'with_flag',
                                        store_modes=True,
                                        constituent_modes=True)).fit(data)
    assert with_flag.modes.shape[1] == 4


def test_conjugate_symmetry(tmp_path):
    '''Real data: ``L(-k,-l) = conj(L(k,l))`` and ``T(-k,-l) = T(k,l)``.'''
    bmd = Standard(params=_params(
        tmp_path, regions=[1, 2, 3, 4, 5, 6, 7, 8], save_modes=False)
    ).fit(_random_data())
    t = bmd.triads
    checked = 0
    for i in range(t.n_triads):
        k, l = int(t.k[i]), int(t.l[i])
        try:
            j = t.find(-k, -l)
        except ValueError:
            continue
        a = bmd.L[t.f1_idx[i], t.f2_idx[i]]
        b = bmd.L[t.f1_idx[j], t.f2_idx[j]]
        assert b == pytest.approx(np.conj(a), rel=1e-8, abs=1e-14)
        assert bmd.T[t.f1_idx[j], t.f2_idx[j]] == \
            pytest.approx(bmd.T[t.f1_idx[i], t.f2_idx[i]], rel=1e-8, abs=1e-14)
        checked += 1
    assert checked > 50


@pytest.mark.parametrize('m', [4, 12, 24, 48])
def test_triad_counts(m):
    '''Closed-form counts away from the Nyquist bound: ``(m+1)^2`` for regions
    {1, 2} (region 1 alone is ``(m+1)(m+2)/2``), and a brute-force count over
    the full plane for all eight regions.'''
    n_dft = 8 * m
    assert bmd_utils.triad_indices(n_dft, regions=(1, 2),
                                   max_freq_idx=m).n_triads == (m + 1) ** 2
    assert bmd_utils.triad_indices(n_dft, regions=(1,),
                                   max_freq_idx=m).n_triads == \
        (m + 1) * (m + 2) // 2

    t = bmd_utils.triad_indices(n_dft, regions=range(1, 9))
    nyq = n_dft // 2
    brute = sum(1 for k in range(-nyq, nyq) for l in range(-nyq, nyq)
                if abs(k + l) < nyq)
    assert t.n_triads == brute
    assert len(np.unique(t.linear_idx)) == t.n_triads


def test_specific_triad_counts():
    '''The figures quoted in the documentation. The all-regions count at
    ``n_dft=128`` is the number of ``(k, l)`` in ``[-64, 63]^2`` with
    ``|k+l| < 64``: 12223 (not the 128^2 * 3/4 = 12288 a continuum estimate
    gives).'''
    assert bmd_utils.triad_indices(128, regions=(1, 2),
                                   max_freq_idx=24).n_triads == 625
    assert bmd_utils.triad_indices(128, regions=(1,),
                                   max_freq_idx=24).n_triads == 325
    assert bmd_utils.triad_indices(256, regions=(1, 2),
                                   max_freq_idx=48).n_triads == 2401
    brute = sum(1 for k in range(-64, 64) for l in range(-64, 64)
                if abs(k + l) < 64)
    assert brute == 12223
    assert bmd_utils.triad_indices(128, regions=range(1, 9)).n_triads == brute


# ---------------------------------------------------------------------------
# regressions
# ---------------------------------------------------------------------------

def test_normalize_weights_leaves_caller_untouched(tmp_path):
    weights = utils_weights.uniform(XSHAPE, n_vars=1)
    before = weights['weights'].copy()
    bmd = Standard(params=_params(tmp_path, normalize_weights=True,
                                  save_modes=False),
                   weights=weights).fit(_random_data() * 3.0)
    np.testing.assert_array_equal(weights['weights'], before)
    # ... while the decomposition did use the normalized weights
    assert bmd.weights.ravel()[0] == pytest.approx(1 / 9.0, rel=0.1)


def test_user_mean_must_have_the_data_layout(tmp_path):
    data = _random_data(nv=2)
    ok = np.zeros((*XSHAPE, 2))
    Standard(params=_params(tmp_path, n_variables=2, save_modes=False),
             mean=ok).fit(data)
    # the reference's variable-first layout has the right element count
    bad = np.zeros((2, *XSHAPE))
    with pytest.raises(ValueError, match='mean has shape'):
        Standard(params=_params(tmp_path, n_variables=2, save_modes=False),
                 mean=bad).fit(data)


def test_store_modes_is_guarded_by_max_modes_gb(tmp_path):
    with pytest.raises(ValueError, match='max_modes_gb'):
        Standard(params=_params(tmp_path, save_modes=False, store_modes=True,
                                max_modes_gb=1e-12)).fit(_random_data())


def test_stale_mode_files_are_removed_on_rerun(tmp_path):
    data = _random_data()
    big = Standard(params=_params(tmp_path, max_freq_idx=4)).fit(data)
    small = Standard(params=_params(tmp_path, max_freq_idx=2)).fit(data)
    assert small.savedir_sim == big.savedir_sim
    assert small.n_triads < big.n_triads
    files = [f for f in os.listdir(small.modes_dir) if f.endswith('.npy')]
    assert len(files) == small.n_triads
    assert get_all_modes(small.savedir_sim).shape[0] == small.n_triads


def test_params_are_not_mutated_and_numpy_values_are_saved(tmp_path):
    params = _params(tmp_path, regions=np.array([1, 2]),
                     time_step=np.float32(0.5), save_modes=False)
    snapshot = copy.deepcopy(params)
    bmd = Standard(params=params).fit(_random_data())
    assert params.keys() == snapshot.keys()
    assert 'n_blocks' not in params and 'results_folder' not in params
    saved = load_results(bmd.savedir_sim).params
    assert saved['regions'] == [1, 2]
    assert saved['n_blocks'] == bmd.n_blocks
    assert os.path.exists(os.path.join(bmd.savedir_sim, 'bispectrum.npz'))


def test_bad_configurations_fail_before_any_work(tmp_path):
    with pytest.raises(ValueError, match='solver_z0'):
        Standard(params=_params(tmp_path, solver_z0=np.ones(3)))
    with pytest.raises(ValueError, match='n_dft'):
        Standard(params=_params(tmp_path, n_dft=2))


def test_normalize_mode_rejects_non_positive_norm():
    psi = np.ones(3, dtype=complex)
    with pytest.raises(ValueError, match='non-positive'):
        bmd_utils.normalize_mode(psi, -np.ones(3))
    with pytest.raises(ValueError, match='non-positive'):
        bmd_utils.normalize_mode(np.zeros(3, dtype=complex), np.ones(3))


def test_normalize_data_standardizes_blocks(tmp_path):
    '''A variable scaled by 10 must give the same bispectrum once
    standardized: division by the *standard deviation*, not the variance.'''
    data = _random_data()
    a = Standard(params=_params(tmp_path / 'a', normalize_data=True,
                                save_modes=False)).fit(data)
    b = Standard(params=_params(tmp_path / 'b', normalize_data=True,
                                save_modes=False)).fit(10.0 * data)
    np.testing.assert_allclose(a.L, b.L, rtol=1e-10)
    # and the standardized quantity is O(1): compare against the unnormalized
    # bispectrum scaled by the third power of the (uniform) std
    c = Standard(params=_params(tmp_path / 'c', save_modes=False)).fit(data)
    ratio = np.nanmedian(np.abs(a.L) / np.abs(c.L))
    assert ratio == pytest.approx(1.0, rel=0.35)


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def _grid_with_signs():
    triads = bmd_utils.triad_indices(16, dt=0.5, regions=(1, 2),
                                     max_freq_idx=4)
    T = np.full((triads.n_freq, triads.n_freq), np.nan)
    vals = np.linspace(-1.0, 1.0, triads.n_triads)
    T[triads.f1_idx, triads.f2_idx] = vals
    return triads, T


def test_plot_energy_transfer_keeps_the_sign(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    triads, T = _grid_with_signs()
    ax = plot_energy_transfer(T, triads.freq, path=str(tmp_path),
                              filename='transfer.png')
    assert os.path.exists(tmp_path / 'transfer.png')
    levels = ax.collections[0].levels
    assert levels.min() < 0 < levels.max()
    assert levels.min() == pytest.approx(-levels.max())
    assert ax.figure.axes[-1].get_ylabel() == '$T$'


def test_plot_mode_bispectrum_saves_on_user_axes(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    triads, T = _grid_with_signs()
    fig, ax = plt.subplots()
    out = plot_mode_bispectrum(np.abs(T), triads.freq, ax=ax, log=False,
                               xlim=np.array([0.0, 0.5]),
                               path=str(tmp_path), filename='L.png')
    assert out is ax
    assert os.path.exists(tmp_path / 'L.png')
    assert plt.fignum_exists(fig.number)      # a caller-owned figure stays open
    plt.close(fig)


def _fake_modes(n_comp, n1=6, n2=5, nv=1, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_comp, n1, n2, nv))
           + 1j * rng.standard_normal((n_comp, n1, n2, nv)))


def test_plot_triad_modes_two_components(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    modes = _fake_modes(2)
    fig = plot_triad_modes(modes, k=5, l=-2, path=str(tmp_path),
                           filename='modes.png')
    assert os.path.exists(tmp_path / 'modes.png')
    assert len(fig.axes) == 2 * 3          # 3 rows, one contour + one colorbar each
    assert fig._suptitle.get_text() == r'$(k,l,k{+}l) = (5,-2,3)$'


def test_plot_triad_modes_four_components(tmp_path):
    '''With constituent_modes, two rows are prepended and the interaction map
    still multiplies modes 0 and 1 (the sum and quadratic-term modes), not
    whatever ends up in the first two rows of the figure.'''
    import matplotlib
    matplotlib.use('Agg')
    modes = _fake_modes(4)
    fig = plot_triad_modes(modes, k=5, l=-2, path=str(tmp_path),
                           filename='modes4.png')
    assert os.path.exists(tmp_path / 'modes4.png')
    assert len(fig.axes) == 2 * 5          # 5 rows now: contour axes, then colorbars
    titles = [ax.get_title() for ax in fig.axes[:5]]
    assert titles[0].startswith(r'$\phi_k$')
    assert titles[1].startswith(r'$\phi_l$')
    assert titles[2].startswith(r'$\phi_{k+l}$')
    assert titles[3].startswith(r'$\phi_{k \circ l}$')
    assert titles[4].startswith(r'$|\phi_{k \circ l} \cdot \phi_{k+l}|$')

    # the interaction-map row must be the product of modes 0 and 1 regardless
    # of the two rows now in front of it
    expected = np.abs(modes[0, ..., 0] * modes[1, ..., 0])
    prod_ax = fig.axes[4]
    np.testing.assert_allclose(
        np.sort(prod_ax.collections[0].levels),
        np.sort(np.max(np.abs(expected)) * np.linspace(0, 1, 257)))
