#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Cross-bispectral mode decomposition against the standard one.'''
import os

import numpy as np
import pytest

from pybmd.bmd.cross import Cross
from pybmd.bmd.standard import Standard
import pybmd.utils.weights as utils_weights


XSHAPE = (6, 5)


def _params(savedir, **kwargs):
    params = dict(n_dft=16, time_step=0.5, n_space_dims=2, n_overlap=8,
                  regions=[1, 2], max_freq_idx=4, store_modes=True,
                  savedir=str(savedir))
    params.update(kwargs)
    return params


def _signal(nt=240, seed=0):
    '''A quadratically coupled field with a spatial structure.'''
    rng = np.random.default_rng(seed)
    t = np.arange(nt) * 0.5
    X = rng.standard_normal(XSHAPE)
    a = (np.cos(2 * np.pi * 0.25 * t)[:, None, None] * X
         + np.cos(2 * np.pi * 0.375 * t)[:, None, None] * X[::-1])
    return a + 0.5 * a ** 2 + 0.1 * rng.standard_normal((nt, *XSHAPE))


def _correlation(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_cbmd_reduces_to_bmd(tmp_path):
    '''With ``q = r = s`` the CBMD is the BMD, triad for triad.'''
    s = _signal()
    cbmd = Cross(params=_params(tmp_path / 'c', n_variables=3, state_idx=[0],
                                qr_idx=[[1, 2]])
                 ).fit(np.stack([s, s, s], axis=-1))
    bmd = Standard(params=_params(tmp_path / 'b', n_variables=1)
                   ).fit(s[..., np.newaxis])
    np.testing.assert_allclose(cbmd.L, bmd.L, rtol=1e-12, atol=0)
    np.testing.assert_allclose(cbmd.T, bmd.T, rtol=1e-12, atol=0)
    for i in range(bmd.n_triads):
        mc, mb = cbmd.get_modes_at_triad(i), bmd.get_modes_at_triad(i)
        assert mc.shape == mb.shape == (2, *XSHAPE, 1)
        assert _correlation(mc[0], mb[0]) == pytest.approx(1.0, abs=1e-12)
        assert _correlation(mc[1], mb[1]) == pytest.approx(1.0, abs=1e-12)


def test_two_identical_states_give_identical_mode_slices(tmp_path):
    '''
    Stacking the same state twice doubles ``B`` (same maximiser, ``L`` twice
    as large) and each state slice of the modes must equal the single-state
    mode. This is the check that the state-slowest flat axis built by
    ``_triad_matrices`` is unflattened in the same order.
    '''
    s = _signal()
    data = np.stack([s, s.copy()], axis=-1)
    two = Cross(params=_params(tmp_path / 'two', n_variables=2,
                               state_idx=[0, 1], qr_idx=[[0, 0], [1, 1]])
                ).fit(data)
    one = Standard(params=_params(tmp_path / 'one', n_variables=1)
                   ).fit(s[..., np.newaxis])
    np.testing.assert_allclose(two.L, 2 * one.L, rtol=1e-12, atol=0)

    for i in range(two.n_triads):
        stored = two.get_modes_at_triad(i)
        on_disk = np.load(os.path.join(two.modes_dir,
                                       f'triad_idx_{i:08d}.npy'))
        assert stored.shape == (2, *XSHAPE, 2)
        np.testing.assert_array_equal(stored, on_disk)
        np.testing.assert_array_equal(stored[..., 0], stored[..., 1])
        single = one.get_modes_at_triad(i)[..., 0]
        for m in range(2):
            assert _correlation(stored[m, ..., 0], single[m]) == \
                pytest.approx(1.0, abs=1e-12)


def test_normalize_weights_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='normalize_weights'):
        Cross(params=_params(tmp_path, n_variables=3, normalize_weights=True))


def test_constituent_modes_is_rejected(tmp_path):
    '''CBMD's quadratic term sums n_terms q*r pairs, so no single mode phi_k
    or phi_l is well defined; this must fail before any work, like the
    normalize_weights rejection above.'''
    with pytest.raises(ValueError, match='constituent_modes'):
        Cross(params=_params(tmp_path, n_variables=3, constituent_modes=True))


def test_non_dict_weights_fall_back_to_uniform_with_a_warning(tmp_path):
    s = _signal()
    data = np.stack([s, s, s], axis=-1)
    with pytest.warns(UserWarning, match='not a dict'):
        cbmd = Cross(params=_params(tmp_path / 'w', n_variables=3,
                                    save_modes=False),
                     weights=np.ones(XSHAPE)).fit(data)
    ref = Cross(params=_params(tmp_path / 'r', n_variables=3,
                               save_modes=False),
                weights=utils_weights.uniform(XSHAPE, n_vars=None)).fit(data)
    np.testing.assert_array_equal(cbmd.L, ref.L)
