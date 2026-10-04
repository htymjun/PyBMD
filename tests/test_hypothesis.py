#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Validation against the "hypothesis testing" surrogate data of Schmidt (2020).

The surrogate data and spectral estimators live in
``examples/example4_hypothesis_testing.py``, which also renders the reference
figures.
'''
import atexit
import os
import shutil
import sys
import tempfile

import numpy as np
import pytest

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../'))

from examples.example4_hypothesis_testing import (
    NONRES, TRIAD, QUARTET, NOISE, surrogate_waves, fit_case,
    amplitude_spectrum, classical_bispectrum)

_TMPDIR = tempfile.mkdtemp(prefix='pybmd_test_bmd_')
atexit.register(shutil.rmtree, _TMPDIR, ignore_errors=True)

_CASES = {}


def _case(name, freqs, **kwargs):
    '''Fit one paper case, memoized -- every test in this module shares the
    same 4 fits (~2-4 s each), rather than re-fitting per assertion.'''
    if name not in _CASES:
        _CASES[name] = fit_case(name, freqs, save_dir=_TMPDIR,
                                store_modes=(name == 'triad'), **kwargs)
    return _CASES[name]


def _bispectrum_grid(bmd, m=40):
    '''``|lambda_1|`` re-indexed onto a dense ``(k, l)`` grid, NaN elsewhere.'''
    t = bmd.triads
    grid = np.full((m, m), np.nan)
    vals = np.abs(bmd.L[t.f1_idx, t.f2_idx])
    for kk, ll, v in zip(t.k, t.l, vals):
        if 0 <= kk < m and 0 <= ll < m:
            grid[kk, ll] = v
    return grid


# ---------------------------------------------------------------------------
# amplitude spectrum
# ---------------------------------------------------------------------------

def test_amplitude_spectrum_normalization():
    '''
    ``A = 2|q_hat(x=0,f)|`` must reach close to the unit input amplitude at
    each driven frequency (fig. 1a): since the frequencies are deliberately
    off-grid, Hann leakage costs a few percent.
    '''
    q, x, k = surrogate_waves(TRIAD['freqs'], seed=0)
    freq, A = amplitude_spectrum(q[:, 0, 0], 128, 1.0)
    pos = freq >= 0
    for f in TRIAD['freqs']:
        peak = A[pos][np.argmin(np.abs(freq[pos] - f))]
        assert 0.8 <= peak <= 1.02


# ---------------------------------------------------------------------------
# triad detection and rejection
# ---------------------------------------------------------------------------

def test_resonant_triad_is_detected():
    '''The peak of the mode bispectrum must sit on the driven triad.'''
    bmd, x, k = _case(**TRIAD)
    t = bmd.triads
    vals = np.abs(bmd.L[t.f1_idx, t.f2_idx])
    i = int(np.argmax(vals))
    assert (t.k[i], t.l[i]) == (26, 6)          # nearest DFT bins to (0.2, 0.05)


def test_triad_peak_matches_published_scale():
    '''The reference figure's |lambda_1| z-axis tops out around 0.05.'''
    bmd, x, k = _case(**TRIAD)
    t = bmd.triads
    peak = np.nanmax(np.abs(bmd.L[t.f1_idx, t.f2_idx]))
    assert 0.01 < peak < 0.15


def test_nonresonant_triplet_is_rejected():
    '''``f1 + f2 != f3``: the mode bispectrum must stay flat (fig. 1a,b).'''
    triad_bmd, _, _ = _case(**TRIAD)
    nonres_bmd, _, _ = _case(**NONRES)
    t = triad_bmd.triads
    peak = np.nanmax(np.abs(triad_bmd.L[t.f1_idx, t.f2_idx]))
    tn = nonres_bmd.triads
    n_peak = np.nanmax(np.abs(nonres_bmd.L[tn.f1_idx, tn.f2_idx]))
    assert n_peak < 0.1 * peak


def test_quartet_is_rejected():
    '''A 4-wave resonance without any triad also leaves the bispectrum flat.'''
    triad_bmd, _, _ = _case(**TRIAD)
    quartet_bmd, _, _ = _case(**QUARTET)
    t = triad_bmd.triads
    peak = np.nanmax(np.abs(triad_bmd.L[t.f1_idx, t.f2_idx]))
    tq = quartet_bmd.triads
    q_peak = np.nanmax(np.abs(quartet_bmd.L[tq.f1_idx, tq.f2_idx]))
    assert q_peak < 0.1 * peak


def test_classical_bispectrum_matches_mode_bispectrum_without_noise():
    '''
    "the classical bispectrum performs the same as the mode bispectrum for
    the non-noisy data" (Schmidt 2020): both must peak at the same triad.
    '''
    bmd, x, k = _case(**TRIAD)
    L_grid = _bispectrum_grid(bmd)
    q, _, _ = surrogate_waves(TRIAD['freqs'], seed=0)
    B = classical_bispectrum(q[:, 0, 0], 128, 40)
    assert (np.unravel_index(np.nanargmax(L_grid), L_grid.shape)
            == np.unravel_index(np.nanargmax(B), B.shape))


# ---------------------------------------------------------------------------
# noise robustness
# ---------------------------------------------------------------------------

def test_triad_survives_unit_snr_noise():
    '''
    At SNR = 1 (noise variance equal to signal variance) the triad must still
    dominate over the rest of the plane -- the paper reports "no significant
    side peaks".
    '''
    bmd, x, k = _case(**NOISE)
    t = bmd.triads
    vals = np.abs(bmd.L[t.f1_idx, t.f2_idx])
    i = int(np.argmax(vals))
    assert (t.k[i], t.l[i]) == (26, 6)
    near = (np.abs(t.k - 26) <= 2) & (np.abs(t.l - 6) <= 2)
    background = vals[~near].max()
    assert vals[i] > 1.5 * background


def test_triad_peak_height_is_stable_with_unit_snr_noise():
    '''
    Adding unit-SNR noise changes the noisy realization, but the dominant BMD
    eigenvalue should remain on the clean-signal scale.
    '''
    clean_bmd, _, _ = _case(**TRIAD)
    noisy_bmd, _, _ = _case(**NOISE)

    clean_t = clean_bmd.triads
    noisy_t = noisy_bmd.triads
    clean_peak = np.nanmax(np.abs(clean_bmd.L[clean_t.f1_idx, clean_t.f2_idx]))
    noisy_peak = np.nanmax(np.abs(noisy_bmd.L[noisy_t.f1_idx, noisy_t.f2_idx]))

    assert noisy_peak == pytest.approx(clean_peak, rel=0.03)


# ---------------------------------------------------------------------------
# mode content
# ---------------------------------------------------------------------------

def test_triad_modes_recover_the_waves():
    '''
    The sum-interaction mode must recover ``e^{-i k3 x}`` and the
    quadratic-term mode ``e^{-i(k1+k2)x}`` -- the ``+f`` DFT bin of a real
    cosine carries the *conjugate* of the physical wave, since the analysis
    kernel is ``e^{-i 2 pi f t}``.
    '''
    bmd, x, k = _case(**TRIAD)
    t = bmd.triads
    i = t.find(26, 6)
    wt = bmd.weights.ravel()
    psi_sum, psi_prod = bmd.get_modes_at_triad(i)
    psi_sum, psi_prod = psi_sum.ravel(), psi_prod.ravel()

    def overlap(p, ref):
        return abs(np.vdot(p, wt * ref)) / np.sqrt(
            np.real(np.vdot(p, wt * p)) * np.real(np.vdot(ref, wt * ref)))

    assert overlap(psi_sum, np.exp(-1j * k[2] * x)) == pytest.approx(1.0, abs=1e-3)
    assert overlap(psi_prod, np.exp(-1j * (k[0] + k[1]) * x)) == pytest.approx(1.0, abs=1e-3)
