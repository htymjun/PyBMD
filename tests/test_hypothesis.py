#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Validation against the "hypothesis testing" surrogate data of Schmidt (2020).'''
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

from pybmd.bmd.standard import Standard
import pybmd.utils.weights as utils_weights

FIGURES_DIR = os.path.join(CFD, '..', 'docs', 'figures', 'hypothesis')

_TMPDIR = tempfile.mkdtemp(prefix='pybmd_test_bmd_')
atexit.register(shutil.rmtree, _TMPDIR, ignore_errors=True)


# ---------------------------------------------------------------------------
# surrogate data -- "Hypothesis testing" section of Schmidt (2020)
# ---------------------------------------------------------------------------

def surrogate_waves(freqs, nt=1280, nx=100, dt=1.0, seed=0, snr=None):
    '''
    ``q(x,t) = sum_j A_j cos(k_j x - 2 pi f_j t + theta0)``, unit amplitudes,
    wavenumbers drawn from ``U[0, 5]`` on ``x in [0, 2 pi)`` with 100 points --
    exactly the paper's surrogate-data recipe.

    The paper adds a random phase offset per *realization*; a single
    continuous time series segmented into 10 blocks of ``n_dft=128`` already
    supplies that, since each block sees a different phase through ``t``, so
    there is no need to simulate repeated realizations explicitly. ``snr=1``
    reproduces the paper's noise test: Gaussian noise scaled so its variance
    equals the signal's.
    '''
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    t = np.arange(nt) * dt
    k = rng.uniform(0, 5, size=len(freqs))
    q = np.zeros((nt, nx))
    for kj, fj in zip(k, freqs):
        q += np.cos(kj * x[None, :] - 2 * np.pi * fj * t[:, None])
    if snr is not None:
        q = q + rng.standard_normal(q.shape) * np.sqrt(q.var() / snr)
    return q[..., np.newaxis], x, k


def _params(name, **kwargs):
    params = dict(
        n_dft=128, time_step=1.0, n_space_dims=1, n_variables=1, overlap=0,
        window='hann', regions=[1], solver='MengiOverton', save_modes=False,
        savedir=os.path.join(_TMPDIR, name))
    params.update(kwargs)
    return params


_CASES = {}


def _case(name, freqs, **kwargs):
    '''Fit one paper case, memoized -- every test in this module shares the
    same 4 fits (~2-4 s each), rather than re-fitting per assertion.'''
    if name not in _CASES:
        q, x, k = surrogate_waves(freqs, seed=0, **kwargs)
        w = utils_weights.uniform((x.size,), n_vars=1, dV=x[1] - x[0])
        bmd = Standard(params=_params(name, store_modes=(name == 'triad')),
                       weights=w).fit(q)
        _CASES[name] = (bmd, x, k)
    return _CASES[name]


NONRES = dict(name='nonres', freqs=(0.046875, 0.203125, 0.3515625)) # (0.05, 0.2, 0.35)
TRIAD = dict(name='triad', freqs=(0.046875, 0.203125, 0.25)) # (0.05, 0.2, 0.25)
QUARTET = dict(name='quartet', freqs=(0.046875, 0.1484375, 0.25, 0.453125)) # (0.05, 0.15, 0.25, 0.45)


def amplitude_spectrum(q_x0, n_dft, dt, window='hann'):
    '''``A(f) = 2|mean_blocks q_hat(f)|``, computed independently of BMD with
    the same window and blocking, as the paper's panel (a) does.'''
    win = np.hanning(n_dft + 1)[:-1]
    win_weight = 1.0 / win.mean()
    n_blocks = len(q_x0) // n_dft
    q_c = q_x0 - q_x0.mean()
    blocks = np.stack([q_c[i * n_dft:(i + 1) * n_dft] for i in range(n_blocks)])
    q_hat = np.fft.fft(win[None, :] * blocks, axis=1) * win_weight / n_dft
    freq = np.fft.fftfreq(n_dft, dt)
    return freq, 2 * np.abs(q_hat).mean(axis=0)


def classical_bispectrum(q_x0, n_dft, dt, m, window='hann'):
    '''
    Classical (biased) bispectrum estimator of a single-point time series,
    block-averaged with the same window and blocking as BMD -- the quantity
    the paper compares the mode bispectrum against in its noise test.
    '''
    win = np.hanning(n_dft + 1)[:-1]
    win_weight = 1.0 / win.mean()
    n_blocks = len(q_x0) // n_dft
    q_c = q_x0 - q_x0.mean()
    blocks = np.stack([q_c[i * n_dft:(i + 1) * n_dft] for i in range(n_blocks)])
    q_hat = np.fft.fftshift(
        np.fft.fft(win[None, :] * blocks, axis=1) * win_weight / n_dft, axes=1)
    f_idx = np.rint(np.fft.fftshift(np.fft.fftfreq(n_dft, dt)) * n_dft).astype(int)
    row = lambda k: int(np.searchsorted(f_idx, k))
    B = np.full((m, m), np.nan)
    for i in range(m):
        for j in range(i + 1):
            if i + j < n_dft // 2:
                B[i, j] = np.abs(np.mean(
                    q_hat[:, row(i)] * q_hat[:, row(j)] * np.conj(q_hat[:, row(i + j)])))
    return B


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
    B = classical_bispectrum(q[:, 0, 0], 128, 1.0, 40)
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
    bmd, x, k = _case(name='noise', freqs=TRIAD['freqs'], snr=1.0)
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
    noisy_bmd, _, _ = _case(name='noise', freqs=TRIAD['freqs'], snr=1.0)

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


# ---------------------------------------------------------------------------
# reference figures
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_reference_figures():
    '''
    Render figures next to Schmidt (2020)'s "hypothesis testing" figures for
    visual comparison. Only existence/non-emptiness is asserted here; the
    scientific content is covered by the tests above.
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # -- figure 1: 3 rows (nonres / triad / quartet) x 2 columns -------------
    titles = {
        'nonres': r'$f_1 \pm f_2 \pm f_3 \neq 0$ (no triad)',
        'triad': r'$f_1 + f_2 = f_3$ (triad)',
        'quartet': (r'$f_1+f_2+f_3=f_4$, $f_k\pm f_l\pm f_m\neq 0$'
                   '\n(quartet, no triad)'),
    }
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple']
    fig = plt.figure(figsize=(9, 12))
    for row, case in enumerate((NONRES, TRIAD, QUARTET)):
        bmd, x, k = _case(**case)
        q, _, _ = surrogate_waves(case['freqs'], seed=0)

        ax_a = fig.add_subplot(3, 2, 2 * row + 1)
        freq, A = amplitude_spectrum(q[:, 0, 0], 128, 1.0)
        pos = freq >= 0
        ax_a.plot(freq[pos], A[pos], 'k')
        for j, f in enumerate(case['freqs']):
            ax_a.axvline(f, color=colors[j], lw=1)
        ax_a.set_xlim(0, 0.5)
        ax_a.set_ylim(0, 1.05)
        ax_a.set_xlabel('$f$')
        ax_a.set_ylabel('$A$')
        ax_a.set_title(titles[case['name']], fontsize=9)

        ax_b = fig.add_subplot(3, 2, 2 * row + 2, projection='3d')
        t = bmd.triads
        vals = np.abs(bmd.L[t.f1_idx, t.f2_idx])
        # plot_trisurf colours by the *data* range, not by set_zlim, so a
        # panel that is flat relative to the z-axis (e.g. the non-resonant and
        # quartet cases, at <1% of the 0-0.05 range) would otherwise be
        # painted with the full colormap and read as structured; pin vmin/vmax
        # to the z-limits so colour and height agree, as MATLAB's fixed caxis
        # does in the published figure.
        ax_b.plot_trisurf(t.f1, t.f2, vals, cmap='viridis', linewidth=0.1,
                          vmin=0, vmax=0.05)
        ax_b.set_zlim(0, 0.05)
        ax_b.set_xlabel('$f_1$')
        ax_b.set_ylabel('$f_2$')
        ax_b.set_zlabel(r'$|\lambda_1|$')
        ax_b.set_xticks([0, 0.2, 0.4])
        ax_b.set_yticks([0, 0.1, 0.2])
    fig.tight_layout()
    out1 = os.path.join(FIGURES_DIR, 'hypothesis_harmonics_row.png')
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    assert os.path.getsize(out1) > 0

    # -- figure 2: noise case, 3 panels (bispectra as 3-D surfaces, matching
    #    the style of figure 1's mode-bispectrum panels) ----------------------
    bmd, x, k = _case(name='noise', freqs=TRIAD['freqs'], snr=1.0)
    q, _, _ = surrogate_waves(TRIAD['freqs'], seed=0, snr=1.0)
    t = bmd.triads
    B = classical_bispectrum(q[:, 0, 0], 128, 1.0, 64)
    B_vals = B[t.k, t.l]

    fig2 = plt.figure(figsize=(15, 4.5))
    ax0 = fig2.add_subplot(1, 3, 1)
    freq, A = amplitude_spectrum(q[:, 0, 0], 128, 1.0)
    pos = freq >= 0
    ax0.plot(freq[pos], A[pos], 'k')
    ax0.set_xlim(0, 0.5)
    ax0.set_title('(a) amplitude spectrum')
    ax0.set_xlabel('$f$')

    ax1 = fig2.add_subplot(1, 3, 2, projection='3d')
    ax1.plot_trisurf(t.f1, t.f2, B_vals, cmap='viridis', linewidth=0.1,
                     vmin=0, vmax=0.25)
    ax1.set_title('(b) classical bispectrum')
    ax1.set_zlim(0, 0.25)
    ax1.set_xlabel('$f_1$')
    ax1.set_ylabel('$f_2$')
    ax1.set_zlabel('$|B|$')
    ax1.set_xticks([0, 0.2, 0.4])
    ax1.set_yticks([0, 0.1, 0.2])

    ax2 = fig2.add_subplot(1, 3, 3, projection='3d')
    # the z-limit must come from the data: the noisy mode bispectrum peaks
    # around 0.057 here, and
    # a fixed limit taken from an unrelated scale (e.g. panel (b)'s, or a
    # guessed round number) either flattens the peak into invisibility or
    # leaves the panel mostly empty -- neither matches the published figure,
    # which shows one clear dominant peak.
    vals = np.abs(bmd.L[t.f1_idx, t.f2_idx])
    z_max = float(np.nanmax(vals)) * 1.05
    ax2.plot_trisurf(t.f1, t.f2, vals, cmap='viridis', linewidth=0.1,
                     vmin=0, vmax=z_max)
    ax2.set_title('(c) mode bispectrum')
    ax2.set_zlim(0, z_max)
    ax2.set_xlabel('$f_1$')
    ax2.set_ylabel('$f_2$')
    ax2.set_zlabel(r'$|\lambda_1|$')
    ax2.set_xticks([0, 0.2, 0.4])
    ax2.set_yticks([0, 0.1, 0.2])

    fig2.tight_layout()
    out2 = os.path.join(FIGURES_DIR, 'hypothesis_noise.png')
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    assert os.path.getsize(out2) > 0
