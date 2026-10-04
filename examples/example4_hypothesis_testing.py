#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Example 4: the "hypothesis testing" surrogate data of Schmidt (2020, Figs. 4
and 5): travelling waves whose frequencies do or do not form a triad, with
and without noise. Renders the figures into ``example4_out/`` for visual
comparison with the paper.

    MPLBACKEND=Agg python examples/example4_hypothesis_testing.py

``tests/test_hypothesis.py`` asserts the scientific content on the same data.
'''
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

CFD = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CFD, '..'))

from pybmd.bmd.standard import Standard
import pybmd.utils.weights as utils_weights

# the paper's frequencies, moved to the nearest bins of n_dft=128
NONRES = dict(name='nonres', freqs=(0.046875, 0.203125, 0.3515625)) # (0.05, 0.2, 0.35)
TRIAD = dict(name='triad', freqs=(0.046875, 0.203125, 0.25)) # (0.05, 0.2, 0.25)
QUARTET = dict(name='quartet', freqs=(0.046875, 0.1484375, 0.25, 0.453125)) # (0.05, 0.15, 0.25, 0.45)
NOISE = dict(name='noise', freqs=TRIAD['freqs'], snr=1.0)


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


def fit_case(name, freqs, snr=None, save_dir='example4_out',
             store_modes=False, **overrides):
    '''BMD of one surrogate case with the paper's settings: n_dft=128, no
    overlap, Hann window, sum interactions only; ``overrides`` go into
    ``params``. Returns ``(bmd, x, k)``.'''
    q, x, k = surrogate_waves(freqs, seed=0, snr=snr)
    params = dict(
        n_dft=128, time_step=1.0, n_space_dims=1, n_variables=1, overlap=0,
        window='hann', regions=[1], solver='MengiOverton', save_modes=False,
        store_modes=store_modes, savedir=os.path.join(save_dir, name))
    params.update(overrides)
    w = utils_weights.uniform((x.size,), n_vars=1, dV=x[1] - x[0])
    return Standard(params=params, weights=w).fit(q), x, k


def _block_dft(q_x0, n_dft):
    '''Hann-windowed DFT of the non-overlapping blocks of a single-point time
    series, normalized like BMD's: ``(n_blocks, n_dft)``, column ``i`` being
    integer frequency ``i`` (not fftshifted).'''
    win = np.hanning(n_dft + 1)[:-1]
    n_blocks = len(q_x0) // n_dft
    blocks = (q_x0 - q_x0.mean())[:n_blocks * n_dft].reshape(n_blocks, n_dft)
    return np.fft.fft(win * blocks, axis=1) / win.mean() / n_dft


def amplitude_spectrum(q_x0, n_dft, dt):
    '''``A(f) = 2|mean_blocks q_hat(f)|``, computed independently of BMD with
    the same window and blocking, as the paper's panel (a) does.'''
    q_hat = _block_dft(q_x0, n_dft)
    return np.fft.fftfreq(n_dft, dt), 2 * np.abs(q_hat).mean(axis=0)


def classical_bispectrum(q_x0, n_dft, m):
    '''
    Classical (biased) bispectrum estimator of a single-point time series on
    the integer-frequency grid ``0 <= j <= i < m``, block-averaged with the
    same window and blocking as BMD -- the quantity the paper compares the
    mode bispectrum against in its noise test.
    '''
    q_hat = _block_dft(q_x0, n_dft)
    B = np.full((m, m), np.nan)
    for i in range(m):
        for j in range(i + 1):
            if i + j < n_dft // 2:
                B[i, j] = np.abs(np.mean(
                    q_hat[:, i] * q_hat[:, j] * np.conj(q_hat[:, i + j])))
    return B


def _save(fig, path):
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'wrote {path}')


def _surface(ax, t, vals, zmax, zlabel=r'$|\lambda_1|$'):
    # plot_trisurf colours by the *data* range, not by set_zlim, so a panel
    # that is flat relative to the z-axis would otherwise be painted with the
    # full colormap and read as structured; pin vmin/vmax to the z-limits so
    # colour and height agree, as MATLAB's fixed caxis does in the paper
    ax.plot_trisurf(t.f1, t.f2, vals, cmap='viridis', linewidth=0.1,
                    vmin=0, vmax=zmax)
    ax.set_zlim(0, zmax)
    ax.set_xlabel('$f_1$')
    ax.set_ylabel('$f_2$')
    ax.set_zlabel(zlabel)
    ax.set_xticks([0, 0.2, 0.4])
    ax.set_yticks([0, 0.1, 0.2])


def _amplitude_panel(ax, q):
    freq, A = amplitude_spectrum(q[:, 0, 0], 128, 1.0)
    pos = freq >= 0
    ax.plot(freq[pos], A[pos], 'k')
    ax.set_xlim(0, 0.5)
    ax.set_xlabel('$f$')


def main(save_dir='example4_out'):
    os.makedirs(save_dir, exist_ok=True)

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
        bmd, _, _ = fit_case(save_dir=save_dir, **case)
        q, _, _ = surrogate_waves(case['freqs'], seed=0)

        ax_a = fig.add_subplot(3, 2, 2 * row + 1)
        _amplitude_panel(ax_a, q)
        for j, f in enumerate(case['freqs']):
            ax_a.axvline(f, color=colors[j], lw=1)
        ax_a.set_ylim(0, 1.05)
        ax_a.set_ylabel('$A$')
        ax_a.set_title(titles[case['name']], fontsize=9)

        t = bmd.triads
        _surface(fig.add_subplot(3, 2, 2 * row + 2, projection='3d'), t,
                 np.abs(bmd.L[t.f1_idx, t.f2_idx]), 0.05)
    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'hypothesis_harmonics_row.png'))

    # -- figure 2: unit-SNR noise; amplitude, classical and mode bispectra ---
    bmd, _, _ = fit_case(save_dir=save_dir, **NOISE)
    q, _, _ = surrogate_waves(NOISE['freqs'], seed=0, snr=NOISE['snr'])
    t = bmd.triads
    B = classical_bispectrum(q[:, 0, 0], 128, 64)

    fig = plt.figure(figsize=(15, 4.5))
    ax0 = fig.add_subplot(1, 3, 1)
    _amplitude_panel(ax0, q)
    ax0.set_title('(a) amplitude spectrum')

    ax1 = fig.add_subplot(1, 3, 2, projection='3d')
    _surface(ax1, t, B[t.k, t.l], 0.25, zlabel='$|B|$')
    ax1.set_title('(b) classical bispectrum')

    ax2 = fig.add_subplot(1, 3, 3, projection='3d')
    # the z-limit comes from the data (peak ~0.057): a fixed one either
    # flattens the peak or leaves the panel mostly empty
    vals = np.abs(bmd.L[t.f1_idx, t.f2_idx])
    _surface(ax2, t, vals, float(np.nanmax(vals)) * 1.05)
    ax2.set_title('(c) mode bispectrum')

    fig.tight_layout()
    _save(fig, os.path.join(save_dir, 'hypothesis_noise.png'))


if __name__ == '__main__':
    main()
