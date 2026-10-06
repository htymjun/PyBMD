#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Regenerate the figures embedded in ``octave_cross_validation.md`` (written
to ``figures/`` next to this file).

Requires ``octave-cli`` on PATH and the ``refs/bmd`` submodule populated
(``git submodule update --init``); run from anywhere, paths are resolved
relative to this file.

    python tests/octave/build_report.py
'''
import os
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

OCTAVE_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.realpath(os.path.join(OCTAVE_DIR, '..', '..'))
FIG_DIR = os.path.join(OCTAVE_DIR, 'figures')
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, OCTAVE_DIR)

from pybmd.bmd.standard import Standard
from pybmd.bmd.postproc import plot_mode_bispectrum
import pybmd.bmd.utils as utils_bmd
import pybmd.utils.weights as utils_weights

import octave_ref as oref
# the paper's surrogate-data recipe and BMD settings, reused rather than
# duplicated
from examples.example4_hypothesis_testing import (surrogate_waves, fit_case,
                                                  TRIAD)


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'wrote {path}')


def _rel(a, ref):
    '''Element-wise relative deviation of ``a`` from ``ref``.'''
    return np.abs(a - ref) / np.maximum(ref, 1e-300)


def _check_prereqs():
    exe = shutil.which('octave-cli') or shutil.which('octave')
    bmd_m = os.path.join(REPO_ROOT, 'refs', 'bmd', 'bmd.m')
    missing = []
    if exe is None:
        missing.append('octave-cli not on PATH')
    if not os.path.exists(bmd_m):
        missing.append(f'{bmd_m} not found -- run `git submodule update --init`')
    if missing:
        sys.exit('Cannot build the report:\n  ' + '\n  '.join(missing))


def _full_dataset_run():
    '''
    PyBMD and reference L, at the config octave_cross_validation.md
    cites; the reference side runs bmd.m's own MengiOverton.
    '''
    mat_path = oref.require_full_dataset()
    d = scipy.io.loadmat(mat_path)
    dt = float(d['dt'][0, 0])
    nt, n1, n2 = d['u'].shape
    x = d['u'].astype(np.float64)[..., np.newaxis]
    dV = float((d['x'][1, 0] - d['x'][0, 0]) * (d['y'][0, 1] - d['y'][0, 0]))

    params = dict(n_dft=256, time_step=dt, n_space_dims=2, n_variables=1,
                 n_overlap=128, regions=[1, 2], max_freq_idx=12,
                 save_modes=False, tol=1e-6, n_it_max=500,
                 savedir=os.path.join(FIG_DIR, '_scratch'))
    w = utils_weights.uniform((n1, n2), 1, dV)
    bmd = Standard(params=params, weights=w).fit(x)
    out = oref.run('bmd', x, window=bmd._window.ravel(), weight=w['weights'],
                   n_overlap=128, dt=dt, regions=[1, 2], max_freq_idx=12,
                   tol=1e-6, n_it_max=500, timeout=280)
    shutil.rmtree(params['savedir'], ignore_errors=True)
    return bmd, out['L']


def fig_bispectrum_comparison(bmd, L_ref):
    '''Side-by-side mode bispectrum: PyBMD vs. the reference, same data.'''
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plot_mode_bispectrum(bmd.L, bmd.freq, ax=axes[0],
                         title='PyBMD (MengiOverton)')
    plot_mode_bispectrum(L_ref, bmd.freq, ax=axes[1],
                         title='Reference bmd.m, run under Octave')
    fig.suptitle('Mode bispectrum $\\log|\\lambda_1|$ -- cylinder wake, '
                 'regions={1,2}, max_freq_idx=12, 169 triads')
    fig.tight_layout()
    _save(fig, 'bispectrum_comparison.png')


def fig_deviation(bmd, L_ref):
    '''Per-triad relative deviation of the reference from PyBMD, in the (k,l) plane.'''
    t = bmd.triads
    vals_py = np.abs(bmd.L[t.f1_idx, t.f2_idx])
    vals_ref = np.abs(L_ref[t.f1_idx, t.f2_idx])
    rel = _rel(vals_ref, vals_py)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    sc = ax.scatter(t.k, t.l, c=100 * rel, cmap='inferno_r', s=45,
                    vmin=0, vmax=max(1.0, float(100 * rel.max())),
                    edgecolors='none')
    over10 = rel > 0.10
    ax.scatter(t.k[over10], t.l[over10], s=110, facecolors='none',
              edgecolors='cyan', linewidths=1.3,
              label=f'off by >10% ({int(over10.sum())}/{t.n_triads})')
    ax.set_xlabel('$k$')
    ax.set_ylabel('$l$')
    ax.set_aspect('equal')
    ax.set_title('Reference deviation from PyBMD, per triad\n'
                 '(never exceeds PyBMD -- always an under-estimate)')
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    fig.colorbar(sc, ax=ax, label='relative deviation, %')
    fig.tight_layout()
    _save(fig, 'deviation_heatmap.png')
    return rel


def _hypothesis_run(freqs, snr, max_freq_idx=40):
    '''
    PyBMD and the reference bmd.m (two solvers) on one
    hypothesis-test surrogate case, through example4's ``fit_case`` (n_dft=128,
    overlap=0, Hann window, regions=[1], 10 blocks).

    :return: ``(results, triads)``, where ``results`` maps
        ``'pybmd_MengiOverton'`` and ``'bmd_<solver>'`` to the respective
        ``L``.
    '''
    q, x, k = surrogate_waves(freqs, seed=0, snr=snr)
    w = utils_weights.uniform((x.size,), n_vars=1, dV=x[1] - x[0])

    name = '_scratch_hyp'
    bmd, _, _ = fit_case(name, freqs, snr=snr, save_dir=FIG_DIR,
                         max_freq_idx=max_freq_idx)
    results = {'pybmd_MengiOverton': bmd.L}
    shutil.rmtree(os.path.join(FIG_DIR, name), ignore_errors=True)

    # bmd.m's HeWatson draws an unseeded random start vector (refs/bmd/bmd.m
    # has no seeding hook this driver can reach), so its numbers -- unlike
    # every other figure in this script -- vary run to run; that variability
    # is itself part of what the figure documents.
    kw = dict(window=bmd._window.ravel(), weight=w['weights'], n_overlap=0,
             dt=1.0, regions=[1], max_freq_idx=max_freq_idx, tol=1e-6,
             n_it_max=500, timeout=280)
    results['bmd_MengiOverton'] = oref.run('bmd', q, solver='MengiOverton', **kw)['L']
    results['bmd_HeWatson'] = oref.run('bmd', q, solver='HeWatson', **kw)['L']
    return results, bmd.triads


def fig_hypothesis_pybmd_vs_matlab():
    '''
    Schmidt (2020) verified BMD by hypothesis testing with He & Watson's
    algorithm, not Mengi-Overton (which postdates the paper's 2020
    publication -- bmd.m switched default solvers on 2023-08-16). This
    reruns that test -- the resonant triad, without noise and at SNR=1 --
    through PyBMD and through the real bmd.m under Octave, so the published
    qualitative conclusion (a clean peak on the driven triad, side peaks
    suppressed) is checked directly rather than inferred from the isolated
    B-matrix comparison in test_octave_reference.py's Tier C.
    '''
    cases = [('no noise', None), ('SNR = 1', 1.0)]
    fig = plt.figure(figsize=(16, 9))
    summary = []

    for row, (label, snr) in enumerate(cases):
        results, t = _hypothesis_run(TRIAD['freqs'], snr)
        panels = [
            ('PyBMD MengiOverton',
            np.abs(results['pybmd_MengiOverton'][t.f1_idx, t.f2_idx])),
            ('bmd.m MengiOverton',
            np.abs(results['bmd_MengiOverton'][t.f1_idx, t.f2_idx])),
            ('bmd.m HeWatson',
            np.abs(results['bmd_HeWatson'][t.f1_idx, t.f2_idx])),
        ]
        vmax = max(v.max() for _, v in panels) * 1.05
        for col, (name, vals) in enumerate(panels):
            ax = fig.add_subplot(2, 4, row * 4 + col + 1, projection='3d')
            ax.plot_trisurf(t.f1, t.f2, vals, cmap='viridis', linewidth=0.1,
                            vmin=0, vmax=vmax)
            i = int(np.argmax(vals))
            ax.set_zlim(0, vmax)
            ax.set_xlabel('$f_1$')
            ax.set_ylabel('$f_2$')
            ax.set_title(f'{name} ({label})\npeak ({t.k[i]},{t.l[i]}) '
                        f'$|\\lambda_1|$={vals[i]:.5f}', fontsize=8)

        py = np.abs(results['pybmd_MengiOverton'][t.f1_idx, t.f2_idx])
        order = np.argsort(py)
        ax = fig.add_subplot(2, 4, row * 4 + 4)
        alternatives = [
            ('bmd.m MengiOverton', 'bmd_MengiOverton', 'crimson'),
            ('bmd.m HeWatson', 'bmd_HeWatson', 'tab:orange'),
        ]
        for name, key, color in alternatives:
            vals = np.abs(results[key][t.f1_idx, t.f2_idx])
            rel = _rel(vals, py)
            ax.semilogy(np.arange(len(py)), np.maximum(rel[order], 1e-16), '.',
                       ms=3, color=color, label=name)
            summary.append((label, name, float(rel.max()),
                           int((rel > 0.01).sum()), len(rel)))
        ax.set_xlabel(f'triad, sorted by PyBMD $|\\lambda_1|$ ({label})')
        ax.set_ylabel('relative deviation from\nPyBMD MengiOverton', fontsize=8)
        ax.legend(fontsize=6, loc='upper left')
        ax.set_title(f'{label}: disagreement lives in the\nnear-zero background',
                    fontsize=8, pad=12)

    fig.suptitle("Hypothesis test (Schmidt 2020) -- resonant triad, PyBMD vs. "
                "bmd.m under Octave\nthe paper's conclusion (peak on the driven "
                "triad) is solver-independent")
    # tight_layout does not reason well about a grid mixing 3D and 2D axes
    # (it under-estimates the space the deviation panels' title/ylabel need);
    # a manual rect plus explicit spacing avoids the overlap tight_layout
    # alone leaves between them.
    fig.subplots_adjust(top=0.86, bottom=0.08, hspace=0.45, wspace=0.5)
    _save(fig, 'hypothesis_pybmd_vs_matlab.png')
    for label, name, mx, n1, n in summary:
        print(f'  [{label}] {name:24s} vs PyBMD MengiOverton: '
             f'max rel {mx:.3e}; >1%: {n1}/{n}')


def fig_scale_equivariance():
    '''
    A correct solver for the numerical radius is exactly scale-equivariant:
    r(cA) = c*r(A). Runs the *unmodified* reference on a random case and on a
    1e-2 rescale of it, and plots |L(X)| against the rescaled-back |L(cX)|/c^3
    -- points on the diagonal would mean perfect equivariance.
    '''
    rng = np.random.default_rng(7)
    x = rng.standard_normal((128, 4, 3))
    window = utils_bmd.hamming_window(32)
    weight = np.ones((4, 3))
    c = 1e-2
    kwargs = dict(window=window, weight=weight, n_overlap=16, dt=1 / 32,
                 regions=[1, 2], max_freq_idx=4, tol=1e-6, n_it_max=500)
    out1 = oref.run('bmd', x, **kwargs)
    out2 = oref.run('bmd', c * x, **kwargs)
    L1, L2 = out1['L'], out2['L']
    finite = np.isfinite(L1) & np.isfinite(L2)
    a = np.abs(L1[finite])
    b = np.abs(L2[finite]) / c**3

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lim = max(a.max(), b.max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', lw=1, label='perfect equivariance')
    ax.scatter(a, b, s=30, color='crimson')
    ax.set_xlabel(r'$|\lambda_1(X)|$')
    ax.set_ylabel(r'$|\lambda_1(10^{-2}X)| \, / \, 10^{-6}$')
    ax.set_title("Reference bmd.m's scale-equivariance error\n"
                 '(run directly under Octave, unmodified)')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', frameon=True)
    fig.tight_layout()
    _save(fig, 'scale_equivariance.png')
    rel = _rel(b, a)
    print(f'  max rel deviation from equivariance: {rel.max():.3f}; '
         f'>1%: {int((rel > 0.01).sum())}/{rel.size}; '
         f'>10%: {int((rel > 0.10).sum())}/{rel.size}')


def main():
    _check_prereqs()
    os.makedirs(FIG_DIR, exist_ok=True)

    bmd, L_ref = _full_dataset_run()
    fig_bispectrum_comparison(bmd, L_ref)
    rel = fig_deviation(bmd, L_ref)
    print(f'full dataset: {int((rel > 0.01).sum())}/{rel.size} triads off by '
         f'>1%, {int((rel > 0.10).sum())}/{rel.size} by >10%')

    fig_hypothesis_pybmd_vs_matlab()

    fig_scale_equivariance()


if __name__ == '__main__':
    main()
