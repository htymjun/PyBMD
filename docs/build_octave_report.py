#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Regenerate the figures embedded in ``docs/octave_cross_validation.md``.

Requires ``octave-cli`` on PATH and the ``refs/bmd`` submodule populated
(``git submodule update --init``); run from anywhere, paths are resolved
relative to this file.

    python docs/build_octave_report.py
'''
import os
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

DOCS_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.realpath(os.path.join(DOCS_DIR, '..'))
FIG_DIR = os.path.join(DOCS_DIR, 'figures', 'octave')
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'tests', 'octave'))

from pybmd.bmd.standard import Standard
from pybmd.bmd.postproc import plot_mode_bispectrum
import pybmd.bmd.utils as utils_bmd
import pybmd.utils.weights as utils_weights

import octave_ref as oref


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
    '''PyBMD and reference L, at the config docs/octave_cross_validation.md cites.'''
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
    path = os.path.join(FIG_DIR, 'bispectrum_comparison.png')
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'wrote {path}')


def fig_deviation(bmd, L_ref):
    '''Per-triad relative deviation of the reference from PyBMD, in the (k,l) plane.'''
    t = bmd.triads
    vals_py = np.abs(bmd.L[t.f1_idx, t.f2_idx])
    vals_ref = np.abs(L_ref[t.f1_idx, t.f2_idx])
    rel = np.abs(vals_ref - vals_py) / np.maximum(vals_py, 1e-300)

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
    path = os.path.join(FIG_DIR, 'deviation_heatmap.png')
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'wrote {path}')
    return rel


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
    path = os.path.join(FIG_DIR, 'scale_equivariance.png')
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'wrote {path}')
    rel = np.abs(b - a) / np.maximum(a, 1e-300)
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

    fig_scale_equivariance()


if __name__ == '__main__':
    main()
