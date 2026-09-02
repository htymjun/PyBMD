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
sys.path.insert(0, os.path.join(REPO_ROOT, 'tests'))

from pybmd.bmd.standard import Standard
from pybmd.bmd.postproc import plot_mode_bispectrum
import pybmd.bmd.utils as utils_bmd
import pybmd.utils.weights as utils_weights

import octave_ref as oref
# the paper's surrogate-data recipe, reused rather than duplicated -- see
# test_hypothesis.py's own docstring for the recipe itself
from test_hypothesis import surrogate_waves, TRIAD


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


def _full_dataset_run(pybmd_solver='MengiOverton'):
    '''
    PyBMD and reference L, at the config docs/octave_cross_validation.md
    cites. ``pybmd_solver`` selects PyBMD's own solver; the reference side
    always runs bmd.m's own MengiOverton -- the fixed point of comparison.
    '''
    mat_path = oref.require_full_dataset()
    d = scipy.io.loadmat(mat_path)
    dt = float(d['dt'][0, 0])
    nt, n1, n2 = d['u'].shape
    x = d['u'].astype(np.float64)[..., np.newaxis]
    dV = float((d['x'][1, 0] - d['x'][0, 0]) * (d['y'][0, 1] - d['y'][0, 0]))

    params = dict(n_dft=256, time_step=dt, n_space_dims=2, n_variables=1,
                 n_overlap=128, regions=[1, 2], max_freq_idx=12,
                 save_modes=False, tol=1e-6, n_it_max=500, solver=pybmd_solver,
                 savedir=os.path.join(FIG_DIR, f'_scratch_{pybmd_solver}'))
    w = utils_weights.uniform((n1, n2), 1, dV)
    bmd = Standard(params=params, weights=w).fit(x)
    # the octave call is identical across pybmd_solver values, so octave_ref's
    # in-process cache turns every call after the first into a no-op
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


def fig_three_way_solver_comparison(bmd, L_ref):
    '''
    Three-way comparison on the config fig_bispectrum_comparison/fig_deviation
    already use: the reference's own (bug-affected) MengiOverton, PyBMD's
    default (corrected) MengiOverton, and PyBMD's MengiOvertonMATLAB -- a
    bug-for-bug port that exists to reproduce the reference. This is the
    live-Octave-checked artifact behind the ablation tables CLAUDE.md's
    Deviations section quotes.
    '''
    bmd_compat, _ = _full_dataset_run(pybmd_solver='MengiOvertonMATLAB')
    t = bmd.triads
    vals_py = np.abs(bmd.L[t.f1_idx, t.f2_idx])
    vals_compat = np.abs(bmd_compat.L[t.f1_idx, t.f2_idx])
    vals_ref = np.abs(L_ref[t.f1_idx, t.f2_idx])

    fig = plt.figure(figsize=(13, 8.5))
    for i, (L, name) in enumerate([
            (L_ref, 'Reference bmd.m\n(own MengiOverton)'),
            (bmd.L, 'PyBMD MengiOverton\n(default, corrected)'),
            (bmd_compat.L, 'PyBMD MengiOvertonMATLAB\n(bug-compatible)')]):
        ax = fig.add_subplot(2, 3, i + 1)
        plot_mode_bispectrum(L, bmd.freq, ax=ax, title=name)

    ax4 = fig.add_subplot(2, 3, 4)
    lim = max(vals_py.max(), vals_ref.max(), vals_compat.max()) * 1.05
    ax4.plot([0, lim], [0, lim], 'k--', lw=1, label='y = x')
    ax4.scatter(vals_py, vals_ref, s=18, color='crimson', label='bmd.m')
    ax4.scatter(vals_py, vals_compat, s=18, color='tab:blue', marker='x',
               label='PyBMD MengiOvertonMATLAB')
    ax4.set_xlabel(r'PyBMD MengiOverton $|\lambda_1|$')
    ax4.set_ylabel(r'$|\lambda_1|$')
    ax4.set_xlim(0, lim)
    ax4.set_ylim(0, lim)
    ax4.set_aspect('equal')
    ax4.legend(fontsize=8, loc='upper left')
    ax4.set_title('bmd.m and MengiOvertonMATLAB land on the\n'
                 'same line, both at or below PyBMD default', fontsize=9)

    for ax_idx, vals, title in (
            (5, vals_ref, 'Reference deviation\nfrom PyBMD default, %'),
            (6, vals_compat, 'MengiOvertonMATLAB deviation\nfrom PyBMD default, %')):
        ax = fig.add_subplot(2, 3, ax_idx)
        rel = np.abs(vals - vals_py) / np.maximum(vals_py, 1e-300)
        sc = ax.scatter(t.k, t.l, c=100 * rel, cmap='inferno_r', s=40,
                        vmin=0, vmax=max(1.0, float(100 * rel.max())),
                        edgecolors='none')
        ax.set_xlabel('$k$')
        ax.set_ylabel('$l$')
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=9)
        fig.colorbar(sc, ax=ax)

    fig.suptitle('Three-way solver comparison -- cylinder wake, regions={1,2}, '
                'max_freq_idx=12, 169 triads')
    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'three_way_solver_comparison.png')
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'wrote {path}')

    rel_ref = np.abs(vals_ref - vals_py) / np.maximum(vals_py, 1e-300)
    rel_compat = np.abs(vals_compat - vals_py) / np.maximum(vals_py, 1e-300)
    rel_compat_vs_ref = np.abs(vals_compat - vals_ref) / np.maximum(vals_ref, 1e-300)
    print(f'  bmd.m               vs PyBMD default: max rel {rel_ref.max():.3e}; '
         f'>1%: {int((rel_ref > 0.01).sum())}/{len(rel_ref)}; '
         f'>10%: {int((rel_ref > 0.10).sum())}/{len(rel_ref)}')
    print(f'  MengiOvertonMATLAB  vs PyBMD default: max rel {rel_compat.max():.3e}; '
         f'>1%: {int((rel_compat > 0.01).sum())}/{len(rel_compat)}')
    print(f'  MengiOvertonMATLAB  vs bmd.m        : max rel {rel_compat_vs_ref.max():.3e}; '
         f'>1%: {int((rel_compat_vs_ref > 0.01).sum())}/{len(rel_compat_vs_ref)}')


def _hypothesis_run(freqs, snr, max_freq_idx=40, n_dft=128, seed=0):
    '''
    PyBMD (three solvers) and the reference bmd.m (two solvers) on one
    hypothesis-test surrogate case -- see test_hypothesis.py's
    ``surrogate_waves`` for the recipe this reproduces exactly (n_dft=128,
    overlap=0, Hann window, regions=[1], 10 blocks).

    :return: ``(results, triads)``, where ``results`` maps
        ``'pybmd_<solver>'`` to a fitted :class:`Standard` and
        ``'bmd_<solver>'`` to the reference's raw ``L``.
    '''
    q, x, k = surrogate_waves(freqs, seed=seed, snr=snr)
    w = utils_weights.uniform((x.size,), n_vars=1, dV=x[1] - x[0])

    results = {}
    for solver in ('MengiOverton', 'MengiOvertonMATLAB', 'simpleIteration'):
        params = dict(n_dft=n_dft, time_step=1.0, n_space_dims=1, n_variables=1,
                     overlap=0, window='hann', regions=[1],
                     max_freq_idx=max_freq_idx, solver=solver, save_modes=False,
                     savedir=os.path.join(FIG_DIR, f'_scratch_hyp_{solver}'))
        bmd = Standard(params=params, weights=w).fit(q)
        results[f'pybmd_{solver}'] = bmd
        shutil.rmtree(params['savedir'], ignore_errors=True)

    bmd = results['pybmd_MengiOverton']
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
    B-matrix comparison in fig_three_way_solver_comparison.
    '''
    cases = [('no noise', None), ('SNR = 1', 1.0)]
    fig = plt.figure(figsize=(16, 9))
    summary = []

    for row, (label, snr) in enumerate(cases):
        results, t = _hypothesis_run(TRIAD['freqs'], snr)
        panels = [
            ('PyBMD MengiOverton',
            np.abs(results['pybmd_MengiOverton'].L[t.f1_idx, t.f2_idx])),
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

        py = np.abs(results['pybmd_MengiOverton'].L[t.f1_idx, t.f2_idx])
        order = np.argsort(py)
        ax = fig.add_subplot(2, 4, row * 4 + 4)
        alternatives = [
            ('bmd.m MengiOverton', 'bmd_MengiOverton', 'crimson'),
            ('bmd.m HeWatson', 'bmd_HeWatson', 'tab:orange'),
            ('PyBMD simpleIteration', 'pybmd_simpleIteration', 'tab:green'),
            ('PyBMD MengiOvertonMATLAB', 'pybmd_MengiOvertonMATLAB', 'tab:blue'),
        ]
        for name, key, color in alternatives:
            vals = (np.abs(results[key].L[t.f1_idx, t.f2_idx]) if key.startswith('pybmd_')
                   else np.abs(results[key][t.f1_idx, t.f2_idx]))
            rel = np.abs(vals - py) / np.maximum(py, 1e-300)
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
    path = os.path.join(FIG_DIR, 'hypothesis_pybmd_vs_matlab.png')
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'wrote {path}')
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

    fig_three_way_solver_comparison(bmd, L_ref)
    fig_hypothesis_pybmd_vs_matlab()

    fig_scale_equivariance()


if __name__ == '__main__':
    main()
