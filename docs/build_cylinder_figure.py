#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Reproduce ``refs/figures/cylinder_bispectrum_sumdiff.pdf`` (Schmidt 2020,
*Nonlinear Dynamics*) with PyBMD on the cylinder-wake dataset.

Requires the ``refs/bmd`` submodule populated (``git submodule update --init``)
for ``refs/bmd/wake_Re500.mat``; run from anywhere, paths are resolved
relative to this file.

    MPLBACKEND=Agg python docs/build_cylinder_figure.py
    MPLBACKEND=Agg mpirun -n 4 python docs/build_cylinder_figure.py --mpi

See docs/cylinder_bispectrum.md for how the parameters below were derived
from the published figure.
'''
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

DOCS_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.realpath(os.path.join(DOCS_DIR, '..'))
FIG_DIR = os.path.join(DOCS_DIR, 'figures', 'cylinder')
sys.path.insert(0, REPO_ROOT)

from pybmd.bmd.standard import Standard
from pybmd.bmd.postproc import load_results, top_triads
import pybmd.utils.weights as utils_weights
from pybmd.utils.io import read_data

DEFAULT_DATA = os.path.join(REPO_ROOT, 'refs', 'bmd', 'wake_Re500.mat')

# the six triads the reference figure circles and labels in panel (b)
TRIADS = ((12, 12), (12, 0), (24, 12), (24, 24), (36, 12), (36, 24))
# per-triad label placement: (dx, dy) offset in points and horizontal
# alignment. The two triads sharing a row (k=24 and k=36) point their labels
# away from each other so the text doesn't collide in the gap between them;
# (12,12) and (12,0) are pushed right, clear of the y-axis and the f2=0 line
LABEL_LAYOUT = {
    (12, 12): ((6, 2), 'left'), (12, 0): ((8, -2), 'left'),
    (24, 12): ((-4, 8), 'right'), (24, 24): ((-4, 8), 'right'),
    (36, 12): ((4, 8), 'left'), (36, 24): ((4, 8), 'left'),
}

# sampled from refs/figures/cylinder_bispectrum_sumdiff.pdf's own colorbar,
# rasterized at 400 dpi: jet, with its low (dark blue) end replaced by a fade
# to white so the featureless background reads as blank rather than "cold"
_CBAR_STOPS = (
    (0.000, (1.000, 1.000, 1.000)), (0.074, (0.835, 0.835, 1.000)),
    (0.152, (0.482, 0.482, 1.000)), (0.230, (0.075, 0.075, 1.000)),
    (0.307, (0.000, 0.345, 1.000)), (0.381, (0.004, 0.733, 1.000)),
    (0.459, (0.051, 1.000, 0.953)), (0.537, (0.306, 1.000, 0.694)),
    (0.615, (0.694, 1.000, 0.306)), (0.689, (1.000, 1.000, 0.000)),
    (0.767, (1.000, 0.545, 0.000)), (0.844, (1.000, 0.090, 0.000)),
    (0.922, (0.776, 0.000, 0.000)), (1.000, (0.502, 0.000, 0.000)),
)


def _check_prereqs(data_path):
    if not os.path.exists(data_path):
        sys.exit(
            f'Cannot build the figure:\n  {data_path} not found -- run '
            '`git submodule update --init` to populate refs/bmd.')


def _cylinder_colormap():
    return LinearSegmentedColormap.from_list('cylinder_jet', _CBAR_STOPS)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-dft', type=int, default=480)
    p.add_argument('--overlap', type=float, default=50,
                   help='percent overlap between DFT blocks (default: 50, '
                        'matching the reference)')
    p.add_argument('--data', default=DEFAULT_DATA)
    p.add_argument('--savedir', default=os.path.join(
        REPO_ROOT, 'cylinder_sumdiff_out'))
    p.add_argument('--out', default=os.path.join(
        FIG_DIR, 'cylinder_bispectrum_sumdiff.png'))
    p.add_argument('--dpi', type=int, default=200)
    p.add_argument('--clim', type=float, nargs=2, default=None,
                   metavar=('VMIN', 'VMAX'),
                   help='override the shared log|lambda_1| colour limits')
    p.add_argument('--reuse', action='store_true',
                   help='load an existing --savedir instead of re-fitting')
    p.add_argument('--mpi', action='store_true',
                   help='fit under mpi4py.MPI.COMM_WORLD')
    return p.parse_args()


def _load_cylinder_wake(data_path):
    d = read_data(data_path)
    dt = float(np.ravel(d['dt'])[0])
    u = np.asarray(d['u'], dtype=np.float64)
    return u, dt


def _fit(args, comm):
    u, dt = _load_cylinder_wake(args.data)
    nt, n1, n2 = u.shape
    is_root = comm is None or comm.rank == 0
    if is_root:
        print(f'cylinder wake: nt={nt}, grid={n1}x{n2}, dt={dt}')

    data = u[..., np.newaxis]  # single variable: streamwise velocity
    params = dict(
        n_dft=args.n_dft,
        time_step=dt,
        n_space_dims=2,
        n_variables=1,
        overlap=args.overlap,
        regions=[1, 2],                 # sum- and difference-interactions
        max_freq_idx=None,              # the whole plane, as in the reference
        solver='MengiOverton',
        save_modes=False,               # ~43k triads: modes would be ~8 GB
        store_modes=False,
        compute_energy_transfer=False,  # T is not part of this figure
        savedir=args.savedir,
    )
    # refs/bmd/bmd.m:279-281 defaults to weight = ones(nx,1) ("uniform") when
    # no weight is passed, and example1.m calls bmd(u) with none -- match that
    # rather than a physically-motivated quadrature weight, since B (and so
    # the colour scale) is linear in the weight
    weights = utils_weights.uniform((n1, n2), n_vars=1, dV=1.0)
    bmd = Standard(params=params, weights=weights, comm=comm).fit(data)
    if is_root:
        df = 1.0 / (args.n_dft * dt)
        print(f'n_dft={args.n_dft}  df={df:.6f}  n_overlap={bmd.n_overlap}  '
              f'n_blocks={bmd.n_blocks}  n_triads={bmd.n_triads}')
    return bmd


def _print_report(results):
    triads = results.triads
    values = np.abs(results.L[triads.f1_idx, triads.f2_idx])
    print('\nlabelled triads:')
    for k, l in TRIADS:
        i = triads.find(k, l)
        print(f'  (k,l,k+l) = ({k:3d},{l:3d},{k + l:3d})  '
              f'(f1,f2,f3) = ({triads.f1[i]:.4f}, {triads.f2[i]:.4f}, '
              f'{triads.f3[i]:.4f})  |lambda_1| = {values[i]:.4e}')

    print('\ntop 15 triads by |lambda_1| (k != 0 and l != 0):')
    for row in top_triads(results, n=15, quantity='L', exclude_zero=True):
        print(f"  (k,l,k+l) = ({row['k']:3d},{row['l']:3d},{row['kl']:3d})  "
              f"|lambda_1| = {row['value']:.4e}")


def _panel(ax, field, freq, df, xlim, ylim, cmap, vmin, vmax):
    edges = np.append(freq - df / 2, freq[-1] + df / 2)
    ax.pcolormesh(edges, edges, field.T, cmap=cmap, vmin=vmin, vmax=vmax,
                 shading='flat')
    ax.set_aspect('equal')
    ax.set_xlabel(r'$f_1$')
    ax.set_ylabel(r'$f_2$')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def _plot(results, args):
    freq, L = results.freq, results.L
    df = freq[1] - freq[0]
    triads = results.triads
    field = np.ma.masked_invalid(np.log(np.abs(L)))
    if args.clim is not None:
        vmin, vmax = args.clim
    else:
        vmin, vmax = float(field.min()), float(field.max())
    cmap = _cylinder_colormap()

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(7.6, 4.0), gridspec_kw=dict(width_ratios=[4, 3]))

    _panel(axa, field, freq, df, (0, freq[-1]), (-freq[-1], freq[-1] / 2),
          cmap, vmin, vmax)
    cax = axa.inset_axes([0.055, 0.026, 0.083, 0.251])
    fig.colorbar(ScalarMappable(Normalize(vmin, vmax), cmap), cax=cax,
                ticks=[0, -10, -20], label=r'$\log(|\lambda_1|)$')

    _panel(axb, field, freq, df, (0, 0.8), (-0.8, 0.8), cmap, vmin, vmax)

    f0 = triads.f1[triads.find(12, 0)]  # the shedding frequency, 12*df
    axb.plot([0, 0.8], [f0, f0 - 0.8], 'k--', lw=0.8)
    for k, l in TRIADS:
        i = triads.find(k, l)
        f1, f2 = triads.f1[i], triads.f2[i]
        axb.plot(f1, f2, 'o', ms=7, mfc='none', mec='k', mew=1.0)
        offset, ha = LABEL_LAYOUT[(k, l)]
        axb.annotate(f'({k},{l})', (f1, f2), textcoords='offset points',
                     xytext=offset, fontsize=8, ha=ha, va='bottom')

    for ax, label in ((axa, '(a)'), (axb, '(b)')):
        ax.text(-0.32, 1.08, label, transform=ax.transAxes,
               fontsize=12, fontweight='bold', va='bottom')

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'\n[vmin, vmax] = [{vmin:.3f}, {vmax:.3f}]')
    print(f'wrote {args.out}')


def main():
    args = _parse_args()
    _check_prereqs(args.data)

    comm = None
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

    if args.reuse:
        results = load_results(args.savedir)
    else:
        _fit(args, comm)
        if comm is not None and comm.rank != 0:
            return
        results = load_results(args.savedir)

    _print_report(results)
    _plot(results, args)


if __name__ == '__main__':
    main()
