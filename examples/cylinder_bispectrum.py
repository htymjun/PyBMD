#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Reproduce the cylinder-wake mode bispectrum of Schmidt (2020, *Nonlinear
Dynamics*), ``refs/figures/cylinder_bispectrum_sumdiff.pdf``, on the full
``refs/bmd/wake_Re500.mat`` (``git submodule update --init``).

    MPLBACKEND=Agg python examples/cylinder_bispectrum.py

See cylinder_bispectrum.md for how the parameters were recovered from the
published figure.
'''
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

CFD = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CFD, '..'))

from pybmd.bmd.standard import Standard
from pybmd.bmd.postproc import top_triads
import pybmd.utils.weights as utils_weights
from pybmd.utils.io import read_data

DATA_PATH = os.path.join(CFD, '..', 'refs', 'bmd', 'wake_Re500.mat')
FIGURE_PATH = os.path.join(CFD, 'figures', 'cylinder',
                           'cylinder_bispectrum_sumdiff.png')

# the six triads circled in the reference's panel (b), with each label's
# offset (points) and alignment chosen so neighbouring labels don't collide
TRIADS = {
    (12, 12): ((6, 2), 'left'), (12, 0): ((8, -2), 'left'),
    (24, 12): ((-4, 8), 'right'), (24, 24): ((-4, 8), 'right'),
    (36, 12): ((4, 8), 'left'), (36, 24): ((4, 8), 'left'),
}

# sampled from the reference's own colorbar: jet, with its dark-blue end
# faded to white
CMAP = LinearSegmentedColormap.from_list('cylinder_jet', (
    (0.000, (1.000, 1.000, 1.000)), (0.074, (0.835, 0.835, 1.000)),
    (0.152, (0.482, 0.482, 1.000)), (0.230, (0.075, 0.075, 1.000)),
    (0.307, (0.000, 0.345, 1.000)), (0.381, (0.004, 0.733, 1.000)),
    (0.459, (0.051, 1.000, 0.953)), (0.537, (0.306, 1.000, 0.694)),
    (0.615, (0.694, 1.000, 0.306)), (0.689, (1.000, 1.000, 0.000)),
    (0.767, (1.000, 0.545, 0.000)), (0.844, (1.000, 0.090, 0.000)),
    (0.922, (0.776, 0.000, 0.000)), (1.000, (0.502, 0.000, 0.000)),
))


def _panel(ax, field, freq, xlim, ylim, vmin, vmax):
    df = freq[1] - freq[0]
    edges = np.append(freq - df / 2, freq[-1] + df / 2)
    ax.pcolormesh(edges, edges, field.T, cmap=CMAP, vmin=vmin, vmax=vmax)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$f_1$')
    ax.set_ylabel(r'$f_2$')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def main(save_dir='cylinder_bispectrum_out'):
    d = read_data(DATA_PATH)
    dt = float(np.ravel(d['dt'])[0])
    u = np.asarray(d['u'], dtype=np.float64)
    nt, n1, n2 = u.shape

    params = dict(
        n_dft=480,                      # the reference's df = 1/57.6
        time_step=dt,
        n_space_dims=2,
        n_variables=1,
        overlap=50,
        regions=[1, 2],                 # sum- and difference-interactions
        solver='MengiOverton',
        save_modes=False,               # ~43k triads: modes would be ~8 GB
        store_modes=False,
        compute_energy_transfer=False,
        savedir=save_dir,
    )
    # bmd.m defaults to a uniform weight and example1.m passes none; B is
    # linear in the weight, so this sets the colour scale
    weights = utils_weights.uniform((n1, n2), n_vars=1, dV=1.0)
    bmd = Standard(params=params, weights=weights).fit(u[..., np.newaxis])

    triads, freq = bmd.triads, bmd.freq
    print('labelled triads:')
    for k, l in TRIADS:
        i = triads.find(k, l)
        print(f'  ({k:3d},{l:3d},{k + l:3d})  '
              f'|lambda_1| = {abs(bmd.L[triads.f1_idx[i], triads.f2_idx[i]]):.4e}')
    print('top 15 triads (k != 0 and l != 0):')
    for row in top_triads(save_dir, n=15):
        print(f"  ({row['k']:3d},{row['l']:3d},{row['kl']:3d})  "
              f"|lambda_1| = {row['value']:.4e}")

    field = np.ma.masked_invalid(np.log(np.abs(bmd.L)))
    vmin, vmax = float(field.min()), float(field.max())
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(7.6, 4.0), gridspec_kw=dict(width_ratios=[4, 3]))

    _panel(axa, field, freq, (0, freq[-1]), (-freq[-1], freq[-1] / 2),
           vmin, vmax)
    cax = axa.inset_axes([0.055, 0.026, 0.083, 0.251])
    fig.colorbar(ScalarMappable(Normalize(vmin, vmax), CMAP), cax=cax,
                 ticks=[0, -10, -20], label=r'$\log(|\lambda_1|)$')

    _panel(axb, field, freq, (0, 0.8), (-0.8, 0.8), vmin, vmax)
    f0 = 12 * (freq[1] - freq[0])  # the shedding frequency
    axb.plot([0, 0.8], [f0, f0 - 0.8], 'k--', lw=0.8)
    for (k, l), (offset, ha) in TRIADS.items():
        i = triads.find(k, l)
        f1, f2 = triads.f1[i], triads.f2[i]
        axb.plot(f1, f2, 'o', ms=7, mfc='none', mec='k', mew=1.0)
        axb.annotate(f'({k},{l})', (f1, f2), textcoords='offset points',
                     xytext=offset, fontsize=8, ha=ha, va='bottom')

    for ax, label in ((axa, '(a)'), (axb, '(b)')):
        ax.text(-0.32, 1.08, label, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='bottom')

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[vmin, vmax] = [{vmin:.3f}, {vmax:.3f}]')
    print(f'wrote {FIGURE_PATH}')
    return bmd


if __name__ == '__main__':
    main()
