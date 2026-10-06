#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Example 5: the cylinder-wake results of Schmidt (2020, *Nonlinear Dynamics*)
on the full ``refs/bmd/wake_Re500.mat`` (``git submodule update --init``):
the mode bispectrum of Fig. 7, ``refs/figures/cylinder_bispectrum_sumdiff.pdf``,
and the bispectral modes of the triad cascade of Figs. 8 and 9.

    MPLBACKEND=Agg python examples/example5_cylinder_paper.py   # ~4.5 min

Writes to ``example5_out/``:

- ``cylinder_bispectrum_sumdiff.png``, Fig. 7 in the paper's layout;
- ``modes_k*_l*.png``, one ``plot_triad_modes`` figure per triad circled in
  Fig. 7b: the u and v components of the bispectral mode phi_{k+l}, the
  cross-frequency field phi_{k o l} and the interaction map
  |phi_{k o l} phi_{k+l}|. Fig. 8 is the phi_{k+l} row of all six; Fig. 9 is
  the whole of ``modes_k12_l12.png``.

Both use the same BMD settings, q = [u, v] as in the paper; the modes come
from a second fit restricted to the cascade (each triad is solved
independently, so the restriction changes nothing -- checked on lambda_1).
See example5_cylinder_paper.md for how the parameters were recovered.
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
from pybmd.bmd.postproc import plot_triad_modes, top_triads
import pybmd.utils.weights as utils_weights
from pybmd.utils.io import read_data

DATA_PATH = os.path.join(CFD, '..', 'refs', 'bmd', 'wake_Re500.mat')

# the six triads circled in the reference's panel (b) -- the cascade of
# Fig. 8 -- with each label's offset (points) and alignment chosen so
# neighbouring labels don't collide
TRIADS = {
    (12, 12): ((6, 2), 'left'), (12, 0): ((8, -2), 'left'),
    (24, 12): ((-4, 8), 'right'), (24, 24): ((-4, 8), 'right'),
    (36, 12): ((4, 8), 'left'), (36, 24): ((4, 8), 'left'),
}
F0_IDX = 12  # the shedding frequency f0, as an index

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


def symmetric_fraction(field):
    '''Share of ``sum |field|^2`` in the part even in y (axis 0), for a grid
    symmetric about y = 0: 1 for a symmetric field, 0 for an antisymmetric one.'''
    even = 0.5 * (field + field[::-1, :])
    return float(np.sum(np.abs(even)**2) / np.sum(np.abs(field)**2))


def streamwise_wavelength(field, dx, n_pad=4096):
    '''Wavelength of the peak of the y-summed streamwise power spectrum of a
    complex mode component, from a zero-padded DFT along x (axis 1).'''
    spec = np.sum(np.abs(np.fft.fft(field, n=n_pad, axis=1))**2, axis=0)
    kx = 2 * np.pi * np.fft.fftfreq(n_pad, dx)
    spec[kx == 0] = 0
    return float(2 * np.pi / abs(kx[np.argmax(spec)]))


def _lambda1(bmd, k, l):
    '''``|lambda_1|`` of the triad ``(k, l, k+l)`` of a fitted decomposition.'''
    i = bmd.find_triad(k, l)
    return abs(bmd.L[bmd.triads.f1_idx[i], bmd.triads.f2_idx[i]])


def plot_bispectrum(bmd, path):
    '''Fig. 7: (a) the sum and difference regions, (b) the low-frequency
    magnification with the cascade circled.'''
    triads, freq = bmd.triads, bmd.freq
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
    f0 = F0_IDX * (freq[1] - freq[0])
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
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[vmin, vmax] = [{vmin:.3f}, {vmax:.3f}]')
    print(f'wrote {path}')


def main(save_dir='example5_out'):
    d = read_data(DATA_PATH)
    dt = float(np.ravel(d['dt'])[0])
    # the .mat file is MATLAB's (nx, ny); PyBMD expects (ny, nx)
    x, y = np.asarray(d['x']).T[0, :], np.asarray(d['y']).T[:, 0]
    data = np.stack([d['u'], d['v']], axis=-1).astype(np.float64)
    data = data.transpose(0, 2, 1, 3)
    ny, nx, nv = data.shape[1:]

    params = dict(
        n_dft=480,                # the reference's df = 1/57.6
        time_step=dt,
        n_space_dims=2,
        n_variables=nv,           # q = [u, v], as in the paper
        overlap=50,
        regions=[1, 2],           # sum- and difference-interactions
        save_modes=False,
        store_modes=False,        # ~43k triads: modes would be ~15 GB
        compute_energy_transfer=False,
        savedir=os.path.join(save_dir, 'bispectrum'),
    )
    # bmd.m defaults to a uniform weight and example1.m passes none; B is
    # linear in the weight, so this sets the colour scale
    weights = utils_weights.uniform((ny, nx), n_vars=nv, dV=1.0)

    # -- Fig. 7: mode bispectrum over the sum and difference regions ---------
    bmd = Standard(params=params, weights=weights).fit(data)
    print('top 15 triads (k != 0 and l != 0):')
    for row in top_triads(bmd, n=15):
        print(f"  ({row['k']:3d},{row['l']:3d},{row['kl']:3d})  "
              f"|lambda_1| = {row['value']:.4e}")
    plot_bispectrum(bmd, os.path.join(save_dir,
                                      'cylinder_bispectrum_sumdiff.png'))

    # -- Figs. 8 and 9: modes, from the same settings on fewer triads --------
    params_modes = dict(
        params,
        regions=[1],              # every triad of Fig. 8 is a sum-interaction
        max_freq_idx=36,          # covers the cascade, 703 triads
        store_modes=True,         # ~250 MB in memory
        savedir=os.path.join(save_dir, 'modes'),
    )
    bmd_modes = Standard(params=params_modes, weights=weights).fit(data)

    dx = x[1] - x[0]
    print('Fig. 8: phi_{k+l}; sym = share of energy even in y '
          '(u: 0 for odd n, 1 for even n; v the opposite)')
    print('   (k, l)   n  |lambda_1|   sym(u)  sym(v)  lambda_x  n*lambda_x')
    max_rel = 0.0
    for k, l in TRIADS:
        lam1 = _lambda1(bmd, k, l)
        max_rel = max(max_rel, abs(_lambda1(bmd_modes, k, l) - lam1) / lam1)
        modes = bmd_modes.get_modes_at_freqs(k, l)
        phi = modes[0]
        n = (k + l) // F0_IDX
        lam = streamwise_wavelength(phi[..., 1], dx)
        print(f'  ({k:2d},{l:2d})  {n:2d}  {lam1:.4e}   '
              f'{symmetric_fraction(phi[..., 0]):.3f}   '
              f'{symmetric_fraction(phi[..., 1]):.3f}   '
              f'{lam:7.3f}   {n * lam:7.3f}')
        plot_triad_modes(
            modes, k, l, x=x, y=y, vars_idx=(0, 1), figsize=(11, 9),
            xlabel='$x$', ylabel='$y$', path=save_dir,
            filename=f'modes_k{k}_l{l}.png')
    print(f'max relative difference of |lambda_1| between the two fits: '
          f'{max_rel:.1e}')
    return bmd, bmd_modes


if __name__ == '__main__':
    main()
