#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Example 1: BMD of a single-variable, two-dimensional flow field -- the
streamwise velocity of the wake behind a cylinder at Re=500.

Mirrors ``example1.m`` of the reference MATLAB implementation.
'''
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

from pybmd.bmd.standard import Standard
from pybmd.bmd.postproc import plot_mode_bispectrum, plot_triad_modes
import pybmd.utils.weights as utils_weights
from examples.data import load_cylinder_wake


def main(save_dir='example1_out'):
    x, y, u, v, dt = load_cylinder_wake()
    nt, n1, n2 = u.shape
    print(f'cylinder wake: nt={nt}, grid={n1}x{n2}, dt={dt}')

    ## single variable: the streamwise velocity
    data = u[..., np.newaxis]

    params = dict(
        n_dft=64,
        time_step=dt,
        n_space_dims=2,
        n_variables=1,
        overlap=50,
        regions=[1, 2],          # sum- and difference-interactions
        max_freq_idx=12,         # restrict to |k|, |l| <= 12
        solver='MengiOverton',
        constituent_modes=True,  # also plot phi_k, phi_l alongside phi_{k+l}, phi_{k.l}
        savedir=save_dir,
    )
    weights = utils_weights.trapz_2d(x[:, 0], y[0, :], n_vars=1)
    bmd = Standard(params=params, weights=weights).fit(data)

    ## the strongest triad, found through the triad map rather than by index
    triads = bmd.triads
    vals = np.abs(bmd.L[triads.f1_idx, triads.f2_idx])
    i_peak = int(np.argmax(vals[triads.k != 0]))
    i_peak = int(np.flatnonzero(triads.k != 0)[i_peak])
    k, l = int(triads.k[i_peak]), int(triads.l[i_peak])
    print(f'strongest triad: (k,l,k+l) = ({k},{l},{k + l}), '
          f'|lambda_1| = {vals[i_peak]:.4e}')

    plot_mode_bispectrum(
        bmd.L, bmd.freq, mark=[(triads.f1[i_peak], triads.f2[i_peak])],
        path=save_dir, filename='bispectrum.png')
    plot_triad_modes(
        bmd.get_modes_at_triad(i_peak), k, l, x1=x[:, 0], x2=y[0, :],
        path=save_dir, filename='modes.png')
    print(f'figures written to {os.path.abspath(save_dir)}')
    return bmd


if __name__ == '__main__':
    main()
