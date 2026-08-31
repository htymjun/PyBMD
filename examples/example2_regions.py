#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Example 2: manual spectral estimation parameters, several regions of the mode
bispectrum, and data with more than one variable (both velocity components of
the cylinder wake).

Mirrors ``example2.m`` of the reference MATLAB implementation. Where the
reference selects a triad interactively, here the triad is looked up by its
frequency doublet.
'''
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

from pybmd.bmd.standard import Standard
from pybmd.bmd.postproc import plot_mode_bispectrum, plot_triad_modes
import pybmd.utils.weights as utils_weights
from examples.data import load_cylinder_wake


def main(save_dir='example2_out', k=5, l=-2):
    x, y, u, v, dt = load_cylinder_wake("../tests/data/wake_Re500_sub.npz")
    nt, n1, n2 = u.shape

    ## two variables, stacked on the last axis
    data = np.stack([u, v], axis=-1)

    params = dict(
        n_dft=64,
        time_step=dt,
        n_space_dims=2,
        n_variables=2,
        n_overlap=32,            # absolute overlap, as in the reference
        regions=[1, 2, 3, 4, 5, 6, 7, 8],
        max_freq_idx=12,
        solver='MengiOverton',
        savedir=save_dir,
    )
    weights = utils_weights.trapz_2d(x[:, 0], y[0, :], n_vars=2)
    bmd = Standard(params=params, weights=weights).fit(data)
    print(f'computed {bmd.n_triads} triads over all eight regions')

    ## look the triad up by its index doublet; the reference examples had no
    ## working way to do this and fell back to a hard-coded triad number
    i = bmd.triads.find(k, l)
    print(f'triad (k,l,k+l) = ({k},{l},{k + l}) is index {i}, '
          f'|lambda_1| = {abs(bmd.L[bmd.triads.f1_idx[i], bmd.triads.f2_idx[i]]):.4e}')

    plot_mode_bispectrum(bmd.L, bmd.freq,
                         mark=[(bmd.triads.f1[i], bmd.triads.f2[i])],
                         path=save_dir, filename='bispectrum_all_regions.png')
    plot_triad_modes(bmd.get_modes_at_triad(i), k, l, x1=x[:, 0], x2=y[0, :],
                     vars_idx=(0, 1), path=save_dir,
                     filename=f'modes_k{k}_l{l}.png')
    print(f'figures written to {os.path.abspath(save_dir)}')
    return bmd


if __name__ == '__main__':
    main()
