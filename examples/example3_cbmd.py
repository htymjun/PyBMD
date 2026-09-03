#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Example 3: cross-bispectral mode decomposition.

For the purpose of demonstration the CBMD is computed in the special case
where all three variables are the same, in which it reduces exactly to the BMD.
When cross-correlating genuinely different variables, compute more regions of
the mode bispectrum: the symmetries that make regions 1 and 2 sufficient are
lost.

Mirrors ``example3.m`` of the reference MATLAB implementation.
'''
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

from pybmd.bmd.standard import Standard
from pybmd.bmd.cross import Cross
from pybmd.bmd.postproc import plot_mode_bispectrum
import pybmd.utils.weights as utils_weights
from examples.data import load_cylinder_wake


def main(save_dir='example3_out'):
    x, y, u, v, dt = load_cylinder_wake()
    nt, n1, n2 = u.shape

    ## q, r and s all set to the streamwise velocity
    data = np.stack([u, u, u], axis=-1)

    common = dict(n_dft=64, time_step=dt, n_space_dims=2, n_overlap=32,
                  regions=[1, 2], max_freq_idx=12, solver='MengiOverton')

    cbmd = Cross(
        params=dict(common, n_variables=3, state_idx=[0], qr_idx=[[1, 2]],
                    savedir=os.path.join(save_dir, 'cbmd')),
        weights=utils_weights.trapz_2d(x[:, 0], y[0, :], n_vars=None),
    ).fit(data)

    ## the same computation through the standard BMD, for comparison
    bmd = Standard(
        params=dict(common, n_variables=1,
                    savedir=os.path.join(save_dir, 'bmd')),
        weights=utils_weights.trapz_2d(x[:, 0], y[0, :], n_vars=1),
    ).fit(u[..., np.newaxis])

    diff = np.nanmax(np.abs(cbmd.L - bmd.L))
    print(f'max |L_cbmd - L_bmd| = {diff:.3e}  '
          f'(CBMD reduces to BMD when q = r = s)')

    plot_mode_bispectrum(cbmd.L, cbmd.freq, title='Mode cross-bispectrum',
                         path=save_dir, filename='cross_bispectrum.png')
    print(f'figures written to {os.path.abspath(save_dir)}')
    return cbmd, bmd


if __name__ == '__main__':
    main()
