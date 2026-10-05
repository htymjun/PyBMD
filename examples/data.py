#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Loader for the cylinder-wake dataset used by the examples.'''
import os

import numpy as np

# the subsampled fixture shipped with the test-suite, resolved against this
# file so the examples run from any working directory
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            '..', 'tests', 'data', 'wake_Re500_sub.npz')


def load_cylinder_wake(path=DEFAULT_PATH):
    '''
    Load the cylinder wake at Re=500.

    The file keeps the MATLAB layout, ``(nx, ny)``; it is transposed here to
    PyBMD's ``(ny, nx)``.

    :param str path: ``.npz`` file holding ``x``, ``y``, ``u``, ``v`` and
        ``dt``. Default is the subsampled fixture in ``tests/data``.

    :return: ``x``, ``y`` of shape ``(ny, nx)``, ``u``, ``v`` of shape
        ``(nt, ny, nx)``, and ``dt``.
    :rtype: tuple
    '''
    with np.load(path) as d:
        return (d['x'].T.astype(np.float64), d['y'].T.astype(np.float64),
                d['u'].transpose(0, 2, 1).astype(np.float64),
                d['v'].transpose(0, 2, 1).astype(np.float64),
                float(d['dt']))
