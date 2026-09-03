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

    :param str path: ``.npz`` file holding ``x``, ``y``, ``u``, ``v`` and
        ``dt``. Default is the subsampled fixture in ``tests/data``.

    :return: ``x``, ``y``, ``u``, ``v``, ``dt``.
    :rtype: tuple
    '''
    with np.load(path) as d:
        return (d['x'].astype(np.float64), d['y'].astype(np.float64),
                d['u'].astype(np.float64), d['v'].astype(np.float64),
                float(d['dt']))
