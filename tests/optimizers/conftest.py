#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Shared helpers for the split-out numerical-radius optimizer tests.'''
import os
import sys

import numpy as np

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from pybmd.bmd.optimizers import max_fov


def brute_force_radius(A, n_theta=40001):
    '''Numerical radius by dense search over the rotation angle.'''
    theta = np.linspace(0, 2 * np.pi, n_theta)
    return float(np.max(max_fov(A, theta)))


def random_matrices(n_cases=10, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_cases):
        n = int(rng.integers(3, 12))
        out.append(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    return out
