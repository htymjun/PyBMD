#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
On a badly-scaled matrix (the regime every real BMD ``B`` lives in),
``matlab_compat=True`` must reproduce ``refs/bmd/bmd.m``'s own under-estimate:
it can never exceed the default solver's answer, and on a matrix this small
it must actually be materially lower -- pinning both the one-sided direction
of deviation 2 and the bug it is deliberately reproducing.
'''
import os
import sys

import numpy as np

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from conftest import random_matrices
from pybmd.bmd.optimizers import mengi_overton


def test_matlab_compat_underestimates_on_tiny_matrix():
    # seed 0 gives a robust ~22% gap at this scale (not every random matrix
    # does -- the unimodularity test can still get lucky and find a crossing
    # near theta=0 -- so the seed is chosen deliberately, not swept)
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))) * 1e-6

    w_default, _ = mengi_overton(A, tol=1e-6, n_it_max=500)
    w_compat, _ = mengi_overton(A, tol=1e-6, n_it_max=500, matlab_compat=True)

    # never exceeds the corrected solver, matching the one-sided invariant
    # measured live against the real bmd.m (docs/octave_cross_validation.md)
    assert abs(w_compat) <= abs(w_default) * (1 + 1e-8)
    # and, at this scale, is materially lower -- not merely numerically equal
    assert abs(w_compat) < abs(w_default) * 0.9


def test_matlab_compat_never_exceeds_default_on_random_matrices():
    '''Same one-sided invariant, swept over the shared random-matrix bank,
    rescaled into the regime every real BMD B occupies.'''
    for A in random_matrices():
        A = A * 1e-5
        w_default, _ = mengi_overton(A, tol=1e-6, n_it_max=500)
        w_compat, _ = mengi_overton(A, tol=1e-6, n_it_max=500, matlab_compat=True)
        assert abs(w_compat) <= abs(w_default) * (1 + 1e-8)
