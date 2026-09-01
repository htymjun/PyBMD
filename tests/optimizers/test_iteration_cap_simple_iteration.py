#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''An exhausted iteration budget must return a finite result, not crash or
propagate NaN -- no test in this directory ever sets n_it_max.'''
import os
import sys

import numpy as np
import pytest

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from pybmd.bmd.optimizers import simple_iteration
from conftest import random_matrices


@pytest.mark.parametrize('A', random_matrices())
def test_simple_iteration_stops_cleanly_at_iteration_cap(A):
    '''n_it_max=1 forces the power iteration to break early (it > n_it_max,
    so up to n_it_max + 1 updates run); the truncated answer must still be a
    legitimate Rayleigh quotient of a unit vector.'''
    w, z = simple_iteration(A, n_it_max=1)
    assert np.isfinite(w)
    assert np.linalg.norm(z) == pytest.approx(1.0)
    assert w == pytest.approx(z.conj() @ A @ z)
    # true regardless of convergence: w is always some z^H A z, so it can
    # never exceed the operator norm
    assert abs(w) <= np.linalg.norm(A, 2) + 1e-8
