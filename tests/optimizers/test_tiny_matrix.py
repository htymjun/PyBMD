#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Regression guard: a badly scaled matrix must not beat the solver.'''
import os
import sys

import numpy as np

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from pybmd.bmd.optimizers import mengi_overton, simple_iteration


def test_tiny_matrix_is_not_a_local_maximum():
    '''Regression guard: a badly scaled matrix must not beat the solver.'''
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((7, 7)) + 1j * rng.standard_normal((7, 7))) * 1e-9
    w_mo, _ = mengi_overton(A)
    w_si, _ = simple_iteration(A)
    assert abs(w_mo) >= abs(w_si) * (1 - 1e-5)
