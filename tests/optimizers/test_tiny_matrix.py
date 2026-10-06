#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Regression guard: a badly scaled matrix must not beat the solver.'''
import os
import sys

import numpy as np
import pytest

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from pybmd.bmd.optimizers import mengi_overton
from conftest import brute_force_radius


def test_tiny_matrix_is_not_a_local_maximum():
    '''Regression guard: a badly scaled matrix must not beat the solver.'''
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((7, 7)) + 1j * rng.standard_normal((7, 7))) * 1e-9
    w, _ = mengi_overton(A)
    assert abs(w) == pytest.approx(brute_force_radius(A), rel=1e-5)
