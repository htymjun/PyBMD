#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Analytic case: the numerical radius of a 2x2 nilpotent Jordan block is 1.'''
import os
import sys

import numpy as np
import pytest

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from pybmd.bmd.optimizers import solve, SOLVERS


def test_jordan_block_radius_is_one():
    '''r([[0,2],[0,0]]) = 2 cos(pi / (2 + 1)) = 2 cos(pi / 3) = 1 exactly.'''
    A = np.array([[0, 2], [0, 0]], dtype=complex)
    for name in SOLVERS:
        w, z = solve(A, solver=name)
        assert abs(w) == pytest.approx(1.0, abs=1e-8), name
        assert np.linalg.norm(z) == pytest.approx(1.0), name
