#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''The smallest possible input, a 1x1 matrix, must work for both solvers.'''
import os
import sys

import numpy as np
import pytest

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from pybmd.bmd.optimizers import solve, SOLVERS


@pytest.mark.parametrize('solver', SOLVERS)
def test_one_by_one_matrix(solver):
    '''
    random_matrices() only ever draws n in [3, 12), so this is the only
    coverage of the n=1 edge: a single unit-modulus z is the only feasible
    point, and the numerical radius is exactly |a|.
    '''
    A = np.array([[3.0 + 4.0j]])
    w, z = solve(A, solver=solver)
    assert abs(w) == pytest.approx(5.0)
    assert np.linalg.norm(z) == pytest.approx(1.0)
    assert w == pytest.approx(z.conj() @ A @ z)
