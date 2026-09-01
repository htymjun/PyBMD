#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''The returned w is the signed/complex Rayleigh quotient z^H A z, not |w|.'''
import os
import sys

import numpy as np
import pytest

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from pybmd.bmd.optimizers import solve, SOLVERS
from conftest import random_matrices


def test_diagonal_case_keeps_its_sign():
    '''
    _dominant_eigvec picks the largest-*modulus* eigenvector, so the
    numerically largest diagonal entry of diag(1, -4, 2.5) -- which is
    negative -- must come back as -4, not +4. base.py stores this complex
    value directly into L, so the sign/phase is load-bearing, not incidental;
    every other test in this directory only ever checks abs(w).
    '''
    A = np.diag([1.0 + 0j, -4.0, 2.5])
    w, _ = solve(A, solver='MengiOverton')
    assert w == pytest.approx(-4.0)


@pytest.mark.parametrize('A', random_matrices())
@pytest.mark.parametrize('solver', SOLVERS)
def test_w_is_exactly_the_rayleigh_quotient(A, solver):
    '''Whatever z the solver settles on, w must equal z^H A z exactly -- this
    is the definition, not an approximation the solver is free to round.'''
    w, z = solve(A, solver=solver, tol=1e-10)
    assert w == pytest.approx(z.conj() @ A @ z, abs=1e-9)
