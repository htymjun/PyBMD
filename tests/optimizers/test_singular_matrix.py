#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''The pencil S = [[A,0],[0,I]] is singular when A is; infinite generalized
eigenvalues must be filtered rather than crashing the level-set search.'''
import os
import sys

import numpy as np
import pytest

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from pybmd.bmd.optimizers import mengi_overton
from conftest import brute_force_radius


def test_singular_matrix():
    '''The pencil S = [[A,0],[0,I]] is singular when A is; infinite generalized
    eigenvalues must be filtered rather than crashing the level-set search.'''
    A = np.outer([1.0, 2.0, 3.0], [1.0, 0.0, 1.0]).astype(complex)
    w, z = mengi_overton(A)
    assert abs(w) == pytest.approx(brute_force_radius(A), rel=1e-6)
    assert np.linalg.norm(z) == pytest.approx(1.0)
