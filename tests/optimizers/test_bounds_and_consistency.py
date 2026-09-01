#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''rho(A) <= r(A) <= ||A||_2, ||z|| == 1, and w == z^H A z.'''
import os
import sys

import numpy as np
import pytest

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from pybmd.bmd.optimizers import mengi_overton
from conftest import random_matrices


@pytest.mark.parametrize('A', random_matrices())
def test_bounds_and_consistency(A):
    '''rho(A) <= r(A) <= ||A||_2, ||z|| == 1, and w == z^H A z.'''
    w, z = mengi_overton(A)
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    assert np.linalg.norm(z) == pytest.approx(1.0)
    assert z.conj() @ A @ z == pytest.approx(w)
    assert abs(w) >= rho - 1e-8
    assert abs(w) <= np.linalg.norm(A, 2) + 1e-8
    assert abs(w) >= np.max(np.abs(np.diag(A))) - 1e-8
