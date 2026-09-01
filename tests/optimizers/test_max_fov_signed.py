#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''max_fov returns the signed largest eigenvalue, not the largest in modulus.'''
import os
import sys

import numpy as np
import pytest

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from pybmd.bmd.optimizers import max_fov


def test_max_fov_is_signed_not_modulus():
    '''
    The signed largest eigenvalue and the largest modulus differ whenever the
    Hermitian part is indefinite. This is the distinction that makes the
    level-set search in mengi_overton correct.
    '''
    A = np.array([[1.0, 0.0], [0.0, -3.0]], dtype=complex)
    assert max_fov(A, 0.0)[0] == pytest.approx(1.0)      # signed lambda_max
    H = 0.5 * (A + A.conj().T)
    assert np.max(np.abs(np.linalg.eigvalsh(H))) == pytest.approx(3.0)
