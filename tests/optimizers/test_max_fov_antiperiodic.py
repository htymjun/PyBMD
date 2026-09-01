#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''max_fov(theta + pi) == -min_fov(theta), the antiperiodicity of the rotated
Hermitian part.'''
import os
import sys

import numpy as np
import pytest

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from pybmd.bmd.optimizers import max_fov
from conftest import random_matrices


def test_max_fov_antiperiodic():
    '''H(theta + pi) == -H(theta), so lambda_max(theta+pi) == -lambda_min(theta).'''
    A = random_matrices(1)[0]
    theta = np.linspace(0, np.pi, 17)
    H_pos = max_fov(A, theta)
    H_neg = max_fov(A, theta + np.pi)
    for t, a, b in zip(theta, H_pos, H_neg):
        A_rot = A * np.exp(1j * t)
        H = 0.5 * (A_rot + A_rot.conj().T)
        assert a == pytest.approx(np.linalg.eigvalsh(H)[-1])
        assert b == pytest.approx(-np.linalg.eigvalsh(H)[0])
