#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Analytic case: for a Hermitian matrix the numerical radius is the spectral
radius.'''
import os
import sys

import numpy as np
import pytest

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from pybmd.bmd.optimizers import mengi_overton


def test_hermitian_radius_equals_spectral_radius():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
    A = A + A.conj().T
    rho = np.max(np.abs(np.linalg.eigvalsh(A)))
    w, _ = mengi_overton(A)
    assert abs(w) == pytest.approx(rho, rel=1e-8)
