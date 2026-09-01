#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Analytic case: for a normal matrix the numerical radius is the spectral
radius, since the numerical range is the convex hull of the spectrum.'''
import os
import sys

import numpy as np
import pytest

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from pybmd.bmd.optimizers import mengi_overton


def test_normal_radius_equals_spectral_radius():
    '''For a normal matrix the numerical range is the convex hull of the
    spectrum, so the numerical radius equals the spectral radius.'''
    rng = np.random.default_rng(4)
    q, _ = np.linalg.qr(rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6)))
    d = rng.standard_normal(6) + 1j * rng.standard_normal(6)
    A = q @ np.diag(d) @ q.conj().T
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    w, _ = mengi_overton(A)
    assert abs(w) == pytest.approx(rho, rel=1e-8)
