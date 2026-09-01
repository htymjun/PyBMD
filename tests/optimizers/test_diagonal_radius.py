#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Analytic case: for a diagonal matrix the numerical radius is the largest
diagonal entry in modulus.'''
import os
import sys

import numpy as np
import pytest

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from pybmd.bmd.optimizers import mengi_overton


def test_diagonal_radius_is_max_abs_entry():
    A = np.diag([1.0 + 0j, -4.0, 2.5])
    w, _ = mengi_overton(A)
    assert abs(w) == pytest.approx(4.0, rel=1e-10)
