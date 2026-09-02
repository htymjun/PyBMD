#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
The gap between ``matlab_compat=True`` and the default solver is purely a
scaling artefact of ``bmd.m``'s absolute unimodularity tolerance, not a
different algorithm: on a matrix already well scaled (``||A||_1 ~ 1``), the
level-set search behaves the same whether or not it is pre-scaled, so the two
must agree to close to machine precision.
'''
import os
import sys

import numpy as np
import pytest

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from conftest import random_matrices
from pybmd.bmd.optimizers import mengi_overton


def test_matlab_compat_agrees_with_default_on_well_scaled_matrices():
    for A in random_matrices():
        A = A / np.linalg.norm(A, 1)   # ||A||_1 == 1
        w_default, _ = mengi_overton(A, tol=1e-8, n_it_max=500)
        w_compat, _ = mengi_overton(A, tol=1e-8, n_it_max=500, matlab_compat=True)
        assert abs(w_compat) == pytest.approx(abs(w_default), rel=1e-6)
