#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''simple_iteration's stopping test on |w - w_old| must act relative to the
scale of A: the matrices BMD produces have ||B||_1 ~ 1e-6 and below, where an
absolute 1e-8 fires after one update. The same start vector on A and on a tiny
multiple of A must therefore reach the same relative accuracy.'''
import os
import sys

import numpy as np
import pytest

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from pybmd.bmd.optimizers import simple_iteration


@pytest.mark.parametrize('scale', [1e-6, 1e-9, 1e-12])
def test_simple_iteration_relative_accuracy_is_scale_invariant(scale):
    rng = np.random.default_rng(3)
    A = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
    # a deliberately poor start, so the iteration has work to do
    z0 = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    w_ref, _ = simple_iteration(A, z_0=z0, tol=1e-10, n_it_max=1000)
    w, _ = simple_iteration(A * scale, z_0=z0, tol=1e-10, n_it_max=1000)
    assert abs(w) / scale == pytest.approx(abs(w_ref), rel=1e-8)
