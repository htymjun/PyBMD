#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''The numerical radius is homogeneous: r(cA) = c r(A).'''
import os
import sys

import numpy as np
import pytest

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from pybmd.bmd.optimizers import mengi_overton
from conftest import brute_force_radius


@pytest.mark.parametrize('scale', [1e6, 1e3, 1.0, 1e-3, 1e-6, 1e-9, 1e-12])
def test_scale_equivariance(scale):
    '''
    The numerical radius is homogeneous, r(cA) = c r(A). This is not automatic:
    the unit-circle test in the level-set solvers is an absolute tolerance
    multiplied by ||A||_1, so without rescaling it collapses for small matrices
    and the search silently returns a local maximum. BMD produces exactly such
    matrices -- B carries a 1/n_blocks and the quadrature weights.
    '''
    rng = np.random.default_rng(0)
    A = rng.standard_normal((7, 7)) + 1j * rng.standard_normal((7, 7))
    ref = brute_force_radius(A * scale)
    w, _ = mengi_overton(A * scale)
    assert abs(w) == pytest.approx(ref, rel=1e-6)
