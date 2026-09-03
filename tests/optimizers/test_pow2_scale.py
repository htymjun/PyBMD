#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''_pow2_scale must land ||A||_1 in (0.5, 1] and rescale exactly, never
approximately -- power-of-two scaling is the whole point of deviation 2.'''
import math
import os
import sys

import numpy as np
import pytest

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from pybmd.bmd.optimizers import _pow2_scale


@pytest.mark.parametrize('scale_factor', [1e6, 1e3, 1.0, 1e-3, 1e-6, 1e-9, 1e-12])
def test_pow2_scale_postcondition_and_exactness(scale_factor):
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((7, 7)) + 1j * rng.standard_normal((7, 7))) * scale_factor
    scaled = _pow2_scale(A)

    assert 0.5 < np.linalg.norm(scaled, 1) <= 1.0

    # scaling by a power of two is exact in binary floating point: dividing
    # then multiplying back by the same power of two must recover A bitwise,
    # not merely to within a tolerance
    k = math.ceil(math.log2(np.linalg.norm(A, 1)))
    expected_scale = np.ldexp(1.0, k)
    assert np.array_equal(scaled * expected_scale, A)


def test_pow2_scale_passes_through_zero_matrix():
    A = np.zeros((3, 3), dtype=complex)
    assert np.array_equal(_pow2_scale(A), A)


def test_pow2_scale_passes_through_non_finite_matrix():
    '''A matrix whose 1-norm is not finite must be returned unchanged rather
    than dividing by an infinite or NaN scale.'''
    A = np.eye(3, dtype=complex)
    A[0, 0] = np.inf
    assert np.array_equal(_pow2_scale(A), A)


@pytest.mark.parametrize('norm', [1.5e-309, 1e-315])
def test_pow2_scale_handles_a_subnormal_norm(norm):
    '''When ||A||_1 is subnormal so is 2**e, and dividing by it overflows to
    inf; the scaling must stay finite (and mengi_overton must not crash).'''
    from pybmd.bmd.optimizers import mengi_overton
    rng = np.random.default_rng(1)
    A = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
    A = A / np.linalg.norm(A, 1) * norm
    scaled = _pow2_scale(A)
    assert np.all(np.isfinite(scaled))
    assert 0.5 < np.linalg.norm(scaled, 1) <= 1.0
    w, z = mengi_overton(A)
    assert np.isfinite(w) and np.linalg.norm(z) == pytest.approx(1.0)
