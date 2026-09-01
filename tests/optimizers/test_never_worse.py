#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''The globally convergent solver must dominate the power iteration.'''
import os
import sys

import pytest

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from pybmd.bmd.optimizers import mengi_overton, simple_iteration
from conftest import random_matrices


@pytest.mark.parametrize('A', random_matrices())
def test_mengi_overton_is_never_worse(A):
    '''
    The globally convergent solver must dominate the power iteration. The
    level-set search inflates its trial level by ``tol`` at each step, so its
    answer is accurate to a relative ``tol`` rather than to machine precision;
    compare with that slack.
    '''
    tol = 1e-6
    w_mo, _ = mengi_overton(A, tol=tol)
    w_si, _ = simple_iteration(A, tol=tol)
    assert abs(w_mo) >= abs(w_si) * (1 - 10 * tol)
