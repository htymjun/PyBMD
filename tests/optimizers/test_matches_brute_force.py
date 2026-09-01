#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''The globally convergent solver must match a dense angular search.'''
import os
import sys

import pytest

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from pybmd.bmd.optimizers import mengi_overton
from conftest import brute_force_radius, random_matrices


@pytest.mark.parametrize('A', random_matrices())
def test_mengi_overton_matches_brute_force(A):
    '''The globally convergent solver must match a dense angular search.'''
    w, _ = mengi_overton(A)
    assert abs(w) == pytest.approx(brute_force_radius(A), rel=1e-6)
