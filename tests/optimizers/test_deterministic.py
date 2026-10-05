#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Repeated solver calls must be bit-identical, so MPI runs reproduce serial
ones.'''
import os
import sys

import numpy as np

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from pybmd.bmd.optimizers import mengi_overton
from conftest import random_matrices


def test_solver_is_deterministic_without_rng():
    '''Repeated calls must be bit-identical, so MPI runs reproduce serial ones.'''
    A = random_matrices(1)[0]
    w1, z1 = mengi_overton(A)
    w2, z2 = mengi_overton(A)
    assert w1 == w2
    assert np.array_equal(z1, z2)
