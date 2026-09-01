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

from pybmd.bmd.optimizers import solve, SOLVERS
from conftest import random_matrices


def test_solvers_are_deterministic_without_rng():
    '''Repeated calls must be bit-identical, so MPI runs reproduce serial ones.'''
    A = random_matrices(1)[0]
    for name in SOLVERS:
        w1, z1 = solve(A, solver=name)
        w2, z2 = solve(A, solver=name)
        assert w1 == w2, name
        assert np.array_equal(z1, z2), name
