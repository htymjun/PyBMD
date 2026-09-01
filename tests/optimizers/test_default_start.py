#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''default_start's coarse angular scan must be deterministic.'''
import os
import sys

import numpy as np
import pytest

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from pybmd.bmd.optimizers import default_start
from conftest import random_matrices


def test_default_start_is_deterministic():
    A = random_matrices(1)[0]
    z1, z2 = default_start(A), default_start(A)
    assert np.array_equal(z1, z2)
    assert np.linalg.norm(z1) == pytest.approx(1.0)
