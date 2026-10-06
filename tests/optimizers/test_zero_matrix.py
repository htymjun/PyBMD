#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Every solver must handle the zero matrix without dividing by zero.'''
import os
import sys

import numpy as np
import pytest

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from pybmd.bmd.optimizers import mengi_overton


def test_zero_matrix():
    A = np.zeros((4, 4), dtype=complex)
    w, z = mengi_overton(A)
    assert abs(w) == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(z) == pytest.approx(1.0)
