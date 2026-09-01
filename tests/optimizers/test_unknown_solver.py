#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''solve() must reject a solver name outside SOLVERS.'''
import os
import sys

import pytest

CF = os.path.realpath(__file__)
CFD = os.path.dirname(CF)
sys.path.append(os.path.join(CFD, '../../'))

from pybmd.bmd.optimizers import solve, SOLVERS


def test_unknown_solver_raises():
    '''An unrecognised solver name must raise, not silently fall through.'''
    with pytest.raises(ValueError, match='Unknown solver') as excinfo:
        solve([[1.0]], solver='nope')
    # the message should still name the two solvers that are actually valid
    for name in SOLVERS:
        assert name in str(excinfo.value)
