#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Complex data must be rejected, not silently cast to its real part.'''
import numpy as np
import pytest

from pybmd.bmd.standard import Standard
from pybmd.utils.io import get_data_array


def test_complex_data_is_rejected(tmp_path):
    data = np.ones((64, 4, 1)) * 1j
    with pytest.raises(TypeError, match='real-valued'):
        get_data_array(data, xdim=1, nv=1)
    params = dict(n_dft=16, time_step=1.0, n_space_dims=1, n_variables=1,
                  savedir=str(tmp_path))
    with pytest.raises(TypeError, match='real-valued'):
        Standard(params=params).fit(data)
