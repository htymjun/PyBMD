#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Curvilinear quadrature weights.'''
import numpy as np
import pytest

import pybmd.utils.weights as utils_weights


def _rectilinear():
    x1 = np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.5])
    x2 = np.array([-1.0, -0.2, 0.0, 0.5, 2.0])
    x, y = np.meshgrid(x1, x2)   # (ny, nx)
    return x1, x2, x, y


def test_curvilinear_reduces_to_trapz_on_rectilinear_grid():
    x1, x2, x, y = _rectilinear()
    w = utils_weights.curvilinear_2d(x, y, n_vars=2)['weights']
    ref = utils_weights.trapz_2d(x1, x2, n_vars=2)['weights']
    np.testing.assert_allclose(w, ref, rtol=1e-14)


def test_curvilinear_is_rotation_invariant():
    x1, x2, x, y = _rectilinear()
    c, s = np.cos(0.7), np.sin(0.7)
    w = utils_weights.curvilinear_2d(c * x - s * y, s * x + c * y)['weights']
    ref = utils_weights.trapz_2d(x1, x2)['weights']
    np.testing.assert_allclose(w, ref, rtol=1e-12)


def test_curvilinear_polar_annulus_area():
    r, th = np.meshgrid(np.linspace(1.0, 2.0, 41),
                        np.linspace(0.0, 0.5 * np.pi, 81))
    w = utils_weights.curvilinear_2d(r * np.cos(th), r * np.sin(th),
                                     n_vars=None)['weights']
    assert w.shape == r.shape
    assert w.sum() == pytest.approx(0.75 * np.pi, rel=1e-3)


def test_curvilinear_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        utils_weights.curvilinear_2d(np.zeros((3, 4)), np.zeros((4, 3)))


def test_trapz_shapes_put_x_last():
    x, y, z = np.arange(5.0), np.arange(4.0), np.arange(3.0)
    assert utils_weights.trapz_2d(x, y)['weights'].shape == (4, 5, 1)
    assert utils_weights.trapz_3d(x, y, z, n_vars=2)['weights'].shape == (3, 4, 5, 2)
