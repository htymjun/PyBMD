#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Readers: both .mat layouts agree, and every format reaches an array.'''
import numpy as np
import pytest

from pybmd.utils.io import _read_mat, get_data_array, read_data


def test_read_mat_v73_matches_v5(tmp_path):
    '''MATLAB writes v7.3 (HDF5) arrays column-major, so h5py reports the
    dimensions reversed, and complex arrays as a (real, imag) compound; the
    reader must return what scipy returns for the same variables in a v5 file.'''
    h5py = pytest.importorskip('h5py')
    scipy_io = pytest.importorskip('scipy.io')
    rng = np.random.default_rng(0)
    u = rng.standard_normal((7, 3, 2))
    z = u + 1j * rng.standard_normal(u.shape)
    scipy_io.savemat(tmp_path / 'v5.mat', {'u': u, 'z': z})

    with h5py.File(tmp_path / 'v73.mat', 'w') as f:
        f['u'] = u.T
        zc = np.empty(z.T.shape, dtype=[('real', '<f8'), ('imag', '<f8')])
        zc['real'], zc['imag'] = z.T.real, z.T.imag
        f['z'] = zc
        f.create_group('#refs#')

    v5 = _read_mat(str(tmp_path / 'v5.mat'))
    v73 = _read_mat(str(tmp_path / 'v73.mat'))
    assert set(v73) == {'u', 'z'}
    for key in ('u', 'z'):
        assert v73[key].shape == v5[key].shape == u.shape
        np.testing.assert_array_equal(v73[key], v5[key])
    assert np.iscomplexobj(v73['z'])


def test_single_array_npz_is_accepted(tmp_path):
    data = np.random.default_rng(0).standard_normal((10, 4, 3))
    np.savez(tmp_path / 'one.npz', u=data)
    out = get_data_array(str(tmp_path / 'one.npz'), xdim=2, nv=1)
    np.testing.assert_array_equal(out[..., 0], data)

    np.savez(tmp_path / 'two.npz', u=data, v=data)
    with pytest.raises(ValueError, match='2 array variables'):
        get_data_array(str(tmp_path / 'two.npz'), xdim=2, nv=1)


def test_netcdf_is_read_into_an_array(tmp_path):
    xr = pytest.importorskip('xarray')
    pytest.importorskip('netCDF4')
    data = np.random.default_rng(0).standard_normal((10, 4, 3, 2))
    xr.DataArray(data, dims=('t', 'x', 'y', 'v'), name='q').to_netcdf(
        tmp_path / 'q.nc')
    assert isinstance(read_data(str(tmp_path / 'q.nc')), xr.Dataset)
    out = get_data_array(str(tmp_path / 'q.nc'), xdim=2, nv=2)
    np.testing.assert_array_equal(out, data)
