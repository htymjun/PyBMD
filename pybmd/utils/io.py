'''Module implementing I/O utils used across the library.'''
import argparse
import os
from os.path import splitext

import numpy as np
import yaml


REQUIRED_KEYS = ['time_step', 'n_space_dims', 'n_variables', 'n_dft']


def read_data(data_file, format=None, comm=None):
    '''
    Read a data file in one of the supported formats.

    :param str data_file: path to the data file.
    :param str format: format to read. Default is inferred from the extension.
    :param MPI.Comm comm: parallel communicator. Default is None.

    :return: the data: an array for ``.npy``, a dict of arrays for ``.npz``
        and ``.mat``, an ``xarray.Dataset`` for ``.nc``.
    :rtype: numpy.ndarray or dict or xarray.Dataset
    '''
    if not format:
        _, format = splitext(data_file)
    format = format.lower().lstrip('.')
    if comm is not None and comm.rank == 0:
        print(f'reading data with format: {format}')
    if format == 'npy':
        return np.load(data_file)
    if format == 'npz':
        with np.load(data_file) as d:
            return {k: d[k] for k in d.files}
    if format == 'mat':
        return _read_mat(data_file)
    if format == 'nc':
        import xarray as xr
        with xr.open_dataset(data_file) as ds:
            return ds.load()
    raise ValueError(f'{format} format not supported')


def _from_matlab_hdf5(arr):
    '''
    Undo the two transformations MATLAB applies when writing a v7.3 array.

    MATLAB stores arrays column-major, so h5py reports the dimensions
    reversed: a MATLAB ``[nt, nx, nv]`` variable reads back as ``(nv, nx,
    nt)`` and must be transposed to match what ``scipy.io.loadmat`` returns
    for the same variable in a v5 file. Complex arrays arrive as a compound
    ``(real, imag)`` dtype.
    '''
    arr = np.asarray(arr)
    if arr.dtype.names and {'real', 'imag'} <= set(arr.dtype.names):
        arr = arr['real'] + 1j * arr['imag']
    return arr.T


def _read_mat(data_file):
    '''Read a .mat file, handling both the v7.3 (HDF5) and v5 layouts.'''
    try:
        import h5py
        with h5py.File(data_file, 'r') as f:
            # '#refs#' holds cell/struct storage, not user variables
            return {k: _from_matlab_hdf5(v[()]) for k, v in f.items()
                    if isinstance(v, h5py.Dataset) and not k.startswith('#')}
    except (ImportError, OSError):
        # OSError: not an HDF5 file, i.e. a pre-v7.3 .mat
        import scipy.io
        d = scipy.io.loadmat(data_file)
        return {k: v for k, v in d.items() if not k.startswith('__')}


def _as_array(obj):
    '''
    Coerce what :func:`read_data` returns into a single array: an array is
    passed through, a dict (``.npz``/``.mat``) or ``xarray.Dataset`` must hold
    exactly one array variable, an ``xarray.DataArray`` gives its values.
    '''
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        arrays = {k: v for k, v in obj.items()
                  if isinstance(v, np.ndarray) and v.ndim >= 2}
        if len(arrays) == 1:
            return next(iter(arrays.values()))
        raise ValueError(
            f'the file holds {len(arrays)} array variables '
            f'{sorted(arrays)}; load it and pass the data array directly.')
    if hasattr(obj, 'data_vars'):   # xarray.Dataset
        names = list(obj.data_vars)
        if len(names) != 1:
            raise ValueError(
                f'the dataset holds {len(names)} variables {names}; select '
                f'one and pass its array directly.')
        obj = obj[names[0]]
    if hasattr(obj, 'dims') and hasattr(obj, 'values'):   # xarray.DataArray
        return np.asarray(obj.values)
    return np.asarray(obj)


def read_config(parsed_file=None):
    '''
    Parse a YAML config file with ``required:`` and ``optional:`` sections,
    each a list of single-key mappings.

    :param str parsed_file: file to parse. Default is None, in which case the
        path is read from the ``--config_file`` command-line argument.

    :return: the parameters read from the config file.
    :rtype: dict
    '''
    parser = argparse.ArgumentParser(description='Config file.')
    parser.add_argument('--config_file', required=True,
                        help='Configuration file.')
    if parsed_file:
        args = parser.parse_args(['--config_file', parsed_file])
    else:
        args = parser.parse_args()

    with open(args.config_file) as file:
        l = yaml.load(file, Loader=yaml.FullLoader)

    params = _parse_yaml(l['required'])
    found, missing = _check_keys(params, REQUIRED_KEYS)
    if not found:
        raise ValueError(f'config file is missing required keys: {missing}')
    if 'optional' in l:
        params = {**params, **_parse_yaml(l['optional'])}
    return params


def get_data_array(data_list, xdim, nv, dtype=np.float64):
    '''
    Assemble the input into a single array of shape ``(nt, *xshape, nv)``.

    Accepts an array, a path, or a list of either; a list of arrays or paths is
    concatenated along time. A trailing singleton variable axis is appended
    when the data has none and ``nv == 1``.

    :param data_list: the data, or path(s) to it.
    :param int xdim: number of spatial dimensions.
    :param int nv: number of variables.
    :param type dtype: floating-point type to cast to. Default is float64.

    :return: the data.
    :rtype: numpy.ndarray
    '''
    if isinstance(data_list, np.ndarray):
        data = data_list
    elif isinstance(data_list, (str, os.PathLike)):
        data = _as_array(read_data(os.fspath(data_list)))
    elif isinstance(data_list, (list, tuple)):
        parts = [_as_array(read_data(os.fspath(d))
                           if isinstance(d, (str, os.PathLike)) else d)
                 for d in data_list]
        data = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
    else:
        data = _as_array(data_list)

    if not isinstance(data, np.ndarray):
        raise TypeError(
            f'could not resolve data_list into an array; got {type(data)}.')

    if nv == 1 and data.ndim == xdim + 1:
        data = data[..., np.newaxis]
    if data.ndim != xdim + 2:
        raise ValueError(
            f'data has {data.ndim} dimensions, expected {xdim + 2} for '
            f'n_space_dims={xdim} and n_variables={nv}: '
            f'(nt, {", ".join(["nx"] * xdim)}, nv). Got shape {data.shape}.')
    if data.shape[-1] != nv:
        raise ValueError(
            f'data has {data.shape[-1]} variables in its last axis but '
            f'n_variables is {nv}.')
    return np.ascontiguousarray(data, dtype=dtype)


def _parse_yaml(l):
    params = dict()
    for d in l:
        k = list(d.keys())[0]
        params[k] = d[k]
    return params


def _check_keys(l, keys):
    if isinstance(keys, str):
        keys = [keys]
    keys_not_found = [k for k in keys if k not in l.keys()]
    return not keys_not_found, keys_not_found
