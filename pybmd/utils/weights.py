'''
Module implementing spatial inner-product weights, usually quadrature weights.

Every constructor returns the dict ``{'weights_name': str, 'weights': ndarray}``.

.. note::

    The expected shape differs between the two decompositions:

    - :class:`pybmd.bmd.standard.Standard` expects ``(*xshape, n_variables)``
      -- the weight covers the variables too;
    - :class:`pybmd.bmd.cross.Cross` expects ``xshape`` -- a purely spatial
      weight, which is tiled internally over the state variables.

    The weight is flattened in the same C order as the data. Supplying a
    weight built in Fortran order attaches each weight to the wrong grid point,
    which silently corrupts the modes without raising, so the classes reject a
    bare flat vector and require the full shape.
'''
import numpy as np


def uniform(xshape, n_vars=1, dV=1.0):
    '''
    Uniform weights, optionally scaled by a constant cell volume.

    :param tuple xshape: spatial shape of the data.
    :param int n_vars: number of variables. Default is 1.
    :param float dV: cell volume. Default is 1.

    :return: the weights.
    :rtype: dict
    '''
    shape = tuple(xshape) + ((n_vars,) if n_vars else ())
    return {'weights_name': 'uniform', 'weights': dV * np.ones(shape)}


def _cell_widths(coord):
    '''Trapezoidal cell widths for a 1-D, possibly non-uniform, coordinate.'''
    coord = np.asarray(coord, dtype=float).ravel()
    if coord.size < 2:
        return np.ones_like(coord)
    d = np.empty_like(coord)
    d[1:-1] = 0.5 * (coord[2:] - coord[:-2])
    d[0] = 0.5 * (coord[1] - coord[0])
    d[-1] = 0.5 * (coord[-1] - coord[-2])
    return np.abs(d)


def trapz_2d(x1, x2, n_vars=1):
    '''
    2-D integration weights on a possibly non-uniform orthogonal grid.

    :param numpy.ndarray x1: first spatial coordinate, 1-D.
    :param numpy.ndarray x2: second spatial coordinate, 1-D.
    :param int n_vars: number of variables. Default is 1.

    :return: the weights, of shape ``(len(x1), len(x2), n_vars)``.
    :rtype: dict
    '''
    dA = np.einsum('i,j->ij', _cell_widths(x1), _cell_widths(x2))
    if n_vars:
        dA = np.repeat(dA[..., np.newaxis], n_vars, axis=-1)
    return {'weights_name': 'trapz_2d', 'weights': dA}


def trapz_3d(x1, x2, x3, n_vars=1):
    '''
    3-D integration weights on a possibly non-uniform orthogonal grid.

    :param numpy.ndarray x1: first spatial coordinate, 1-D.
    :param numpy.ndarray x2: second spatial coordinate, 1-D.
    :param numpy.ndarray x3: third spatial coordinate, 1-D.
    :param int n_vars: number of variables. Default is 1.

    :return: the weights, of shape ``(len(x1), len(x2), len(x3), n_vars)``.
    :rtype: dict
    '''
    dV = np.einsum('i,j,k->ijk', _cell_widths(x1), _cell_widths(x2),
                   _cell_widths(x3))
    if n_vars:
        dV = np.repeat(dV[..., np.newaxis], n_vars, axis=-1)
    return {'weights_name': 'trapz_3d', 'weights': dV}


def custom(**kwargs):
    '''
    Customized weights, to be implemented by the user if required. The returned
    array must have the shape documented at the top of this module.
    '''
    pass


def apply_normalization(data, weights, n_vars, method='variance', comm=None):
    '''
    Normalize the weights variable-wise by the data variance.

    :param numpy.ndarray data: the data.
    :param numpy.ndarray weights: the weights.
    :param int n_vars: number of variables.
    :param str method: normalization method. Default is 'variance'.
    :param MPI.Comm comm: parallel communicator. Default is None.

    :return: the normalized weights.
    :rtype: numpy.ndarray
    '''
    if method.lower() != 'variance':
        return weights
    axis = tuple(np.arange(0, data[..., 0].ndim))
    for i in range(0, n_vars):
        var = np.nanvar(data[..., i], axis=axis)
        if comm is not None:
            # every rank holds the same replicated data, so no reduction is
            # needed; kept explicit so the intent is not mistaken for an
            # omission
            pass
        weights[..., i] = weights[..., i] / var
    return weights
