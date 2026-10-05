'''
Module implementing spatial inner-product weights, usually quadrature weights.

Every constructor returns the dict ``{'weights_name': str, 'weights': ndarray}``.

.. note::

    The expected shape differs between the two decompositions:

    - :class:`pybmd.bmd.standard.Standard` expects ``(*xshape, n_variables)``
      -- the weight covers the variables too;
    - :class:`pybmd.bmd.cross.Cross` expects ``xshape`` -- a purely spatial
      weight, which is tiled internally over the state variables.

    Spatial axes follow the NumPy/matplotlib image convention, ``(ny, nx)``
    in 2-D and ``(nz, ny, nx)`` in 3-D: x is the *last* spatial axis. Arrays
    from MATLAB or Fortran, stored ``(nx, ny)``, must be transposed first.

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


def trapz_2d(x, y, n_vars=1):
    '''
    2-D integration weights on a possibly non-uniform orthogonal grid.

    :param numpy.ndarray x: x coordinate, 1-D, of length ``nx``.
    :param numpy.ndarray y: y coordinate, 1-D, of length ``ny``.
    :param int n_vars: number of variables. Default is 1.

    :return: the weights, of shape ``(ny, nx, n_vars)``.
    :rtype: dict
    '''
    dA = np.outer(_cell_widths(y), _cell_widths(x))
    if n_vars:
        dA = np.repeat(dA[..., np.newaxis], n_vars, axis=-1)
    return {'weights_name': 'trapz_2d', 'weights': dA}


def curvilinear_2d(x, y, n_vars=1):
    '''
    2-D integration weights on a structured curvilinear grid.

    The cell area is the Jacobian of the map from the index space
    ``(xi, eta)`` to ``(x, y)``, ``|x_xi y_eta - x_eta y_xi|``, with
    second-order differences (one-sided at the edges), times the trapezoidal
    weights of the index space (1/2 on the edges). On a rectilinear grid this
    reduces exactly to :func:`trapz_2d`.

    :param numpy.ndarray x: x coordinate of every grid point, ``(ny, nx)``.
    :param numpy.ndarray y: y coordinate of every grid point, ``(ny, nx)``.
    :param int n_vars: number of variables. Default is 1.

    :return: the weights, of shape ``(ny, nx, n_vars)``.
    :rtype: dict
    '''
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or x.shape != y.shape or min(x.shape) < 2:
        raise ValueError(
            f'x and y must be 2-D arrays of the same shape, at least 2 points '
            f'in each direction; got {x.shape} and {y.shape}.')
    x_xi, x_eta = np.gradient(x)
    y_xi, y_eta = np.gradient(y)
    jac = np.abs(x_xi * y_eta - x_eta * y_xi)
    w_xi, w_eta = np.ones(x.shape[0]), np.ones(x.shape[1])
    w_xi[[0, -1]] = 0.5
    w_eta[[0, -1]] = 0.5
    dA = jac * np.outer(w_xi, w_eta)
    if n_vars:
        dA = np.repeat(dA[..., np.newaxis], n_vars, axis=-1)
    return {'weights_name': 'curvilinear_2d', 'weights': dA}


def trapz_3d(x, y, z, n_vars=1):
    '''
    3-D integration weights on a possibly non-uniform orthogonal grid.

    :param numpy.ndarray x: x coordinate, 1-D, of length ``nx``.
    :param numpy.ndarray y: y coordinate, 1-D, of length ``ny``.
    :param numpy.ndarray z: z coordinate, 1-D, of length ``nz``.
    :param int n_vars: number of variables. Default is 1.

    :return: the weights, of shape ``(nz, ny, nx, n_vars)``.
    :rtype: dict
    '''
    dV = np.einsum('k,j,i->kji', _cell_widths(z), _cell_widths(y),
                   _cell_widths(x))
    if n_vars:
        dV = np.repeat(dV[..., np.newaxis], n_vars, axis=-1)
    return {'weights_name': 'trapz_3d', 'weights': dV}
