'''
Optimizer for the numerical radius, i.e. the maximum of the modulus of the
field of values

.. math::

    r(A) = \\max_{\\|z\\|=1} |z^H A z|,

which is the quantity maximised at every triad of the bispectral mode
decomposition. :func:`mengi_overton` implements the globally convergent
level-set algorithm of Mengi & Overton (2005), as in the 17-Aug-2023 revision
of the reference ``bmd.m``, with the fixes listed in ``pybmd/bmd/CLAUDE.md``.
'''
import numpy as np
import scipy.linalg as sla


__all__ = ['mengi_overton', 'max_fov']

# unit-circle / level-set detection tolerance, as in the reference implementation
_SQRT_EPS = np.sqrt(np.finfo(float).eps)


def max_fov(A, theta):
    '''
    Maximum of the field of values of ``A`` in the direction ``theta``, that is
    the largest eigenvalue of the Hermitian part of the rotated matrix

    .. math::

        \\lambda_{max}\\left(\\frac{1}{2}\\left(Ae^{i\\theta}
        + (Ae^{i\\theta})^H\\right)\\right).

    :param numpy.ndarray A: square complex matrix.
    :param theta: angle(s) at which to evaluate, in radians.
    :type theta: float or numpy.ndarray

    :return: the maximum field of value at each angle.
    :rtype: numpy.ndarray

    .. note::

        The *signed* largest eigenvalue is returned, not the largest in
        modulus as in the reference's ``maxFOV``. Both give the same maximum
        over all angles, but not the same *level sets*: the crossings located
        by :func:`mengi_overton` are where the signed ``lambda_max`` equals
        the current level, and filtering them by modulus rejects valid ones.
    '''
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    out = np.empty(theta.shape[0], dtype=float)
    for i, th in enumerate(theta):
        A_rot = A * np.exp(1j * th)
        H = 0.5 * (A_rot + A_rot.conj().T)
        out[i] = np.linalg.eigvalsh(H)[-1]
    return out


def _dominant_eigvec(A, phi):
    '''
    Unit vector maximising the field of values of ``A`` in direction ``phi``,
    together with the corresponding (complex) value of ``z^H A z``.
    '''
    A_rot = A * np.exp(1j * phi)
    H = 0.5 * (A_rot + A_rot.conj().T)
    eigval, eigvec = np.linalg.eigh(H)
    z = eigvec[:, int(np.argmax(np.abs(eigval)))]
    return z.conj() @ A @ z, z


def _pow2_scale(A):
    '''
    Rescale ``A`` so that ``||A||_1`` lies in ``(0.5, 1]``.

    The unit-circle test in the level-set solver is ``abs(abs(D) - 1) <=
    sqrt(eps) * ||A||_1`` -- an *absolute* tolerance on a dimensionless
    quantity, scaled by the norm.  For the matrices BMD actually produces that
    norm is small (``B`` carries a ``1/n_blocks`` and the weights, and runs at
    1e-3 or below on real data), which drives the tolerance below the accuracy
    of the badly scaled pencil.  Every crossing is then rejected, the search
    terminates at once, and the solver silently returns a local maximum.

    Scaling by a power of two is exact in binary floating point, so this only
    re-conditions the problem and changes no value.  The numerical radius is
    homogeneous, ``r(cA) = c r(A)`` for real ``c > 0``, with the same
    maximiser, so the caller recovers the answer by evaluating the Rayleigh
    quotient on the original matrix.

    :param numpy.ndarray A: square complex matrix.

    :return: the scaled matrix.
    :rtype: numpy.ndarray
    '''
    norm_1 = float(np.linalg.norm(A, 1))
    if norm_1 == 0.0 or not np.isfinite(norm_1):
        return A
    # ldexp applies the power of two exactly and, unlike dividing by 2**e,
    # cannot overflow when the norm is subnormal (2**e is then itself
    # subnormal, and complex division by it produces inf)
    e = int(np.ceil(np.log2(norm_1)))
    if np.iscomplexobj(A):
        return np.ldexp(A.real, -e) + 1j * np.ldexp(A.imag, -e)
    return np.ldexp(A, -e)


def mengi_overton(A, tol=1e-8, n_it_max=500):
    '''
    Level-set algorithm of Mengi & Overton (2005) for the numerical radius.
    Globally convergent and deterministic.

    The maximum field of value ``max_fov(A, theta)`` is maximised over
    ``theta``.  At each iteration the unimodular eigenvalues of a matrix pencil
    give the angles at which the current level ``w`` is crossed; the midpoint
    of each interval between consecutive crossings is tested, and any midpoint
    lying above ``w`` becomes a candidate for the next iteration.  The search
    stops when no interval lies above the current level.

    :param numpy.ndarray A: square complex matrix.
    :param float tol: level inflation factor and stopping tolerance.
        Default is 1e-8.
    :param int n_it_max: maximum number of level-set iterations. Default is 500.

    :return: the value ``w = z^H A z`` and the maximiser ``z``.
    :rtype: tuple(complex, numpy.ndarray)
    '''
    A_in = np.asarray(A)
    n = A_in.shape[0]
    # work on a rescaled copy so the unit-circle tolerance below stays
    # meaningful; see _pow2_scale
    A = _pow2_scale(A_in)
    norm_A = np.linalg.norm(A, 1)
    if norm_A == 0.0:
        z = np.zeros(n, dtype=complex)
        z[0] = 1.0
        return 0j, z
    zeros = np.zeros((n, n))
    eye = np.eye(n)
    S = np.block([[A, zeros], [zeros, eye]])

    phi = np.zeros(1)
    phi_max = 0.0
    it = 0
    while phi.size:
        # highest level found so far, and the angle attaining it
        levels = max_fov(A, phi)
        i_max = int(np.argmax(levels))
        phi_max = phi[i_max]
        w = levels[i_max] * (1 + tol)

        # angles at which the level curve crosses the level w
        R = np.block([[2 * w * eye, -A.conj().T], [eye, zeros]])
        eigval = sla.eig(R, S, right=False)
        eigval = eigval[np.isfinite(eigval)]
        on_circle = np.abs(np.abs(eigval) - 1) <= (_SQRT_EPS * norm_A)
        theta = np.angle(eigval[on_circle])
        if theta.size:
            keep = np.abs(max_fov(A, theta) - w) <= _SQRT_EPS * max(w, 1.0)
            theta = theta[keep]
        if theta.size == 0:
            break
        # np.unique sorts, which the interval sweep below relies on; round
        # first, as exact float equality would leave near-duplicates in place
        theta = np.unique(np.round(theta, 10))

        # descend into every interval whose midpoint lies above the level
        candidates = []
        for i, lower in enumerate(theta):
            if i < theta.size - 1:
                mid = 0.5 * (lower + theta[i + 1])
            else:
                mid = np.mod(0.5 * (lower + theta[0] + 2 * np.pi), 2 * np.pi)
            if max_fov(A, mid)[0] > w:
                candidates.append(mid)
        phi = np.asarray(candidates, dtype=float)

        it += 1
        if it >= n_it_max:
            break

    # the maximiser is unaffected by the rescaling; evaluate the Rayleigh
    # quotient on the original matrix to recover the unscaled value
    _, z = _dominant_eigvec(A, phi_max)
    return z.conj() @ A_in @ z, z
