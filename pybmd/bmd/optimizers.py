'''
Optimizers for the numerical radius, i.e. the maximum of the modulus of the
field of values

.. math::

    r(A) = \\max_{\\|z\\|=1} |z^H A z|,

which is the quantity maximised at every triad of the bispectral mode
decomposition.  Two algorithms are provided:

- :func:`mengi_overton` -- level-set algorithm of Mengi & Overton (2005);
  globally convergent, and the default.
- :func:`simple_iteration` -- Watson's simple power iteration; cheap, but not
  guaranteed to find the global optimum.

.. note::

    The reference MATLAB implementation, ``bmd.m``, originally shipped
    with the paper (Schmidt 2020) uses He & Watson's (1997) nested-iteration
    algorithm -- see the appendix of ``Schmidt_2020_NODY_r2.tex``, whose
    Algorithm 1 is :func:`simple_iteration` here. A later revision of
    ``bmd.m`` (17-Aug-2023) replaced that default with Mengi & Overton's
    globally convergent level-set algorithm, which is what this module makes
    the default too. He & Watson's algorithm itself is not ported: it is only
    locally convergent per restart, and PyBMD has no use for a solver that
    needs a random start vector to escape local optima when
    :func:`mengi_overton` finds the global one directly and deterministically.
'''
import numpy as np
import scipy.linalg as sla


__all__ = ['solve', 'mengi_overton', 'simple_iteration', 'max_fov',
           'default_start', 'SOLVERS']

SOLVERS = ('MengiOverton', 'simpleIteration')

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
        modulus. Since the Hermitian part at ``theta + pi`` is the negative
        of the one at ``theta``, both give the same maximum over all angles,
        and so the same numerical radius.  They do not, however, give the
        same *level sets*: the unimodular generalized eigenvalues located by
        :func:`mengi_overton` are the angles at which the signed
        ``lambda_max`` equals the current level, and filtering them with the
        modulus would reject valid angles.
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
    scale = np.ldexp(1.0, int(np.ceil(np.log2(norm_1))))
    return A / scale


def default_start(A, n_scan=16):
    '''
    Deterministic start vector: the maximiser of the field of values over a
    coarse scan of rotation angles.

    The reference implementation starts :func:`simple_iteration` from a
    random vector, which makes results depend on the state of a global
    generator -- and, under MPI, on how triads happen to be distributed
    across ranks.  Scanning instead is deterministic, costs ``n_scan``
    eigendecompositions of an ``(n_blocks, n_blocks)`` matrix, and starts the
    search at or above the largest sampled local maximum.

    :param numpy.ndarray A: square complex matrix.
    :param int n_scan: number of angles to scan over ``[0, pi)``. Default is 16.

    :return: unit-norm start vector.
    :rtype: numpy.ndarray

    .. note::

        Only ``[0, pi)`` is scanned: the Hermitian part at ``theta + pi`` is
        the negative of the one at ``theta``, so its largest eigenvalue is
        already covered by the smallest one at ``theta``.
    '''
    best_val, best_vec = -np.inf, None
    for th in np.linspace(0, np.pi, n_scan, endpoint=False):
        A_rot = A * np.exp(1j * th)
        H = 0.5 * (A_rot + A_rot.conj().T)
        eigval, eigvec = np.linalg.eigh(H)
        for j in (0, -1):                    # most negative and most positive
            if np.abs(eigval[j]) > best_val:
                best_val, best_vec = np.abs(eigval[j]), eigvec[:, j]
    return best_vec.astype(complex)


def simple_iteration(A, z_0=None, tol=1e-8, n_it_max=100):
    '''
    Watson's simple power iteration for the numerical radius (Algorithm 1 of
    the appendix of ``Schmidt_2020_NODY_r2.tex``).  Cheap, but it is not
    guaranteed to find the global optimum -- on random matrices it
    under-estimates the numerical radius fairly often.  Prefer
    :func:`mengi_overton` unless reproducing legacy results.

    :param numpy.ndarray A: square complex matrix.
    :param numpy.ndarray z_0: start vector. Default is :func:`default_start`.
    :param float tol: convergence tolerance on ``|w - w_old|``. Default is 1e-8.
    :param int n_it_max: maximum number of iterations. Default is 100.

    :return: the value ``w = z^H A z`` and the maximiser ``z``.
    :rtype: tuple(complex, numpy.ndarray)
    '''
    A = np.asarray(A)
    if z_0 is None:
        z_0 = default_start(A)
    z = np.asarray(z_0, dtype=complex).ravel()
    z = z / np.sqrt(z.conj() @ z)

    w = np.inf + 0j
    w_err = np.inf
    it = 0
    while w_err > tol:
        w_old = w
        w = z.conj() @ A @ z
        w_err = np.abs(w - w_old)
        z_new = w * (A.conj().T @ z) + np.conj(w) * (A @ z)
        norm = np.sqrt(np.real(np.vdot(z_new, z_new)))
        # the update collapses whenever w is 0 (a nilpotent or zero A, or a
        # start vector on which the form vanishes); the direction is then
        # undetermined, so keep the last unit iterate rather than adopting a
        # zero vector that would leave z unnormalized
        if not np.isfinite(norm) or norm == 0:
            break
        z = z_new / norm
        it += 1
        if it > n_it_max:
            break
    return z.conj() @ A @ z, z


def mengi_overton(A, tol=1e-8, n_it_max=500):
    '''
    Level-set algorithm of Mengi & Overton (2005) for the numerical radius.
    Globally convergent, deterministic, and the default solver.

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
            level_tol = _SQRT_EPS * max(w, 1.0)
            keep = np.abs(max_fov(A, theta) - w) <= level_tol
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


def solve(A, solver='MengiOverton', tol=1e-8, n_it_max=500, z0=None):
    '''
    Maximise ``|z^H A z|`` over unit vectors ``z`` with the requested solver.

    :param numpy.ndarray A: square complex matrix.
    :param str solver: one of 'MengiOverton', 'simpleIteration'.
        Default is 'MengiOverton'.
    :param float tol: solver tolerance. Default is 1e-8.
    :param int n_it_max: maximum number of iterations. Default is 500.
    :param numpy.ndarray z0: explicit start vector for the iterative solvers,
        overriding :func:`default_start`. Mainly of use for reproducing
        results from another implementation. Not supported by 'MengiOverton',
        which is deterministic and needs no start vector.

    :return: the value ``w = z^H A z`` and the maximiser ``z``.
    :rtype: tuple(complex, numpy.ndarray)
    '''
    if solver == 'MengiOverton':
        if z0 is not None:
            raise ValueError('z0 is not supported by the MengiOverton solver.')
        return mengi_overton(A, tol=tol, n_it_max=n_it_max)
    elif solver == 'simpleIteration':
        return simple_iteration(A, z_0=z0, tol=tol, n_it_max=n_it_max)
    raise ValueError(f'Unknown solver {solver!r}; must be one of {SOLVERS}.')
