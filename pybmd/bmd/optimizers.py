'''
Optimizers for the numerical radius, i.e. the maximum of the modulus of the
field of values

.. math::

    r(A) = \\max_{\\|z\\|=1} |z^H A z|,

which is the quantity maximised at every triad of the bispectral mode
decomposition.  Three algorithms are provided:

- :func:`mengi_overton` -- level-set algorithm of Mengi & Overton (2005);
  globally convergent, and the default.
- :func:`he_watson` -- 'An Algorithm' of He & Watson (1997); globally optimal
  upon convergence, but slower and less reliable.
- :func:`simple_iteration` -- Watson's simple power iteration; cheap, but not
  guaranteed to find the global optimum.

.. note::

    :func:`max_fov` uses the *signed* largest eigenvalue of the Hermitian part.
    The reference MATLAB implementation uses ``max(abs(eig(H)))`` instead,
    which makes the level-set intersection filter in :func:`mengi_overton`
    discard valid angles and terminate at a local maximum.  See :func:`max_fov`.
'''
import numpy as np
import scipy.linalg as sla


__all__ = ['solve', 'mengi_overton', 'he_watson', 'simple_iteration', 'max_fov']

SOLVERS = ('MengiOverton', 'HeWatson', 'simpleIteration')

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
        modulus.  Since the Hermitian part at ``theta + pi`` is the negative of
        the one at ``theta``, both give the same maximum over all angles, and
        so the same numerical radius.  They do not, however, give the same
        *level sets*: the unimodular generalized eigenvalues located by
        :func:`mengi_overton` are the angles at which the signed
        ``lambda_max`` equals the current level, and filtering them with the
        modulus rejects valid angles.  The level-set search then finds no new
        interval to descend into and returns a local maximum.
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
    Rescale ``A`` so that ``||A||_1`` lies in ``(0.5, 1]``, returning the scaled
    matrix and the factor used.

    The unit-circle test in the level-set solvers is ``abs(abs(D) - 1) <=
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

    :return: the scaled matrix and the scale factor.
    :rtype: tuple(numpy.ndarray, float)
    '''
    norm_1 = float(np.linalg.norm(A, 1))
    if norm_1 == 0.0 or not np.isfinite(norm_1):
        return A, 1.0
    scale = np.ldexp(1.0, int(np.ceil(np.log2(norm_1))))
    return A / scale, scale


def _random_start(n, rng):
    '''Random complex start vector, matching the reference ``rand + 1i*rand``.'''
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(size=n) + 1j * rng.uniform(size=n)


def default_start(A, n_scan=16):
    '''
    Deterministic start vector: the maximiser of the field of values over a
    coarse scan of rotation angles.

    The reference implementation starts :func:`simple_iteration` and
    :func:`he_watson` from a random vector, which makes results depend on the
    state of a global generator -- and, under MPI, on how triads happen to be
    distributed across ranks.  Scanning instead is deterministic, costs
    ``n_scan`` eigendecompositions of an ``(n_blocks, n_blocks)`` matrix, and
    starts the search at or above the largest sampled local maximum.

    The last point matters for :func:`he_watson`, which stops when the level
    curve has no crossings.  Started too low -- below the *minimum* of
    ``lambda_max`` over angle -- the level is never crossed at all, and that
    stopping test fires while the true maximum is still far above.

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


def simple_iteration(A, z_0=None, tol=1e-6, n_it_max=100, rng=None):
    '''
    Watson's simple power iteration for the numerical radius.  Cheap, but it is
    not guaranteed to find the global optimum -- on random matrices it
    under-estimates the numerical radius fairly often.  Prefer
    :func:`mengi_overton` unless reproducing legacy results.

    :param numpy.ndarray A: square complex matrix.
    :param numpy.ndarray z_0: start vector. Default is :func:`default_start`,
        or a random vector when ``rng`` is given.
    :param float tol: convergence tolerance on ``|w - w_old|``. Default is 1e-6.
    :param int n_it_max: maximum number of iterations. Default is 100.
    :param numpy.random.Generator rng: if given, start from a random vector
        drawn from this generator instead of :func:`default_start`.

    :return: the value ``w = z^H A z`` and the maximiser ``z``.
    :rtype: tuple(complex, numpy.ndarray)
    '''
    A = np.asarray(A)
    if z_0 is None:
        z_0 = default_start(A) if rng is None else _random_start(A.shape[0], rng)
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


def he_watson(A, tol=1e-6, n_it_max=500, rng=None):
    '''
    He & Watson's (1997) algorithm, which is guaranteed to find the global
    optimum upon convergence.

    :param numpy.ndarray A: square complex matrix.
    :param float tol: convergence tolerance. Default is 1e-6.
    :param int n_it_max: maximum number of iterations. Default is 500.
    :param numpy.random.Generator rng: if given, start from a random vector
        drawn from this generator instead of :func:`default_start`. A generator
        is required for the restart-on-stall path; one is created if needed.

    :return: the value ``w = z^H A z`` and the maximiser ``z``.
    :rtype: tuple(complex, numpy.ndarray)
    '''
    A_in = np.asarray(A)
    n = A_in.shape[0]
    # as in mengi_overton, rescale so the unit-circle tolerance stays meaningful
    A, _ = _pow2_scale(A_in)
    norm_A = np.linalg.norm(A, 1)
    if norm_A == 0.0:
        z = np.zeros(n, dtype=complex)
        z[0] = 1.0
        return 0j, z
    z = default_start(A) if rng is None else _random_start(n, rng)

    zeros = np.zeros((n, n))
    eye = np.eye(n)
    S = np.block([[A, zeros], [zeros, eye]])

    lb, ub = 0.0, norm_A
    w_best, z_best = 0.0 + 0j, z
    it = 0
    while (ub - lb) > tol or it == 0:
        it += 1
        w, z = simple_iteration(A, z, tol=tol, n_it_max=n_it_max)
        # the reference returns the last iterate; track the best one instead,
        # since a restart or a stalled step can otherwise lose a better point
        if np.abs(w) > np.abs(w_best):
            w_best, z_best = w, z
        lb = max(lb, np.abs(w))
        alpha = lb + tol
        R = np.block([[2 * alpha * eye, -A.conj().T], [eye, zeros]])
        eigval, eigvec = sla.eig(R, S)
        finite = np.isfinite(eigval)
        on_circle = np.zeros(eigval.shape, dtype=bool)
        on_circle[finite] = \
            np.abs(np.abs(eigval[finite]) - 1) < (_SQRT_EPS * norm_A)
        if not on_circle.any():
            break
        if it >= n_it_max:
            break
        if it % 100 == 0:
            # stalled; restart from a fresh direction, as in the reference
            if rng is None:
                rng = np.random.default_rng(it)
            z = _random_start(n, rng)
        else:
            z = eigvec[-n:, int(np.flatnonzero(on_circle)[0])]
    nrm = np.linalg.norm(z_best)
    z_best = z_best / nrm if nrm > 0 else z_best
    return z_best.conj() @ A_in @ z_best, z_best


def mengi_overton(A, tol=1e-6, n_it_max=500):
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
        Default is 1e-6.
    :param int n_it_max: maximum number of level-set iterations. Default is 500.

    :return: the value ``w = z^H A z`` and the maximiser ``z``.
    :rtype: tuple(complex, numpy.ndarray)
    '''
    A_in = np.asarray(A)
    n = A_in.shape[0]
    # work on a rescaled copy so the unit-circle tolerance below stays meaningful
    A, _ = _pow2_scale(A_in)
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


def solve(A, solver='MengiOverton', tol=1e-6, n_it_max=500, rng=None, z0=None):
    '''
    Maximise ``|z^H A z|`` over unit vectors ``z`` with the requested solver.

    :param numpy.ndarray A: square complex matrix.
    :param str solver: one of 'MengiOverton', 'HeWatson', 'simpleIteration'.
        Default is 'MengiOverton'.
    :param float tol: solver tolerance. Default is 1e-6.
    :param int n_it_max: maximum number of iterations. Default is 500.
    :param numpy.random.Generator rng: generator for solvers that need a random
        start vector. Passing an explicit generator keeps results reproducible
        and independent of how work is distributed across MPI ranks.
    :param numpy.ndarray z0: explicit start vector for the iterative solvers,
        overriding both ``rng`` and :func:`default_start`. Mainly of use for
        reproducing results from another implementation.

    :return: the value ``w = z^H A z`` and the maximiser ``z``.
    :rtype: tuple(complex, numpy.ndarray)
    '''
    if solver == 'MengiOverton':
        # deterministic and globally convergent: no start vector is needed
        return mengi_overton(A, tol=tol, n_it_max=n_it_max)
    elif solver == 'HeWatson':
        if z0 is not None:
            raise ValueError('z0 is not supported by the HeWatson solver.')
        return he_watson(A, tol=tol, n_it_max=n_it_max, rng=rng)
    elif solver == 'simpleIteration':
        return simple_iteration(A, z_0=z0, tol=tol, n_it_max=n_it_max, rng=rng)
    raise ValueError(f'Unknown solver {solver!r}; must be one of {SOLVERS}.')
