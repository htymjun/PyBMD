'''Derived module from base.py for cross-bispectral mode decomposition.'''
import numpy as np

from pybmd.bmd.base import Base


class Cross(Base):
    '''
    Class that implements the Cross-Bispectral Mode Decomposition.

    CBMD has the same inputs and outputs as :class:`pybmd.bmd.standard.Standard`,
    but the data must contain all the variables to be cross-correlated. The
    form of the quadratic nonlinear term is set by two parameters:

    - ``state_idx``: variable indices of the states ``s``, one per row of the
      nonlinear term;
    - ``qr_idx``: variable indices of the products ``q*r``, with columns
      alternating between ``q`` and ``r``; one row per entry of ``state_idx``.

    The defaults ``state_idx = [0]`` and ``qr_idx = [[1, 2]]`` reproduce the
    reference, i.e. the single term ``s_0 <- q_1 * r_2``.

    .. note::

        Unlike the reference MATLAB ``cbmd.m``, which expects the variable
        index immediately after time, PyBMD keeps the variable index **last**
        for both decompositions, so that the same data array works with either
        class. ``state_idx`` and ``qr_idx`` are **0-based**, whereas ``regions``
        stays 1-based since those are labels on the published octant figure.

    :References:

        Schmidt, O. T., *Bispectral mode decomposition of nonlinear flows*,
        Nonlinear Dynamics, 2020. DOI 10.1007/s11071-020-06037-z
    '''

    _label = 'CBMD'

    def __init__(self, params, weights=None, comm=None):
        super().__init__(params, weights=weights, comm=comm)
        self._state_idx = np.atleast_1d(
            np.asarray(params.get('state_idx', [0]), dtype=int))
        self._qr_idx = np.atleast_2d(
            np.asarray(params.get('qr_idx', [[1, 2]]), dtype=int))
        self._validate_var_idx()
        if self._constituent_modes:
            raise ValueError(
                'constituent_modes is not supported by Cross: the quadratic '
                'term sums n_terms q*r pairs, so no single constituent mode '
                'phi_k or phi_l is well defined.')

    @property
    def n_state(self):
        '''Number of states in the quadratic nonlinear term.'''
        return self._state_idx.size

    @property
    def n_terms(self):
        '''Number of ``q*r`` products per state.'''
        return self._qr_idx.shape[1] // 2

    def _validate_var_idx(self):
        '''Bounds-check the 0-based variable indices.'''
        if self._qr_idx.shape[0] != self._state_idx.size:
            raise ValueError(
                f'qr_idx has {self._qr_idx.shape[0]} rows but state_idx has '
                f'{self._state_idx.size} entries; one row per state is '
                f'required.')
        if self._qr_idx.shape[1] % 2 != 0:
            raise ValueError(
                f'qr_idx must have an even number of columns, alternating q '
                f'and r; got {self._qr_idx.shape[1]}.')
        for name, idx in (('state_idx', self._state_idx),
                          ('qr_idx', self._qr_idx)):
            bad = idx[(idx < 0) | (idx >= self._nv)]
            if bad.size:
                raise ValueError(
                    f'{name} contains out-of-range variable indices '
                    f'{np.unique(bad).tolist()}; these are 0-based and must '
                    f'lie in 0..{self._nv - 1}.')

    def _expected_weights_shape(self):
        '''
        CBMD weights are purely spatial, with no variable axis: the same weight
        applies to every state, and is tiled internally.
        '''
        return tuple(self._xshape)

    @property
    def _mode_shape(self):
        return (*self._xshape, self.n_state)

    def _block_shape(self):
        # (nx, nv), so that q_hat[f][:, v] is a contiguous slice
        return (self._nx, self._nv)

    def _post_initialize(self):
        # the same spatial weight applies to every state; tile the whole
        # spatial vector n_state times (a per-element repeat would scramble it)
        self._weights_tiled = np.tile(self._weights, (self.n_state, 1))
        self._pr0(f'State indices            : {self._state_idx.tolist()}')
        self._pr0(f'q*r indices              : {self._qr_idx.tolist()}')

    def _unflatten_modes(self, psi):
        '''
        :meth:`_triad_matrices` stacks the states along the flat axis with the
        state as the *slowest* index (``flat = j*nx + p``), matching the
        reference's ``repmat``. A C-order reshape straight into
        ``(*xshape, n_state)`` would read the state as the fastest index and
        scramble the modes, so unflatten as ``(n_state, *xshape)`` first and
        move the state axis last.
        '''
        psi = psi.reshape((psi.shape[0], self.n_state, *self._xshape))
        return np.moveaxis(psi, 1, -1)

    def _triad_matrices(self, q_hat, i_triad):
        '''
        Assemble the state realizations and the quadratic term for one triad,
        stacking the states along the flattened spatial axis.

        :param dict q_hat: Fourier realizations by frequency row, each of shape
            ``(nx, nv, n_blocks)``.
        :param int i_triad: index into the per-triad arrays.

        :return: ``(q_s, q_qr, weights)``, the first two of shape
            ``(n_state*nx, n_blocks)``.
        :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
        '''
        t = self._triads
        q1 = q_hat[int(t.f1_idx[i_triad])]
        q2 = q_hat[int(t.f2_idx[i_triad])]
        q3 = q_hat[int(t.f3_idx[i_triad])]

        n = self.n_state * self._nx
        q_s = np.empty((n, self._n_blocks), dtype=self._complex)
        q_qr = np.zeros((n, self._n_blocks), dtype=self._complex)
        for j in range(self.n_state):
            sl = slice(j * self._nx, (j + 1) * self._nx)
            q_s[sl] = q3[:, self._state_idx[j], :]
            for k in range(0, 2 * self.n_terms, 2):
                q_qr[sl] += (q1[:, self._qr_idx[j, k], :]
                             * q2[:, self._qr_idx[j, k + 1], :])
        return q_s, q_qr, self._weights_tiled
