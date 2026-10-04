'''Derived module from base.py for standard BMD.'''
from pybmd.bmd.base import Base


class Standard(Base):
    '''
    Class that implements the Bispectral Mode Decomposition of Schmidt (2020).

    The computation is performed on the *data* passed to the `fit` method of
    the `Standard` class, derived from the `Base` class.

    The data must have time as its first dimension and the variable index as
    its last; any number of spatial dimensions may sit in between.

    :References:

        Schmidt, O. T., *Bispectral mode decomposition of nonlinear flows*,
        Nonlinear Dynamics, 2020. DOI 10.1007/s11071-020-06037-z
    '''

    def _triad_matrices(self, q_hat, i_triad):
        '''
        Assemble the realizations of the sum interaction and of the quadratic
        term for one triad.

        :param dict q_hat: Fourier realizations by frequency row.
        :param int i_triad: index into the per-triad arrays.

        :return: ``(q_sum, q_prod, weights)``, the first two of shape
            ``(nx*nv, n_blocks)``.
        :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
        '''
        t = self._triads
        q1 = q_hat[int(t.f1_idx[i_triad])]
        q2 = q_hat[int(t.f2_idx[i_triad])]
        q3 = q_hat[int(t.f3_idx[i_triad])]
        return q3, q1 * q2, self._weights

    def _constituent_matrices(self, q_hat, i_triad):
        '''
        :return: ``(q_k, q_l)``, the DFT rows at ``f1`` and ``f2``, of shape
            ``(nx*nv, n_blocks)``.
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
        '''
        t = self._triads
        return q_hat[int(t.f1_idx[i_triad])], q_hat[int(t.f2_idx[i_triad])]
