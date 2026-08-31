'''Derived module from base.py for standard BMD.'''
import time

from pybmd.bmd.base import Base
import pybmd.utils.parallel as utils_par


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

    def fit(self, data_list, variables=None):
        '''
        Class-specific method to fit the data matrix using the BMD algorithm.

        :param data_list: data matrix of shape ``(nt, *xshape, n_variables)``,
            or path(s) to it.

        :return: the fitted object.
        :rtype: Standard
        '''
        start0 = time.time()

        start = time.time()
        self._initialize(data_list, variables)
        self._mode_shape = (*self._xshape, self._nv)
        self._pr0(f'Time to initialize: {time.time() - start} s.')

        start = time.time()
        q_hat = self._compute_qhat(block_shape=(self._nxv,))
        self._pr0(f'Time to compute DFT: {time.time() - start} s.')
        del self.data
        utils_par.barrier(self._comm)

        start = time.time()
        self._triad_loop(q_hat)
        del q_hat
        self._pr0(f'------------------------------------')
        self._pr0(f'Time to compute BMD: {time.time() - start} s.')

        self._store_and_save()
        self._pr0(f' ')
        self._pr0(f'Results saved in folder {self._savedir_sim}')
        self._pr0(f'Total time: {time.time() - start0} s.')
        utils_par.barrier(self._comm)
        return self

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
