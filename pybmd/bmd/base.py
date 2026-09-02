'''
Base module for the BMD:
    - The `Base.fit` method must be implemented in inherited classes
'''
from __future__ import division

import os
import time
import warnings

import numpy as np
import yaml

import pybmd.bmd.utils as utils_bmd
import pybmd.bmd.optimizers as optimizers
import pybmd.utils.io as utils_io
import pybmd.utils.parallel as utils_par
import pybmd.utils.weights as utils_weights

CWD = os.getcwd()
B2GB = 9.3132257461548e-10


class Base():
    '''
    Bispectral Mode Decomposition base class.

    :param dict params: parameters of the decomposition. Required keys are
        ``n_dft``, ``time_step``, ``n_space_dims`` and ``n_variables``; see
        the class documentation for the optional ones.
    :param dict weights: spatial inner-product weights, as returned by the
        constructors in :mod:`pybmd.utils.weights`. Default is uniform.
    :param MPI.Comm comm: parallel communicator. Default is None (serial).
    :param numpy.ndarray mean: long-time mean to subtract instead of the one
        computed from the data. Default is None.
    '''

    def __init__(self, params, weights=None, comm=None, mean=None):
        ##--- required
        self._n_dft = params['n_dft']
        self._dt = params['time_step']
        self._xdim = params['n_space_dims']
        self._nv = params['n_variables']
        if not isinstance(self._n_dft, (int, np.integer)):
            raise TypeError('n_dft must be an integer.')
        self._n_dft = int(self._n_dft)

        ##--- optional: spectral estimation
        # percentage overlap; note the default is 50, not PySPOD's 0, because
        # the reference BMD uses floor(n_dft/2)
        self._overlap = params.get('overlap', 50)
        # absolute overlap in snapshots; takes precedence over `overlap`
        self._n_overlap_req = params.get('n_overlap', None)
        self._window_req = params.get('window', 'hamming')
        self._mean_type = params.get('mean_type', 'longtime')

        ##--- optional: bispectrum
        self._regions = params.get('regions', [1, 2])
        self._max_freq_idx = params.get('max_freq_idx', None)

        ##--- optional: solver
        self._solver = params.get('solver', 'MengiOverton')
        self._solver_tol = params.get('tol', 1e-6)
        self._solver_n_it_max = params.get('n_it_max', 500)
        # explicit start vector for the iterative solvers, used to reproduce
        # results from another implementation
        self._solver_z0 = params.get('solver_z0', None)

        ##--- optional: storage
        self._dtype = params.get('dtype', 'double')
        self._normalize_weights = params.get('normalize_weights', False)
        self._normalize_data = params.get('normalize_data', False)
        self._savedir = os.path.join(CWD, params.get('savedir', 'bmd_results'))
        params['savedir'] = self._savedir
        self._save_modes = params.get('save_modes', True)
        self._store_modes = params.get('store_modes', False)
        self._max_modes_gb = params.get('max_modes_gb', 8.0)
        self._compute_transfer = params.get('compute_energy_transfer', True)

        self._params = params
        self._weights_tmp = weights
        self._mean_user = mean
        self._comm = comm
        self._float, self._complex = utils_bmd._get_dtype(self._dtype)

        ## define rank and size for both parallel and serial
        if self._comm is not None:
            self._rank = comm.rank
            self._size = comm.size
        else:
            self._rank = 0
            self._size = 1

        ## validate eagerly, so a bad configuration fails before any I/O
        if self._solver not in optimizers.SOLVERS:
            raise ValueError(
                f'solver must be one of {optimizers.SOLVERS}; '
                f'got {self._solver!r}.')
        if self._mean_type.lower() not in ('longtime', 'blockwise', 'zero',
                                           'none'):
            raise ValueError(f'{self._mean_type} not recognized.')

        ## window and overlap
        self._window, self._window_name = utils_bmd.get_window(
            self._window_req, self._n_dft)
        self._window = self._set_dtype(self._window)
        self._resolve_overlap()

    # --------------------------------------------------------------------------
    # to be implemented by inherited classes
    # --------------------------------------------------------------------------

    def fit(self, data_list, *args, **kwargs):
        '''
        Fit the data using BMD.

        :param list data_list: data matrix for which to compute the BMD.
        '''
        raise NotImplementedError  # pragma: no cover

    def _triad_matrices(self, q_hat, i_triad):
        '''
        Assemble the two ``(n, n_blocks)`` matrices whose cross-spectral
        density gives the bispectral matrix ``B`` for a triad, together with
        the weight vector to use.

        :return: ``(q_sum, q_prod, weights)``, where ``q_sum`` collects the
            realizations at ``f1 + f2`` and ``q_prod`` the quadratic term.
        '''
        raise NotImplementedError  # pragma: no cover

    # --------------------------------------------------------------------------
    # basic getters
    # --------------------------------------------------------------------------

    @property
    def savedir_sim(self):
        '''Directory where results are saved.'''
        return self._savedir_sim

    @property
    def modes_dir(self):
        '''Directory where modes are saved.'''
        return self._modes_dir

    @property
    def dim(self):
        '''Number of dimensions of the data matrix.'''
        return self._dim

    @property
    def shape(self):
        '''Shape of the data matrix.'''
        return self._shape

    @property
    def nt(self):
        '''Number of time-steps of the data matrix.'''
        return self._nt

    @property
    def nx(self):
        '''Number of spatial points of the data matrix.'''
        return self._nx

    @property
    def nv(self):
        '''Number of variables of the data matrix.'''
        return self._nv

    @property
    def xdim(self):
        '''Number of spatial dimensions of the data matrix.'''
        return self._xdim

    @property
    def xshape(self):
        '''Spatial shape of the data matrix.'''
        return self._xshape

    @property
    def comm(self):
        '''The MPI communicator.'''
        return self._comm

    @property
    def dt(self):
        '''The time-step.'''
        return self._dt

    @property
    def n_dft(self):
        '''Number of DFT points per block.'''
        return self._n_dft

    @property
    def n_overlap(self):
        '''Number of overlapping snapshots between consecutive blocks.'''
        return self._n_overlap

    @property
    def n_blocks(self):
        '''Number of blocks.'''
        return self._n_blocks

    @property
    def freq(self):
        '''The two-sided, fftshifted frequency axis.'''
        return self._triads.freq

    @property
    def f_idx(self):
        '''The signed integer frequency index axis.'''
        return self._triads.f_idx

    @property
    def n_freq(self):
        '''Number of frequencies.'''
        return self._triads.n_freq

    @property
    def triads(self):
        '''The :class:`pybmd.bmd.utils.Triads` computed.'''
        return self._triads

    @property
    def n_triads(self):
        '''Number of triads.'''
        return self._triads.n_triads

    @property
    def weights(self):
        '''Weights used to compute the inner product.'''
        return self._weights

    @property
    def bispectrum(self):
        '''
        The mode bispectrum ``L``, of shape ``(n_freq, n_freq)``. Entries that
        do not correspond to a computed triad are NaN.
        '''
        return self._L

    @property
    def L(self):
        '''Alias of :attr:`bispectrum`.'''
        return self._L

    @property
    def energy_transfer(self):
        '''
        The energy-transfer term ``T``, of shape ``(n_freq, n_freq)``. Entries
        that do not correspond to a computed triad are NaN.
        '''
        return self._T

    @property
    def T(self):
        '''Alias of :attr:`energy_transfer`.'''
        return self._T

    @property
    def coeffs(self):
        '''
        The expansion coefficients, of shape ``(n_triads, n_blocks)``. These
        are the maximisers of the numerical radius, from which the modes of any
        triad can be recomputed without re-running the optimizer.
        '''
        return self._coeffs

    @property
    def modes(self):
        '''
        All modes, of shape ``(n_triads, 2, *xshape, nv)``. Only available when
        ``store_modes`` was set.
        '''
        if not self._store_modes:
            raise ValueError(
                'Modes were not retained in memory; set params["store_modes"] '
                '= True, or read them from disk with get_modes_at_triad().')
        return self._modes

    # --------------------------------------------------------------------------
    # common methods
    # --------------------------------------------------------------------------

    def _resolve_overlap(self):
        '''Resolve the overlap, which may be given in percent or in snapshots.'''
        if self._n_overlap_req is not None:
            self._n_overlap = int(self._n_overlap_req)
            self._overlap = 100.0 * self._n_overlap / self._n_dft
        else:
            # floor, not ceil/round: matches the reference's own
            # nOvlp = floor(nDFT/2) (bmd.m:268) at the 50% default, and for
            # any percentage that does not divide n_dft evenly -- ceil would
            # silently pick a different n_overlap (hence n_blocks) than the
            # reference on the same request.
            self._n_overlap = int(np.floor(self._n_dft * self._overlap / 100))
        if self._n_overlap > self._n_dft - 1:
            raise ValueError('Overlap is too large.')

    def _initialize(self, data_list, variables=None):
        '''Set up dimensions, weights, mean, frequency axis and triads.'''
        self._pr0(f' ')
        self._pr0(f'Initialize data')
        self._pr0(f'------------------------------------')

        st = time.time()
        self.data = utils_io.get_data_array(
            data_list, self._xdim, self._nv, dtype=self._float)
        self._pr0(f'- loaded data into memory: {time.time() - st} s.')

        self._shape = self.data.shape
        self._dim = self.data.ndim
        self._nt = self._shape[0]
        self._xshape = self._shape[1:-1]
        self._nx = int(np.prod(self._xshape))
        self._nxv = self._nx * self._nv

        self._pr0(f'nx: {self._nx}')
        self._pr0(f'dim: {self._dim}')
        self._pr0(f'shape: {self._shape}')
        self._pr0(f'xdim: {self._xdim}')
        self._pr0(f'xshape: {self._xshape}')
        self._pr0(f'nt: {self._nt}')

        # define number of blocks
        num = self._nt - self._n_overlap
        den = self._n_dft - self._n_overlap
        self._n_blocks = num // den

        # test feasibility
        if (self._n_dft < 4) or (self._n_blocks < 2):
            raise ValueError('Spectral estimation parameters not meaningful.')

        ## define and check weights
        self.define_weights()

        ## apply mean
        st = time.time()
        self.select_mean(self.data)
        self._pr0(f'- computed mean: {time.time() - st} s.')

        ## normalize weights if required
        if self._normalize_weights:
            self._pr0('- normalizing weights')
            self._weights = utils_weights.apply_normalization(
                data=self.data, weights=self._weights,
                n_vars=self._nv, comm=self._comm)

        ## flatten weights, in the same C order the data is flattened in
        self._weights = np.reshape(self._weights, [-1, 1])
        self._weights = self._set_dtype(self._weights)

        # determine correction for FFT window gain
        self._win_weight = 1 / np.mean(self._window)
        self._window = self._window.reshape(self._window.shape[0], 1)

        # get frequency axis and triads
        self._triads = utils_bmd.triad_indices(
            n_dft=self._n_dft, dt=self._dt, regions=self._regions,
            max_freq_idx=self._max_freq_idx)
        if self._triads.n_triads == 0:
            raise ValueError(
                f'No triads to compute for regions={self._regions} and '
                f'max_freq_idx={self._max_freq_idx}.')

        ## create folders to save results
        self._savedir_sim = os.path.join(
            self._savedir,
            'nfft' + str(self._n_dft)
            + '_novlp' + str(self._n_overlap)
            + '_nblks' + str(self._n_blocks))
        self._modes_dir = os.path.join(self._savedir_sim, 'modes')
        if self._rank == 0:
            os.makedirs(self._modes_dir, exist_ok=True)
        utils_par.barrier(self._comm)

        # problem size accounting; check the mode footprint before the DFT, so
        # an unaffordable configuration fails in seconds rather than in hours
        self._pb_size_f = self.data.size * self._float(1).nbytes * B2GB
        self._qhat_size_gb = (self._triads.freq_needed.size * self._nxv
                              * self._complex(1).nbytes * B2GB)
        self._modes_size_gb = (2 * self.n_triads * self._nxv
                               * self._complex(1).nbytes * B2GB)
        if self._save_modes and self._modes_size_gb > self._max_modes_gb:
            raise ValueError(
                f'Saving all modes would need {self._modes_size_gb:.2f} GB, '
                f'above the max_modes_gb limit of {self._max_modes_gb:.2f} GB. '
                f'Set params["save_modes"] = False to store only the '
                f'coefficients, or raise params["max_modes_gb"].')

        self._print_parameters()
        self._pr0(f'------------------------------------')

    def define_weights(self):
        '''Define and check weights.'''
        self._pr0('- checking weight dimensions')
        expected = tuple(self._xshape) + (self._nv,)
        if isinstance(self._weights_tmp, dict):
            self._weights = np.asarray(self._weights_tmp['weights'])
            self._weights_name = self._weights_tmp['weights_name']
            self._check_weights_shape(expected)
        else:
            if self._weights_tmp is not None and self._rank == 0:
                warnings.warn(
                    'Parameter `weights` is not a dict as returned by '
                    'pybmd.utils.weights; using default uniform weighting.')
            self._weights = np.ones(expected)
            self._weights_name = 'uniform'

    def _check_weights_shape(self, expected):
        '''
        Require the full spatial shape rather than a flat vector.

        A flat weight vector carries no record of the order it was built in.
        Passing one built in Fortran order attaches each weight to the wrong
        grid point: the bispectrum stays plausible, because it is a full
        reduction over space and so is insensitive to the permutation, while
        the modes come out scrambled. Requiring the shape removes the ambiguity.
        '''
        if self._weights.shape != tuple(expected):
            raise ValueError(
                f'weights have shape {self._weights.shape} but '
                f'{tuple(expected)} is required. Pass an array with the full '
                f'spatial shape rather than a flattened vector, so that it is '
                f'unambiguous which weight belongs to which grid point.')

    def select_mean(self, data):
        '''Select the mean to subtract from every block.'''
        mean_type = self._mean_type.lower()
        self._lt_mean = self.long_t_mean(data)
        if self._mean_user is not None:
            self._t_mean = np.reshape(
                np.asarray(self._mean_user), [-1])
            self._mean_type = 'user'
        elif mean_type == 'longtime':
            self._t_mean = self._lt_mean
        elif mean_type in ('blockwise', 'zero', 'none'):
            self._t_mean = 0.0
            if mean_type in ('zero', 'none') and self._rank == 0:
                warnings.warn(
                    'No mean subtracted. Consider using longtime mean.')
        return self._t_mean

    def long_t_mean(self, data):
        '''Compute the long-time mean, flattened over space and variables.'''
        t_mean = np.mean(data, axis=0)
        return self._set_dtype(np.reshape(t_mean, [-1]))

    def _get_block(self, i_blk):
        '''Snapshots of block ``i_blk``, flattened to ``(n_dft, nxv)``.'''
        offset = min(i_blk * (self._n_dft - self._n_overlap) + self._n_dft,
                     self._nt) - self._n_dft
        q_blk = self.data[offset:offset + self._n_dft].copy()
        return q_blk.reshape(self._n_dft, -1), offset

    def _compute_blocks(self, i_blk):
        '''
        Windowed, mean-subtracted DFT of one block.

        The full two-sided spectrum is always required: BMD couples ``f1``,
        ``f2`` and ``f1 + f2``, and the difference-interaction regions need
        negative frequencies. There is therefore no real-signal ``rfft`` path.
        '''
        q_blk, offset = self._get_block(i_blk)
        q_blk = q_blk - self._t_mean

        if self._mean_type.lower() == 'blockwise':
            q_blk = q_blk - np.mean(q_blk, axis=0)

        if self._normalize_data:
            den = self._n_dft - 1
            q_var = np.sum((q_blk - np.mean(q_blk, axis=0))**2, axis=0) / den
            q_var[q_var < 4 * np.finfo(float).eps] = 1
            q_blk = q_blk / q_var

        q_blk = q_blk * self._window
        q_blk = self._set_dtype(q_blk)
        q_blk_hat = (self._win_weight / self._n_dft) * np.fft.fft(q_blk, axis=0)
        return np.fft.fftshift(q_blk_hat, axes=0), offset

    def _compute_qhat(self, block_shape):
        '''
        Fourier realizations for every frequency row any triad refers to.

        Only the rows in ``triads.freq_needed`` are retained; for a bispectrum
        restricted by ``max_freq_idx`` that is a small fraction of ``n_dft``.

        :param tuple block_shape: shape of one frequency row of a block.

        :return: mapping from frequency row to its ``(*block_shape, n_blocks)``
            array of realizations.
        :rtype: dict
        '''
        self._pr0(f' ')
        self._pr0(f'Calculating temporal DFT')
        self._pr0(f'------------------------------------')

        needed = self._triads.freq_needed
        q_hat = {int(f): np.empty((*block_shape, self._n_blocks),
                                 dtype=self._complex) for f in needed}
        for i_blk in range(0, self._n_blocks):
            st = time.time()
            q_blk_hat, offset = self._compute_blocks(i_blk)
            for f in needed:
                q_hat[int(f)][..., i_blk] = q_blk_hat[f].reshape(block_shape)
            self._pr0(f'block {i_blk + 1}/{self._n_blocks} '
                      f'({offset}:{self._n_dft + offset});  '
                      f'Elapsed time: {time.time() - st} s.')
        self._pr0(f'------------------------------------')
        return q_hat

    def _triad_loop(self, q_hat):
        '''
        Solve every triad, distributing them across ranks, and reduce.

        Sets ``self._L``, ``self._T`` and ``self._coeffs``, and writes the modes
        of the triads owned by this rank.
        '''
        self._pr0(f' ')
        self._pr0(f'Calculating BMD')
        self._pr0(f'------------------------------------')

        n_freq, n_triads = self.n_freq, self.n_triads
        # accumulate into zeros rather than NaN: NaN would propagate through the
        # sum-reduction below and poison every entry. The NaN mask that the
        # reference uses is restored afterwards.
        L = np.zeros((n_freq, n_freq), dtype=self._complex)
        T = np.zeros((n_freq, n_freq), dtype=self._float)
        coeffs = np.zeros((n_triads, self._n_blocks), dtype=self._complex)
        if self._store_modes:
            self._modes = np.zeros(
                (n_triads, 2, *self._mode_shape), dtype=self._complex)

        my_triads = utils_par.distribute_indices(n_triads, self._comm)
        st = time.time()
        for n, i in enumerate(my_triads):
            i = int(i)
            q_sum, q_prod, weights = self._triad_matrices(q_hat, i)

            # cross-spectral density between the sum interaction and the
            # quadratic term; (n_blocks, n_blocks)
            B = q_sum.conj().T @ (q_prod * weights) / self._n_blocks

            r, a = optimizers.solve(
                B, solver=self._solver, tol=self._solver_tol,
                n_it_max=self._solver_n_it_max, z0=self._solver_z0)

            # the optimizer works in double precision regardless of the
            # requested dtype -- B is only (n_blocks, n_blocks), so the accuracy
            # is free -- but the results are stored at the requested precision
            a = a.astype(self._complex)
            psi_sum = q_sum @ a
            psi_prod = q_prod @ a
            L[self._triads.f1_idx[i], self._triads.f2_idx[i]] = r
            if self._compute_transfer:
                # note the energy transfer carries no weight, unlike B; this
                # follows the reference implementation
                T[self._triads.f1_idx[i], self._triads.f2_idx[i]] = \
                    np.real(np.vdot(psi_sum, psi_prod)) / self._n_blocks
            coeffs[i, :] = a

            psi = np.stack([utils_bmd.normalize_mode(psi_sum, weights),
                            utils_bmd.normalize_mode(psi_prod, weights)])
            if self._store_modes:
                self._modes[i] = psi.reshape((2, *self._mode_shape))
            if self._save_modes:
                self._save_modes_at_triad(i, psi)

            if n % 100 == 0 or n == my_triads.size - 1:
                self._pr0(
                    f'triad {n + 1}/{my_triads.size} on rank {self._rank}; '
                    f'(k,l,k+l) = ({self._triads.k[i]},{self._triads.l[i]},'
                    f'{self._triads.kl[i]});  '
                    f'Elapsed time: {time.time() - st:.3f} s.')

        # one sum-reduction each: every entry is written by exactly one rank,
        # so all other ranks contribute an exact zero
        self._L = utils_par.allreduce(L, self._comm)
        self._T = utils_par.allreduce(T, self._comm)
        self._coeffs = utils_par.allreduce(coeffs, self._comm)
        if self._store_modes:
            self._modes = utils_par.allreduce(self._modes, self._comm)

        # restore the reference semantics: entries that are not triads are NaN
        outside = ~self._triads.mask
        self._L[outside] = np.nan
        self._T[outside] = np.nan
        if not self._compute_transfer:
            self._T[:] = np.nan
        utils_par.barrier(self._comm)

    def _save_modes_at_triad(self, i_triad, psi):
        '''Write the two modes of one triad to ``modes/triad_idx_{i:08d}.npy``.'''
        path = os.path.join(self._modes_dir, f'triad_idx_{i_triad:08d}.npy')
        np.save(path, psi.reshape((2, *self._mode_shape)))

    def get_modes_at_triad(self, triad_idx):
        '''
        Load the modes of one triad.

        :param int triad_idx: index into the per-triad arrays, as returned by
            ``self.triads.find(k, l)``.

        :return: the modes, of shape ``(2, *xshape, nv)``. Index 0 is the
            sum-interaction mode and index 1 the quadratic-term mode.
        :rtype: numpy.ndarray
        '''
        if self._store_modes:
            return self._modes[triad_idx]
        path = os.path.join(self._modes_dir, f'triad_idx_{triad_idx:08d}.npy')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'No modes stored for triad {triad_idx}. Was fit() run with '
                f'save_modes enabled?')
        return np.load(path)

    def get_modes_at_freqs(self, k, l):
        '''
        Load the modes of the triad ``(k, l, k+l)``.

        :param int k: integer frequency index of f1.
        :param int l: integer frequency index of f2.

        :return: the modes, of shape ``(2, *xshape, nv)``.
        :rtype: numpy.ndarray
        '''
        return self.get_modes_at_triad(self._triads.find(k, l))

    def find_triad(self, k, l):
        '''See :meth:`pybmd.bmd.utils.Triads.find`.'''
        return self._triads.find(k, l)

    def _store_and_save(self):
        '''Store and save results.'''
        self._params['n_freq'] = int(self.n_freq)
        self._params['n_triads'] = int(self.n_triads)
        self._params['results_folder'] = str(self._savedir_sim)
        self._params['time_step'] = float(self._dt)
        self._params['n_dft'] = int(self._n_dft)
        self._params['n_blocks'] = int(self._n_blocks)
        self._params['n_overlap'] = int(self._n_overlap)
        self._params['overlap'] = float(self._overlap)
        self._params['solver'] = str(self._solver)

        if self._rank == 0:
            path_params = os.path.join(self._savedir_sim, 'params_modes.yaml')
            with open(path_params, 'w') as f:
                yaml.dump(self._params, f)
            np.save(os.path.join(self._savedir_sim, 'weights.npy'),
                    self._weights)
            np.save(os.path.join(self._savedir_sim, 'ltm_modes.npy'),
                    self._lt_mean)
            np.save(os.path.join(self._savedir_sim, 'coeffs.npy'), self._coeffs)
            self._triads.to_npz(os.path.join(self._savedir_sim, 'triads.npz'))
            np.savez(os.path.join(self._savedir_sim, 'bispectrum.npz'),
                     L=self._L, T=self._T, freq=self.freq, f_idx=self.f_idx)
            print(f'Parameters dictionary saved in: {path_params}')
            print(f'Bispectrum saved in: '
                  f'{os.path.join(self._savedir_sim, "bispectrum.npz")}')
        utils_par.barrier(self._comm)

    def _pr0(self, string):
        '''Print rank 0 only.'''
        utils_par.pr0(string=string, comm=self._comm)

    def _set_dtype(self, d):
        '''Set data type.'''
        if np.issubdtype(d.dtype, np.complexfloating):
            return d.astype(self._complex)
        if np.issubdtype(d.dtype, np.floating):
            return d.astype(self._float)
        return d

    def _print_parameters(self):
        '''Display parameter summary.'''
        self._pr0(f'')
        self._pr0(f'BMD parameters')
        self._pr0(f'------------------------------------')
        self._pr0(f'Problem size (real)      : {self._pb_size_f:.2f} GB')
        self._pr0(f'Q_hat size               : {self._qhat_size_gb:.2f} GB')
        self._pr0(f'Modes size (all triads)  : {self._modes_size_gb:.2f} GB')
        self._pr0(f'Data type for real       : {self._float}')
        self._pr0(f'Data type for complex    : {self._complex}')
        self._pr0(f'No. snapshots per block  : {self._n_dft}')
        self._pr0(f'Block overlap            : {self._n_overlap}')
        self._pr0(f'No. of blocks            : {self._n_blocks}')
        self._pr0(f'Windowing fct. (time)    : {self._window_name}')
        self._pr0(f'Weighting fct. (space)   : {self._weights_name}')
        self._pr0(f'Mean                     : {self._mean_type}')
        self._pr0(f'Time-step                : {self._dt}')
        self._pr0(f'Time snapshots           : {self._nt}')
        self._pr0(f'Space dimensions         : {self._xdim}')
        self._pr0(f'Number of variables      : {self._nv}')
        self._pr0(f'Number of frequencies    : {self.n_freq} '
                  f'(rows retained: {self._triads.freq_needed.size})')
        self._pr0(f'Regions                  : {list(self._regions)}')
        self._pr0(f'Max frequency index      : {self._max_freq_idx} '
                  f'(Nyquist index {self._triads.f_nyq_idx})')
        self._pr0(f'Number of triads         : {self.n_triads}')
        # flag the bug-compatible solver loudly: it exists only to reproduce
        # a specific published MATLAB result and always under-estimates, so
        # a run must not be able to use it silently
        compat_note = (' [MATLAB-COMPATIBLE: reproduces refs/bmd/bmd.m\'s '
                       'known under-estimation bug -- see optimizers.py]'
                       if self._solver == 'MengiOvertonMATLAB' else '')
        self._pr0(f'Solver (x*Ax)            : {self._solver} '
                  f'(tol {self._solver_tol}, n_it_max {self._solver_n_it_max})'
                  f'{compat_note}')
        self._pr0(f'MPI ranks                : {self._size}')
        self._pr0(f'Results to be saved in   : {self._savedir}')
        self._pr0(f'------------------------------------')
        self._pr0(f'')
