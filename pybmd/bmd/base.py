'''
Base module for the BMD: parameters, weights, mean, DFT blocking, the triad
loop and storage. :class:`~pybmd.bmd.standard.Standard` and
:class:`~pybmd.bmd.cross.Cross` only supply the per-triad matrices and the
shape hooks below.
'''
import glob
import math
import os
import time
import warnings

import numpy as np
import yaml

import pybmd.bmd.utils as utils_bmd
import pybmd.bmd.optimizers as optimizers
import pybmd.bmd.postproc as postproc
import pybmd.utils.io as utils_io
import pybmd.utils.parallel as utils_par

B2GB = 9.3132257461548e-10
# default for params['max_modes_gb']: above this, keeping every mode
# (save_modes/store_modes) is refused
MAX_MODES_GB = 8.0


def _yaml_safe(obj):
    '''
    Convert numpy scalars and arrays, recursively through containers, into
    plain Python values that ``yaml.dump`` can represent.

    Users routinely leave numpy values in ``params`` (``regions=np.array([1,
    2])``, a float32 time step); the dump must not fail on them after the
    whole decomposition has run.
    '''
    if isinstance(obj, dict):
        return {str(k): _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_yaml_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _yaml_safe(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, complex):
        return str(obj)
    return obj


def _check_save_modes_top(top):
    '''Validate ``params['save_modes_top']``: None, an int >= 1 (a count) or a
    float in (0, 1] (a fraction of the candidate triads).'''
    if top is None:
        return None
    if isinstance(top, (bool, np.bool_)):
        raise TypeError('save_modes_top must be an int, a float or None.')
    if isinstance(top, (int, np.integer)):
        if top < 1:
            raise ValueError(
                f'save_modes_top as a count must be >= 1; got {top}.')
        return int(top)
    if isinstance(top, (float, np.floating)):
        if not 0 < top <= 1:
            raise ValueError(
                f'save_modes_top as a fraction must be in (0, 1]; got {top}.')
        return float(top)
    raise TypeError('save_modes_top must be an int, a float or None.')


class Base():
    '''
    Bispectral Mode Decomposition base class.

    :param dict params: parameters of the decomposition. Required keys are
        ``n_dft``, ``time_step``, ``n_space_dims`` and ``n_variables``; see
        the class documentation for the optional ones.
    :param dict weights: spatial inner-product weights, as returned by the
        constructors in :mod:`pybmd.utils.weights`. Default is uniform.
    :param MPI.Comm comm: parallel communicator. Default is None (serial).
    '''

    _label = 'BMD'   # name used in the timing print

    def __init__(self, params, weights=None, comm=None):
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

        ##--- optional: bispectrum
        self._regions = params.get('regions', [1, 2])
        self._max_freq_idx = params.get('max_freq_idx', None)

        ##--- optional: solver (Mengi-Overton)
        self._solver_tol = params.get('tol', 1e-6)
        self._solver_n_it_max = params.get('n_it_max', 500)

        ##--- optional: storage
        self._dtype = params.get('dtype', 'double')
        # precision of the arrays written to disk; default follows `dtype`.
        # 'single' halves the files while the computation stays at `dtype`
        self._save_dtype = params.get('save_dtype', None) or self._dtype
        # resolve against the working directory at construction, not at import
        self._savedir = os.path.abspath(params.get('savedir', 'bmd_results'))
        self._save_modes = params.get('save_modes', True)
        self._store_modes = params.get('store_modes', False)
        # write the modes of the strongest triads only: an int is a count, a
        # float in (0, 1] a fraction; None writes every triad
        self._save_modes_top = _check_save_modes_top(
            params.get('save_modes_top', None))
        # None disables the check
        self._max_modes_gb = params.get('max_modes_gb', MAX_MODES_GB)
        # opt-in: also compute and store the two constituent modes phi_k,
        # phi_l alongside the sum and quadratic-term modes
        self._constituent_modes = params.get('constituent_modes', False)
        self._n_mode_comp = 4 if self._constituent_modes else 2
        self._compute_transfer = params.get('compute_energy_transfer', True)

        # a copy: the run records derived quantities (n_blocks, the results
        # folder, ...) into it, and that must not leak into the caller's dict
        self._params = dict(params)
        self._params['savedir'] = self._savedir
        self._weights_tmp = weights
        self._comm = comm
        self._float, self._complex = utils_bmd._get_dtype(self._dtype)
        self._save_float, self._save_complex = utils_bmd._get_dtype(
            self._save_dtype)

        ## define rank and size for both parallel and serial
        if self._comm is not None:
            self._rank = comm.rank
            self._size = comm.size
        else:
            self._rank = 0
            self._size = 1
        # the ranks sharing memory with this one: the large arrays are held
        # once per node, not per rank; see _shared_empty
        self._node_comm = utils_par.node_comm(comm)
        self._wins = {}

        ## validate eagerly, so a bad configuration fails before any I/O
        if self._n_dft < 4:
            raise ValueError(
                f'n_dft must be at least 4; got {self._n_dft}.')

        ## window and overlap
        self._window, self._window_name = utils_bmd.get_window(
            self._window_req, self._n_dft)
        self._window = self._set_dtype(self._window)
        self._resolve_overlap()

    def __getstate__(self):
        '''
        Pickle without the MPI handles, which cannot be serialized: the
        loaded object is serial (``comm=None``), and arrays held in shared
        memory, such as ``modes``, are restored as private copies.
        '''
        state = self.__dict__.copy()
        state.update(_comm=None, _node_comm=None, _wins={}, _rank=0, _size=1)
        return state

    def fit(self, data_list):
        '''
        Fit the data: initialize, DFT every block, solve every triad, save.

        :param data_list: data matrix of shape ``(nt, *xshape, n_variables)``,
            or path(s) to it. Under MPI only the first rank of each node reads
            it, into memory shared by the ranks of that node; the other ranks
            ignore it, so they may pass None instead of loading the data.

        :return: the fitted object.
        '''
        start0 = time.time()

        start = time.time()
        self._initialize(data_list)
        self._pr0(f'Time to initialize: {time.time() - start} s.')

        start = time.time()
        q_hat = self._compute_qhat(comm=self._node_comm)
        self._pr0(f'Time to compute DFT: {time.time() - start} s.')
        del self.data
        self._free_shared('data')
        utils_par.barrier(self._comm)

        start = time.time()
        self._triad_loop(q_hat)
        del q_hat
        self._free_shared('q_hat')
        self._pr0(f'------------------------------------')
        self._pr0(f'Time to compute {self._label}: {time.time() - start} s.')

        self._store_and_save()
        self._pr0(f' ')
        self._pr0(f'Results saved in folder {self._savedir_sim}')
        self._pr0(f'Total time: {time.time() - start0} s.')
        utils_par.barrier(self._comm)
        return self

    # --------------------------------------------------------------------------
    # hooks for inherited classes
    # --------------------------------------------------------------------------

    def _triad_matrices(self, q_hat, i_triad):
        '''
        Assemble the two ``(n, n_blocks)`` matrices whose cross-spectral
        density gives the bispectral matrix ``B`` for a triad, together with
        the weight vector to use.

        :return: ``(q_sum, q_prod, weights)``, where ``q_sum`` collects the
            realizations at ``f1 + f2`` and ``q_prod`` the quadratic term.
        '''
        raise NotImplementedError  # pragma: no cover

    def _constituent_matrices(self, q_hat, i_triad):
        '''
        Assemble the two ``(n, n_blocks)`` matrices for the constituent
        frequencies ``k`` and ``l`` of a triad, used only when
        ``params['constituent_modes']`` is set.

        :return: ``(q_k, q_l)``, on the same flat axis as ``q_sum`` in
            :meth:`_triad_matrices`, so the triad's own expansion vector
            ``a`` applies unchanged.
        '''
        raise NotImplementedError  # pragma: no cover

    def _expected_weights_shape(self):
        '''Shape a user weight array must have: spatial shape plus variables.'''
        return tuple(self._xshape) + (self._nv,)

    @property
    def _mode_shape(self):
        '''Shape of one mode: the flat axis :meth:`_triad_matrices` builds,
        unflattened.'''
        return (*self._xshape, self._nv)

    def _block_shape(self):
        '''Shape of one frequency row of a block, as stored in ``q_hat``.'''
        return (self._nxv,)

    def _post_initialize(self):
        '''Subclass set-up that needs the flattened weights; runs last in
        :meth:`_initialize`.'''

    def _unflatten_modes(self, psi):
        '''
        Reshape the flat modes ``(n_comp, n)`` of a triad to
        ``(n_comp, *mode_shape)``, where ``n_comp`` is 2 (sum, quadratic-term)
        or 4 with ``params['constituent_modes']`` (plus phi_k, phi_l).

        The flat axis is the one :meth:`_triad_matrices` builds. Here it is
        the C-order flattening of ``(*xshape, nv)``, so a plain reshape undoes
        it; a subclass that stacks that axis in a different order must override
        this, or its modes come out scrambled while ``L`` stays correct.
        '''
        return psi.reshape((psi.shape[0], *self._mode_shape))

    # --------------------------------------------------------------------------
    # basic getters
    # --------------------------------------------------------------------------

    @property
    def savedir_sim(self):
        '''Directory where results are saved.'''
        return self._savedir_sim

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
    def L(self):
        '''
        The mode bispectrum, of shape ``(n_freq, n_freq)``. Entries that do
        not correspond to a computed triad are NaN.
        '''
        return self._L

    @property
    def T(self):
        '''
        The energy-transfer term, of shape ``(n_freq, n_freq)``. Entries that
        do not correspond to a computed triad are NaN.
        '''
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
        All modes, of shape ``(n_triads, n_comp, *xshape, nv)`` where
        ``n_comp`` is 2, or 4 with ``params['constituent_modes']`` set. Only
        available when ``store_modes`` was set.
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

    def _initialize(self, data_list):
        '''Set up dimensions, weights, mean, frequency axis and triads.'''
        self._pr0(f' ')
        self._pr0(f'Initialize data')
        self._pr0(f'------------------------------------')

        st = time.time()
        self.data = self._load_shared(data_list)
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
        if self._n_blocks < 2:
            raise ValueError(
                f'Spectral estimation parameters not meaningful: nt={self._nt}, '
                f'n_dft={self._n_dft}, n_overlap={self._n_overlap} give '
                f'{self._n_blocks} block(s), at least 2 are needed.')

        ## define and check weights
        self.define_weights()

        ## long-time mean, subtracted from every block; each rank of a node
        ## averages its own spatial points
        st = time.time()
        t_mean = self._shared_empty((self._nxv,), self._float, 't_mean',
                                    self._node_comm)
        cols = self._my_cols(self._node_comm)
        t_mean[cols] = self.long_t_mean(
            self.data.reshape(self._nt, -1)[:, cols])
        utils_par.barrier(self._node_comm)
        # a private copy: reconstruct_modes needs it after fit returns
        self._t_mean = t_mean.copy()
        del t_mean
        self._free_shared('t_mean')
        self._pr0(f'- computed mean: {time.time() - st} s.')

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

        # the triads whose modes are written; resolved to a count here so the
        # size check below can use it, selected by |L| after the triad loop
        self._n_saved_triads = self._resolve_save_modes_top()

        ## create folders to save results
        self._savedir_sim = os.path.join(
            self._savedir,
            'nfft' + str(self._n_dft)
            + '_novlp' + str(self._n_overlap)
            + '_nblks' + str(self._n_blocks))
        self._modes_dir = os.path.join(self._savedir_sim, 'modes')
        if self._rank == 0:
            os.makedirs(self._modes_dir, exist_ok=True)
            # the directory name encodes only the blocking, so a previous run
            # with more triads may have left mode files here that this run
            # will not overwrite; every other file is rewritten, so these
            # would be the only stale state a loader could pick up
            for stale in (glob.glob(os.path.join(self._modes_dir,
                                                 'triad_idx_*.npy'))
                          + glob.glob(os.path.join(self._modes_dir,
                                                   'saved_triad_idx.npy'))):
                os.remove(stale)
        utils_par.barrier(self._comm)

        # problem size accounting; check the mode footprint before the DFT, so
        # an unaffordable configuration fails in seconds rather than in hours
        self._pb_size_f = self.data.size * self._float(1).nbytes * B2GB
        self._qhat_size_gb = (self._triads.freq_needed.size * self._nxv
                              * self._n_blocks
                              * self._complex(1).nbytes * B2GB)
        # modes are held in memory at `dtype` but written at `save_dtype`;
        # size the check by the larger of the footprints actually incurred
        triad_gb = (self._n_mode_comp * int(np.prod(self._mode_shape))
                    * B2GB)
        footprints = []
        if self._store_modes:
            footprints.append(
                self.n_triads * self._complex(1).nbytes * triad_gb)
        if self._save_modes:
            footprints.append(
                self._n_saved_triads * self._save_complex(1).nbytes * triad_gb)
        self._modes_size_gb = max(footprints, default=0.0)
        if (self._max_modes_gb is not None
                and self._modes_size_gb > self._max_modes_gb):
            raise ValueError(
                f'Keeping all modes would need {self._modes_size_gb:.2f} GB '
                f'(on disk with save_modes, and in memory on every node with '
                f'store_modes), above the limit of {self._max_modes_gb:.2f} '
                f'GB. Raise params["max_modes_gb"] (None disables the check), '
                f'set params["save_dtype"] to "single", or set '
                f'params["save_modes"] and params["store_modes"] to False to '
                f'store only the coefficients.')

        self._print_parameters()
        self._pr0(f'------------------------------------')
        self._post_initialize()

    def define_weights(self):
        '''
        Define and check weights. Under MPI, like the data, they are taken
        from the first rank of each node and broadcast to the others, whose
        own ``weights`` argument is ignored: every triad must see the same
        weights, whichever rank solves it.
        '''
        self._pr0('- checking weight dimensions')
        self._weights, self._weights_name = self._on_node_root(
            self._resolve_weights)

    def _resolve_weights(self):
        '''The weights and their name, from the ``weights`` argument.'''
        expected = self._expected_weights_shape()
        if isinstance(self._weights_tmp, dict):
            weights = np.asarray(self._weights_tmp['weights'])
            self._check_weights_shape(weights, expected)
            return weights, self._weights_tmp['weights_name']
        if self._weights_tmp is not None:
            warnings.warn(
                'Parameter `weights` is not a dict as returned by '
                'pybmd.utils.weights; using default uniform weighting.')
        return np.ones(expected), 'uniform'

    def _check_weights_shape(self, weights, expected):
        '''
        Require the full spatial shape rather than a flat vector.

        A flat weight vector carries no record of the order it was built in.
        Passing one built in Fortran order attaches each weight to the wrong
        grid point: the bispectrum stays plausible, because it is a full
        reduction over space and so is insensitive to the permutation, while
        the modes come out scrambled. Requiring the shape removes the ambiguity.
        '''
        if weights.shape != tuple(expected):
            raise ValueError(
                f'weights have shape {weights.shape} but '
                f'{tuple(expected)} is required. Pass an array with the full '
                f'spatial shape rather than a flattened vector, so that it is '
                f'unambiguous which weight belongs to which grid point.')

    def long_t_mean(self, data):
        '''Compute the long-time mean, flattened over space and variables.'''
        t_mean = np.mean(data, axis=0)
        return self._set_dtype(np.reshape(t_mean, [-1]))

    def _get_block(self, i_blk, cols=slice(None)):
        '''Snapshots of block ``i_blk``, flattened to ``(n_dft, nxv)``, at
        the flat spatial columns ``cols``.'''
        offset = min(i_blk * (self._n_dft - self._n_overlap) + self._n_dft,
                     self._nt) - self._n_dft
        q_blk = self.data.reshape(self._nt, -1)[offset:offset + self._n_dft,
                                                cols].copy()
        return q_blk, offset

    def _compute_blocks(self, i_blk, cols=slice(None)):
        '''
        Windowed, mean-subtracted DFT of one block.

        The full two-sided spectrum is always required: BMD couples ``f1``,
        ``f2`` and ``f1 + f2``, and the difference-interaction regions need
        negative frequencies. There is therefore no real-signal ``rfft`` path.
        '''
        q_blk, offset = self._get_block(i_blk, cols)
        q_blk = q_blk - self._t_mean[cols]

        q_blk = q_blk * self._window
        q_blk = self._set_dtype(q_blk)
        q_blk_hat = (self._win_weight / self._n_dft) * np.fft.fft(q_blk, axis=0)
        return np.fft.fftshift(q_blk_hat, axes=0), offset

    def _compute_qhat(self, needed=None, comm=None):
        '''
        Fourier realizations for every frequency row any triad refers to.

        Only the rows in ``triads.freq_needed`` are retained; for a bispectrum
        restricted by ``max_freq_idx`` that is a small fraction of ``n_dft``.

        :param needed: frequency rows to retain instead, as used by
            :meth:`reconstruct_modes`. Default is ``triads.freq_needed``.
        :param comm: node communicator. With one, ``q_hat`` is held once in
            the node's shared memory and every rank transforms its own share
            of the spatial points; ``self.data`` must then be shared too.
            Default is None (private arrays, all points on this rank).

        :return: mapping from frequency row to its ``(*block_shape, n_blocks)``
            array of realizations, ``block_shape`` being :meth:`_block_shape`.
        :rtype: dict
        '''
        self._pr0(f' ')
        self._pr0(f'Calculating temporal DFT')
        self._pr0(f'------------------------------------')

        block_shape = self._block_shape()
        if needed is None:
            needed = self._triads.freq_needed
        # one shared allocation for every row; q_hat maps a row to its view
        q_all = self._shared_empty(
            (len(needed), *block_shape, self._n_blocks), self._complex,
            'q_hat', comm)
        q_hat = {int(f): q_all[j] for j, f in enumerate(needed)}
        # each row viewed as (nxv, n_blocks), whatever its block_shape
        q_flat = {f: q.reshape(self._nxv, self._n_blocks)
                  for f, q in q_hat.items()}
        cols = self._my_cols(comm)
        for i_blk in range(0, self._n_blocks):
            st = time.time()
            q_blk_hat, offset = self._compute_blocks(i_blk, cols)
            for f in needed:
                q_flat[int(f)][cols, i_blk] = q_blk_hat[f]
            self._pr0(f'block {i_blk + 1}/{self._n_blocks} '
                      f'({offset}:{self._n_dft + offset});  '
                      f'Elapsed time: {time.time() - st} s.')
        utils_par.barrier(comm)
        self._pr0(f'------------------------------------')
        return q_hat

    def _my_cols(self, comm):
        '''The flat ``(nxv,)`` columns of this rank's share of the spatial
        points: all variables of a contiguous range of points.'''
        start, stop = utils_par.split_range(self._nx, comm)
        return slice(start * self._nv, stop * self._nv)

    def _shared_empty(self, shape, dtype, key, comm):
        '''An array held once in the shared memory of the node communicator
        ``comm`` (private if None); its window is kept under ``key`` until
        :meth:`_free_shared`.'''
        arr, win = utils_par.shared_empty(shape, dtype, comm)
        if win is not None:
            self._wins.setdefault(key, []).append(win)
        return arr

    def _free_shared(self, key):
        '''Release the shared windows under ``key``; every array mapping
        them must have been dropped.'''
        for win in self._wins.pop(key, []):
            utils_par.free_shared(win, self._node_comm)

    def _load_shared(self, data_list):
        '''
        Read the data on the first rank of each node, into memory shared by
        the ranks of that node. Serial, or alone on the node, the data is
        returned as read, without the copy into shared memory.
        '''
        node = self._node_comm
        if node is None or node.size == 1:
            return utils_io.get_data_array(
                data_list, self._xdim, self._nv, dtype=self._float)
        loaded = []

        def read():
            loaded.append(utils_io.get_data_array(
                data_list, self._xdim, self._nv, dtype=self._float))
            return loaded[0].shape

        shape = self._on_node_root(read)
        shared = self._shared_empty(shape, self._float, 'data', node)
        if loaded:
            shared[...] = loaded.pop()
        utils_par.barrier(node)
        return shared

    def _on_node_root(self, func):
        '''
        Call ``func()`` on the first rank of each node only and broadcast its
        result to the other ranks of the node, so that an input given on that
        rank alone is seen by all. An exception it raises is re-raised on
        every rank of the node, rather than leaving the others waiting.
        '''
        node = self._node_comm
        if node is None:
            return func()
        result = None
        if node.rank == 0:
            try:
                result = func()
            except Exception as err:
                result = err
        result = node.bcast(result, root=0)
        if isinstance(result, Exception):
            raise result
        return result

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
        # with save_modes_top the modes to write are known only once L is
        # complete, so they are written after the reduction instead
        save_in_loop = self._save_modes and self._save_modes_top is None
        if self._store_modes:
            # once per node, each rank writing the rows of its own triads;
            # the window is never freed: the caller may still hold a view of
            # self.modes, so it lives until MPI finalizes
            self._modes, win = utils_par.shared_zeros(
                (n_triads, self._n_mode_comp, *self._mode_shape),
                self._complex, self._node_comm)
            if win is not None:
                self._wins.setdefault('modes', []).append(win)

        my_triads = utils_par.distribute_indices(n_triads, self._comm)
        st = time.time()
        for n, i in enumerate(my_triads):
            i = int(i)
            q_sum, q_prod, weights = self._triad_matrices(q_hat, i)

            # cross-spectral density between the sum interaction and the
            # quadratic term; (n_blocks, n_blocks)
            B = q_sum.conj().T @ (q_prod * weights) / self._n_blocks

            # the optimizer works in double precision regardless of the
            # requested dtype -- B is only (n_blocks, n_blocks), so the accuracy
            # is free -- but the results are stored at the requested precision
            B = B.astype(np.complex128, copy=False)
            r, a = optimizers.mengi_overton(
                B, tol=self._solver_tol, n_it_max=self._solver_n_it_max)
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

            if self._store_modes or save_in_loop:
                psi = self._triad_modes(q_hat, i, a, psi_sum, psi_prod,
                                        weights)
                if self._store_modes:
                    self._modes[i] = psi
                if save_in_loop:
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
            utils_par.allreduce_across_nodes(self._modes, self._comm,
                                             self._node_comm)

        # restore the reference semantics: entries that are not triads are NaN
        outside = ~self._triads.mask
        self._L[outside] = np.nan
        self._T[outside] = np.nan
        if not self._compute_transfer:
            self._T[:] = np.nan

        if self._save_modes and self._save_modes_top is not None:
            self._save_top_modes(q_hat, my_triads)
        utils_par.barrier(self._comm)

    def _triad_modes(self, q_hat, i, a, psi_sum, psi_prod, weights):
        '''The normalized modes ``(n_comp, *mode_shape)`` of triad ``i`` for
        the expansion vector ``a``.'''
        mode_stack = [utils_bmd.normalize_mode(psi_sum, weights),
                      utils_bmd.normalize_mode(psi_prod, weights)]
        if self._constituent_modes:
            q_k, q_l = self._constituent_matrices(q_hat, i)
            mode_stack.append(utils_bmd.normalize_mode(q_k @ a, weights))
            mode_stack.append(utils_bmd.normalize_mode(q_l @ a, weights))
        return self._unflatten_modes(np.stack(mode_stack))

    def _save_top_modes(self, q_hat, my_triads):
        '''
        Write the modes of the ``save_modes_top`` strongest triads, ranked as
        :func:`~pybmd.bmd.postproc.top_triads` ranks them, and their indices
        to ``modes/saved_triad_idx.npy``. Each rank writes the triads it
        owns, rebuilding the modes from ``q_hat`` and the reduced ``coeffs``.
        '''
        saved = np.sort(postproc.top_triads(
            self, n=self._n_saved_triads)['triad_idx'])
        for i in np.intersect1d(saved, my_triads):
            i = int(i)
            psi = (self._modes[i] if self._store_modes
                   else self._modes_from_coeffs(q_hat, i))
            self._save_modes_at_triad(i, psi)
        if self._rank == 0:
            np.save(os.path.join(self._modes_dir, 'saved_triad_idx.npy'),
                    saved)
        self._saved_triad_idx = saved

    def _modes_from_coeffs(self, q_hat, i):
        '''The modes of triad ``i`` rebuilt from ``q_hat`` and the reduced
        ``coeffs``, identical to those formed in the triad loop.'''
        q_sum, q_prod, weights = self._triad_matrices(q_hat, i)
        a = self._coeffs[i]
        return self._triad_modes(q_hat, i, a, q_sum @ a, q_prod @ a, weights)

    def reconstruct_modes(self, data_list, triad_idx):
        '''
        Rebuild the modes of triads whose modes were not saved, from the data
        and ``coeffs``, without re-running the optimizer.

        Only the DFT rows of the requested triads are recomputed, by the same
        blocking, window and mean as :meth:`fit`, so the result is identical to
        the modes ``fit`` would have saved (at the computation ``dtype``, not
        ``save_dtype``). Must be called on the fitted object.

        :param data_list: the data passed to :meth:`fit`, or path(s) to it.
        :param triad_idx: an index into the per-triad arrays, as returned by
            ``self.triads.find(k, l)``, or a sequence of them.

        :return: the modes of shape ``(n_comp, *mode_shape)`` for one index,
            or ``(len(triad_idx), n_comp, *mode_shape)`` for a sequence. See
            :meth:`get_modes_at_triad` for ``n_comp`` and the index order.
        :rtype: numpy.ndarray
        '''
        if not hasattr(self, '_coeffs'):
            raise RuntimeError('reconstruct_modes needs a fitted object; '
                               'call fit() first.')
        scalar = np.ndim(triad_idx) == 0
        idx = np.atleast_1d(np.asarray(triad_idx, dtype=int))
        if idx.size and (idx.min() < 0 or idx.max() >= self.n_triads):
            raise IndexError(
                f'triad_idx must be in [0, {self.n_triads}); got {triad_idx}.')

        data = utils_io.get_data_array(
            data_list, self._xdim, self._nv, dtype=self._float)
        if data.shape != self._shape:
            raise ValueError(
                f'data has shape {data.shape}, but fit() was run on '
                f'{self._shape}; pass the same data.')
        t = self._triads
        needed = np.unique(np.concatenate(
            [t.f1_idx[idx], t.f2_idx[idx], t.f3_idx[idx]]))
        # _compute_blocks reads self.data, which fit() deletes after the DFT
        self.data = data
        try:
            # private, on this rank alone: reconstruct_modes need not be
            # called collectively
            q_hat = self._compute_qhat(needed, comm=None)
        finally:
            del self.data
        psi = np.stack([self._modes_from_coeffs(q_hat, int(i)) for i in idx])
        return psi[0] if scalar else psi

    def _resolve_save_modes_top(self):
        '''Number of triads whose modes are written: every triad, or
        ``save_modes_top`` of the candidates ranked by ``top_triads`` (those
        with ``k != 0`` and ``l != 0``).'''
        top = self._save_modes_top
        if top is None:
            return self.n_triads
        n_cand = int(np.count_nonzero(
            (self._triads.k != 0) & (self._triads.l != 0)))
        if isinstance(top, float):
            return min(math.ceil(top * n_cand), n_cand)
        return min(top, n_cand)

    def _save_modes_at_triad(self, i_triad, psi):
        '''Write the modes ``(n_comp, *mode_shape)`` of one triad to
        ``modes/triad_idx_{i:08d}.npy``.'''
        path = os.path.join(self._modes_dir, f'triad_idx_{i_triad:08d}.npy')
        np.save(path, self._to_save_dtype(psi))

    def get_modes_at_triad(self, triad_idx, data=None):
        '''
        Load the modes of one triad.

        :param int triad_idx: index into the per-triad arrays, as returned by
            ``self.triads.find(k, l)``.
        :param data: the data passed to :meth:`fit`, or path(s) to it. If
            given, modes that were not saved (``save_modes=False``, or outside
            ``save_modes_top``) are rebuilt with :meth:`reconstruct_modes`
            instead of raising. Default is None.

        :return: the modes, of shape ``(n_comp, *xshape, nv)`` (``n_state`` in
            place of ``nv`` for :class:`~pybmd.bmd.cross.Cross`). ``n_comp`` is
            2, or 4 with ``params['constituent_modes']`` set. Index 0 is the
            sum-interaction mode :math:`\\phi_{k+l}` and index 1 the
            quadratic-term mode :math:`\\phi_{k \\circ l}`; with
            ``constituent_modes``, indices 2 and 3 are the constituent modes
            :math:`\\phi_k` and :math:`\\phi_l`.
        :rtype: numpy.ndarray
        '''
        if self._store_modes:
            return self._modes[triad_idx]
        path = os.path.join(self._modes_dir, f'triad_idx_{triad_idx:08d}.npy')
        not_top = (self._save_modes and self._save_modes_top is not None
                   and triad_idx not in self._saved_triad_idx)
        if not not_top and os.path.exists(path):
            return np.load(path)
        if data is not None:
            return self.reconstruct_modes(data, triad_idx)
        if not_top:
            raise FileNotFoundError(
                f'Triad {triad_idx} is not among the '
                f'{self._n_saved_triads} strongest triads written with '
                f'save_modes_top={self._save_modes_top!r}; see '
                f'modes/saved_triad_idx.npy, or pass data= to rebuild it.')
        raise FileNotFoundError(
            f'No modes stored for triad {triad_idx}. Was fit() run with '
            f'save_modes enabled? Pass data= to rebuild them.')

    def get_modes_at_freqs(self, k, l, data=None):
        '''
        Load the modes of the triad ``(k, l, k+l)``.

        :param int k: integer frequency index of f1.
        :param int l: integer frequency index of f2.
        :param data: see :meth:`get_modes_at_triad`.

        :return: the modes, of shape ``(n_comp, *xshape, nv)``. See
            :meth:`get_modes_at_triad` for ``n_comp`` and the index order.
        :rtype: numpy.ndarray
        '''
        return self.get_modes_at_triad(self._triads.find(k, l), data=data)

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

        if self._rank == 0:
            # arrays first: the YAML dump is the one step that can fail on an
            # unexpected value type, and it must not take the results with it
            np.savez(os.path.join(self._savedir_sim, 'bispectrum.npz'),
                     L=self._to_save_dtype(self._L),
                     T=self._to_save_dtype(self._T),
                     freq=self.freq, f_idx=self.f_idx)
            self._triads.to_npz(os.path.join(self._savedir_sim, 'triads.npz'))
            np.save(os.path.join(self._savedir_sim, 'coeffs.npy'),
                    self._to_save_dtype(self._coeffs))
            np.save(os.path.join(self._savedir_sim, 'weights.npy'),
                    self._to_save_dtype(self._weights))
            np.save(os.path.join(self._savedir_sim, 'ltm_modes.npy'),
                    self._to_save_dtype(self._t_mean))
            path_params = os.path.join(self._savedir_sim, 'params_modes.yaml')
            with open(path_params, 'w') as f:
                yaml.dump(_yaml_safe(self._params), f)
            print(f'Parameters dictionary saved in: {path_params}',
                  flush=True)
            print(f'Bispectrum saved in: '
                  f'{os.path.join(self._savedir_sim, "bispectrum.npz")}',
                  flush=True)
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

    def _to_save_dtype(self, d):
        '''Cast a result array to the on-disk precision (``save_dtype``).'''
        if np.issubdtype(d.dtype, np.complexfloating):
            return d.astype(self._save_complex, copy=False)
        if np.issubdtype(d.dtype, np.floating):
            return d.astype(self._save_float, copy=False)
        return d

    def _print_parameters(self):
        '''Display parameter summary.'''
        self._pr0(f'')
        self._pr0(f'BMD parameters')
        self._pr0(f'------------------------------------')
        self._pr0(f'Problem size (real)      : {self._pb_size_f:.2f} GB')
        self._pr0(f'Q_hat size               : {self._qhat_size_gb:.2f} GB')
        self._pr0(f'Modes size (all triads)  : {self._modes_size_gb:.2f} GB')
        self._pr0(f'Constituent modes        : {self._constituent_modes}')
        self._pr0(f'Triads with saved modes  : {self._n_saved_triads}'
                  f'{"" if self._save_modes else " (save_modes off)"}')
        self._pr0(f'Data type for real       : {self._float}')
        self._pr0(f'Data type for complex    : {self._complex}')
        self._pr0(f'Data type on disk        : {self._save_dtype}')
        self._pr0(f'No. snapshots per block  : {self._n_dft}')
        self._pr0(f'Block overlap            : {self._n_overlap}')
        self._pr0(f'No. of blocks            : {self._n_blocks}')
        self._pr0(f'Windowing fct. (time)    : {self._window_name}')
        self._pr0(f'Weighting fct. (space)   : {self._weights_name}')
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
        self._pr0(f'Solver (x*Ax)            : MengiOverton '
                  f'(tol {self._solver_tol}, n_it_max {self._solver_n_it_max})')
        self._pr0(f'MPI ranks                : {self._size}')
        self._pr0(f'Results to be saved in   : {self._savedir}')
        self._pr0(f'------------------------------------')
        self._pr0(f'')
