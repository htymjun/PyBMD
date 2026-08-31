'''
Module implementing the frequency and triad machinery of the BMD, plus the
temporal windows.  These are free functions, usable without instantiating a
decomposition class.

Triads are frequency triplets ``{f1, f2, f1+f2}``, equivalently index triplets
``(k, l, k+l)``.  The region of the ``f1``-``f2`` plane each triplet falls in
is numbered as in Schmidt (2020)::

          f2 or l
             ^
     ________|
     |\\      |\\
     |  \\  7 |  \\
     | 6  \\  | 8 /\\
     |      \\| / 1  \\
 ----+-------+-------+-> f1 or k
      \\  5 / |\\      |
        \\/ 4 |  \\  2 |
          \\  | 3  \\  |
            \\|______\\|
             |
'''
from dataclasses import dataclass

import numpy as np


__all__ = ['Triads', 'triad_indices', 'freq_axis', 'get_window', 'boxcar_window',
           'hamming_window', 'hann_window']

N_REGIONS = 8


def hamming_window(n_dft):
    '''
    Standard Hamming window of length ``n_dft``.

    :param int n_dft: length of the window.

    :return: the window.
    :rtype: numpy.ndarray
    '''
    x = np.arange(0, n_dft, 1)
    return 0.54 - 0.46 * np.cos(2 * np.pi * x / (n_dft - 1))


def hann_window(n_dft):
    '''
    Standard Hann window of length ``n_dft``.

    :param int n_dft: length of the window.

    :return: the window.
    :rtype: numpy.ndarray
    '''
    x = np.arange(0, n_dft, 1)
    return 0.5 * (1 - np.cos(2 * np.pi * x / (n_dft - 1)))


def boxcar_window(n_dft):
    '''
    Rectangular window of length ``n_dft``, i.e. no windowing at all.

    Useful when the signal is periodic in the block length, since it then
    introduces no spectral leakage.

    :param int n_dft: length of the window.

    :return: the window.
    :rtype: numpy.ndarray
    '''
    return np.ones(n_dft)


def get_window(window, n_dft):
    '''
    Resolve the ``window`` parameter into an array and a name.

    :param window: 'hamming', 'hann', 'boxcar', or an array of length ``n_dft``.
    :type window: str or numpy.ndarray
    :param int n_dft: number of snapshots per block.

    :return: the window and its name.
    :rtype: tuple(numpy.ndarray, str)
    '''
    if isinstance(window, str):
        name = window.lower()
        if name == 'hamming':
            return hamming_window(n_dft), 'hamming'
        if name == 'hann':
            return hann_window(n_dft), 'hann'
        if name in ('boxcar', 'rectangular', 'none'):
            return boxcar_window(n_dft), 'boxcar'
        raise ValueError(
            f'Unknown window {window!r}; use "hamming", "hann", "boxcar", '
            f'or an array.')
    w = np.asarray(window, dtype=float).ravel()
    if w.size != n_dft:
        raise ValueError(
            f'window has length {w.size} but n_dft is {n_dft}.')
    return w, 'user specified'


def freq_axis(n_dft, dt=None):
    '''
    Two-sided, ``fftshift``-ed frequency axis.

    :param int n_dft: number of snapshots per block.
    :param float dt: time-step between snapshots. If None, ``1/n_dft`` is used,
        so that ``freq`` coincides with the integer frequency index.

    :return: the physical frequency, the signed integer frequency index, and
        the Nyquist index.
    :rtype: tuple(numpy.ndarray, numpy.ndarray, int)
    '''
    if dt is None:
        dt = 1.0 / n_dft
    f_idx = np.fft.fftshift(np.fft.fftfreq(n_dft, d=1.0 / n_dft))
    f_idx = np.rint(f_idx).astype(np.int64)
    freq = f_idx / dt / n_dft
    f_nyq_idx = int(-f_idx[0])
    return freq, f_idx, f_nyq_idx


def _region_masks(k, l, regions):
    '''
    Boolean mask per requested region over the ``(k, l)`` index grid, using the
    predicates of Schmidt's ``bmd.m``. ``k`` and ``l`` are 2-D index grids.
    '''
    predicates = {
        1: (k >= 0) & (l >= 0) & (k >= l),
        2: (k >= 0) & (l <= 0) & (k >= np.abs(l)),
        3: (k >= 0) & (l <= 0) & (k <= np.abs(l)),
        4: (k <= 0) & (l <= 0) & (k >= l),
        5: (k <= 0) & (l <= 0) & (k <= l),
        6: (k <= 0) & (l >= 0) & (np.abs(k) >= l),
        7: (k <= 0) & (l >= 0) & (np.abs(k) <= l),
        8: (k >= 0) & (l >= 0) & (k <= l),
    }
    # ascending order matters: the reference evaluates its eight `if` blocks in
    # sequence, so where regions overlap the higher-numbered one wins
    return {r: predicates[r] for r in sorted(regions)}


@dataclass(frozen=True)
class Triads:
    '''
    The set of frequency triplets a decomposition is computed over, together
    with the frequency axis they index into.

    All per-triad arrays have length :attr:`n_triads` and are ordered
    row-major over the ``(i, j)`` index grid, matching the loop order of the
    reference MATLAB implementation.
    '''
    freq: np.ndarray        #: (n_freq,) physical frequency, fftshifted
    f_idx: np.ndarray       #: (n_freq,) signed integer frequency index
    f_nyq_idx: int          #: Nyquist frequency index
    f1_idx: np.ndarray      #: (n_triads,) row index of f1 into the axis
    f2_idx: np.ndarray      #: (n_triads,) row index of f2
    f3_idx: np.ndarray      #: (n_triads,) row index of f1 + f2
    region: np.ndarray      #: (n_triads,) region 1..8 of each triad
    triad_map: np.ndarray   #: (n_freq, n_freq) triad index, -1 where absent

    @property
    def n_freq(self):
        '''Number of frequencies.'''
        return self.freq.size

    @property
    def n_triads(self):
        '''Number of triads.'''
        return self.f1_idx.size

    @property
    def k(self):
        '''(n_triads,) signed integer frequency index of f1.'''
        return self.f_idx[self.f1_idx]

    @property
    def l(self):
        '''(n_triads,) signed integer frequency index of f2.'''
        return self.f_idx[self.f2_idx]

    @property
    def kl(self):
        '''(n_triads,) signed integer frequency index of f1 + f2.'''
        return self.f_idx[self.f3_idx]

    @property
    def f1(self):
        '''(n_triads,) physical frequency f1.'''
        return self.freq[self.f1_idx]

    @property
    def f2(self):
        '''(n_triads,) physical frequency f2.'''
        return self.freq[self.f2_idx]

    @property
    def f3(self):
        '''(n_triads,) physical frequency f1 + f2.'''
        return self.freq[self.f3_idx]

    @property
    def mask(self):
        '''(n_freq, n_freq) boolean mask of the computed entries.'''
        return self.triad_map >= 0

    @property
    def freq_needed(self):
        '''Sorted unique frequency rows referenced by any triad.'''
        return np.unique(np.concatenate(
            [self.f1_idx, self.f2_idx, self.f3_idx]))

    @property
    def linear_idx(self):
        '''
        (n_triads,) C-order linear index into an ``(n_freq, n_freq)`` array.

        .. note::

            The MATLAB implementation returns the Fortran-order ``sub2ind``
            value instead. Use :meth:`find` rather than a linear index.
        '''
        return np.ravel_multi_index(
            (self.f1_idx, self.f2_idx), (self.n_freq, self.n_freq))

    def row_of(self, k):
        '''
        Row of the frequency axis holding integer frequency index ``k``.

        :param int k: signed integer frequency index.

        :return: the row index.
        :rtype: int
        '''
        i = int(np.searchsorted(self.f_idx, k))
        if i >= self.n_freq or self.f_idx[i] != k:
            raise ValueError(
                f'frequency index {k} is off the axis '
                f'[{self.f_idx[0]}, {self.f_idx[-1]}].')
        return i

    def find(self, k, l):
        '''
        Triad index of the triplet ``(k, l, k+l)``.

        :param int k: integer frequency index of f1.
        :param int l: integer frequency index of f2.

        :return: index into the per-triad arrays.
        :rtype: int
        '''
        t = int(self.triad_map[self.row_of(k), self.row_of(l)])
        if t < 0:
            raise ValueError(
                f'triad (k={k}, l={l}, k+l={k + l}) was not computed. It falls '
                f'outside the requested regions '
                f'{sorted(set(self.region.tolist()))} or exceeds the frequency '
                f'index bound {int(np.abs(self.k).max())}.')
        return t

    def find_freq(self, f1, f2):
        '''
        Triad index nearest to the physical frequency doublet ``(f1, f2)``.

        :param float f1: requested first frequency.
        :param float f2: requested second frequency.

        :return: index into the per-triad arrays.
        :rtype: int
        '''
        i = int(np.argmin(np.abs(self.freq - f1)))
        j = int(np.argmin(np.abs(self.freq - f2)))
        return self.find(int(self.f_idx[i]), int(self.f_idx[j]))

    def to_npz(self, path):
        '''Save to ``path`` as a .npz archive.'''
        np.savez(path, freq=self.freq, f_idx=self.f_idx,
                 f_nyq_idx=self.f_nyq_idx, f1_idx=self.f1_idx,
                 f2_idx=self.f2_idx, f3_idx=self.f3_idx, region=self.region,
                 triad_map=self.triad_map)

    @classmethod
    def from_npz(cls, path):
        '''Load from a .npz archive written by :meth:`to_npz`.'''
        with np.load(path) as d:
            return cls(freq=d['freq'], f_idx=d['f_idx'],
                       f_nyq_idx=int(d['f_nyq_idx']), f1_idx=d['f1_idx'],
                       f2_idx=d['f2_idx'], f3_idx=d['f3_idx'],
                       region=d['region'], triad_map=d['triad_map'])


def triad_indices(n_dft, dt=None, regions=(1, 2), max_freq_idx=None):
    '''
    Build the set of triads of the ``f1``-``f2`` plane to be computed.

    A triplet ``(k, l, k+l)`` is admitted when ``|k+l| < f_nyq_idx``,
    ``|k| <= max_freq_idx``, ``|l| <= max_freq_idx``, and ``(k, l)`` falls in
    one of the requested ``regions``.

    :param int n_dft: number of snapshots per block.
    :param float dt: time-step between snapshots. Default is ``1/n_dft``.
    :param regions: regions of the bispectrum to compute, in 1..8.
        Default is ``(1, 2)``.
    :type regions: sequence(int)
    :param int max_freq_idx: bound on ``|k|`` and ``|l|``. Default is the
        Nyquist index, i.e. no restriction.

    :return: the triads.
    :rtype: Triads
    '''
    regions = tuple(int(r) for r in np.atleast_1d(regions))
    bad = [r for r in regions if r < 1 or r > N_REGIONS]
    if bad:
        raise ValueError(f'regions must lie in 1..{N_REGIONS}; got {bad}.')

    freq, f_idx, f_nyq_idx = freq_axis(n_dft, dt)
    n_freq = f_idx.size
    if max_freq_idx is None:
        max_freq_idx = f_nyq_idx
    max_freq_idx = int(max_freq_idx)

    # index grids over the (f1, f2) plane
    k, l = np.meshgrid(f_idx, f_idx, indexing='ij')
    admissible = ((np.abs(k + l) < f_nyq_idx)
                  & (np.abs(k) <= max_freq_idx)
                  & (np.abs(l) <= max_freq_idx))

    # later regions overwrite earlier ones where they overlap, matching the
    # sequence of independent `if` statements in the reference implementation
    region_grid = np.zeros((n_freq, n_freq), dtype=np.int64)
    for r, mask in _region_masks(k, l, regions).items():
        region_grid[admissible & mask] = r

    # row-major over (i, j), matching the MATLAB loop order
    f1_idx, f2_idx = np.nonzero(region_grid)
    region = region_grid[f1_idx, f2_idx]
    f3_idx = np.searchsorted(f_idx, k[f1_idx, f2_idx] + l[f1_idx, f2_idx])

    triad_map = np.full((n_freq, n_freq), -1, dtype=np.int64)
    triad_map[f1_idx, f2_idx] = np.arange(f1_idx.size)

    return Triads(freq=freq, f_idx=f_idx, f_nyq_idx=f_nyq_idx,
                  f1_idx=f1_idx, f2_idx=f2_idx, f3_idx=f3_idx,
                  region=region, triad_map=triad_map)


def _get_dtype(dtype):
    '''Real and complex numpy types for a 'double'/'single' precision name.'''
    if dtype == 'double':
        return np.float64, np.complex128
    if dtype in ('single', 'float'):
        return np.float32, np.complex64
    raise ValueError(f"dtype must be 'double' or 'single'; got {dtype!r}.")


def normalize_mode(psi, weights):
    '''
    Normalize a mode by the weighted inner product, ``psi / sqrt(psi^H W psi)``.

    :param numpy.ndarray psi: the mode, shape ``(nx,)``.
    :param numpy.ndarray weights: the weights, shape ``(nx, 1)`` or ``(nx,)``.

    :return: the normalized mode.
    :rtype: numpy.ndarray
    '''
    w = np.asarray(weights).reshape(-1)
    # vdot conjugates its first argument; take the real part explicitly, since
    # round-off can otherwise leave a tiny imaginary residue in a quantity that
    # is real and positive by construction
    nrm = np.sqrt(np.real(np.vdot(psi, psi * w)))
    if nrm <= 0:
        raise ValueError(
            'mode has non-positive weighted norm; check that weights are '
            'strictly positive.')
    return psi / nrm
