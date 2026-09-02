'''
Module implementing bispectrum-specific post-processing: result-directory
loaders, the mode bispectrum map over the ``f1``-``f2`` plane, and the mode
panels for a chosen triad.

This is the only module that knows about triads and regions; generic getters
and field plots live in :mod:`pybmd.utils.postproc`.
'''
from dataclasses import dataclass
import os

import numpy as np
import yaml

from pybmd.bmd.utils import Triads
from pybmd.utils.postproc import get_bispectrum, get_modes_at_triad
from pybmd.utils.postproc import _save_show_plots, _symmetric_levels


__all__ = [
    'BMDResults',
    'find_result_directories',
    'resolve_results_path',
    'load_results',
    'top_triads',
    'plot_mode_bispectrum',
    'plot_energy_transfer',
    'plot_triad_modes',
    'plot_mode_bispectrum_from_dir',
    'plot_energy_transfer_from_dir',
    'plot_triad_modes_from_dir',
    'plot_peak_triad_from_dir',
]


@dataclass(frozen=True)
class BMDResults:
    '''
    Results loaded from a BMD/CBMD results directory.

    ``path`` is the simulation directory containing ``bispectrum.npz`` and
    ``triads.npz``. For a standard run this is the nested directory named like
    ``nfft64_novlp32_nblks10``.
    '''
    path: str
    L: np.ndarray
    T: np.ndarray
    freq: np.ndarray
    f_idx: np.ndarray
    triads: Triads
    params: dict

    @property
    def bispectrum(self):
        '''Alias of :attr:`L`.'''
        return self.L

    @property
    def energy_transfer(self):
        '''Alias of :attr:`T`.'''
        return self.T

    def find_triad(self, k, l):
        '''Triad index of ``(k, l, k+l)``.'''
        return self.triads.find(k, l)

    def get_modes_at_triad(self, triad_idx):
        '''Load the two modes of one triad from disk.'''
        return get_modes_at_triad(self.path, triad_idx)

    def get_modes_at_freqs(self, k, l):
        '''Load the modes of the triad ``(k, l, k+l)`` from disk.'''
        return self.get_modes_at_triad(self.find_triad(k, l))


def find_result_directories(path):
    '''
    Find BMD/CBMD simulation result directories under ``path``.

    :param str path: either a simulation directory itself or a parent
        ``savedir`` containing ``nfft*_novlp*_nblks*`` subdirectories.

    :return: sorted absolute paths containing ``bispectrum.npz``.
    :rtype: list(str)
    '''
    path = os.path.abspath(os.fspath(path))
    if os.path.exists(os.path.join(path, 'bispectrum.npz')):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f'no such results directory: {path}')

    matches = []
    for root, dirs, files in os.walk(path):
        if 'bispectrum.npz' in files:
            matches.append(root)
            dirs[:] = []
    return sorted(matches)


def resolve_results_path(path, latest=False):
    '''
    Resolve ``path`` to one concrete simulation result directory.

    If ``path`` already contains ``bispectrum.npz`` it is returned as-is. If it
    is a parent directory with exactly one result below it, that child is used.
    With ``latest=True``, the newest matching result directory is selected.
    '''
    matches = find_result_directories(path)
    if not matches:
        raise FileNotFoundError(
            f'no bispectrum.npz found under {os.path.abspath(os.fspath(path))}')
    if len(matches) == 1:
        return matches[0]
    if latest:
        return max(matches, key=os.path.getmtime)
    msg = '\n'.join(matches[:10])
    more = '' if len(matches) <= 10 else f'\n... and {len(matches) - 10} more'
    raise ValueError(
        f'found {len(matches)} result directories. Pass one of them directly, '
        f'or set latest=True:\n{msg}{more}')


def load_results(path, latest=False):
    '''
    Load BMD/CBMD arrays and triad metadata from a results directory.

    :param str path: the simulation directory, or a parent savedir containing
        one simulation directory.
    :param bool latest: choose the newest result when ``path`` contains more
        than one. Default is False.

    :return: a :class:`BMDResults` bundle.
    :rtype: BMDResults
    '''
    results_path = resolve_results_path(path, latest=latest)
    L, T, freq, f_idx = get_bispectrum(results_path)
    triads_path = os.path.join(results_path, 'triads.npz')
    if not os.path.exists(triads_path):
        raise FileNotFoundError(f'missing triads metadata: {triads_path}')
    triads = Triads.from_npz(triads_path)

    params_path = os.path.join(results_path, 'params_modes.yaml')
    params = {}
    if os.path.exists(params_path):
        with open(params_path) as f:
            params = yaml.load(f, Loader=yaml.FullLoader) or {}
    return BMDResults(results_path, L, T, freq, f_idx, triads, params)


def top_triads(results, n=10, quantity='L', exclude_zero=True):
    '''
    Return the strongest triads in a loaded result.

    :param BMDResults or str results: loaded results, or a directory accepted
        by :func:`load_results`.
    :param int n: number of triads to return.
    :param str quantity: ``'L'`` for mode bispectrum or ``'T'`` for energy
        transfer magnitude.
    :param bool exclude_zero: exclude triads with ``k == 0`` or ``l == 0``.

    :return: a structured array with triad index, integer/physical
        frequencies, region, and value.
    :rtype: numpy.ndarray
    '''
    if not isinstance(results, BMDResults):
        results = load_results(results)

    quantity = quantity.upper()
    if quantity == 'L':
        grid = results.L
    elif quantity == 'T':
        grid = results.T
    else:
        raise ValueError("quantity must be 'L' or 'T'.")

    t = results.triads
    values = np.abs(grid[t.f1_idx, t.f2_idx])
    keep = np.isfinite(values)
    if exclude_zero:
        keep &= (t.k != 0) & (t.l != 0)
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return np.empty(0, dtype=_top_triad_dtype())

    order = idx[np.argsort(values[idx])[::-1][:int(n)]]
    out = np.empty(order.size, dtype=_top_triad_dtype())
    out['triad_idx'] = order
    out['k'] = t.k[order]
    out['l'] = t.l[order]
    out['kl'] = t.kl[order]
    out['f1'] = t.f1[order]
    out['f2'] = t.f2[order]
    out['f3'] = t.f3[order]
    out['region'] = t.region[order]
    out['value'] = values[order]
    return out


def _top_triad_dtype():
    return [
        ('triad_idx', np.int64),
        ('k', np.int64),
        ('l', np.int64),
        ('kl', np.int64),
        ('f1', np.float64),
        ('f2', np.float64),
        ('f3', np.float64),
        ('region', np.int64),
        ('value', np.float64),
    ]


def triad_label(k, l):
    '''
    LaTeX label for the triplet ``(k, l, k+l)``.

    :param int k: integer frequency index of f1.
    :param int l: integer frequency index of f2.

    :return: the label.
    :rtype: str
    '''
    return rf'$(k,l,k{{+}}l) = ({k},{l},{k + l})$'


def plot_mode_bispectrum(L, freq, log=True, levels=None, xlim=None, ylim=None,
                         cmap='viridis', mark=None, figsize=(6, 6), title='',
                         xlabel=r'$f_1$', ylabel=r'$f_2$', path=None,
                         filename=None, ax=None, extend='both',
                         extendrect=True):
    '''
    Contour the mode bispectrum over the ``f1``-``f2`` plane.

    :param numpy.ndarray L: the bispectrum, of shape ``(n_freq, n_freq)``.
        Entries outside the computed triads are expected to be NaN and are
        masked out.
    :param numpy.ndarray freq: the frequency axis.
    :param bool log: plot ``log|L|`` rather than ``|L|``. Default is True.
    :param levels: contour levels, or the number of them. Default is 100.
    :param mark: triads to annotate, as a list of ``(f1, f2)`` pairs.
    :param str extend: contour extension mode. Default is ``'both'`` so values
        outside explicit contour levels are still colored.
    :param bool extendrect: draw colorbar extensions as rectangles rather than
        triangles. Default is True.
    :param matplotlib.axes.Axes ax: axes to draw on. A new figure is created
        if omitted.

    :return: the axes drawn on.
    :rtype: matplotlib.axes.Axes
    '''
    import matplotlib.pyplot as plt

    f1, f2 = np.meshgrid(freq, freq, indexing='ij')
    field = np.abs(L)
    if log:
        with np.errstate(divide='ignore', invalid='ignore'):
            field = np.log(field)
    field = np.ma.masked_invalid(field)

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=figsize)
    im = ax.contourf(f1, f2, field,
                     levels=levels if levels is not None else 100,
                     cmap=cmap, extend=extend)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or 'Mode bispectrum')
    ax.set_aspect('equal')

    # default to the extent of the computed triads, which is usually a small
    # part of the full plane
    valid = np.isfinite(np.asarray(np.abs(L)))
    if xlim is None and valid.any():
        xlim = (f1[valid].min(), f1[valid].max())
    if ylim is None and valid.any():
        ylim = (f2[valid].min(), f2[valid].max())
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    if mark:
        for m_f1, m_f2 in mark:
            ax.plot(m_f1, m_f2, 'o', ms=8, mfc='none', mec='r', mew=1.5)

    ax.figure.colorbar(im, ax=ax, extendrect=extendrect,
                       label=r'$\log|\lambda_1|$' if log else r'$|\lambda_1|$')
    if created:
        _save_show_plots(filename, path, plt)
    return ax


def plot_energy_transfer(T, freq, **kwargs):
    '''
    Contour the energy-transfer term over the ``f1``-``f2`` plane.

    :param numpy.ndarray T: the energy transfer, of shape ``(n_freq, n_freq)``.
    :param numpy.ndarray freq: the frequency axis.

    See :func:`plot_mode_bispectrum` for the remaining arguments.
    '''
    kwargs.setdefault('log', False)
    kwargs.setdefault('title', 'Energy transfer')
    return plot_mode_bispectrum(T.astype(complex), freq, **kwargs)


def plot_triad_modes(modes, k, l, x1=None, x2=None, vars_idx=(0,),
                     cmap='RdBu_r', cmap_prod='bone_r', figsize=(10, 9),
                     facecolor=None, xlim=None, ylim=None,
                     xlabel=r'$x_1$', ylabel=r'$x_2$',
                     tight_layout=True, extend='both', extendrect=True,
                     path=None, filename=None):
    '''
    Plot the two bispectral modes of a triad and their interaction map.

    Rows are the sum-interaction mode :math:`\\phi_{k+l}`, the quadratic-term
    mode :math:`\\phi_{k \\circ l}`, and the magnitude of their product, which
    localizes where the triadic interaction takes place. Columns are variables.

    :param numpy.ndarray modes: modes of one triad, of shape
        ``(2, n1, n2, nv)``, as returned by ``get_modes_at_triad``.
    :param int k: integer frequency index of f1, used for the title.
    :param int l: integer frequency index of f2, used for the title.
    :param numpy.ndarray x1: first coordinate. Default is the index.
    :param numpy.ndarray x2: second coordinate. Default is the index.
    :param vars_idx: variables to plot.
    :param facecolor: background color for the figure and axes. Default is
        None, leaving Matplotlib's default unchanged.
    :param xlim: x-axis limits. Default is Matplotlib's auto limits.
    :param ylim: y-axis limits. Default is Matplotlib's auto limits.
    :param str xlabel: x-axis label. Default is ``'$x_1$'``.
    :param str ylabel: y-axis label. Default is ``'$x_2$'``.
    :param bool tight_layout: call ``fig.tight_layout()``. Default is True.
    :param str extend: contour extension mode. Default is ``'both'`` so values
        outside explicit contour levels are still colored.
    :param bool extendrect: draw colorbar extensions as rectangles rather than
        triangles. Default is True.

    :return: the figure.
    :rtype: matplotlib.figure.Figure
    '''
    import matplotlib.pyplot as plt

    if modes.ndim != 4:
        raise ValueError(
            f'plot_triad_modes needs modes of shape (2, n1, n2, nv); got '
            f'{modes.shape}. Only two-dimensional data can be contoured.')
    vars_idx = list(vars_idx)
    if x1 is None:
        x1 = np.arange(modes.shape[1])
    if x2 is None:
        x2 = np.arange(modes.shape[2])

    titles = [r'$\phi_{k+l}$', r'$\phi_{k \circ l}$',
              r'$|\phi_{k \circ l} \cdot \phi_{k+l}|$']
    fig, axes = plt.subplots(3, len(vars_idx), figsize=figsize, squeeze=False)
    if facecolor is not None:
        fig.patch.set_facecolor(facecolor)
    for c, iv in enumerate(vars_idx):
        fields = [np.real(modes[0, ..., iv]),
                  np.real(modes[1, ..., iv]),
                  np.abs(modes[0, ..., iv] * modes[1, ..., iv])]
        for r, field in enumerate(fields):
            ax = axes[r][c]
            if facecolor is not None:
                ax.set_facecolor(facecolor)
            if r < 2:
                lv, cm = _symmetric_levels(field), cmap
            else:
                m = np.max(np.abs(field)) or 1.0
                lv, cm = m * np.linspace(0, 1, 257), cmap_prod
            im = ax.contourf(x1, x2, field.T, levels=lv, cmap=cm,
                             extend=extend)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.set_aspect('equal')
            ax.set_title(f'{titles[r]}  var {iv}')
            fig.colorbar(im, ax=ax, extendrect=extendrect)
    fig.suptitle(triad_label(k, l))
    if tight_layout:
        fig.tight_layout()
    _save_show_plots(filename, path, plt)
    return fig


def plot_mode_bispectrum_from_dir(results_path, latest=False, **kwargs):
    '''
    Load a results directory and contour its mode bispectrum.

    ``kwargs`` are forwarded to :func:`plot_mode_bispectrum`.
    '''
    results = load_results(results_path, latest=latest)
    return plot_mode_bispectrum(results.L, results.freq, **kwargs)


def plot_energy_transfer_from_dir(results_path, latest=False, **kwargs):
    '''
    Load a results directory and contour its energy-transfer map.

    ``kwargs`` are forwarded to :func:`plot_energy_transfer`.
    '''
    results = load_results(results_path, latest=latest)
    return plot_energy_transfer(results.T, results.freq, **kwargs)


def plot_triad_modes_from_dir(results_path, k=None, l=None, triad_idx=None,
                              latest=False, x1=None, x2=None, vars_idx=(0,),
                              **kwargs):
    '''
    Load and plot modes for one triad from a results directory.

    Select the triad either by ``triad_idx`` or by the integer frequency pair
    ``k, l``. ``kwargs`` are forwarded to :func:`plot_triad_modes`.
    '''
    results = load_results(results_path, latest=latest)
    if triad_idx is None:
        if k is None or l is None:
            raise ValueError('pass either triad_idx or both k and l.')
        triad_idx = results.find_triad(k, l)
    else:
        triad_idx = int(triad_idx)
        k = int(results.triads.k[triad_idx])
        l = int(results.triads.l[triad_idx])
    modes = results.get_modes_at_triad(triad_idx)
    return plot_triad_modes(modes, k, l, x1=x1, x2=x2, vars_idx=vars_idx,
                            **kwargs)


def plot_peak_triad_from_dir(results_path, n=1, quantity='L', latest=False,
                             exclude_zero=True, x1=None, x2=None,
                             vars_idx=(0,), mark=True,
                             bispectrum_kwargs=None, modes_kwargs=None):
    '''
    Plot the bispectrum and modes for the strongest triad in a result.

    :return: ``(bispectrum_axes, modes_figure, top)``, where ``top`` is the
        one-row structured array returned by :func:`top_triads`.
    '''
    results = load_results(results_path, latest=latest)
    top = top_triads(results, n=n, quantity=quantity,
                    exclude_zero=exclude_zero)
    if top.size == 0:
        raise ValueError('no finite triads found to plot.')

    peak = top[0]
    bispectrum_kwargs = dict(bispectrum_kwargs or {})
    modes_kwargs = dict(modes_kwargs or {})
    if mark:
        bispectrum_kwargs.setdefault('mark', [(float(peak['f1']),
                                               float(peak['f2']))])

    ax = plot_mode_bispectrum(results.L, results.freq, **bispectrum_kwargs)
    fig = plot_triad_modes_from_dir(
        results.path, triad_idx=int(peak['triad_idx']), x1=x1, x2=x2,
        vars_idx=vars_idx, **modes_kwargs)
    return ax, fig, top
