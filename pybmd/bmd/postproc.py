'''
Module implementing bispectrum-specific post-processing of a fitted
``Standard``/``Cross``: the strongest triads, the mode bispectrum map over the
``f1``-``f2`` plane, and the mode panels for a chosen triad.
'''
import os

import numpy as np


__all__ = [
    'top_triads',
    'plot_mode_bispectrum',
    'plot_triad_modes',
]


def top_triads(results, n=10, quantity='L', exclude_zero=True):
    '''
    Return the strongest triads of a fitted decomposition.

    :param results: a fitted ``Standard`` or ``Cross``.
    :param int n: number of triads to return.
    :param str quantity: ``'L'`` for mode bispectrum or ``'T'`` for energy
        transfer magnitude.
    :param bool exclude_zero: exclude triads with ``k == 0`` or ``l == 0``.

    :return: a structured array with triad index, integer/physical
        frequencies, region, and value.
    :rtype: numpy.ndarray
    '''
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

    # stable sort, so ties keep their triad order and the ranking is
    # reproducible across numpy builds
    order = idx[np.argsort(-values[idx], kind='stable')[:int(n)]]
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


def _save_figure(fig, filename, path=None):
    '''Save ``fig`` as ``path/filename``, creating ``path`` (default: the
    working directory) if needed.'''
    if path is None:
        path = os.getcwd()
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, filename), dpi=200, bbox_inches='tight')


def _save_show_plots(filename, path, plt):
    '''Save the current figure if a filename is given, otherwise show it.'''
    if filename:
        _save_figure(plt.gcf(), filename, path)
        plt.close()
    else:
        plt.show()


def _symmetric_levels(field, n_levels=257, scale=0.5):
    '''Contour levels symmetric about zero, as used for real mode fields.'''
    m = scale * np.max(np.abs(field))
    if m == 0:
        m = 1.0
    return m * np.linspace(-1, 1, n_levels)


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
                         cmap='jet', mark=None, figsize=(6, 6), title='',
                         xlabel=r'$f_1$', ylabel=r'$f_2$', path=None,
                         filename=None, ax=None, extend='both',
                         extendrect=True, cbar_label=None):
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
        if omitted. ``filename`` is honoured either way; the figure is only
        closed when this call created it.
    :param str cbar_label: colorbar label. Default names ``|lambda_1|``.

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
    if levels is None:
        levels = 100
    if cbar_label is None:
        cbar_label = r'$\log|\lambda_1|$' if log else r'$|\lambda_1|$'

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=figsize)
    im = ax.contourf(f1, f2, field, levels=levels, cmap=cmap, extend=extend)
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
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if mark:
        for m_f1, m_f2 in mark:
            ax.plot(m_f1, m_f2, 'o', ms=8, mfc='none', mec='k', mew=1.5)

    ax.figure.colorbar(im, ax=ax, extendrect=extendrect, label=cbar_label)
    if filename:
        _save_figure(ax.figure, filename, path)
        if created:
            plt.close(ax.figure)
    elif created:
        plt.show()
    return ax


def plot_triad_modes(modes, k, l, x=None, y=None, vars_idx=(0,),
                     cmap='RdBu_r', cmap_prod='bone_r', figsize=None,
                     facecolor=None, xlim=None, ylim=None,
                     xlabel=r'$x$', ylabel=r'$y$',
                     tight_layout=True, extend='both', extendrect=True,
                     path=None, filename=None):
    '''
    Plot the bispectral modes of a triad and their interaction map.

    With the default two modes per triad, rows are the sum-interaction mode
    :math:`\\phi_{k+l}`, the quadratic-term mode :math:`\\phi_{k \\circ l}`,
    and the magnitude of their product, which localizes where the triadic
    interaction takes place. With ``params['constituent_modes']`` set when
    fitting (four modes per triad), two more rows come first: the constituent
    modes :math:`\\phi_k` and :math:`\\phi_l` at the triad's own frequencies.
    Columns are variables.

    :param numpy.ndarray modes: modes of one triad, of shape
        ``(2, ny, nx, nv)`` or ``(4, ny, nx, nv)``, as returned by
        ``get_modes_at_triad``; pass it ``data=`` to rebuild a triad whose
        modes were not saved.
    :param int k: integer frequency index of f1, used for the title.
    :param int l: integer frequency index of f2, used for the title.
    :param numpy.ndarray x: x coordinate, 1-D of length ``nx`` or 2-D
        ``(ny, nx)``. Default is the index.
    :param numpy.ndarray y: y coordinate, 1-D of length ``ny`` or 2-D
        ``(ny, nx)``. Default is the index.
    :param vars_idx: variables to plot.
    :param facecolor: background color for the figure and axes. Default is
        None, leaving Matplotlib's default unchanged.
    :param figsize: figure size. Default is ``(10, 9)`` for two modes and
        ``(10, 15)`` for four.
    :param xlim: x-axis limits. Default is Matplotlib's auto limits.
    :param ylim: y-axis limits. Default is Matplotlib's auto limits.
    :param str xlabel: x-axis label. Default is ``'$x$'``.
    :param str ylabel: y-axis label. Default is ``'$y$'``.
    :param bool tight_layout: call ``fig.tight_layout()``. Default is True.
    :param str extend: contour extension mode. Default is ``'both'`` so values
        outside explicit contour levels are still colored.
    :param bool extendrect: draw colorbar extensions as rectangles rather than
        triangles. Default is True.

    :return: the figure.
    :rtype: matplotlib.figure.Figure
    '''
    import matplotlib.pyplot as plt

    if modes.ndim != 4 or modes.shape[0] not in (2, 4):
        raise ValueError(
            f'plot_triad_modes needs modes of shape (2, ny, nx, nv) or '
            f'(4, ny, nx, nv); got {modes.shape}. Only two-dimensional data '
            f'can be contoured.')
    has_constituents = modes.shape[0] == 4
    if figsize is None:
        figsize = (10, 15) if has_constituents else (10, 9)
    vars_idx = list(vars_idx)
    if x is None:
        x = np.arange(modes.shape[2])
    if y is None:
        y = np.arange(modes.shape[1])

    # rows, keyed into `modes` by name rather than position, so inserting the
    # constituent rows ahead of the interaction map can't silently mis-scale
    # it against the wrong row
    rows = []
    if has_constituents:
        rows += [(r'$\phi_k$', 2, 'signed'), (r'$\phi_l$', 3, 'signed')]
    rows += [(r'$\phi_{k+l}$', 0, 'signed'),
             (r'$\phi_{k \circ l}$', 1, 'signed'),
             (r'$|\phi_{k \circ l} \cdot \phi_{k+l}|$', 'prod', 'mag')]

    fig, axes = plt.subplots(len(rows), len(vars_idx), figsize=figsize,
                             squeeze=False)
    if facecolor is not None:
        fig.patch.set_facecolor(facecolor)
    for c, iv in enumerate(vars_idx):
        for r, (title, comp, kind) in enumerate(rows):
            if kind == 'mag':
                field = np.abs(modes[0, ..., iv] * modes[1, ..., iv])
            else:
                field = np.real(modes[comp, ..., iv])
            ax = axes[r][c]
            if facecolor is not None:
                ax.set_facecolor(facecolor)
            if kind == 'signed':
                lv, cm = _symmetric_levels(field), cmap
            else:
                m = np.max(np.abs(field)) or 1.0
                lv, cm = m * np.linspace(0, 1, 257), cmap_prod
            im = ax.contourf(x, y, field, levels=lv, cmap=cm,
                             extend=extend)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.set_aspect('equal')
            ax.set_title(f'{title}  var {iv}')
            fig.colorbar(im, ax=ax, extendrect=extendrect)
    fig.suptitle(triad_label(k, l))
    if tight_layout:
        fig.tight_layout()
    _save_show_plots(filename, path, plt)
    return fig
