'''
Module implementing bispectrum-specific post-processing: the mode bispectrum
map over the ``f1``-``f2`` plane, and the mode panels for a chosen triad.

This is the only module that knows about triads and regions; generic getters
and field plots live in :mod:`pybmd.utils.postproc`.
'''
import numpy as np

from pybmd.utils.postproc import _save_show_plots, _symmetric_levels


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
                         filename=None, ax=None):
    '''
    Contour the mode bispectrum over the ``f1``-``f2`` plane.

    :param numpy.ndarray L: the bispectrum, of shape ``(n_freq, n_freq)``.
        Entries outside the computed triads are expected to be NaN and are
        masked out.
    :param numpy.ndarray freq: the frequency axis.
    :param bool log: plot ``log|L|`` rather than ``|L|``. Default is True.
    :param levels: contour levels, or the number of them. Default is 100.
    :param mark: triads to annotate, as a list of ``(f1, f2)`` pairs.
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
                     cmap=cmap, extend='both')
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

    ax.figure.colorbar(im, ax=ax,
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
    for c, iv in enumerate(vars_idx):
        fields = [np.real(modes[0, ..., iv]),
                  np.real(modes[1, ..., iv]),
                  np.abs(modes[0, ..., iv] * modes[1, ..., iv])]
        for r, field in enumerate(fields):
            ax = axes[r][c]
            if r < 2:
                lv, cm = _symmetric_levels(field), cmap
            else:
                m = np.max(np.abs(field)) or 1.0
                lv, cm = m * np.linspace(0, 1, 257), cmap_prod
            im = ax.contourf(x1, x2, field.T, levels=lv, cmap=cm, extend='both')
            ax.set_xlabel(r'$x_1$')
            ax.set_ylabel(r'$x_2$')
            ax.set_title(f'{titles[r]}  var {iv}')
            fig.colorbar(im, ax=ax)
    fig.suptitle(triad_label(k, l))
    fig.tight_layout()
    _save_show_plots(filename, path, plt)
    return fig
