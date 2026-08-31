'''
Module implementing generic post-processing utils: getters for results written
by a decomposition, and plots of the data and of the modes as fields.

Bispectrum-specific plotting lives in :mod:`pybmd.bmd.postproc`, which is the
only module that knows about triads and the regions of the ``f1``-``f2`` plane.
'''
import os

import numpy as np


def find_nearest_freq(freq_req, freq):
    '''
    Nearest frequency on an axis.

    :param float freq_req: requested frequency.
    :param numpy.ndarray freq: the frequency axis.

    :return: the nearest frequency and its index.
    :rtype: tuple(float, int)
    '''
    freq = np.asarray(freq)
    idx = int(np.argmin(np.abs(freq - freq_req)))
    return float(freq[idx]), idx


def find_nearest_coords(coords, x):
    '''
    Nearest point of a coordinate grid.

    :param tuple coords: requested coordinates, one per spatial dimension.
    :param list x: coordinate arrays, one per spatial dimension.

    :return: the nearest coordinates and their indices.
    :rtype: tuple(tuple, tuple)
    '''
    idx = tuple(int(np.argmin(np.abs(np.asarray(xi) - c)))
                for xi, c in zip(x, coords))
    xi = tuple(float(np.asarray(x[d]).ravel()[i]) for d, i in enumerate(idx))
    return xi, idx


def get_modes_at_triad(results_path, triad_idx):
    '''
    Load the modes of one triad from a results directory.

    :param str results_path: the ``savedir_sim`` of a fitted decomposition.
    :param int triad_idx: index into the per-triad arrays.

    :return: the modes, of shape ``(2, *xshape, nv)``.
    :rtype: numpy.ndarray
    '''
    path = os.path.join(results_path, 'modes',
                        f'triad_idx_{triad_idx:08d}.npy')
    if not os.path.exists(path):
        raise FileNotFoundError(f'no modes stored for triad {triad_idx}.')
    return np.load(path)


def get_all_modes(results_path):
    '''
    Load every saved mode from a results directory.

    :param str results_path: the ``savedir_sim`` of a fitted decomposition.

    :return: the modes, of shape ``(n_saved, 2, *xshape, nv)``.
    :rtype: numpy.ndarray
    '''
    modes_dir = os.path.join(results_path, 'modes')
    files = sorted(f for f in os.listdir(modes_dir) if f.endswith('.npy'))
    if not files:
        raise FileNotFoundError(f'no modes found in {modes_dir}.')
    return np.stack([np.load(os.path.join(modes_dir, f)) for f in files])


def get_bispectrum(results_path):
    '''
    Load the mode bispectrum written by a decomposition.

    :param str results_path: the ``savedir_sim`` of a fitted decomposition.

    :return: ``L``, ``T``, ``freq`` and ``f_idx``.
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
    '''
    with np.load(os.path.join(results_path, 'bispectrum.npz')) as d:
        return d['L'], d['T'], d['freq'], d['f_idx']


def _save_show_plots(filename, path, plt):
    '''Save the current figure if a filename is given, otherwise show it.'''
    if filename:
        if path is None:
            path = os.getcwd()
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, filename), dpi=200,
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def _symmetric_levels(field, n_levels=257, scale=0.5):
    '''Contour levels symmetric about zero, as used for real mode fields.'''
    m = scale * np.max(np.abs(field))
    if m == 0:
        m = 1.0
    return m * np.linspace(-1, 1, n_levels)


def plot_2d_data(data, time_idx=(0,), vars_idx=(0,), x1=None, x2=None,
                 title='', figsize=(12, 8), path=None, filename=None,
                 cmap='RdBu_r'):
    '''
    Plot snapshots of two-dimensional data.

    :param numpy.ndarray data: data of shape ``(nt, n1, n2, nv)``.
    :param time_idx: snapshots to plot.
    :param vars_idx: variables to plot.
    :param numpy.ndarray x1: first coordinate. Default is the index.
    :param numpy.ndarray x2: second coordinate. Default is the index.
    '''
    import matplotlib.pyplot as plt

    time_idx, vars_idx = list(time_idx), list(vars_idx)
    if x1 is None:
        x1 = np.arange(data.shape[1])
    if x2 is None:
        x2 = np.arange(data.shape[2])

    fig, axes = plt.subplots(len(time_idx), len(vars_idx), figsize=figsize,
                             squeeze=False)
    for r, it in enumerate(time_idx):
        for c, iv in enumerate(vars_idx):
            ax = axes[r][c]
            field = np.real(data[it, ..., iv])
            im = ax.contourf(x1, x2, field.T,
                             levels=_symmetric_levels(field, 65, 1.0),
                             cmap=cmap, extend='both')
            ax.set_xlabel(r'$x_1$')
            ax.set_ylabel(r'$x_2$')
            ax.set_title(f't={it}, var={iv}')
            fig.colorbar(im, ax=ax)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    _save_show_plots(filename, path, plt)
