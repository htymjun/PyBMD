'''
The discrete mode bispectrum map, and the triad viewer (``pybmd-viewer``) on
the directories written by example 5's ``export_triads``. The viewer tests need PyQt5 and run
headless (``QT_QPA_PLATFORM=offscreen``).
'''
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import QuadMesh
from matplotlib.contour import ContourSet

from examples.example5_cylinder_paper import export_triads
from pybmd.bmd.postproc import plot_mode_bispectrum
from pybmd.bmd.standard import Standard


def _fit(savedir):
    rng = np.random.default_rng(0)
    params = dict(n_dft=16, time_step=0.5, n_space_dims=2, n_variables=1,
                  n_overlap=8, regions=[1], max_freq_idx=2,
                  store_modes=True, save_modes=False,
                  constituent_modes=True,
                  compute_energy_transfer=False,
                  savedir=str(savedir / 'fit'))
    return Standard(params=params).fit(rng.standard_normal((200, 6, 5, 1)))


def _viewer_module():
    pytest.importorskip('PyQt5')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    import pybmd.viewer as viewer
    return viewer


def test_plot_mode_bispectrum_is_discrete():
    freq = np.arange(5) * 0.1
    L = np.full((5, 5), np.nan)
    L[1:4, 0:3] = np.arange(1.0, 10.0).reshape(3, 3)
    ax = plot_mode_bispectrum(L, freq, normalize=True, title='t',
                              filename=None, ax=plt.subplots()[1])
    meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
    assert len(meshes) == 1
    assert not any(isinstance(c, ContourSet) for c in ax.collections)
    values = meshes[0].get_array()
    assert values.size == L.size
    assert np.isclose(values.max(), 0.0)          # log(max / max)
    assert ax.get_title() == 't'
    # the view spans whole cells of the computed triads
    assert np.allclose(ax.get_xlim(), (0.05, 0.35))
    assert np.allclose(ax.get_ylim(), (-0.05, 0.25))
    plt.close('all')


def test_example5_export_matches_viewer_format(tmp_path):
    viewer = _viewer_module()
    bmd = _fit(tmp_path)
    export_triads(bmd, str(tmp_path), np.arange(5), np.arange(6))

    triads = viewer.find_triads(tmp_path / 'triads')
    assert sorted(t.name for t in triads) == sorted(
        f'triad_k{k}_l{l}' for k, l in zip(bmd.triads.k, bmd.triads.l))
    for t in triads:
        assert t.warning() == ''
    loaded = viewer.load_bmd(tmp_path / 'triads')
    assert np.array_equal(loaded.L, bmd.L, equal_nan=True)


def test_viewer_navigation(tmp_path):
    viewer = _viewer_module()
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    bmd = _fit(tmp_path)
    export_triads(bmd, str(tmp_path), np.arange(5), np.arange(6))
    triads_dir = tmp_path / 'triads'
    app = QApplication.instance() or QApplication([])
    liked = tmp_path / 'liked.txt'
    w = viewer.TriadViewer(viewer.find_triads(triads_dir), liked,
                           viewer.load_bmd(triads_dir))
    w.show()
    # shortcuts fire only for the active window
    QApplication.setActiveWindow(w)
    QTest.qWaitForWindowActive(w)
    QTest.keyClick(w, Qt.Key_K)
    assert w.index == 1
    QTest.keyClick(w, Qt.Key_L)
    t = w.triads[1]
    assert liked.read_text().split('\t')[0] == t.name

    # the marker outlines the current triad's cell
    i = bmd.find_triad(*t.kl_idx)
    df = bmd.freq[1] - bmd.freq[0]
    x0, y0 = w.bispec.marker.get_xy()
    assert np.allclose((x0 + df / 2, y0 + df / 2),
                       (bmd.triads.f1[i], bmd.triads.f2[i]))
    w.close()
    app.processEvents()
