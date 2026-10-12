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
    assert liked.read_text().split('\t')[-1].strip() == f'triads/{t.name}'  # relative

    # the marker outlines the current triad's cell
    i = bmd.find_triad(*t.kl_idx)
    df = bmd.freq[1] - bmd.freq[0]
    x0, y0 = w.bispec.marker.get_xy()
    assert np.allclose((x0 + df / 2, y0 + df / 2),
                       (bmd.triads.f1[i], bmd.triads.f2[i]))
    assert w.bispec.marker.get_edgecolor()[:3] == (0, 0, 0)

    # export goes next to the triad directory, named after it
    QTest.keyClick(w, Qt.Key_E)
    assert plt.imread(triads_dir / f'{t.name}.png').ndim == 3
    w.close()
    app.processEvents()


def test_zoom_window():
    viewer = _viewer_module()
    n, df = viewer.ZOOM_CELLS, 0.1
    home = ((-0.05, 9.95), (-0.05, 9.95))
    (x0, x1), (y0, y1) = viewer.zoom_window((5.0, 0.0), df, home)
    assert np.isclose(x1 - x0, n * df) and x0 < 5.0 < x1
    assert np.isclose(y0, -0.05) and np.isclose(y1 - y0, n * df)   # shifted inside
    small = ((0.0, 0.5), (0.0, 0.5))
    assert viewer.zoom_window((0.2, 0.2), df, small) == small


def test_export_png_and_liked_command(tmp_path):
    viewer = _viewer_module()
    bmd = _fit(tmp_path)
    export_triads(bmd, str(tmp_path), np.arange(5), np.arange(6))
    triads_dir = tmp_path / 'triads'
    triads = viewer.find_triads(triads_dir)

    # the close-up outlines the triad's cell, the whole map outlines the close-up
    fig, (ax_full, ax_zoom) = plt.subplots(2)
    t = triads[0]
    point = viewer.bmd_point(bmd, t)
    region, cell = viewer.draw_bispectrum(ax_full, ax_zoom, bmd, point)
    x0, y0 = region.get_xy()
    assert np.allclose((x0, x0 + region.get_width()), ax_zoom.get_xlim())
    assert np.allclose((y0, y0 + region.get_height()), ax_zoom.get_ylim())
    df = bmd.freq[1] - bmd.freq[0]
    assert np.allclose(np.add(cell.get_xy(), df / 2), point)
    plt.close(fig)

    assert viewer.export_triad_png(t, None) == triads_dir / f'{t.name}.png'

    liked = tmp_path / 'liked.txt'
    liked.write_text(''.join(viewer.liked_line(u.path, liked) + '\n' for u in triads[1:3]))
    assert '\ttriads/triad_' in liked.read_text()     # relative to the liked file
    written = viewer.export_liked(triads_dir, liked, bmd)
    assert written == [triads_dir / f'{u.name}.png' for u in triads[1:3]]
    viewer.main([str(triads_dir), '--export-liked', '-o', str(liked)])
    for path in written:
        assert plt.imread(path).ndim == 3

    # likes recorded before the triad directories were moved still match, by name
    moved = tmp_path / 'old_location' / triads[1].name
    liked.write_text(f'{triads[1].name}\t\t{moved}\n')
    assert viewer.match_liked(viewer.read_liked(liked), triads) == [triads[1].path]
    assert viewer.export_liked(triads_dir, liked, bmd) == [written[0]]
