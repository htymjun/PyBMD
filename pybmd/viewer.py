"""Triad viewer: browse triad_* directories (Mode_{k,l,kl}_f*.png + psi.png) and mark favourites.

The directories are the ones written by ``pybmd.bmd.postproc.export_triads``. Needs PyQt5
(``pip install -e '.[viewer]'``).

Usage:
    pybmd-viewer ROOT_DIR [-o liked_triads.txt]

If ROOT_DIR/bmd.pkl or its parent's (a pickled, fitted pybmd Standard/Cross) exists, its mode
bispectrum, normalised by its maximum, is shown with the current triad marked. Click a cell to
jump to that triad; drag a rectangle to zoom in.

Keys:
    k / Right : next        j / Left : previous
    l         : like toggle Home / End : first / last
    Esc       : quit (liked list is saved and printed)
"""
import argparse
import pickle
import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QGridLayout, QHBoxLayout, QLabel, QListWidget, QMainWindow,
    QPushButton, QShortcut, QSizePolicy, QSlider, QSplitter, QVBoxLayout, QWidget,
)

from pybmd.bmd.postproc import plot_mode_bispectrum

MODE_RE = re.compile(r"^Mode_(k|l|kl)_f(-?\d+\.\d+)\.png$")
TRIAD_DIR_RE = re.compile(r"^triad_k(-?\d+)_l(-?\d+)$")
ROLES = ("k", "l", "kl")  # f_k + f_l = f_kl
FREQ_TOL = 2e-3  # three frequencies each rounded to 3 decimals: |error| <= 1.5e-3
CACHE_SIZE = 64
DRAG_PX = 5  # a press/release closer than this (pixels) is a click, not a zoom drag
KEY_GUIDE = ("Keys:   k / → : next     j / ← : previous     l : like     "
             "Home / End : first / last     Esc : quit     "
             "(bispectrum: click to jump, drag to zoom)")


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


class Triad:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.name = path.name
        m = TRIAD_DIR_RE.match(path.name)
        self.kl_idx = (int(m.group(1)), int(m.group(2))) if m else None  # integer (k, l)
        self.modes = {}  # role -> (freq, freq_str, path)
        self.duplicates = []
        for p in sorted(path.iterdir()):
            m = MODE_RE.match(p.name)
            if m:
                role, fs = m.groups()
                if role in self.modes:
                    self.duplicates.append(role)
                self.modes[role] = (float(fs), fs, p)
        psi = path / "psi.png"
        self.psi = psi if psi.exists() else None

    def warning(self):
        missing = [r for r in ROLES if r not in self.modes]
        if missing:
            return "Missing: " + ", ".join(f"Mode_{r}_f*.png" for r in missing)
        if self.duplicates:
            return "Duplicated: " + ", ".join(f"Mode_{r}_f*.png" for r in self.duplicates)
        fk, fl, fkl = (self.modes[r][0] for r in ROLES)
        if abs(fk + fl - fkl) > FREQ_TOL:
            return f"f_k + f_l = {fk + fl:.3f} ≠ f_kl = {fkl:.3f}"
        if self.psi is None:
            return "psi.png is missing"
        return ""

    def freq_text(self):
        return " ".join(f"{r}={self.modes[r][1]}" for r in ROLES if r in self.modes)


def find_triads(root: Path):
    if root.name.startswith("triad_") and root.is_dir():
        dirs = [root]
    else:
        dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("triad_")]
    dirs.sort(key=lambda p: natural_key(p.name))
    return [Triad(d) for d in dirs]


BMD_ATTRS = ("L", "freq", "f_idx")  # what the viewer reads from a fitted pybmd Standard


def _find_bmd(obj):
    """The fitted pybmd object itself, or the first one inside a tuple/list/dict."""
    if all(hasattr(obj, a) for a in BMD_ATTRS):
        return obj
    items = obj.values() if isinstance(obj, dict) else obj if isinstance(obj, (tuple, list)) else ()
    for x in items:
        found = _find_bmd(x)
        if found is not None:
            return found
    return None


def load_bmd(root: Path):
    """Load the pickled BMD fit placed next to the triad_* directories, or None."""
    for d in (root, root.parent):
        pkl = d / "bmd.pkl"
        if pkl.exists():
            try:
                with open(pkl, "rb") as f:
                    obj = pickle.load(f)
            except Exception as e:  # pybmd not importable, broken file, ...
                print(f"Warning: cannot load {pkl} ({e}); starting without the bispectrum.",
                      file=sys.stderr)
                return None
            bmd = _find_bmd(obj)
            if bmd is None:
                print(f"Warning: {pkl} holds no BMD object with {'/'.join(BMD_ATTRS)}"
                      f" (got {type(obj).__name__}); starting without the bispectrum.",
                      file=sys.stderr)
            return bmd
    return None


class ImagePanel(QWidget):
    """Caption + image that rescales with the window, keeping aspect ratio."""

    def __init__(self):
        super().__init__()
        self.caption = QLabel()
        self.caption.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.caption.setFont(font)
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image.setMinimumSize(100, 100)
        self.image.setStyleSheet("background: white; border: 1px solid #bbb;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.caption)
        layout.addWidget(self.image, 1)
        self.pixmap = None

    def set_content(self, caption, pixmap):
        self.caption.setText(caption)
        self.pixmap = pixmap
        self._rescale()

    def _rescale(self):
        if self.pixmap is None or self.pixmap.isNull():
            self.image.setPixmap(QPixmap())
            self.image.setText("(no image)")
            return
        self.image.setPixmap(self.pixmap.scaled(
            self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()


class BispectrumPanel(QWidget):
    """Discrete map of log(|L| / max|L|) over the (f1, f2) plane; one cell per frequency pair.

    Click a cell to emit `clicked`; drag a rectangle to zoom in. Zooms are stacked so
    they can be undone one at a time, or reset to the full view.
    """

    clicked = pyqtSignal(float, float)  # data coordinates (f1, f2)

    def __init__(self, bmd):
        super().__init__()
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Rectangle

        caption = QLabel("Mode bispectrum  log(|λ₁| / max|λ₁|)")
        caption.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        caption.setFont(font)

        freq = np.asarray(bmd.freq, dtype=float)
        self.df = freq[1] - freq[0]
        fig = Figure(figsize=(4, 4), tight_layout=True)
        ax = fig.add_subplot()
        plot_mode_bispectrum(bmd.L, freq, normalize=True, ax=ax, extend='neither')
        self.marker = Rectangle((0, 0), self.df, self.df, fill=False,
                                edgecolor="magenta", linewidth=2.5, visible=False)
        ax.add_patch(self.marker)
        self.rubber = Rectangle((0, 0), 0, 0, fill=True, facecolor=(0, 0, 0, 0.1),
                                edgecolor="black", linestyle="--", visible=False)
        ax.add_patch(self.rubber)
        self.ax = ax
        self._home = (ax.get_xlim(), ax.get_ylim())
        self._history = []  # previous (xlim, ylim), most recent last
        self._press = None  # (pixel x, pixel y, data x, data y) of the current drag

        self.canvas = FigureCanvasQTAgg(fig)
        self.canvas.setFocusPolicy(Qt.NoFocus)
        self.canvas.setToolTip("Click: jump to the nearest triad\nDrag: zoom into the rectangle")
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        self.undo_btn = QPushButton("Undo zoom")
        self.undo_btn.setToolTip("Go back to the previous zoom level")
        self.undo_btn.clicked.connect(self.undo_zoom)
        self.reset_btn = QPushButton("Reset view")
        self.reset_btn.setToolTip("Show the whole bispectrum")
        self.reset_btn.clicked.connect(self.reset_zoom)
        buttons = QHBoxLayout()
        for b in (self.undo_btn, self.reset_btn):
            b.setFocusPolicy(Qt.NoFocus)
            buttons.addWidget(b)
        self._update_buttons()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(caption)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(buttons)

    def set_current(self, point):
        if point is None:
            self.marker.set_visible(False)
        else:
            self.marker.set_xy((point[0] - self.df / 2, point[1] - self.df / 2))
            self.marker.set_visible(True)
        self.canvas.draw_idle()

    # ---- zoom ---------------------------------------------------------------
    def _set_view(self, xlim, ylim):
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self._update_buttons()
        self.canvas.draw_idle()

    def zoom_to(self, xlim, ylim):
        self._history.append((self.ax.get_xlim(), self.ax.get_ylim()))
        self._set_view(xlim, ylim)

    def undo_zoom(self):
        if self._history:
            self._set_view(*self._history.pop())

    def reset_zoom(self):
        self._history.clear()
        self._set_view(*self._home)

    def _update_buttons(self):
        self.undo_btn.setEnabled(bool(self._history))
        self.reset_btn.setEnabled((self.ax.get_xlim(), self.ax.get_ylim()) != self._home)

    # ---- mouse --------------------------------------------------------------
    def _on_press(self, event):
        if event.button == 1 and event.inaxes is self.ax and event.xdata is not None:
            self._press = (event.x, event.y, event.xdata, event.ydata)

    def _on_motion(self, event):
        if self._press is None:
            return
        _, _, x0, y0 = self._press
        x1, y1 = self.ax.transData.inverted().transform((event.x, event.y))
        self.rubber.set_bounds(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        self.rubber.set_visible(True)
        self.canvas.draw_idle()

    def _on_release(self, event):
        if self._press is None:
            return
        px, py, x0, y0 = self._press
        self._press = None
        self.rubber.set_visible(False)
        if abs(event.x - px) < DRAG_PX and abs(event.y - py) < DRAG_PX:
            self.canvas.draw_idle()
            self.clicked.emit(x0, y0)
            return
        # pixel -> data, so a release outside the axes still gives a rectangle
        x1, y1 = self.ax.transData.inverted().transform((event.x, event.y))
        self.zoom_to(tuple(sorted((x0, x1))), tuple(sorted((y0, y1))))


class TriadViewer(QMainWindow):
    def __init__(self, triads, liked_path: Path, bmd=None):
        super().__init__()
        self.triads = triads
        self.bmd = bmd
        self.liked_path = liked_path
        self.index = 0
        self.cache = OrderedDict()
        self.liked = self._load_liked()  # list of resolved Path, in like order

        self.setWindowTitle("Triad Viewer")
        self.resize(1400, 950)

        # header
        self.title = QLabel()
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.title.setFont(font)
        self.warn = QLabel()
        self.warn.setStyleSheet("color: #c00; font-weight: bold;")

        # 2x2 grid: k l / kl psi
        self.panels = [ImagePanel() for _ in range(4)]
        grid_w = QWidget()
        grid = QGridLayout(grid_w)
        grid.setContentsMargins(0, 0, 0, 0)
        for i, panel in enumerate(self.panels):
            grid.addWidget(panel, i // 2, i % 2)

        # liked list
        self.liked_list = QListWidget()
        self.liked_list.setFocusPolicy(Qt.NoFocus)
        self.liked_list.itemDoubleClicked.connect(self._jump_to_liked)
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(QLabel("♥ Liked triads (double-click to jump)"))
        side_layout.addWidget(self.liked_list)

        self.bispec = None
        if bmd is not None:
            self.bispec = BispectrumPanel(bmd)
            self.bispec.clicked.connect(self._jump_to_freq)
            self._points = [self._bmd_point(t) for t in triads]
            self._by_kl = {t.kl_idx: i for i, t in enumerate(triads) if t.kl_idx}
            right = QSplitter(Qt.Vertical)
            right.addWidget(self.bispec)
            right.addWidget(side)
            right.setSizes([600, 300])
            side = right

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(grid_w)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1050, 330])

        # controls (NoFocus so that key shortcuts always reach the window)
        self.prev_btn = QPushButton("◀ Prev (j)")
        self.prev_btn.setToolTip("Previous triad  [j / ←]")
        self.prev_btn.clicked.connect(lambda: self.go(self.index - 1))
        self.next_btn = QPushButton("Next ▶ (k)")
        self.next_btn.setToolTip("Next triad  [k / →]")
        self.next_btn.clicked.connect(lambda: self.go(self.index + 1))
        self.like_btn = QPushButton()
        self.like_btn.setToolTip("Toggle like  [l]")
        self.like_btn.clicked.connect(self.toggle_like)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, max(len(triads) - 1, 0))
        self.slider.setToolTip("Drag to move  [Home / End: first / last]")
        self.slider.valueChanged.connect(self.go)
        for w in (self.prev_btn, self.next_btn, self.like_btn, self.slider):
            w.setFocusPolicy(Qt.NoFocus)
        controls = QHBoxLayout()
        controls.addWidget(self.prev_btn)
        controls.addWidget(self.next_btn)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.like_btn)

        guide = QLabel(KEY_GUIDE)
        guide.setStyleSheet("color: #555; padding: 4px; background: #eee; border-radius: 4px;")

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.title)
        layout.addWidget(self.warn)
        layout.addWidget(splitter, 1)
        layout.addLayout(controls)
        layout.addWidget(guide)
        self.setCentralWidget(central)

        shortcuts = [
            (("K", "Right"), lambda: self.go(self.index + 1)),
            (("J", "Left"), lambda: self.go(self.index - 1)),
            (("L",), self.toggle_like),
            (("Home",), lambda: self.go(0)),
            (("End",), lambda: self.go(len(self.triads) - 1)),
            (("Esc",), self.close),
        ]
        for keys, slot in shortcuts:
            for key in keys:
                QShortcut(QKeySequence(key), self, activated=slot)

        self._refresh_liked_list()
        self.show_current()

    # ---- navigation -------------------------------------------------------
    def go(self, i):
        i = max(0, min(i, len(self.triads) - 1))
        if i == self.index:
            return
        self.index = i
        self.show_current()

    def show_current(self):
        t = self.triads[self.index]
        mark = "  ♥" if t.path in self.liked else ""
        self.title.setText(f"{t.name}    ({self.index + 1} / {len(self.triads)}){mark}")
        self.warn.setText(t.warning())
        self.warn.setVisible(bool(t.warning()))
        self.like_btn.setText("♥ Liked (l)" if t.path in self.liked else "♡ Like (l)")
        self.like_btn.setStyleSheet(
            "background: #f8c; font-weight: bold;" if t.path in self.liked else "")

        contents = []
        for r in ROLES:
            if r in t.modes:
                _, s, p = t.modes[r]
                contents.append((f"{r}:  f = {s}", p))
            else:
                contents.append((f"{r}:  (none)", None))
        contents.append(("psi", t.psi))
        for panel, (caption, path) in zip(self.panels, contents):
            panel.set_content(caption, self._pixmap(path))

        if self.bispec is not None:
            self.bispec.set_current(self._points[self.index])

        self.slider.blockSignals(True)
        self.slider.setValue(self.index)
        self.slider.blockSignals(False)
        QTimer.singleShot(0, self._preload_neighbours)

    def _pixmap(self, path):
        if path is None:
            return None
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        pm = QPixmap(str(path))
        self.cache[path] = pm
        if len(self.cache) > CACHE_SIZE:
            self.cache.popitem(last=False)
        return pm

    def _preload_neighbours(self):
        for j in (self.index + 1, self.index - 1):
            if 0 <= j < len(self.triads):
                t = self.triads[j]
                for p in [m[2] for m in t.modes.values()] + [t.psi]:
                    self._pixmap(p)

    # ---- bispectrum -------------------------------------------------------
    def _bmd_point(self, t):
        """(f1, f2) of a triad on the bispectrum: from the dir name, else the file names."""
        if t.kl_idx is not None:
            f_idx = np.asarray(self.bmd.f_idx)
            ik, il = (np.flatnonzero(f_idx == v) for v in t.kl_idx)
            if ik.size and il.size:
                return float(self.bmd.freq[ik[0]]), float(self.bmd.freq[il[0]])
        if "k" in t.modes and "l" in t.modes:
            return t.modes["k"][0], t.modes["l"][0]
        return None

    def _jump_to_freq(self, x, y):
        freq = np.asarray(self.bmd.freq)
        f_idx = np.asarray(self.bmd.f_idx)
        kl = (int(f_idx[np.argmin(np.abs(freq - x))]), int(f_idx[np.argmin(np.abs(freq - y))]))
        if kl in self._by_kl:
            self.go(self._by_kl[kl])
            return
        dists = [np.inf if p is None else (p[0] - x) ** 2 + (p[1] - y) ** 2
                 for p in self._points]
        if np.isfinite(min(dists)):
            self.go(int(np.argmin(dists)))

    # ---- likes ------------------------------------------------------------
    def toggle_like(self):
        p = self.triads[self.index].path
        if p in self.liked:
            self.liked.remove(p)
        else:
            self.liked.append(p)
        self._save_liked()
        self._refresh_liked_list()
        self.show_current()

    def _liked_lines(self):
        by_path = {t.path: t for t in self.triads}
        lines = []
        for p in self.liked:
            t = by_path.get(p)
            freq = t.freq_text() if t else ""
            lines.append(f"{p.name}\t{freq}\t{p}")
        return lines

    def _save_liked(self):
        self.liked_path.write_text("".join(l + "\n" for l in self._liked_lines()))

    def _load_liked(self):
        if not self.liked_path.exists():
            return []
        liked = []
        for line in self.liked_path.read_text().splitlines():
            if line.strip():
                liked.append(Path(line.split("\t")[-1]).resolve())
        return liked

    def _refresh_liked_list(self):
        self.liked_list.clear()
        for line in self._liked_lines():
            name, freq, _ = line.split("\t")
            self.liked_list.addItem(f"{name}   {freq}")

    def _jump_to_liked(self, item):
        p = self.liked[self.liked_list.row(item)]
        for i, t in enumerate(self.triads):
            if t.path == p:
                self.go(i)
                return

    def closeEvent(self, event):
        self._save_liked()
        print(f"=== Liked ({len(self.liked)}) -> {self.liked_path} ===")
        for line in self._liked_lines():
            print(line)
        super().closeEvent(event)


def view_triads(root, liked_path=None, bmd=None):
    """Open the viewer on ROOT (a directory of triad_* directories, or one of them) and
    block until it is closed. BMD defaults to ROOT/bmd.pkl or its parent's, if present."""
    root = Path(root).resolve()
    triads = find_triads(root)
    if not triads:
        sys.exit(f"No triad_* directories found in: {root}")
    liked_path = Path(liked_path) if liked_path else root / "liked_triads.txt"
    if bmd is None:
        bmd = load_bmd(root)

    app = QApplication.instance() or QApplication(sys.argv)
    viewer = TriadViewer(triads, liked_path.resolve(), bmd)
    viewer.show()
    return app.exec_()


def main():
    ap = argparse.ArgumentParser(
        prog="pybmd-viewer", description="Browse triad_* directories and mark favourites.")
    ap.add_argument("root", type=Path, help="directory containing triad_* directories")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="liked list file (default: ROOT/liked_triads.txt)")
    args = ap.parse_args()
    sys.exit(view_triads(args.root, args.output))


if __name__ == "__main__":
    main()
