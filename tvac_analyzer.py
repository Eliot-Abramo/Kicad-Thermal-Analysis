#!/usr/bin/env python3
"""
TVAC Thermal Analyzer (refactor)

This entrypoint keeps the existing simulation core but replaces:
- KiCad parser: now in tvac.kicad (fixed footprint/pad transforms, dynamic layers)
- UI layer isolation: per-layer checkboxes (no 4-layer hard limit; no dead toggles)

Simulation physics remains in tvac.core (your existing implementation).
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

from tvac.core import *  # noqa: F401,F403 (keeps your current physics + solver)
from tvac.kicad import read_pcb

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget,
        QPushButton, QLabel, QFileDialog, QMessageBox, QProgressBar, QCheckBox, QGroupBox
    )
    from PyQt5.QtCore import Qt
    HAS_QT = True
except Exception:
    HAS_QT = False

if HAS_QT:
    from tvac.ui import PCBView, LayerPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TVAC Thermal Analyzer (KiCad 9)")
        self.resize(1400, 900)

        self.pcb = None
        self.cfg = Config()

        self._build_ui()

    def _build_ui(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        act_open = file_menu.addAction("&Open PCB...")
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._on_open)

        splitter = QSplitter(Qt.Horizontal)

        # Left pane: controls + layers
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(10, 10, 10, 10)
        title = QLabel("TVAC Thermal Analyzer")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        ll.addWidget(title)

        self.lbl_board = QLabel("No board loaded")
        self.lbl_board.setStyleSheet("color: #a0a8b4;")
        ll.addWidget(self.lbl_board)

        self.layer_panel = LayerPanel()
        self.layer_panel.layersChanged.connect(self._on_layers_changed)
        ll.addWidget(self.layer_panel, 1)

        # Feature toggles
        gb = QGroupBox("Overlays")
        gl = QVBoxLayout(gb)
        self.cb_grid = QCheckBox("Grid"); self.cb_grid.setChecked(True)
        self.cb_outline = QCheckBox("Outline"); self.cb_outline.setChecked(True)
        self.cb_traces = QCheckBox("Traces"); self.cb_traces.setChecked(True)
        self.cb_zones = QCheckBox("Zones"); self.cb_zones.setChecked(True)
        self.cb_vias = QCheckBox("Vias"); self.cb_vias.setChecked(True)
        self.cb_pads = QCheckBox("Pads"); self.cb_pads.setChecked(True)
        self.cb_comps = QCheckBox("Components"); self.cb_comps.setChecked(True)
        for cb, attr in [
            (self.cb_grid, "show_grid"),
            (self.cb_outline, "show_outline"),
            (self.cb_traces, "show_traces"),
            (self.cb_zones, "show_zones"),
            (self.cb_vias, "show_vias"),
            (self.cb_pads, "show_pads"),
            (self.cb_comps, "show_components"),
        ]:
            cb.toggled.connect(lambda v, a=attr: self._set_view_attr(a, v))
            gl.addWidget(cb)
        ll.addWidget(gb)

        splitter.addWidget(left)

        # Right pane: board view
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        hdr = QWidget()
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(10, 8, 10, 8)
        hl.addWidget(QLabel("PCB Preview"))
        hl.addStretch()
        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color: #a0a8b4;")
        hl.addWidget(self.lbl_info)
        rl.addWidget(hdr)

        self.pcb_view = PCBView()
        rl.addWidget(self.pcb_view, 1)

        # bottom bar
        bottom = QWidget()
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(10, 6, 10, 6)
        self.status_label = QLabel("● Ready")
        self.status_label.setStyleSheet("color: #66bb6a;")
        bl.addWidget(self.status_label)
        bl.addStretch()
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(220)
        bl.addWidget(self.progress)
        rl.addWidget(bottom)

        splitter.addWidget(right)
        splitter.setSizes([380, 1020])

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(splitter, 1)
        self.setCentralWidget(central)

    def _set_view_attr(self, name, value):
        setattr(self.pcb_view, name, value)
        self.pcb_view.update()

    def _on_layers_changed(self, layers: set):
        self.pcb_view.set_layers(layers)

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open KiCad board", "", "KiCad PCB (*.kicad_pcb)")
        if not path:
            return
        try:
            self.pcb = read_pcb(path)
            self.lbl_board.setText(os.path.basename(path))
            self.lbl_info.setText(f"{len(self.pcb.copper_layers)} Cu layers • "
                                  f"{len(self.pcb.components)} comps • "
                                  f"{len(self.pcb.traces)+len(self.pcb.arc_traces)} traces • "
                                  f"{len(self.pcb.vias)} vias • {len(self.pcb.zones)} zones")
            self.pcb_view.set_data(self.pcb, self.cfg)
            self.layer_panel.set_layers(self.pcb.copper_layers, checked=True)
            self.status_label.setText("● Board loaded")
            self.status_label.setStyleSheet("color: #66bb6a;")
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Failed to open board", f"{e}\n\n{tb}")


def main():
    if not HAS_QT:
        print("PyQt5 not installed. Install with: pip install PyQt5")
        sys.exit(1)
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
