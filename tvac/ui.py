from __future__ import annotations
import math
from typing import Dict, List, Optional, Set

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QCheckBox, QPushButton, QLabel, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QPointF
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont

from .geometry import PCBData, Pt


def layer_color(name: str, alpha: int = 180) -> QColor:
    """
    Deterministic color per layer name.
    Avoids hardcoding to 4 inner layers; supports arbitrary KiCad stackups.
    """
    # simple hash -> hue
    h = 0
    for ch in name:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    hue = (h % 360) / 360.0
    # HSV -> RGB
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.85)
    return QColor(int(r * 255), int(g * 255), int(b * 255), int(alpha))


class PCBView(QWidget):
    componentClicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pcb: Optional[PCBData] = None
        self.cfg = None
        self.result = None
        self.mesh = None
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._last_mouse = None
        self._selected_ref = ""

        self.visible_layers: Set[str] = set()
        self.show_outline = True
        self.show_grid = True
        self.show_thermal = False
        self.show_components = True
        self.show_mounting = True
        self.show_traces = True
        self.show_zones = True
        self.show_vias = True
        self.show_pads = True

        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    def set_data(self, pcb: PCBData, cfg):
        self.pcb = pcb
        self.cfg = cfg
        self.visible_layers = set(pcb.copper_layers) if pcb else set()
        self._fit_view()
        self.update()

    def set_layer_visible(self, layer: str, visible: bool):
        if visible:
            self.visible_layers.add(layer)
        else:
            self.visible_layers.discard(layer)
        self.update()

    def set_layers(self, layers: Set[str]):
        self.visible_layers = set(layers)
        self.update()

    def is_layer_visible(self, layer_name: str) -> bool:
        return layer_name in self.visible_layers

    def _fit_view(self):
        if not self.pcb:
            return
        o = self.pcb.outline
        bw = max(1, o.max_x - o.min_x)
        bh = max(1, o.max_y - o.min_y)
        w, h = self.width(), self.height()
        self._zoom = min((w - 40) / bw, (h - 40) / bh)
        cx = (o.min_x + o.max_x) / 2
        cy = (o.min_y + o.max_y) / 2
        self._pan = QPointF(w / 2 - cx * self._zoom, h / 2 - cy * self._zoom)

    def _to_screen(self, x, y):
        return (x * self._zoom + self._pan.x(), y * self._zoom + self._pan.y())

    def _from_screen(self, sx, sy):
        return ((sx - self._pan.x()) / self._zoom, (sy - self._pan.y()) / self._zoom)

    def resizeEvent(self, e):
        self._fit_view()

    def wheelEvent(self, e):
        factor = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        pos = e.pos() if hasattr(e, 'pos') else e.position().toPoint()
        mx, my = pos.x(), pos.y()
        wx, wy = self._from_screen(mx, my)
        self._zoom *= factor
        self._pan = QPointF(mx - wx * self._zoom, my - wy * self._zoom)
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._last_mouse = e.pos()
            if self.pcb:
                wx, wy = self._from_screen(e.pos().x(), e.pos().y())
                for comp in self.pcb.components:
                    bb0, bb1 = comp.bbox()
                    if bb0.x <= wx <= bb1.x and bb0.y <= wy <= bb1.y:
                        self._selected_ref = comp.ref
                        self.componentClicked.emit(comp.ref)
                        self.update()
                        return
        elif e.button() == Qt.MiddleButton:
            self._last_mouse = e.pos()

    def mouseMoveEvent(self, e):
        if self._last_mouse and (e.buttons() & (Qt.LeftButton | Qt.MiddleButton)):
            d = e.pos() - self._last_mouse
            self._pan += QPointF(d.x(), d.y())
            self._last_mouse = e.pos()
            self.update()

    def mouseReleaseEvent(self, e):
        self._last_mouse = None

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(30, 33, 39))

        if not self.pcb:
            p.setPen(QPen(QColor(160, 168, 180)))
            p.setFont(QFont("sans-serif", 14))
            p.drawText(self.rect(), Qt.AlignCenter, "Load a .kicad_pcb file")
            p.end()
            return

        if self.show_grid:
            self._draw_grid(p)
        if self.show_outline:
            self._draw_outline(p)
        if self.show_zones:
            self._draw_zones(p)
        if self.show_traces:
            self._draw_traces(p)
        if self.show_vias:
            self._draw_vias(p)
        if self.show_pads:
            self._draw_pads(p)
        if self.show_components:
            self._draw_components(p)

        p.end()

    def _draw_grid(self, p):
        o = self.pcb.outline
        step = 10.0
        p.setPen(QPen(QColor(50, 55, 65), 1))
        x = o.min_x
        while x <= o.max_x:
            sx1, sy1 = self._to_screen(x, o.min_y)
            sx2, sy2 = self._to_screen(x, o.max_y)
            p.drawLine(int(sx1), int(sy1), int(sx2), int(sy2))
            x += step
        y = o.min_y
        while y <= o.max_y:
            sx1, sy1 = self._to_screen(o.min_x, y)
            sx2, sy2 = self._to_screen(o.max_x, y)
            p.drawLine(int(sx1), int(sy1), int(sx2), int(sy2))
            y += step

    def _draw_outline(self, p):
        pts = self.pcb.outline.outline
        o = self.pcb.outline
        p.setPen(QPen(QColor(200, 200, 200), 2))
        if not pts:
            x1, y1 = self._to_screen(o.min_x, o.min_y)
            x2, y2 = self._to_screen(o.max_x, o.max_y)
            p.setBrush(QBrush(QColor(20, 80, 20, 60)))
            p.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            return

        # draw as polyline
        from PyQt5.QtGui import QPolygonF
        poly = QPolygonF()
        for pt in pts:
            sx, sy = self._to_screen(pt.x, pt.y)
            poly.append(QPointF(sx, sy))
        p.drawPolyline(poly)

    def _draw_traces(self, p):
        for tr in self.pcb.traces:
            if not self.is_layer_visible(tr.layer):
                continue
            color = layer_color(tr.layer, 200)
            w = max(1, int(tr.width_mm * self._zoom))
            p.setPen(QPen(color, w))
            x1, y1 = self._to_screen(tr.start.x, tr.start.y)
            x2, y2 = self._to_screen(tr.end.x, tr.end.y)
            p.drawLine(int(x1), int(y1), int(x2), int(y2))

        # arc traces approximated by segments
        for arc in self.pcb.arc_traces:
            if not self.is_layer_visible(arc.layer):
                continue
            color = layer_color(arc.layer, 200)
            w = max(1, int(arc.width_mm * self._zoom))
            p.setPen(QPen(color, w))
            pts = self._arc_polyline(arc.start.x, arc.start.y, arc.mid.x, arc.mid.y, arc.end.x, arc.end.y, n=24)
            for (x1,y1),(x2,y2) in zip(pts, pts[1:]):
                sx1, sy1 = self._to_screen(x1,y1)
                sx2, sy2 = self._to_screen(x2,y2)
                p.drawLine(int(sx1), int(sy1), int(sx2), int(sy2))

    def _arc_polyline(self, x1,y1,xm,ym,x2,y2,n=24):
        # Circle through three points -> angles
        import numpy as np
        p1 = np.array([x1,y1], float)
        pm = np.array([xm,ym], float)
        p2 = np.array([x2,y2], float)
        def perp_bis(a,b):
            mid=(a+b)/2
            d=b-a
            return mid,d
        m1,d1=perp_bis(p1,pm)
        m2,d2=perp_bis(pm,p2)
        A=np.array([[d1[0], d1[1]],[d2[0], d2[1]]],float)
        B=np.array([np.dot(d1,m1), np.dot(d2,m2)],float)
        try:
            c=np.linalg.solve(A,B)
        except Exception:
            return [(x1,y1),(x2,y2)]
        r=float(np.linalg.norm(p1-c))
        a1=math.atan2(y1-c[1], x1-c[0])
        am=math.atan2(ym-c[1], xm-c[0])
        a2=math.atan2(y2-c[1], x2-c[0])
        # choose direction so that am is between a1 and a2
        def norm(a):
            while a<0: a+=2*math.pi
            while a>=2*math.pi: a-=2*math.pi
            return a
        a1n,amn,a2n = norm(a1),norm(am),norm(a2)
        # try CCW span
        def ccw_span(a,b):
            return (b-a)%(2*math.pi)
        span_ccw = ccw_span(a1n,a2n)
        span_ccw_mid = ccw_span(a1n,amn)
        ccw_ok = span_ccw_mid <= span_ccw
        if ccw_ok:
            angles=[a1n + span_ccw*(i/(n-1)) for i in range(n)]
        else:
            span_cw = ccw_span(a2n,a1n)
            angles=[a1n - span_cw*(i/(n-1)) for i in range(n)]
        return [(float(c[0]+r*math.cos(a)), float(c[1]+r*math.sin(a))) for a in angles]

    def _draw_vias(self, p):
        for v in self.pcb.vias:
            # show if either end layer visible
            if not (self.is_layer_visible(v.layers[0]) or self.is_layer_visible(v.layers[1])):
                continue
            color = QColor(220, 220, 220, 200)
            p.setPen(QPen(color, 1))
            p.setBrush(QBrush(QColor(220, 220, 220, 120)))
            x, y = self._to_screen(v.pos.x, v.pos.y)
            r = max(2, int((v.diam_mm/2) * self._zoom))
            p.drawEllipse(int(x - r), int(y - r), int(2*r), int(2*r))

    def _draw_zones(self, p):
        from PyQt5.QtGui import QPolygonF
        for z in self.pcb.zones:
            if not self.is_layer_visible(z.layer):
                continue
            if not z.polygon:
                continue
            color = layer_color(z.layer, 80)
            p.setPen(QPen(layer_color(z.layer, 160), 1))
            p.setBrush(QBrush(color))
            poly = QPolygonF()
            for pt in z.polygon:
                sx, sy = self._to_screen(pt.x, pt.y)
                poly.append(QPointF(sx, sy))
            p.drawPolygon(poly)

    def _draw_pads(self, p):
        for comp in self.pcb.components:
            for pad in comp.pads:
                # draw if any pad layer visible (fallback to comp side)
                pad_layers = pad.layers or [comp.layer]
                if not any(self.is_layer_visible(l) for l in pad_layers):
                    continue
                color = QColor(245, 200, 120, 160)
                p.setPen(QPen(QColor(245, 200, 120, 220), 1))
                p.setBrush(QBrush(color))
                x, y = self._to_screen(pad.pos.x, pad.pos.y)
                w = max(2, int(pad.w * self._zoom))
                h = max(2, int(pad.h * self._zoom))
                p.drawRect(int(x - w/2), int(y - h/2), int(w), int(h))

    def _draw_components(self, p):
        p.setPen(QPen(QColor(120, 160, 245, 220), 2))
        for comp in self.pcb.components:
            bb0, bb1 = comp.bbox()
            x1, y1 = self._to_screen(bb0.x, bb0.y)
            x2, y2 = self._to_screen(bb1.x, bb1.y)
            if comp.ref == self._selected_ref:
                p.setBrush(QBrush(QColor(0, 150, 136, 80)))
            else:
                p.setBrush(Qt.NoBrush)
            p.drawRect(int(x1), int(y1), int(x2-x1), int(y2-y1))
            p.setFont(QFont("sans-serif", 9))
            p.drawText(int(x1)+2, int(y1)-2, comp.ref)


class LayerPanel(QWidget):
    """
    Scrollable list of per-layer checkboxes + quick actions.
    """
    layersChanged = pyqtSignal(set)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layers: List[str] = []
        self._cbs: Dict[str, QCheckBox] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(6)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("Copper layers"))
        hdr.addStretch()
        btn_all = QPushButton("All")
        btn_none = QPushButton("None")
        btn_all.clicked.connect(self._all_on)
        btn_none.clicked.connect(self._all_off)
        hdr.addWidget(btn_all)
        hdr.addWidget(btn_none)
        root.addLayout(hdr)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(self.scroll.NoFrame)

        self.inner = QWidget()
        self.inner_layout = QVBoxLayout(self.inner)
        self.inner_layout.setContentsMargins(6,6,6,6)
        self.inner_layout.setSpacing(4)
        self.inner_layout.addStretch(1)

        self.scroll.setWidget(self.inner)
        root.addWidget(self.scroll, 1)

    def set_layers(self, layers: List[str], checked: bool = True):
        self._layers = list(layers)
        # clear
        for cb in self._cbs.values():
            cb.setParent(None)
        self._cbs.clear()

        # rebuild
        # remove stretch at end
        while self.inner_layout.count():
            item = self.inner_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        for name in self._layers:
            cb = QCheckBox(name)
            cb.setChecked(checked)
            cb.toggled.connect(lambda _v, _n=name: self._emit())
            self.inner_layout.addWidget(cb)
            self._cbs[name] = cb
        self.inner_layout.addStretch(1)
        self._emit()

    def layers(self) -> set:
        return {n for n, cb in self._cbs.items() if cb.isChecked()}

    def _emit(self):
        self.layersChanged.emit(self.layers())

    def _all_on(self):
        for cb in self._cbs.values():
            cb.setChecked(True)
        self._emit()

    def _all_off(self):
        for cb in self._cbs.values():
            cb.setChecked(False)
        self._emit()
