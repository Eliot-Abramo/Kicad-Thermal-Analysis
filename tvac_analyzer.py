#!/usr/bin/env python3
"""
TVAC Thermal Analyzer v4 — Standalone Edition
==============================================
Physically rigorous thermal analysis for space-electronics PCBs in TVAC.

Reads KiCad 9 .kicad_pcb files directly (no pcbnew dependency).
Generates adaptive thermal meshes with via modeling, solves heat conduction
+ radiation in TVAC conditions with energy balance verification.

Features:
  - Adaptive mesh refinement near vias, heat sources, copper transitions
  - Explicit via thermal modeling between copper layers
  - Proper copper fraction computation per mesh cell
  - Linearized radiation Picard iteration with IC(0) preconditioner
  - Separate top/bottom surface emissivity
  - Energy balance verification
  - KiCad 9 full support (arcs, filled_polygon, property nodes)

Usage:
    python tvac_analyzer.py                     # opens file dialog
    python tvac_analyzer.py board.kicad_pcb     # open specific file

Dependencies:
    PyQt5, numpy, scipy  (pip install PyQt5 numpy scipy)

Version: 4.0.0
"""
from __future__ import annotations

import sys, os, re, json, math, time, ctypes, traceback
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, cg as scipy_cg

# ═══════════════════════════════════════════════════════════════════════
# §1  S-Expression Parser  (reads .kicad_pcb)
# ═══════════════════════════════════════════════════════════════════════

def _tokenize(text: str):
    """Tokenize KiCad S-expression format. Handles quoted strings with escapes."""
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in ' \t\r\n':
            i += 1
        elif c == '(':
            yield '('
            i += 1
        elif c == ')':
            yield ')'
            i += 1
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                if text[j] == '\\':
                    j += 1
                j += 1
            yield text[i+1:j]
            i = j + 1
        else:
            j = i
            while j < n and text[j] not in ' \t\r\n()':
                j += 1
            yield text[i:j]
            i = j


def parse_sexpr(text: str):
    """Parse S-expression text into nested list structure."""
    tokens = list(_tokenize(text))
    pos = [0]

    def _parse():
        if pos[0] >= len(tokens):
            return None
        tok = tokens[pos[0]]
        if tok == '(':
            pos[0] += 1
            lst = []
            while pos[0] < len(tokens) and tokens[pos[0]] != ')':
                item = _parse()
                if item is not None:
                    lst.append(item)
            pos[0] += 1  # skip ')'
            return lst
        elif tok == ')':
            pos[0] += 1
            return None
        else:
            pos[0] += 1
            return tok
    return _parse()


def _find(node, tag):
    """Find first child node with given tag."""
    if not isinstance(node, list):
        return None
    for c in node:
        if isinstance(c, list) and c and c[0] == tag:
            return c
    return None


def _find_all(node, tag):
    """Find all child nodes with given tag."""
    if not isinstance(node, list):
        return []
    return [c for c in node if isinstance(c, list) and c and c[0] == tag]


def _val(node, tag, default=None):
    """Get string value of a tagged child."""
    c = _find(node, tag)
    return c[1] if c and len(c) > 1 else default


def _float(node, tag, default=0.0):
    """Get float value of a tagged child."""
    try:
        return float(_val(node, tag, default))
    except (TypeError, ValueError):
        return default


# ═══════════════════════════════════════════════════════════════════════
# §2  Data Classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Pt:
    x: float = 0.0
    y: float = 0.0


@dataclass
class Pad:
    ref: str = ""
    name: str = ""
    net_name: str = ""
    net_code: int = 0
    pos: Pt = field(default_factory=Pt)
    w: float = 0.0
    h: float = 0.0
    shape: str = "rect"  # rect, circle, oval, roundrect, custom
    layers: List[str] = field(default_factory=list)
    drill: float = 0.0
    pad_type: str = "smd"  # smd, thru_hole, np_thru_hole, connect


@dataclass
class Component:
    ref: str = ""
    value: str = ""
    footprint: str = ""
    pos: Pt = field(default_factory=Pt)
    rotation: float = 0.0
    layer: str = "F.Cu"
    pads: List[Pad] = field(default_factory=list)
    power_w: float = 0.0

    def bbox(self) -> Tuple[Pt, Pt]:
        if not self.pads:
            s = 2.0
            return Pt(self.pos.x - s, self.pos.y - s), Pt(self.pos.x + s, self.pos.y + s)
        xs = [p.pos.x for p in self.pads]
        ys = [p.pos.y for p in self.pads]
        ws = [p.w / 2 for p in self.pads]
        hs = [p.h / 2 for p in self.pads]
        return (Pt(min(x - w for x, w in zip(xs, ws)),
                   min(y - h for y, h in zip(ys, hs))),
                Pt(max(x + w for x, w in zip(xs, ws)),
                   max(y + h for y, h in zip(ys, hs))))


@dataclass
class Trace:
    net_name: str = ""
    net_code: int = 0
    start: Pt = field(default_factory=Pt)
    end: Pt = field(default_factory=Pt)
    width_mm: float = 0.0
    layer: str = ""

    @property
    def length_mm(self):
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return math.sqrt(dx * dx + dy * dy)


@dataclass
class ArcTrace:
    """Arc segment on a copper layer."""
    net_name: str = ""
    net_code: int = 0
    start: Pt = field(default_factory=Pt)
    mid: Pt = field(default_factory=Pt)
    end: Pt = field(default_factory=Pt)
    width_mm: float = 0.0
    layer: str = ""


@dataclass
class Via:
    pos: Pt = field(default_factory=Pt)
    drill_mm: float = 0.3
    diam_mm: float = 0.6
    net_name: str = ""
    net_code: int = 0
    layers: Tuple[str, str] = ("F.Cu", "B.Cu")
    via_type: str = "through"  # through, blind, micro


@dataclass
class Zone:
    net_name: str = ""
    net_code: int = 0
    layer: str = ""
    outline: List[Pt] = field(default_factory=list)
    filled_polygons: List[List[Pt]] = field(default_factory=list)
    priority: int = 0
    fill_percent: float = 1.0  # estimated copper fill fraction


@dataclass
class BoardOutline:
    outline: List[Pt] = field(default_factory=list)
    min_x: float = 0.0
    max_x: float = 100.0
    min_y: float = 0.0
    max_y: float = 100.0

    def area_mm2(self):
        pts = self.outline
        if len(pts) < 3:
            return (self.max_x - self.min_x) * (self.max_y - self.min_y)
        a = sum(pts[i].x * pts[(i + 1) % len(pts)].y -
                pts[(i + 1) % len(pts)].x * pts[i].y
                for i in range(len(pts)))
        return abs(a) / 2.0


@dataclass
class PCBData:
    board_thickness_mm: float = 1.6
    copper_layers: List[str] = field(default_factory=list)
    copper_thickness_um: Dict[str, float] = field(default_factory=dict)
    outline: BoardOutline = field(default_factory=BoardOutline)
    components: List[Component] = field(default_factory=list)
    traces: List[Trace] = field(default_factory=list)
    arc_traces: List[ArcTrace] = field(default_factory=list)
    vias: List[Via] = field(default_factory=list)
    zones: List[Zone] = field(default_factory=list)
    nets: Dict[int, str] = field(default_factory=dict)
    mounting_holes: List[Pt] = field(default_factory=list)
    layer_stackup: List[dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# §3  KiCad 9 PCB File Reader
# ═══════════════════════════════════════════════════════════════════════

class KiCadPCBReader:
    """
    Reads KiCad 9 .kicad_pcb files.

    KiCad 9 format changes vs older versions:
    - Uses 'property' nodes for Reference/Value (not fp_text)
    - Zone fills stored in 'filled_polygon' under 'fill' node
    - Arc segments in 'arc' nodes with start/mid/end
    - UUIDs for identification
    - Stackup info in setup
    """

    def read(self, path: str) -> PCBData:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        tree = parse_sexpr(text)
        if not tree or tree[0] != 'kicad_pcb':
            raise ValueError("Not a valid kicad_pcb file")
        return self._extract(tree)

    def _extract(self, root) -> PCBData:
        pcb = PCBData()
        pcb.nets = self._read_nets(root)
        pcb.copper_layers = self._read_layers(root)
        pcb.outline = self._read_outline(root)
        pcb.components = self._read_footprints(root, pcb.nets)
        pcb.traces = self._read_segments(root, pcb.nets)
        pcb.arc_traces = self._read_arcs(root, pcb.nets)
        pcb.vias = self._read_vias(root, pcb.nets)
        pcb.zones = self._read_zones(root, pcb.nets)
        pcb.mounting_holes = self._find_mounting_holes(pcb.components)

        # Read stackup for copper thickness info
        self._read_stackup(root, pcb)

        # Compute board bounds
        if pcb.outline.outline:
            pcb.outline.min_x = min(p.x for p in pcb.outline.outline)
            pcb.outline.max_x = max(p.x for p in pcb.outline.outline)
            pcb.outline.min_y = min(p.y for p in pcb.outline.outline)
            pcb.outline.max_y = max(p.y for p in pcb.outline.outline)
        elif pcb.components:
            xs = [c.pos.x for c in pcb.components]
            ys = [c.pos.y for c in pcb.components]
            m = 15.0
            pcb.outline.min_x = min(xs) - m
            pcb.outline.max_x = max(xs) + m
            pcb.outline.min_y = min(ys) - m
            pcb.outline.max_y = max(ys) + m

        return pcb

    def _read_nets(self, root):
        nets = {}
        for n in _find_all(root, 'net'):
            if len(n) >= 3:
                try:
                    nets[int(n[1])] = n[2]
                except (ValueError, TypeError):
                    pass
        return nets

    def _read_layers(self, root):
        layers_node = _find(root, 'layers')
        cu = []
        if layers_node:
            for c in layers_node[1:]:
                if isinstance(c, list) and len(c) >= 3:
                    name = c[1] if isinstance(c[1], str) else str(c[1])
                    ltype = c[2] if len(c) > 2 else ""
                    if isinstance(ltype, str) and ltype in ("signal", "power", "mixed"):
                        cu.append(name)
        return cu or ["F.Cu", "B.Cu"]

    def _read_stackup(self, root, pcb):
        """Read stackup for layer thicknesses."""
        setup = _find(root, 'setup')
        if not setup:
            return
        st = _find(setup, 'stackup')
        if not st:
            return

        total_thickness = 0.0
        for layer_node in _find_all(st, 'layer'):
            if len(layer_node) < 2:
                continue
            layer_name = layer_node[1]
            thickness = _float(layer_node, 'thickness', 0.0)
            total_thickness += thickness

            # Store copper thickness
            if isinstance(layer_name, str) and '.Cu' in layer_name:
                pcb.copper_thickness_um[layer_name] = thickness * 1000  # mm to μm

            pcb.layer_stackup.append({
                'name': layer_name,
                'thickness_mm': thickness,
                'type': _val(layer_node, 'type', ''),
            })

        if total_thickness > 0:
            pcb.board_thickness_mm = total_thickness

    def _read_outline(self, root):
        bo = BoardOutline()
        pts = []

        # gr_line on Edge.Cuts
        for line in _find_all(root, 'gr_line'):
            if _val(line, 'layer', '') == 'Edge.Cuts':
                s, e = _find(line, 'start'), _find(line, 'end')
                if s and e and len(s) >= 3 and len(e) >= 3:
                    try:
                        pts.append(Pt(float(s[1]), float(s[2])))
                        pts.append(Pt(float(e[1]), float(e[2])))
                    except (ValueError, TypeError):
                        pass

        # gr_rect on Edge.Cuts
        for rect in _find_all(root, 'gr_rect'):
            if _val(rect, 'layer', '') == 'Edge.Cuts':
                s, e = _find(rect, 'start'), _find(rect, 'end')
                if s and e and len(s) >= 3 and len(e) >= 3:
                    try:
                        x1, y1 = float(s[1]), float(s[2])
                        x2, y2 = float(e[1]), float(e[2])
                        pts.extend([Pt(x1, y1), Pt(x2, y1), Pt(x2, y2), Pt(x1, y2)])
                    except (ValueError, TypeError):
                        pass

        # gr_poly on Edge.Cuts
        for poly in _find_all(root, 'gr_poly'):
            if _val(poly, 'layer', '') == 'Edge.Cuts':
                ptsnode = _find(poly, 'pts')
                if ptsnode:
                    for xy in _find_all(ptsnode, 'xy'):
                        if len(xy) >= 3:
                            try:
                                pts.append(Pt(float(xy[1]), float(xy[2])))
                            except (ValueError, TypeError):
                                pass

        # gr_arc on Edge.Cuts — approximate with line segments
        for arc in _find_all(root, 'gr_arc'):
            if _val(arc, 'layer', '') == 'Edge.Cuts':
                arc_pts = self._arc_to_points(arc)
                pts.extend(arc_pts)

        if pts:
            # Deduplicate and order
            seen = set()
            unique = []
            for p in pts:
                key = (round(p.x, 3), round(p.y, 3))
                if key not in seen:
                    seen.add(key)
                    unique.append(p)
            if len(unique) >= 3:
                cx = sum(p.x for p in unique) / len(unique)
                cy = sum(p.y for p in unique) / len(unique)
                unique.sort(key=lambda p: math.atan2(p.y - cy, p.x - cx))
            bo.outline = unique

        return bo

    def _arc_to_points(self, arc_node, n_segments=16):
        """Convert arc to series of points for approximation."""
        pts = []
        # KiCad 9: arc has start, mid, end
        start_n = _find(arc_node, 'start')
        mid_n = _find(arc_node, 'mid')
        end_n = _find(arc_node, 'end')

        if not start_n or not end_n:
            return pts

        try:
            sx, sy = float(start_n[1]), float(start_n[2])
            ex, ey = float(end_n[1]), float(end_n[2])
        except (ValueError, TypeError, IndexError):
            return pts

        if mid_n and len(mid_n) >= 3:
            try:
                mx, my = float(mid_n[1]), float(mid_n[2])
                # Compute circle center from 3 points
                center = self._circle_center(sx, sy, mx, my, ex, ey)
                if center:
                    cx, cy = center
                    r = math.sqrt((sx - cx)**2 + (sy - cy)**2)
                    a_start = math.atan2(sy - cy, sx - cx)
                    a_end = math.atan2(ey - cy, ex - cx)
                    a_mid = math.atan2(my - cy, mx - cx)

                    # Determine direction
                    def normalize_angle(a):
                        while a < -math.pi: a += 2 * math.pi
                        while a > math.pi: a -= 2 * math.pi
                        return a

                    da1 = normalize_angle(a_mid - a_start)
                    da2 = normalize_angle(a_end - a_mid)

                    if da1 > 0 and da2 > 0:
                        # CCW
                        total = normalize_angle(a_end - a_start)
                        if total <= 0:
                            total += 2 * math.pi
                    else:
                        # CW
                        total = normalize_angle(a_end - a_start)
                        if total >= 0:
                            total -= 2 * math.pi

                    for i in range(n_segments + 1):
                        t = i / n_segments
                        a = a_start + total * t
                        pts.append(Pt(cx + r * math.cos(a), cy + r * math.sin(a)))
                    return pts
            except (ValueError, TypeError, IndexError):
                pass

        # Fallback: just use start and end
        pts.append(Pt(sx, sy))
        pts.append(Pt(ex, ey))
        return pts

    @staticmethod
    def _circle_center(x1, y1, x2, y2, x3, y3):
        """Compute circle center from 3 points."""
        ax, ay = x1, y1
        bx, by = x2, y2
        cx, cy = x3, y3
        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-10:
            return None
        ux = ((ax*ax + ay*ay) * (by - cy) +
              (bx*bx + by*by) * (cy - ay) +
              (cx*cx + cy*cy) * (ay - by)) / d
        uy = ((ax*ax + ay*ay) * (cx - bx) +
              (bx*bx + by*by) * (ax - cx) +
              (cx*cx + cy*cy) * (bx - ax)) / d
        return (ux, uy)

    def _read_footprints(self, root, nets):
        comps = []
        for fp in _find_all(root, 'footprint'):
            c = Component()
            if len(fp) > 1:
                c.footprint = fp[1] if isinstance(fp[1], str) else str(fp[1])

            at = _find(fp, 'at')
            if at and len(at) >= 3:
                try:
                    c.pos = Pt(float(at[1]), float(at[2]))
                    if len(at) > 3:
                        c.rotation = float(at[3])
                except (ValueError, TypeError):
                    pass

            c.layer = _val(fp, 'layer', 'F.Cu')

            # KiCad 9: properties
            for prop in _find_all(fp, 'property'):
                if len(prop) >= 3:
                    if prop[1] == 'Reference':
                        c.ref = prop[2]
                    elif prop[1] == 'Value':
                        c.value = prop[2]
                    elif prop[1] == 'POWER_DISSIPATION':
                        try:
                            c.power_w = float(prop[2])
                        except (ValueError, TypeError):
                            pass

            # Fallback: fp_text (KiCad 5/6 compat)
            if not c.ref:
                for ft in _find_all(fp, 'fp_text'):
                    if len(ft) >= 3:
                        if ft[1] == 'reference':
                            c.ref = ft[2]
                        elif ft[1] == 'value' and not c.value:
                            c.value = ft[2]

            # Determine if component is on back side (for pad mirroring)
            is_back = c.layer.startswith('B.')

            # Parse pads
            for pad_node in _find_all(fp, 'pad'):
                pad = Pad()
                if len(pad_node) >= 2:
                    pad.name = pad_node[1] if isinstance(pad_node[1], str) else str(pad_node[1])
                if len(pad_node) >= 3:
                    pad.pad_type = pad_node[2] if isinstance(pad_node[2], str) else "smd"
                if len(pad_node) >= 4:
                    pad.shape = pad_node[3] if isinstance(pad_node[3], str) else "rect"
                pad.ref = c.ref

                at_p = _find(pad_node, 'at')
                if at_p and len(at_p) >= 3:
                    try:
                        lx, ly = float(at_p[1]), float(at_p[2])

                        # Mirror X for back-side components
                        if is_back:
                            lx = -lx

                        # Apply rotation
                        rot = math.radians(c.rotation)
                        pad.pos = Pt(
                            c.pos.x + lx * math.cos(rot) - ly * math.sin(rot),
                            c.pos.y + lx * math.sin(rot) + ly * math.cos(rot)
                        )
                    except (ValueError, TypeError):
                        pad.pos = Pt(c.pos.x, c.pos.y)

                sz = _find(pad_node, 'size')
                if sz and len(sz) >= 3:
                    try:
                        pad.w, pad.h = float(sz[1]), float(sz[2])
                    except (ValueError, TypeError):
                        pass

                dr = _find(pad_node, 'drill')
                if dr and len(dr) >= 2:
                    try:
                        pad.drill = float(dr[1])
                    except (ValueError, TypeError):
                        pass

                net_node = _find(pad_node, 'net')
                if net_node and len(net_node) >= 3:
                    try:
                        pad.net_code = int(net_node[1])
                        pad.net_name = net_node[2]
                    except (ValueError, TypeError):
                        pass

                layers_node = _find(pad_node, 'layers')
                if layers_node:
                    pad.layers = [l for l in layers_node[1:] if isinstance(l, str)]

                c.pads.append(pad)

            if c.ref:
                comps.append(c)
        return comps

    def _read_segments(self, root, nets):
        traces = []
        for seg in _find_all(root, 'segment'):
            t = Trace()
            s, e = _find(seg, 'start'), _find(seg, 'end')
            if s and len(s) >= 3 and e and len(e) >= 3:
                try:
                    t.start = Pt(float(s[1]), float(s[2]))
                    t.end = Pt(float(e[1]), float(e[2]))
                except (ValueError, TypeError):
                    continue
            t.width_mm = _float(seg, 'width', 0.25)
            t.layer = _val(seg, 'layer', '')
            net_node = _find(seg, 'net')
            if net_node and len(net_node) >= 2:
                try:
                    t.net_code = int(net_node[1])
                    t.net_name = nets.get(t.net_code, '')
                except (ValueError, TypeError):
                    pass
            traces.append(t)
        return traces

    def _read_arcs(self, root, nets):
        """Read arc segments (KiCad 7+)."""
        arcs = []
        for arc in _find_all(root, 'arc'):
            # Check this is a trace arc (has net), not a graphic arc
            net_node = _find(arc, 'net')
            if not net_node:
                continue
            a = ArcTrace()
            s = _find(arc, 'start')
            m = _find(arc, 'mid')
            e = _find(arc, 'end')
            if s and len(s) >= 3 and e and len(e) >= 3:
                try:
                    a.start = Pt(float(s[1]), float(s[2]))
                    a.end = Pt(float(e[1]), float(e[2]))
                except (ValueError, TypeError):
                    continue
            if m and len(m) >= 3:
                try:
                    a.mid = Pt(float(m[1]), float(m[2]))
                except (ValueError, TypeError):
                    pass
            a.width_mm = _float(arc, 'width', 0.25)
            a.layer = _val(arc, 'layer', '')
            if net_node and len(net_node) >= 2:
                try:
                    a.net_code = int(net_node[1])
                    a.net_name = nets.get(a.net_code, '')
                except (ValueError, TypeError):
                    pass
            arcs.append(a)
        return arcs

    def _read_vias(self, root, nets):
        vias = []
        for v in _find_all(root, 'via'):
            via = Via()
            at = _find(v, 'at')
            if at and len(at) >= 3:
                try:
                    via.pos = Pt(float(at[1]), float(at[2]))
                except (ValueError, TypeError):
                    continue
            via.diam_mm = _float(v, 'size', 0.6)
            via.drill_mm = _float(v, 'drill', 0.3)

            # Via type
            vtype = _val(v, 'type', 'through')
            if vtype:
                via.via_type = vtype

            net_node = _find(v, 'net')
            if net_node and len(net_node) >= 2:
                try:
                    via.net_code = int(net_node[1])
                    via.net_name = nets.get(via.net_code, '')
                except (ValueError, TypeError):
                    pass

            layers_node = _find(v, 'layers')
            if layers_node and len(layers_node) >= 3:
                via.layers = (layers_node[1], layers_node[2])

            vias.append(via)
        return vias

    def _read_zones(self, root, nets):
        zones = []
        for z in _find_all(root, 'zone'):
            zone = Zone()

            net_node = _find(z, 'net')
            if net_node and len(net_node) >= 2:
                try:
                    zone.net_code = int(net_node[1])
                except (ValueError, TypeError):
                    pass

            nn = _find(z, 'net_name')
            zone.net_name = nn[1] if nn and len(nn) >= 2 else nets.get(zone.net_code, '')

            zone.layer = _val(z, 'layer', '')
            if not zone.layer:
                ln = _find(z, 'layers')
                if ln and len(ln) > 1:
                    zone.layer = ln[1]

            try:
                zone.priority = int(_val(z, 'priority', '0'))
            except (ValueError, TypeError):
                pass

            # Read the zone outline (user-drawn boundary)
            poly = _find(z, 'polygon')
            if poly:
                ptsnode = _find(poly, 'pts')
                if ptsnode:
                    for xy in _find_all(ptsnode, 'xy'):
                        if len(xy) >= 3:
                            try:
                                zone.outline.append(Pt(float(xy[1]), float(xy[2])))
                            except (ValueError, TypeError):
                                pass

            # Read filled polygons (actual copper fill) — KiCad 6+
            # These are under (filled_polygon (pts (xy ...)))
            for fp in _find_all(z, 'filled_polygon'):
                fill_layer = _val(fp, 'layer', zone.layer)
                ptsnode = _find(fp, 'pts')
                if ptsnode:
                    poly_pts = []
                    for xy in _find_all(ptsnode, 'xy'):
                        if len(xy) >= 3:
                            try:
                                poly_pts.append(Pt(float(xy[1]), float(xy[2])))
                            except (ValueError, TypeError):
                                pass
                    if poly_pts:
                        zone.filled_polygons.append(poly_pts)

            if zone.outline or zone.filled_polygons:
                zones.append(zone)

        return zones

    @staticmethod
    def _find_mounting_holes(comps):
        holes = []
        for c in comps:
            fp = c.footprint.lower()
            ref = c.ref.lower()
            if ('mountinghole' in fp or 'mounting_hole' in fp or
                'mounting-hole' in fp or ref.startswith('h')):
                holes.append(c.pos)
        return holes


# ═══════════════════════════════════════════════════════════════════════
# §4  Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CompPower:
    ref: str = ""
    power_w: float = 0.0
    source: str = "manual"


@dataclass
class HeatsinkCfg:
    hs_id: str = ""
    material: str = "ALUMINUM_6061"
    polygon: List[Tuple[float, float]] = field(default_factory=list)
    thickness_mm: float = 3.0
    emissivity: Optional[float] = None


@dataclass
class MountingCfg:
    mp_id: str = ""
    x_mm: float = 0.0
    y_mm: float = 0.0
    diameter_mm: float = 3.2
    fixed_temp_c: Optional[float] = None
    thermal_resistance: float = 0.0


@dataclass
class CurrentPathCfg:
    path_id: str = ""
    source_net: str = ""
    sink_net: str = ""
    current_a: float = 1.0
    description: str = ""


@dataclass
class SimCfg:
    resolution_mm: float = 1.0
    mode: str = "steady_state"
    heat_source_mode: str = "component_power"
    include_radiation: bool = True
    use_adaptive_mesh: bool = True
    ambient_temp_c: float = 25.0
    chamber_wall_temp_c: float = 25.0
    initial_temp_c: float = 25.0
    duration_s: float = 600.0
    timestep_s: float = 1.0
    convergence: float = 1e-6
    max_iterations: int = 50000
    min_cell_mm: float = 0.15
    max_cell_mm: float = 3.0


@dataclass
class Config:
    comp_power: List[CompPower] = field(default_factory=list)
    heatsinks: List[HeatsinkCfg] = field(default_factory=list)
    mounting: List[MountingCfg] = field(default_factory=list)
    current_paths: List[CurrentPathCfg] = field(default_factory=list)
    sim: SimCfg = field(default_factory=SimCfg)

    def get_power(self, ref):
        for cp in self.comp_power:
            if cp.ref == ref:
                return cp.power_w
        return 0.0

    def set_power(self, ref, pw, src="manual"):
        for cp in self.comp_power:
            if cp.ref == ref:
                cp.power_w = pw
                cp.source = src
                return
        self.comp_power.append(CompPower(ref=ref, power_w=pw, source=src))

    def total_power(self):
        return sum(cp.power_w for cp in self.comp_power)

    def save(self, path):
        data = {
            'version': '4.0.0',
            'comp_power': [asdict(cp) for cp in self.comp_power],
            'heatsinks': [asdict(hs) for hs in self.heatsinks],
            'mounting': [asdict(mp) for mp in self.mounting],
            'current_paths': [asdict(cp) for cp in self.current_paths],
            'sim': asdict(self.sim),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        cfg = cls()
        for cp in data.get('comp_power', []):
            cfg.comp_power.append(CompPower(**{k: v for k, v in cp.items()
                                               if k in CompPower.__dataclass_fields__}))
        for hs in data.get('heatsinks', []):
            cfg.heatsinks.append(HeatsinkCfg(**{k: v for k, v in hs.items()
                                                if k in HeatsinkCfg.__dataclass_fields__}))
        for mp in data.get('mounting', []):
            cfg.mounting.append(MountingCfg(**{k: v for k, v in mp.items()
                                               if k in MountingCfg.__dataclass_fields__}))
        for cp in data.get('current_paths', []):
            cfg.current_paths.append(CurrentPathCfg(**{k: v for k, v in cp.items()
                                                       if k in CurrentPathCfg.__dataclass_fields__}))
        sim = data.get('sim', {})
        cfg.sim = SimCfg(**{k: v for k, v in sim.items()
                            if k in SimCfg.__dataclass_fields__})
        return cfg


# ═══════════════════════════════════════════════════════════════════════
# §5  Physical Constants & Materials
# ═══════════════════════════════════════════════════════════════════════

SIGMA = 5.670374419e-8  # Stefan-Boltzmann W/(m²·K⁴)
C2K = 273.15
CU_RESISTIVITY = 1.724e-8  # Ω·m at 20°C (IACS standard, IEC 60028)
CU_TEMP_COEFF = 0.00393   # 1/K, for temperature-dependent resistivity

MATERIALS = {
    'FR4':         dict(k=0.29,  cp=1100, rho=1850, emiss=0.90),
    'FR4_HIGH_TG': dict(k=0.35,  cp=1100, rho=1900, emiss=0.90),
    'COPPER':      dict(k=401.0, cp=385,  rho=8960, emiss=0.03),
    'CU_OXIDIZED': dict(k=401.0, cp=385,  rho=8960, emiss=0.65),
    'SOLDER_MASK': dict(k=0.25,  cp=1200, rho=1200, emiss=0.90),
    'PREPREG':     dict(k=0.29,  cp=1100, rho=1850, emiss=0.90),
}

HEATSINK_MATERIALS = {
    'ALUMINUM_6061':   dict(k=167.0, cp=896,  rho=2700, emiss=0.09),
    'ALUMINUM_6063':   dict(k=200.0, cp=900,  rho=2690, emiss=0.09),
    'AL_ANODIZED':     dict(k=167.0, cp=896,  rho=2700, emiss=0.85),
    'COPPER_HEATSINK': dict(k=385.0, cp=385,  rho=8960, emiss=0.65),
}


# ═══════════════════════════════════════════════════════════════════════
# §6  Thermal Mesh & Node
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TNode:
    """Thermal mesh node with full physical properties."""
    idx: int = 0
    x: float = 0.0        # mm
    y: float = 0.0        # mm
    z: float = 0.0        # mm
    layer: int = 0
    dx: float = 1.0       # cell width in mm
    dy: float = 1.0       # cell height in mm
    dz: float = 0.2       # cell depth in mm

    # Material properties
    k: float = 0.29       # W/(m·K) effective in-plane conductivity (parallel mixture)
    k_z: float = 0.29     # W/(m·K) effective through-plane conductivity (series mixture)
    cp: float = 1100.0    # J/(kg·K)
    rho: float = 1850.0   # kg/m³
    emiss_top: float = 0.9
    emiss_bot: float = 0.9
    copper_frac: float = 0.0  # copper area fraction [0,1]

    # Geometric quantities
    vol: float = 0.0      # m³
    surf_top: float = 0.0 # m² (top surface for radiation)
    surf_bot: float = 0.0 # m² (bottom surface for radiation)
    heat: float = 0.0     # W

    # Boundary conditions
    fixed: bool = False
    T_fixed: float = 25.0 # °C

    # Connectivity
    nbrs: Dict[int, float] = field(default_factory=dict)  # {node_idx: conductance_W_per_K}


@dataclass
class TMesh:
    nodes: List[TNode] = field(default_factory=list)
    nx: int = 0
    ny: int = 0
    nz: int = 0
    bmin_x: float = 0.0
    bmax_x: float = 100.0
    bmin_y: float = 0.0
    bmax_y: float = 100.0
    # Adaptive mesh: nodes may not be on regular grid
    is_adaptive: bool = False
    # For regular grid, provide mapping
    _grid_map: Optional[np.ndarray] = field(default=None, repr=False)

    def idx(self, ix, iy, iz):
        return iz * self.nx * self.ny + iy * self.nx + ix


@dataclass
class TResult:
    temps: np.ndarray = field(default_factory=lambda: np.array([]))
    min_t: float = 0.0
    max_t: float = 0.0
    avg_t: float = 0.0
    iters: int = 0
    picard_iters: int = 0
    time_s: float = 0.0
    converged: bool = True
    energy_balance: float = 0.0
    error: str = ""

    def grid_2d(self, nx, ny, nz, layer=0):
        g = np.zeros((ny, nx))
        for iy in range(ny):
            for ix in range(nx):
                i = layer * nx * ny + iy * nx + ix
                if i < len(self.temps):
                    g[iy, ix] = self.temps[i]
        return g


# ═══════════════════════════════════════════════════════════════════════
# §7  Adaptive Mesh Generator
#
# Key improvements over v3:
#   1. Adaptive refinement near vias, heat sources, copper transitions
#   2. Explicit via thermal modeling between layers
#   3. Proper copper fraction per cell (geometric intersection)
#   4. Correct Joule heating distribution
#   5. Separate top/bottom emissivity
# ═══════════════════════════════════════════════════════════════════════

class MeshGen:
    def __init__(self, pcb: PCBData, cfg: Config):
        self.pcb = pcb
        self.cfg = cfg

    def generate(self, cb=None) -> TMesh:
        mesh = TMesh()
        o = self.pcb.outline
        bx0, bx1 = o.min_x, o.max_x
        by0, by1 = o.min_y, o.max_y
        if bx1 - bx0 < 1 or by1 - by0 < 1:
            bx0, bx1, by0, by1 = 0, 100, 0, 100
        mesh.bmin_x, mesh.bmax_x = bx0, bx1
        mesh.bmin_y, mesh.bmax_y = by0, by1

        # Determine base resolution
        base_res = self.cfg.sim.resolution_mm
        min_res = self.cfg.sim.min_cell_mm
        max_res = self.cfg.sim.max_cell_mm

        if cb:
            cb(5, "Computing adaptive refinement map...")

        # Build refinement map: a function that returns desired cell size at (x,y)
        refine_pts = self._compute_refinement_points(min_res, base_res)

        if self.cfg.sim.use_adaptive_mesh and refine_pts:
            xs, ys = self._build_adaptive_grid_1d(bx0, bx1, by0, by1,
                                                   base_res, min_res, max_res,
                                                   refine_pts)
        else:
            nx = max(2, int((bx1 - bx0) / base_res) + 1)
            ny = max(2, int((by1 - by0) / base_res) + 1)
            MAX_2D = 15000
            while nx * ny > MAX_2D:
                base_res *= 1.15
                nx = max(2, int((bx1 - bx0) / base_res) + 1)
                ny = max(2, int((by1 - by0) / base_res) + 1)
            xs = np.linspace(bx0, bx1, nx)
            ys = np.linspace(by0, by1, ny)

        nx = len(xs)
        ny = len(ys)
        nz = max(1, min(4, len(self.pcb.copper_layers)))

        # Limit total nodes
        MAX_TOTAL = 200000
        while nx * ny * nz > MAX_TOTAL:
            # Coarsen
            xs = xs[::2] if len(xs) > 4 else xs
            ys = ys[::2] if len(ys) > 4 else ys
            nx = len(xs)
            ny = len(ys)

        mesh.nx, mesh.ny, mesh.nz = nx, ny, nz

        # Compute layer z-positions from stackup
        if nz > 1:
            zs = np.linspace(0, self.pcb.board_thickness_mm, nz)
        else:
            zs = [0.0]

        if cb:
            cb(15, f"Creating {nx}x{ny}x{nz} = {nx*ny*nz} nodes...")

        # Create nodes with per-cell dimensions
        nodes = []
        idx = 0
        for iz in range(nz):
            is_top = (iz == 0)
            is_bot = (iz == nz - 1)
            for iy in range(ny):
                for ix in range(nx):
                    n = TNode(idx=idx, x=xs[ix], y=ys[iy], z=zs[iz], layer=iz)

                    # Compute cell dimensions
                    if ix > 0 and ix < nx - 1:
                        n.dx = (xs[ix + 1] - xs[ix - 1]) / 2.0
                    elif ix == 0 and nx > 1:
                        n.dx = xs[1] - xs[0]
                    elif ix == nx - 1 and nx > 1:
                        n.dx = xs[-1] - xs[-2]
                    else:
                        n.dx = base_res

                    if iy > 0 and iy < ny - 1:
                        n.dy = (ys[iy + 1] - ys[iy - 1]) / 2.0
                    elif iy == 0 and ny > 1:
                        n.dy = ys[1] - ys[0]
                    elif iy == ny - 1 and ny > 1:
                        n.dy = ys[-1] - ys[-2]
                    else:
                        n.dy = base_res

                    n.dz = self.pcb.board_thickness_mm / max(1, nz)

                    # Convert to meters for physical quantities
                    dx_m = n.dx * 1e-3
                    dy_m = n.dy * 1e-3
                    dz_m = n.dz * 1e-3

                    n.vol = dx_m * dy_m * dz_m
                    cell_area = dx_m * dy_m

                    # Surface areas for radiation
                    if is_top and is_bot:
                        n.surf_top = cell_area
                        n.surf_bot = cell_area
                    elif is_top:
                        n.surf_top = cell_area
                        n.surf_bot = 0.0
                    elif is_bot:
                        n.surf_top = 0.0
                        n.surf_bot = cell_area
                    else:
                        n.surf_top = 0.0
                        n.surf_bot = 0.0

                    # Default material: FR4
                    m = MATERIALS['FR4']
                    n.k = m['k']
                    n.cp = m['cp']
                    n.rho = m['rho']
                    n.emiss_top = m['emiss']
                    n.emiss_bot = m['emiss']

                    nodes.append(n)
                    idx += 1

        mesh.nodes = nodes

        if cb:
            cb(25, "Computing copper fractions...")
        self._compute_copper_fractions(mesh, xs, ys)

        if cb:
            cb(40, "Applying material properties...")
        self._apply_material_properties(mesh)

        if cb:
            cb(50, "Applying heatsinks...")
        self._apply_heatsinks(mesh, xs, ys)

        if cb:
            cb(60, "Computing heat sources...")
        self._add_heat_sources(mesh, xs, ys)

        if cb:
            cb(70, "Computing conductances...")
        self._calc_conductances(mesh, xs, ys)

        if cb:
            cb(80, "Adding via thermal paths...")
        self._add_via_conductances(mesh, xs, ys)

        if cb:
            cb(90, "Applying boundary conditions...")
        self._apply_bc(mesh, xs, ys)

        if cb:
            cb(100, f"Mesh complete: {len(nodes)} nodes")

        return mesh

    def _compute_refinement_points(self, min_res, base_res):
        """Compute points that need fine mesh resolution."""
        pts = []
        # Vias — primary thermal conduits, need fine mesh
        for via in self.pcb.vias:
            pts.append((via.pos.x, via.pos.y, min_res, via.diam_mm * 3))

        # Component pads with power
        for cp in self.cfg.comp_power:
            if cp.power_w <= 0:
                continue
            comp = next((c for c in self.pcb.components if c.ref == cp.ref), None)
            if comp:
                r = min_res * 2
                for pad in comp.pads:
                    pts.append((pad.pos.x, pad.pos.y, r, max(pad.w, pad.h) * 2))
                pts.append((comp.pos.x, comp.pos.y, r, 5.0))

        # Mounting holes (thermal boundaries)
        for mp in self.cfg.mounting:
            if mp.fixed_temp_c is not None:
                pts.append((mp.x_mm, mp.y_mm, min_res * 2, mp.diameter_mm * 3))

        return pts

    def _build_adaptive_grid_1d(self, bx0, bx1, by0, by1,
                                 base_res, min_res, max_res,
                                 refine_pts):
        """Build non-uniform 1D grids in x and y with local refinement."""

        def build_axis(a0, a1, base, min_r, max_r, pts_on_axis, is_x):
            """Build a single adaptive 1D grid."""
            # Start with coarse grid
            positions = set()
            positions.add(a0)
            positions.add(a1)

            # Add points at base resolution
            n_base = max(2, int((a1 - a0) / max_r) + 1)
            for p in np.linspace(a0, a1, n_base):
                positions.add(round(p, 4))

            # Refine near special points
            for (px, py, res, radius) in refine_pts:
                center = px if is_x else py
                if center < a0 - radius or center > a1 + radius:
                    continue

                # Add refined points near this feature
                r_start = max(a0, center - radius)
                r_end = min(a1, center + radius)
                n_fine = max(2, int((r_end - r_start) / res) + 1)
                for p in np.linspace(r_start, r_end, n_fine):
                    positions.add(round(p, 4))

                # Transition zone
                trans_radius = radius * 2
                t_start = max(a0, center - trans_radius)
                t_end = min(a1, center + trans_radius)
                trans_res = (res + base) / 2
                n_trans = max(2, int((t_end - t_start) / trans_res) + 1)
                for p in np.linspace(t_start, t_end, n_trans):
                    positions.add(round(p, 4))

            return np.array(sorted(positions))

        xs = build_axis(bx0, bx1, base_res, min_res, max_res, refine_pts, True)
        ys = build_axis(by0, by1, base_res, min_res, max_res, refine_pts, False)

        # Limit sizes
        MAX_PER_AXIS = 300
        while len(xs) > MAX_PER_AXIS:
            xs = xs[::2]
            if xs[-1] != bx1:
                xs = np.append(xs, bx1)
        while len(ys) > MAX_PER_AXIS:
            ys = ys[::2]
            if ys[-1] != by1:
                ys = np.append(ys, by1)

        return xs, ys

    def _compute_copper_fractions(self, mesh, xs, ys):
        """Compute copper area fraction for each mesh cell.

        Performance note:
        The original implementation was O(nx*ny*(zones+traces+vias)) per layer,
        which becomes impractical once the adaptive grid reaches a few 10^4–10^5
        cells. This version limits work using bounding boxes and grid-index
        range lookups (searchsorted), which turns most cases into
        O(k * local_cells) with k the number of copper features.
        """
        nx, ny = mesh.nx, mesh.ny

        # Map KiCad copper layer names to mesh z indices (only layers present in mesh)
        layer_map = {name: i for i, name in enumerate(self.pcb.copper_layers) if i < mesh.nz}

        # Precompute axis metadata for fast bbox->index mapping
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        # Cell size upper bounds used to slightly expand bboxes for robust coverage
        max_dx = float(np.max(np.diff(xs))) if len(xs) > 1 else 0.0
        max_dy = float(np.max(np.diff(ys))) if len(ys) > 1 else 0.0

        def clamp(v, lo, hi):
            return lo if v < lo else hi if v > hi else v

        def bbox_to_ix(x0, x1):
            # xs are cell boundaries, valid cell indices are [0, len(xs)-2]
            if x1 < xs[0] or x0 > xs[-1]:
                return None
            ix0 = int(np.searchsorted(xs, x0, side='right') - 1)
            ix1 = int(np.searchsorted(xs, x1, side='left') - 1)
            ix0 = clamp(ix0, 0, nx - 1)
            ix1 = clamp(ix1, 0, nx - 1)
            if ix1 < ix0:
                return None
            return ix0, ix1

        def bbox_to_iy(y0, y1):
            if y1 < ys[0] or y0 > ys[-1]:
                return None
            iy0 = int(np.searchsorted(ys, y0, side='right') - 1)
            iy1 = int(np.searchsorted(ys, y1, side='left') - 1)
            iy0 = clamp(iy0, 0, ny - 1)
            iy1 = clamp(iy1, 0, ny - 1)
            if iy1 < iy0:
                return None
            return iy0, iy1

        # Initialize copper fraction to 0
        for n in mesh.nodes:
            n.copper_frac = 0.0

        # ─────────────────────────────────────────────────────────────
        # Zones
        # ─────────────────────────────────────────────────────────────
        for z in self.pcb.zones:
            iz = layer_map.get(z.layer, -1)
            if iz < 0:
                continue

            polys = z.filled_polygons if z.filled_polygons else ([z.outline] if z.outline else [])
            for poly in polys:
                if not poly or len(poly) < 3:
                    continue

                xs_poly = [p.x for p in poly]
                ys_poly = [p.y for p in poly]
                xmin, xmax = min(xs_poly), max(xs_poly)
                ymin, ymax = min(ys_poly), max(ys_poly)

                # Expand slightly to catch cells whose center is near edges
                pad_x = max_dx * 0.6
                pad_y = max_dy * 0.6
                ixr = bbox_to_ix(xmin - pad_x, xmax + pad_x)
                iyr = bbox_to_iy(ymin - pad_y, ymax + pad_y)
                if ixr is None or iyr is None:
                    continue
                ix0, ix1 = ixr
                iy0, iy1 = iyr

                for iy in range(iy0, iy1 + 1):
                    for ix in range(ix0, ix1 + 1):
                        n = mesh.nodes[mesh.idx(ix, iy, iz)]
                        if self._pip(n.x, n.y, poly):
                            # Zone fill: typical copper pour coverage (empirical default)
                            n.copper_frac = max(n.copper_frac, 0.85)

        # ─────────────────────────────────────────────────────────────
        # Traces (line segments)
        # ─────────────────────────────────────────────────────────────
        for tr in self.pcb.traces:
            iz = layer_map.get(tr.layer, -1)
            if iz < 0:
                continue
            hw = tr.width_mm / 2.0
            if hw < 0.01:
                continue

            # Bounding box around segment, expanded by half-width + cell padding
            pad = hw + 0.6 * max(max_dx, max_dy)
            xmin = min(tr.start.x, tr.end.x) - pad
            xmax = max(tr.start.x, tr.end.x) + pad
            ymin = min(tr.start.y, tr.end.y) - pad
            ymax = max(tr.start.y, tr.end.y) + pad
            ixr = bbox_to_ix(xmin, xmax)
            iyr = bbox_to_iy(ymin, ymax)
            if ixr is None or iyr is None:
                continue
            ix0, ix1 = ixr
            iy0, iy1 = iyr

            for iy in range(iy0, iy1 + 1):
                for ix in range(ix0, ix1 + 1):
                    n = mesh.nodes[mesh.idx(ix, iy, iz)]
                    dist = self._pt_seg_dist(n.x, n.y,
                                             tr.start.x, tr.start.y,
                                             tr.end.x, tr.end.y)
                    if dist <= hw + n.dx * 0.5:
                        # Estimate overlap fraction (fast heuristic)
                        overlap = max(0.0, hw - dist + n.dx * 0.3) / max(n.dx, 1e-9)
                        overlap = min(1.0, overlap)
                        n.copper_frac = max(n.copper_frac, overlap * 0.95)

        # ─────────────────────────────────────────────────────────────
        # Arc traces (approximated as segments)
        # ─────────────────────────────────────────────────────────────
        for arc in self.pcb.arc_traces:
            iz = layer_map.get(arc.layer, -1)
            if iz < 0:
                continue
            hw = arc.width_mm / 2.0
            if hw < 0.01:
                continue

            pts = self._arc_to_segments(arc)
            pad = hw + 0.6 * max(max_dx, max_dy)
            for seg_start, seg_end in pts:
                xmin = min(seg_start.x, seg_end.x) - pad
                xmax = max(seg_start.x, seg_end.x) + pad
                ymin = min(seg_start.y, seg_end.y) - pad
                ymax = max(seg_start.y, seg_end.y) + pad
                ixr = bbox_to_ix(xmin, xmax)
                iyr = bbox_to_iy(ymin, ymax)
                if ixr is None or iyr is None:
                    continue
                ix0, ix1 = ixr
                iy0, iy1 = iyr

                for iy in range(iy0, iy1 + 1):
                    for ix in range(ix0, ix1 + 1):
                        n = mesh.nodes[mesh.idx(ix, iy, iz)]
                        dist = self._pt_seg_dist(n.x, n.y,
                                                 seg_start.x, seg_start.y,
                                                 seg_end.x, seg_end.y)
                        if dist <= hw + n.dx * 0.5:
                            overlap = max(0.0, hw - dist + n.dx * 0.3) / max(n.dx, 1e-9)
                            overlap = min(1.0, overlap)
                            n.copper_frac = max(n.copper_frac, overlap * 0.95)

        # ─────────────────────────────────────────────────────────────
        # Vias (pads / barrels approximated as disks per layer)
        # ─────────────────────────────────────────────────────────────
        for via in self.pcb.vias:
            r = via.diam_mm / 2.0
            pad = r + 0.6 * max(max_dx, max_dy)
            xmin = via.pos.x - pad
            xmax = via.pos.x + pad
            ymin = via.pos.y - pad
            ymax = via.pos.y + pad
            ixr = bbox_to_ix(xmin, xmax)
            iyr = bbox_to_iy(ymin, ymax)
            if ixr is None or iyr is None:
                continue
            ix0, ix1 = ixr
            iy0, iy1 = iyr

            # Only touch the layers the via spans
            for layer_name in [via.layers[0], via.layers[1]]:
                iz = layer_map.get(layer_name, -1)
                if iz < 0:
                    continue
                for iy in range(iy0, iy1 + 1):
                    for ix in range(ix0, ix1 + 1):
                        n = mesh.nodes[mesh.idx(ix, iy, iz)]
                        dx = n.x - via.pos.x
                        dy = n.y - via.pos.y
                        if (dx * dx + dy * dy) <= (r + 0.3 * n.dx) ** 2:
                            n.copper_frac = max(n.copper_frac, 0.9)

    def _arc_to_segments(self, arc, n_seg=8):
        """Convert arc trace to line segments."""
        segments = []
        if arc.mid.x == 0 and arc.mid.y == 0:
            segments.append((arc.start, arc.end))
            return segments

        center = KiCadPCBReader._circle_center(
            arc.start.x, arc.start.y,
            arc.mid.x, arc.mid.y,
            arc.end.x, arc.end.y)

        if not center:
            segments.append((arc.start, arc.end))
            return segments

        cx, cy = center
        r = math.sqrt((arc.start.x - cx)**2 + (arc.start.y - cy)**2)
        a_start = math.atan2(arc.start.y - cy, arc.start.x - cx)
        a_end = math.atan2(arc.end.y - cy, arc.end.x - cx)

        da = a_end - a_start
        while da > math.pi:
            da -= 2 * math.pi
        while da < -math.pi:
            da += 2 * math.pi

        prev = arc.start
        for i in range(1, n_seg + 1):
            t = i / n_seg
            a = a_start + da * t
            curr = Pt(cx + r * math.cos(a), cy + r * math.sin(a))
            segments.append((prev, curr))
            prev = curr

        return segments

    def _apply_material_properties(self, mesh):
        """Apply effective thermal properties based on copper fraction."""
        cu = MATERIALS['COPPER']
        fr4 = MATERIALS['FR4']
        sm = MATERIALS['SOLDER_MASK']

        for n in mesh.nodes:
            f = n.copper_frac

            # Effective in-plane conductivity: parallel (Voigt) mixture rule
            # k_xy = f·k_cu + (1-f)·k_fr4
            # Ref: Incropera §3.1, Dede et al. IEEE Trans CPMT 2015
            n.k = f * cu['k'] + (1.0 - f) * fr4['k']

            # Effective through-plane conductivity: series (Reuss) mixture rule
            # 1/k_z = f/k_cu + (1-f)/k_fr4
            # For f=0.5: k_xy ≈ 201 W/(m·K), k_z ≈ 0.58 W/(m·K)  (ratio ~347:1)
            if f > 0 and f < 1.0:
                n.k_z = 1.0 / (f / cu['k'] + (1.0 - f) / fr4['k'])
            else:
                n.k_z = n.k  # pure material is isotropic

            # Effective volumetric properties
            n.cp = f * cu['cp'] + (1.0 - f) * fr4['cp']
            n.rho = f * cu['rho'] + (1.0 - f) * fr4['rho']

            # Emissivity: top and bottom surfaces
            # Top surface: solder mask over copper (or bare FR4)
            if n.layer == 0:  # top layer
                n.emiss_top = sm['emiss']  # solder mask
                n.emiss_bot = 0.0  # internal interface, no radiation
            elif n.layer == mesh.nz - 1:  # bottom layer
                n.emiss_top = 0.0  # internal interface
                n.emiss_bot = sm['emiss']  # solder mask
            else:
                n.emiss_top = 0.0
                n.emiss_bot = 0.0

            # If this is a single-layer model
            if mesh.nz == 1:
                n.emiss_top = sm['emiss']
                n.emiss_bot = sm['emiss']

    def _apply_heatsinks(self, mesh, xs, ys):
        for hs in self.cfg.heatsinks:
            if len(hs.polygon) < 3:
                continue
            mat = HEATSINK_MATERIALS.get(hs.material,
                                         HEATSINK_MATERIALS['ALUMINUM_6061'])
            emiss = hs.emissivity if hs.emissivity is not None else mat['emiss']
            poly_pts = [Pt(p[0], p[1]) for p in hs.polygon]

            for iy in range(mesh.ny):
                for ix in range(mesh.nx):
                    n = mesh.nodes[mesh.idx(ix, iy, 0)]
                    if self._pip(n.x, n.y, poly_pts):
                        n.k = max(n.k, mat['k'])
                        n.k_z = max(n.k_z, mat['k'])  # heatsinks are isotropic
                        n.emiss_top = emiss
                        # Heatsink adds extra radiating surface area
                        hs_thickness_m = hs.thickness_mm * 1e-3
                        dx_m = n.dx * 1e-3
                        dy_m = n.dy * 1e-3
                        # Side area contribution
                        n.surf_top += 2 * (dx_m + dy_m) * hs_thickness_m * 0.3

    @staticmethod
    def _pip(px, py, poly):
        """Point-in-polygon test (ray casting)."""
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            yi, yj = poly[i].y, poly[j].y
            xi, xj = poly[i].x, poly[j].x
            if ((yi > py) != (yj > py)) and \
               (px < (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi):
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _pt_seg_dist(px, py, x1, y1, x2, y2):
        """Point-to-line-segment distance."""
        dx, dy = x2 - x1, y2 - y1
        lsq = dx * dx + dy * dy
        if lsq < 1e-10:
            return math.sqrt((px - x1)**2 + (py - y1)**2)
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / lsq))
        return math.sqrt((px - x1 - t * dx)**2 + (py - y1 - t * dy)**2)

    def _add_heat_sources(self, mesh, xs, ys):
        nx, ny = mesh.nx, mesh.ny
        mode = self.cfg.sim.heat_source_mode

        if mode == "current_injection":
            self._add_joule_heating(mesh, xs, ys)
        else:
            for cp in self.cfg.comp_power:
                if cp.power_w <= 0:
                    continue
                comp = next((c for c in self.pcb.components if c.ref == cp.ref), None)
                if not comp:
                    continue

                bb = comp.bbox()
                affected = []
                for iy in range(ny):
                    for ix in range(nx):
                        n = mesh.nodes[mesh.idx(ix, iy, 0)]
                        if (bb[0].x - 0.5 <= n.x <= bb[1].x + 0.5 and
                            bb[0].y - 0.5 <= n.y <= bb[1].y + 0.5):
                            affected.append(n)

                if not affected:
                    best = min(
                        (mesh.nodes[mesh.idx(ix, iy, 0)]
                         for iy in range(ny) for ix in range(nx)),
                        key=lambda n: (n.x - comp.pos.x)**2 + (n.y - comp.pos.y)**2,
                        default=None)
                    if best:
                        affected = [best]

                if affected:
                    # Distribute power proportional to cell area
                    total_area = sum(n.dx * n.dy for n in affected)
                    if total_area > 0:
                        for n in affected:
                            n.heat += cp.power_w * (n.dx * n.dy) / total_area
                    else:
                        per = cp.power_w / len(affected)
                        for n in affected:
                            n.heat += per

    def _add_joule_heating(self, mesh, xs, ys):
        """Compute I²R Joule heating from current paths."""
        nx, ny = mesh.nx, mesh.ny
        cu_thickness = 35e-6  # 35μm = 1oz copper default

        for cp in self.cfg.current_paths:
            if cp.current_a <= 0:
                continue

            # Find traces on the path nets
            net_traces = [t for t in self.pcb.traces
                          if t.net_name == cp.source_net or t.net_name == cp.sink_net]
            if not net_traces:
                continue

            for tr in net_traces:
                if tr.length_mm < 0.01:
                    continue

                # Get copper thickness for this layer
                layer_cu = self.pcb.copper_thickness_um.get(tr.layer, 35.0)
                t_cu = layer_cu * 1e-6  # μm to m

                # Trace resistance: R = ρ·L / (w·t)
                length_m = tr.length_mm * 1e-3
                width_m = max(tr.width_mm * 1e-3, 0.1e-3)
                R = CU_RESISTIVITY * length_m / (width_m * t_cu)
                P = cp.current_a**2 * R

                # Distribute heat along the trace to nearby mesh nodes
                affected = []
                for iy in range(ny):
                    for ix in range(nx):
                        n = mesh.nodes[mesh.idx(ix, iy, 0)]
                        d = self._pt_seg_dist(n.x, n.y,
                                              tr.start.x, tr.start.y,
                                              tr.end.x, tr.end.y)
                        if d <= tr.width_mm / 2 + n.dx * 0.6:
                            affected.append(n)

                if affected:
                    total_area = sum(n.dx * n.dy for n in affected)
                    if total_area > 0:
                        for n in affected:
                            n.heat += P * (n.dx * n.dy) / total_area

    def _calc_conductances(self, mesh, xs, ys):
        """Compute thermal conductance between adjacent nodes."""
        nx, ny, nz = mesh.nx, mesh.ny, mesh.nz

        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    i = mesh.idx(ix, iy, iz)
                    n = mesh.nodes[i]

                    for dix, diy, diz, direction in [
                        (-1, 0, 0, 'x'), (1, 0, 0, 'x'),
                        (0, -1, 0, 'y'), (0, 1, 0, 'y'),
                        (0, 0, -1, 'z'), (0, 0, 1, 'z')
                    ]:
                        jx, jy, jz = ix + dix, iy + diy, iz + diz
                        if 0 <= jx < nx and 0 <= jy < ny and 0 <= jz < nz:
                            j = mesh.idx(jx, jy, jz)
                            nb = mesh.nodes[j]
                            G = self._conductance(n, nb, direction)
                            if G > 0:
                                n.nbrs[j] = G

    @staticmethod
    def _conductance(a, b, direction):
        """
        Compute thermal conductance G = k_eff * A / L  [W/K]

        For in-plane (x,y): uses parallel mixture conductivity
        For through-plane (z): uses series (harmonic mean) conductivity
        """
        if direction == 'x':
            dist_m = abs(b.x - a.x) * 1e-3
            if dist_m < 1e-9:
                dist_m = max(a.dx, b.dx) * 1e-3
            # Cross-section area: dy * dz
            area = min(a.dy, b.dy) * 1e-3 * min(a.dz, b.dz) * 1e-3
        elif direction == 'y':
            dist_m = abs(b.y - a.y) * 1e-3
            if dist_m < 1e-9:
                dist_m = max(a.dy, b.dy) * 1e-3
            area = min(a.dx, b.dx) * 1e-3 * min(a.dz, b.dz) * 1e-3
        else:  # z
            dist_m = abs(b.z - a.z) * 1e-3
            if dist_m < 1e-9:
                dist_m = max(a.dz, b.dz) * 1e-3
            area = min(a.dx, b.dx) * 1e-3 * min(a.dy, b.dy) * 1e-3

        if dist_m < 1e-12:
            return 0.0

        # Interface conductivity: harmonic mean of the two half-cells
        # For x,y: use in-plane k (parallel mixture, already correct)
        # For z:   use through-plane k_z (series mixture)
        # Ref: Patankar 1980 §4.2, NASA Passive Thermal Control Guidebook Fig. 14
        if direction == 'z':
            ka, kb = a.k_z, b.k_z
        else:
            ka, kb = a.k, b.k
        if ka + kb < 1e-30:
            return 0.0
        k_eff = 2.0 * ka * kb / (ka + kb)

        return k_eff * area / dist_m

    def _add_via_conductances(self, mesh, xs, ys):
        """
        Add explicit thermal conductance for vias between layers.

        Via thermal conductance:
          G_via = k_cu * A_barrel / L

        where:
          A_barrel = π * d_drill * t_plating  (cylindrical shell)
                   or π/4 * (d_outer² - d_drill²)  (full annular area)
          L = distance between connected layers

        For filled vias: A = π/4 * d_drill²
        """
        nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
        if nz < 2:
            return

        layer_map = {name: i for i, name in enumerate(self.pcb.copper_layers)
                     if i < nz}
        k_cu = MATERIALS['COPPER']['k']

        # Standard plating thickness
        plating_thickness = 25e-6  # 25μm typical

        for via in self.pcb.vias:
            # Determine which layers this via connects
            top_layer = layer_map.get(via.layers[0], -1)
            bot_layer = layer_map.get(via.layers[1], -1)
            if top_layer < 0 or bot_layer < 0:
                continue
            if top_layer > bot_layer:
                top_layer, bot_layer = bot_layer, top_layer

            # Via barrel cross-section area
            # Plated barrel is a cylindrical shell on the inside of the drilled hole
            # A = π · t · (d_drill - t)   [exact annular cross-section of plating]
            # NOT the annular pad area π/4·(d_outer² - d_drill²)
            # Ref: IPC-6012 Class 3, TI SLPA015
            d_drill = via.drill_mm * 1e-3
            A_barrel = math.pi * plating_thickness * (d_drill - plating_thickness)
            if A_barrel <= 0:  # degenerate: drill smaller than plating
                A_barrel = math.pi * d_drill * plating_thickness  # thin-wall approx

            # Find nearest mesh node at via position
            best_ix, best_iy = 0, 0
            best_dist = 1e30
            for iy in range(ny):
                for ix in range(nx):
                    n = mesh.nodes[mesh.idx(ix, iy, 0)]
                    d = (n.x - via.pos.x)**2 + (n.y - via.pos.y)**2
                    if d < best_dist:
                        best_dist = d
                        best_ix, best_iy = ix, iy

            # Add conductance between each pair of connected layers
            for iz in range(top_layer, bot_layer):
                i = mesh.idx(best_ix, best_iy, iz)
                j = mesh.idx(best_ix, best_iy, iz + 1)
                n_i = mesh.nodes[i]
                n_j = mesh.nodes[j]

                # Distance between layer midpoints
                L = abs(n_j.z - n_i.z) * 1e-3
                if L < 1e-9:
                    L = self.pcb.board_thickness_mm * 1e-3 / max(1, nz)

                G_via = k_cu * A_barrel / L

                # Add to existing conductance (may already have FR4 conduction)
                if j in n_i.nbrs:
                    n_i.nbrs[j] += G_via
                else:
                    n_i.nbrs[j] = G_via

                if i in n_j.nbrs:
                    n_j.nbrs[i] += G_via
                else:
                    n_j.nbrs[i] = G_via

    def _apply_bc(self, mesh, xs, ys):
        """Apply boundary conditions from mounting points."""
        for mp in self.cfg.mounting:
            if mp.fixed_temp_c is not None:
                # Find nearest surface nodes
                best = None
                best_dist = 1e30
                for n in mesh.nodes:
                    if n.layer == 0:
                        d = (n.x - mp.x_mm)**2 + (n.y - mp.y_mm)**2
                        if d < best_dist:
                            best_dist = d
                            best = n
                if best:
                    best.fixed = True
                    best.T_fixed = mp.fixed_temp_c
                    # Also fix corresponding nodes on other layers
                    if mesh.nz > 1:
                        for iz in range(mesh.nz):
                            idx = best.idx - best.layer * mesh.nx * mesh.ny + iz * mesh.nx * mesh.ny
                            if 0 <= idx < len(mesh.nodes):
                                mesh.nodes[idx].fixed = True
                                mesh.nodes[idx].T_fixed = mp.fixed_temp_c

        # Auto-detect mounting holes
        for hp in self.pcb.mounting_holes:
            if not any(abs(mp.x_mm - hp.x) < 2 and abs(mp.y_mm - hp.y) < 2
                       for mp in self.cfg.mounting):
                self.cfg.mounting.append(MountingCfg(
                    mp_id=f"auto_{len(self.cfg.mounting)}",
                    x_mm=hp.x, y_mm=hp.y))


# ═══════════════════════════════════════════════════════════════════════
# §8  Native C Engine Interface
# ═══════════════════════════════════════════════════════════════════════

class NativeEngine:
    def __init__(self):
        self._lib = None
        self._load()

    def _load(self):
        names = ['libthermal_engine.so', 'libthermal_engine.dylib', 'thermal_engine.dll']
        try:
            script_dir = Path(__file__).parent
        except NameError:
            script_dir = Path.cwd()
        paths = [script_dir, Path.cwd(), script_dir / 'native']

        # On Windows + Python 3.8+, ctypes no longer searches PATH for DLL
        # dependencies. We must explicitly add the directory.
        _dll_dirs = []
        if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
            for p in paths:
                try:
                    _dll_dirs.append(os.add_dll_directory(str(p.resolve())))
                except OSError:
                    pass

        last_err = None
        for p in paths:
            for n in names:
                fp = p / n
                if fp.exists():
                    try:
                        # winmode=0 restores pre-3.8 search behavior on Windows
                        if sys.platform == 'win32':
                            self._lib = ctypes.CDLL(str(fp.resolve()), winmode=0)
                        else:
                            self._lib = ctypes.CDLL(str(fp))
                        self._bind()
                        return
                    except Exception as e:
                        last_err = e
                        self._lib = None

        # Clean up dll directories
        for d in _dll_dirs:
            try: d.close()
            except: pass

        if last_err:
            print(f"[TVAC] C engine found but failed to load: {last_err}", file=sys.stderr)

    def _bind(self):
        L = self._lib

        # Create/destroy
        L.thermal_create_state.argtypes = [ctypes.c_int] * 4
        L.thermal_create_state.restype = ctypes.c_void_p
        L.thermal_destroy_state.argtypes = [ctypes.c_void_p]
        L.thermal_destroy_state.restype = None

        # Node configuration — v1 API (backward compat)
        L.thermal_set_node.argtypes = [ctypes.c_void_p, ctypes.c_int] + [ctypes.c_double] * 7
        L.thermal_set_node.restype = None

        # v2 API with separate top/bottom
        try:
            L.thermal_set_node_v2.argtypes = [ctypes.c_void_p, ctypes.c_int] + [ctypes.c_double] * 9
            L.thermal_set_node_v2.restype = None
            self._has_v2 = True
        except AttributeError:
            self._has_v2 = False

        L.thermal_set_fixed_temp.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
        L.thermal_set_fixed_temp.restype = None
        L.thermal_set_initial_temp.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
        L.thermal_set_initial_temp.restype = None
        L.thermal_set_chamber_temp.argtypes = [ctypes.c_void_p, ctypes.c_double]
        L.thermal_set_chamber_temp.restype = None
        L.thermal_get_temp.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.thermal_get_temp.restype = ctypes.c_double

        # Picard tolerance
        try:
            L.thermal_set_picard_tol.argtypes = [ctypes.c_void_p, ctypes.c_double]
            L.thermal_set_picard_tol.restype = None
            self._has_picard_tol = True
        except AttributeError:
            self._has_picard_tol = False

        # Neighbors
        L.thermal_alloc_neighbors.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.thermal_alloc_neighbors.restype = ctypes.c_int
        L.thermal_set_row_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        L.thermal_set_row_ptr.restype = None
        L.thermal_set_neighbor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_double]
        L.thermal_set_neighbor.restype = None

        # Solvers
        L.thermal_solve_steady.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        L.thermal_solve_steady.restype = ctypes.c_int
        L.thermal_solve_transient.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                               ctypes.c_double, ctypes.c_double, ctypes.c_int]
        L.thermal_solve_transient.restype = ctypes.c_int

    @property
    def ok(self):
        return self._lib is not None


# ═══════════════════════════════════════════════════════════════════════
# §9  Thermal Solver
# ═══════════════════════════════════════════════════════════════════════

class ThermalSolver:
    def __init__(self):
        self.engine = NativeEngine()

    @property
    def backend(self):
        return "C Engine (IC0+PCG)" if self.engine.ok else "SciPy (Python)"

    def solve_steady(self, mesh, cfg, cb=None):
        N = len(mesh.nodes)
        if N == 0:
            return TResult(error="Empty mesh")
        if self.engine.ok:
            return self._solve_c(mesh, cfg, cb)
        return self._solve_scipy(mesh, cfg, cb)

    def solve_transient(self, mesh, cfg, cb=None):
        if self.engine.ok:
            return self._solve_c_transient(mesh, cfg, cb)
        return self._solve_scipy_transient(mesh, cfg, cb)

    # Extended C result structure matching v4.0
    class _CResult(ctypes.Structure):
        _fields_ = [
            ("min_temp", ctypes.c_double),
            ("max_temp", ctypes.c_double),
            ("avg_temp", ctypes.c_double),
            ("iterations", ctypes.c_int),
            ("picard_iters", ctypes.c_int),
            ("compute_time", ctypes.c_double),
            ("converged", ctypes.c_int),
            ("energy_balance", ctypes.c_double),
            ("max_residual", ctypes.c_double),
            ("error", ctypes.c_char * 256),
        ]

    def _solve_c(self, mesh, cfg, cb=None):
        L = self.engine._lib
        N = len(mesh.nodes)
        t0 = time.time()

        state = L.thermal_create_state(N, mesh.nx, mesh.ny, mesh.nz)
        if not state:
            return TResult(error="Failed to create C state")

        try:
            L.thermal_set_chamber_temp(state, ctypes.c_double(cfg.chamber_wall_temp_c + C2K))

            if hasattr(self.engine, '_has_picard_tol') and self.engine._has_picard_tol:
                L.thermal_set_picard_tol(state, ctypes.c_double(cfg.convergence * 10))

            # Set up neighbor connectivity
            total_nbrs = sum(len(n.nbrs) for n in mesh.nodes)
            L.thermal_alloc_neighbors(state, total_nbrs)
            ptr = 0
            for i, node in enumerate(mesh.nodes):
                L.thermal_set_row_ptr(state, i, ptr)
                for off, (j, G) in enumerate(node.nbrs.items()):
                    L.thermal_set_neighbor(state, i, off, j, ctypes.c_double(G))
                ptr += len(node.nbrs)
            L.thermal_set_row_ptr(state, N, ptr)

            # Set node properties
            for node in mesh.nodes:
                if hasattr(self.engine, '_has_v2') and self.engine._has_v2:
                    L.thermal_set_node_v2(
                        state, node.idx,
                        ctypes.c_double(node.k),
                        ctypes.c_double(node.cp),
                        ctypes.c_double(node.rho),
                        ctypes.c_double(node.emiss_top),
                        ctypes.c_double(node.emiss_bot),
                        ctypes.c_double(node.vol),
                        ctypes.c_double(node.surf_top),
                        ctypes.c_double(node.surf_bot),
                        ctypes.c_double(node.heat))
                else:
                    # Fallback: combine surfaces
                    total_surf = node.surf_top + node.surf_bot
                    avg_emiss = (node.emiss_top + node.emiss_bot) / 2.0 if total_surf > 0 else 0.9
                    L.thermal_set_node(
                        state, node.idx,
                        ctypes.c_double(node.k),
                        ctypes.c_double(node.cp),
                        ctypes.c_double(node.rho),
                        ctypes.c_double(avg_emiss),
                        ctypes.c_double(node.vol),
                        ctypes.c_double(total_surf),
                        ctypes.c_double(node.heat))

                if node.fixed:
                    L.thermal_set_fixed_temp(state, node.idx,
                                             ctypes.c_double(node.T_fixed + C2K))
                L.thermal_set_initial_temp(state, node.idx,
                                            ctypes.c_double(cfg.ambient_temp_c + C2K))

            res = self._CResult()
            err = L.thermal_solve_steady(state, ctypes.byref(res),
                                          1 if cfg.include_radiation else 0)

            if err != 0:
                return TResult(error=f"C solver error {err}: {res.error.decode('utf-8', errors='replace')}")

            temps = np.array([L.thermal_get_temp(state, i) - C2K for i in range(N)])
            return TResult(
                temps=temps,
                min_t=res.min_temp, max_t=res.max_temp, avg_t=res.avg_temp,
                iters=res.iterations,
                picard_iters=res.picard_iters,
                time_s=time.time() - t0,
                converged=bool(res.converged),
                energy_balance=res.energy_balance)

        finally:
            L.thermal_destroy_state(state)

    def _solve_c_transient(self, mesh, cfg, cb=None):
        """Transient solve via C engine."""
        L = self.engine._lib
        N = len(mesh.nodes)
        t0 = time.time()

        state = L.thermal_create_state(N, mesh.nx, mesh.ny, mesh.nz)
        if not state:
            return TResult(error="Failed to create C state")

        try:
            L.thermal_set_chamber_temp(state, ctypes.c_double(cfg.chamber_wall_temp_c + C2K))

            total_nbrs = sum(len(n.nbrs) for n in mesh.nodes)
            L.thermal_alloc_neighbors(state, total_nbrs)
            ptr = 0
            for i, node in enumerate(mesh.nodes):
                L.thermal_set_row_ptr(state, i, ptr)
                for off, (j, G) in enumerate(node.nbrs.items()):
                    L.thermal_set_neighbor(state, i, off, j, ctypes.c_double(G))
                ptr += len(node.nbrs)
            L.thermal_set_row_ptr(state, N, ptr)

            for node in mesh.nodes:
                if hasattr(self.engine, '_has_v2') and self.engine._has_v2:
                    L.thermal_set_node_v2(
                        state, node.idx,
                        ctypes.c_double(node.k), ctypes.c_double(node.cp),
                        ctypes.c_double(node.rho),
                        ctypes.c_double(node.emiss_top), ctypes.c_double(node.emiss_bot),
                        ctypes.c_double(node.vol),
                        ctypes.c_double(node.surf_top), ctypes.c_double(node.surf_bot),
                        ctypes.c_double(node.heat))
                else:
                    L.thermal_set_node(
                        state, node.idx,
                        ctypes.c_double(node.k), ctypes.c_double(node.cp),
                        ctypes.c_double(node.rho),
                        ctypes.c_double((node.emiss_top + node.emiss_bot)/2),
                        ctypes.c_double(node.vol),
                        ctypes.c_double(node.surf_top + node.surf_bot),
                        ctypes.c_double(node.heat))

                if node.fixed:
                    L.thermal_set_fixed_temp(state, node.idx,
                                             ctypes.c_double(node.T_fixed + C2K))
                L.thermal_set_initial_temp(state, node.idx,
                                            ctypes.c_double(cfg.initial_temp_c + C2K))

            res = self._CResult()
            err = L.thermal_solve_transient(
                state, ctypes.byref(res),
                ctypes.c_double(cfg.duration_s),
                ctypes.c_double(cfg.timestep_s),
                1 if cfg.include_radiation else 0)

            if err != 0:
                return TResult(error=f"C transient error {err}")

            temps = np.array([L.thermal_get_temp(state, i) - C2K for i in range(N)])
            return TResult(
                temps=temps,
                min_t=res.min_temp, max_t=res.max_temp, avg_t=res.avg_temp,
                iters=res.iterations, time_s=time.time() - t0,
                converged=bool(res.converged))

        finally:
            L.thermal_destroy_state(state)

    def _solve_scipy(self, mesh, cfg, cb=None):
        """Pure Python/SciPy solver with proper physics."""
        N = len(mesh.nodes)
        t0 = time.time()

        if cb:
            cb(10, "Building matrix...")

        is_fixed = np.array([n.fixed for n in mesh.nodes], dtype=bool)
        T_fixed_k = np.array([n.T_fixed + C2K for n in mesh.nodes])
        surf_top = np.array([n.surf_top for n in mesh.nodes])
        surf_bot = np.array([n.surf_bot for n in mesh.nodes])
        emiss_top = np.array([n.emiss_top for n in mesh.nodes])
        emiss_bot = np.array([n.emiss_bot for n in mesh.nodes])
        heat = np.array([n.heat for n in mesh.nodes])

        rows, cols, vals = [], [], []
        base_rhs = np.zeros(N)

        for i, node in enumerate(mesh.nodes):
            if node.fixed:
                rows.append(i)
                cols.append(i)
                vals.append(1.0)
                base_rhs[i] = node.T_fixed + C2K
            else:
                diag = 0.0
                for j, G in node.nbrs.items():
                    rows.append(i)
                    cols.append(j)
                    vals.append(-G)
                    diag += G
                rows.append(i)
                cols.append(i)
                vals.append(max(diag, 1e-10))
                base_rhs[i] = node.heat

        K = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))

        if cb:
            cb(30, "Initial solve...")

        Tw4 = (cfg.chamber_wall_temp_c + C2K) ** 4

        # Initial solve without radiation
        try:
            T = spsolve(K, base_rhs)
        except Exception:
            T = np.full(N, cfg.ambient_temp_c + C2K)

        T[is_fixed] = T_fixed_k[is_fixed]
        T = np.clip(T, 1.0, 3000.0)

        if not cfg.include_radiation:
            T_c = T - C2K
            return TResult(
                temps=T_c,
                min_t=float(np.nanmin(T_c)),
                max_t=float(np.nanmax(T_c)),
                avg_t=float(np.nanmean(T_c)),
                iters=1, time_s=time.time() - t0, converged=True)

        # Picard iteration with linearized radiation
        relax = 0.4
        prev_diff = 1e30
        picard_iters = 0

        # Get diagonal indices for fast modification
        K_diag = K.diagonal().copy()

        for pic in range(100):
            T_prev = T.copy()

            # Linearized radiation
            Ti3_top = np.where(~is_fixed & (surf_top > 0) & (emiss_top > 0),
                               T**3, 0.0)
            Ti3_bot = np.where(~is_fixed & (surf_bot > 0) & (emiss_bot > 0),
                               T**3, 0.0)

            G_rad = (4.0 * emiss_top * SIGMA * surf_top * Ti3_top +
                     4.0 * emiss_bot * SIGMA * surf_bot * Ti3_bot)

            rhs_rad = (emiss_top * SIGMA * surf_top * (3 * Ti3_top * T + Tw4) +
                       emiss_bot * SIGMA * surf_bot * (3 * Ti3_bot * T + Tw4))

            # Modified matrix: K + diag(G_rad)
            K_mod = K.copy()
            K_mod.setdiag(K_diag + G_rad)

            # RHS
            rhs_full = base_rhs.copy()
            rhs_full[~is_fixed] = heat[~is_fixed] + rhs_rad[~is_fixed]
            rhs_full[is_fixed] = T_fixed_k[is_fixed]

            try:
                T_new = spsolve(K_mod, rhs_full)
            except Exception:
                try:
                    T_new, _ = scipy_cg(K_mod, rhs_full, x0=T,
                                        tol=cfg.convergence, maxiter=cfg.max_iterations)
                except Exception as e:
                    return TResult(error=f"Solver failed: {e}")

            T = relax * T_new + (1.0 - relax) * T_prev
            T[is_fixed] = T_fixed_k[is_fixed]
            T = np.clip(T, 1.0, 3000.0)

            diff = np.max(np.abs(T - T_prev))
            picard_iters = pic + 1

            if cb:
                cb(int(30 + 60 * (pic + 1) / 100),
                   f"Picard {pic+1}/100, ΔT={diff:.4e} K, ω={relax:.2f}")

            # Adaptive relaxation
            if diff < prev_diff * 0.95:
                relax = min(0.95, relax + 0.04)
            elif diff > prev_diff * 1.05:
                relax = max(0.15, relax * 0.7)
            prev_diff = diff

            if diff < cfg.convergence * 10 and pic > 2:
                break

        T_c = T - C2K
        return TResult(
            temps=T_c,
            min_t=float(np.nanmin(T_c)),
            max_t=float(np.nanmax(T_c)),
            avg_t=float(np.nanmean(T_c)),
            iters=picard_iters,
            picard_iters=picard_iters,
            time_s=time.time() - t0,
            converged=(diff < cfg.convergence * 100))

    def _solve_scipy_transient(self, mesh, cfg, cb=None):
        N = len(mesh.nodes)
        t0 = time.time()
        dt = cfg.timestep_s
        nsteps = max(1, int(cfg.duration_s / dt))
        Tw4 = (cfg.chamber_wall_temp_c + C2K) ** 4
        theta = 0.5

        rows, cols, kvals = [], [], []
        Q = np.zeros(N)
        C = np.zeros(N)

        for i, node in enumerate(mesh.nodes):
            # Zero capacitance for fixed nodes so LHS row stays as identity
            C[i] = 0.0 if node.fixed else node.rho * node.cp * node.vol
            Q[i] = node.heat
            if node.fixed:
                rows.append(i); cols.append(i); kvals.append(1.0)
            else:
                diag = 0.0
                for j, G in node.nbrs.items():
                    rows.append(i); cols.append(j); kvals.append(-G)
                    diag += G
                rows.append(i); cols.append(i); kvals.append(max(diag, 1e-10))

        K = sparse.csr_matrix((kvals, (rows, cols)), shape=(N, N))
        C_diag = sparse.diags(C / dt)
        LHS = C_diag + theta * K

        T = np.full(N, cfg.initial_temp_c + C2K)
        is_fixed = np.array([n.fixed for n in mesh.nodes])
        surf_top = np.array([n.surf_top for n in mesh.nodes])
        surf_bot = np.array([n.surf_bot for n in mesh.nodes])
        emiss_top = np.array([n.emiss_top for n in mesh.nodes])
        emiss_bot = np.array([n.emiss_bot for n in mesh.nodes])
        T_fixed_k = np.array([n.T_fixed + C2K for n in mesh.nodes])

        for step in range(nsteps):
            # Picard sub-iterations for radiation nonlinearity within each timestep
            # Typically converges in 2-3 iterations since dT per step is small
            n_sub = 3 if cfg.include_radiation else 1
            T_step_start = T.copy()

            for sub in range(n_sub):
                Qrad = np.zeros(N)
                if cfg.include_radiation:
                    Ti4 = T ** 4
                    dT4 = Tw4 - Ti4
                    mask_top = (~is_fixed) & (surf_top > 0) & (emiss_top > 0)
                    mask_bot = (~is_fixed) & (surf_bot > 0) & (emiss_bot > 0)
                    Qrad += np.where(mask_top, emiss_top * SIGMA * surf_top * dT4, 0.0)
                    Qrad += np.where(mask_bot, emiss_bot * SIGMA * surf_bot * dT4, 0.0)

                rhs = (C / dt) * T_step_start - (1 - theta) * K.dot(T_step_start) + Q + Qrad
                rhs[is_fixed] = T_fixed_k[is_fixed]

                try:
                    T_new, _ = scipy_cg(LHS, rhs, x0=T, tol=cfg.convergence,
                                        maxiter=cfg.max_iterations)
                except Exception:
                    T_new = spsolve(LHS, rhs)

                T_new[is_fixed] = T_fixed_k[is_fixed]
                T_new = np.clip(T_new, 1.0, 3000.0)

                # Check sub-iteration convergence
                if sub > 0 and np.max(np.abs(T_new - T)) < cfg.convergence * 100:
                    T = T_new
                    break
                T = T_new
            T = np.clip(T, 1.0, 3000.0)

            if cb and step % max(1, nsteps // 100) == 0:
                cb(int(step * 100 / nsteps), f"Step {step}/{nsteps}")

        T_c = T - C2K
        return TResult(
            temps=T_c,
            min_t=float(np.min(T_c)),
            max_t=float(np.max(T_c)),
            avg_t=float(np.mean(T_c)),
            iters=nsteps,
            time_s=time.time() - t0,
            converged=True)

# ═══════════════════════════════════════════════════════════════════════
# §10  PyQt5 GUI
# ═══════════════════════════════════════════════════════════════════════

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QSplitter, QTabWidget, QTableWidget, QTableWidgetItem,
        QPushButton, QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
        QComboBox, QCheckBox, QGroupBox, QFormLayout, QFileDialog,
        QMessageBox, QHeaderView, QAbstractItemView,
        QDialog, QDialogButtonBox, QFrame, QSizePolicy,
        QRadioButton, QButtonGroup, QTextEdit, QScrollArea,
        QProgressBar, QInputDialog,
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF, QPointF
    from PyQt5.QtGui import (
        QPainter, QColor, QPen, QBrush, QFont, QWheelEvent, QMouseEvent,
        QPolygonF, QLinearGradient,
    )
    HAS_QT = True
except ImportError:
    HAS_QT = False

if HAS_QT:
    # ── Theme ──────────────────────────────────────────────────────
    STYLE = """
    QMainWindow, QWidget { background-color: #1e2127; color: #dce1e8; }
    QTabWidget::pane { border: 1px solid #3c414b; background: #262a32; }
    QTabBar::tab { background: #262a32; color: #a0a8b4; padding: 8px 18px;
                   border: 1px solid #3c414b; border-bottom: none; margin-right: 2px; }
    QTabBar::tab:selected { background: #303644; color: #dce1e8;
                            border-bottom: 2px solid #009688; }
    QTableWidget { background: #262a32; alternate-background-color: #2c3040;
                   gridline-color: #3c414b; color: #dce1e8; border: 1px solid #3c414b; }
    QTableWidget::item:selected { background: rgba(0,150,136,0.3); }
    QHeaderView::section { background: #303644; color: #a0a8b4; padding: 6px;
                           border: 1px solid #3c414b; font-weight: bold; }
    QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QTextEdit {
        background: #303644; color: #dce1e8; border: 1px solid #3c414b;
        border-radius: 4px; padding: 5px; }
    QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
        border: 1px solid #009688; }
    QPushButton { background: #303644; color: #dce1e8; border: 1px solid #3c414b;
                  border-radius: 4px; padding: 7px 16px; font-weight: bold; }
    QPushButton:hover { background: #3c4150; border-color: #009688; }
    QPushButton:pressed { background: #009688; }
    QPushButton#runBtn { background: #009688; color: white; font-size: 14px;
                         padding: 10px 24px; border: none; }
    QPushButton#runBtn:hover { background: #00897B; }
    QGroupBox { border: 1px solid #3c414b; border-radius: 6px;
                margin-top: 12px; padding-top: 18px; font-weight: bold; color: #a0a8b4; }
    QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
    QLabel { color: #a0a8b4; } QLabel#title { color: #dce1e8; font-size: 18px; font-weight: bold; }
    QLabel#subtitle { color: #a0a8b4; font-size: 12px; }
    QCheckBox { color: #a0a8b4; } QRadioButton { color: #a0a8b4; }
    QCheckBox::indicator, QRadioButton::indicator { width: 16px; height: 16px;
        border: 1px solid #3c414b; background: #303644; }
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {
        background: #009688; border-color: #009688; }
    QProgressBar { background: #303644; border: 1px solid #3c414b; border-radius: 4px;
                   text-align: center; color: #dce1e8; }
    QProgressBar::chunk { background: #009688; border-radius: 3px; }
    QStatusBar { background: #1e2127; color: #a0a8b4; border-top: 1px solid #3c414b; }
    QSplitter::handle { background: #3c414b; }
    QScrollBar:vertical { background: #1e2127; width: 10px; }
    QScrollBar::handle:vertical { background: #3c414b; border-radius: 5px; min-height: 20px; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
    QComboBox::drop-down { border: none; }
    QComboBox QAbstractItemView { background: #303644; color: #dce1e8; }
    """

    # Layer color definitions
    LAYER_COLORS = {
        'F.Cu': QColor(200, 50, 50, 180),
        'B.Cu': QColor(50, 50, 200, 180),
        'In1.Cu': QColor(200, 200, 50, 180),
        'In2.Cu': QColor(50, 200, 50, 180),
        'In3.Cu': QColor(200, 50, 200, 180),
        'In4.Cu': QColor(50, 200, 200, 180),
        'Edge.Cuts': QColor(200, 200, 200, 255),
    }

    def get_layer_color(layer_name, alpha=180):
        if layer_name in LAYER_COLORS:
            c = LAYER_COLORS[layer_name]
            return QColor(c.red(), c.green(), c.blue(), alpha)
        return QColor(160, 160, 170, alpha)

    # ── PCB Visualization ──────────────────────────────────────────
    class PCBView(QWidget):
        componentClicked = pyqtSignal(str)

        def __init__(self, parent=None):
            super().__init__(parent)
            self.pcb = None
            self.cfg = None
            self.result = None
            self.mesh = None
            self._zoom = 1.0
            self._pan = QPointF(0, 0)
            self._last_mouse = None
            self._selected_ref = ""

            # Layer visibility — individual per copper layer
            self.visible_layers = set()
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

        def set_data(self, pcb, cfg):
            self.pcb = pcb
            self.cfg = cfg
            # Make all copper layers visible by default
            self.visible_layers = set(pcb.copper_layers) if pcb else set()
            self._fit_view()
            self.update()

        def set_thermal(self, result, mesh):
            self.result = result
            self.mesh = mesh
            self.show_thermal = True
            self.update()

        def select(self, ref):
            self._selected_ref = ref
            self.update()

        def is_layer_visible(self, layer_name):
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
                        bb = comp.bbox()
                        if bb[0].x <= wx <= bb[1].x and bb[0].y <= wy <= bb[1].y:
                            self._selected_ref = comp.ref
                            self.componentClicked.emit(comp.ref)
                            self.update()
                            return
            elif e.button() == Qt.MiddleButton:
                self._last_mouse = e.pos()

        def mouseMoveEvent(self, e):
            if self._last_mouse and (e.buttons() & (Qt.MiddleButton | Qt.LeftButton)):
                delta = e.pos() - self._last_mouse
                self._pan += QPointF(delta.x(), delta.y())
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
            if self.show_thermal and self.result is not None and self.mesh is not None:
                self._draw_thermal(p)
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
            if self.show_mounting:
                self._draw_mounting(p)
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
            poly = QPolygonF()
            for pt in pts:
                sx, sy = self._to_screen(pt.x, pt.y)
                poly.append(QPointF(sx, sy))
            p.setBrush(QBrush(QColor(20, 80, 20, 60)))
            p.drawPolygon(poly)

        def _draw_zones(self, p):
            """Draw zone fills with proper per-layer colors."""
            for z in self.pcb.zones:
                if not self.is_layer_visible(z.layer):
                    continue
                color = get_layer_color(z.layer, 60)
                border = get_layer_color(z.layer, 120)

                # Prefer filled_polygons (actual copper)
                polys = z.filled_polygons if z.filled_polygons else ([z.outline] if z.outline else [])
                for poly_pts in polys:
                    if len(poly_pts) < 3:
                        continue
                    poly = QPolygonF()
                    for pt in poly_pts:
                        sx, sy = self._to_screen(pt.x, pt.y)
                        poly.append(QPointF(sx, sy))
                    p.setPen(QPen(border, 1))
                    p.setBrush(QBrush(color))
                    p.drawPolygon(poly)

        def _draw_traces(self, p):
            for tr in self.pcb.traces:
                if not self.is_layer_visible(tr.layer):
                    continue
                color = get_layer_color(tr.layer, 200)
                w = max(1, int(tr.width_mm * self._zoom))
                p.setPen(QPen(color, w))
                x1, y1 = self._to_screen(tr.start.x, tr.start.y)
                x2, y2 = self._to_screen(tr.end.x, tr.end.y)
                p.drawLine(int(x1), int(y1), int(x2), int(y2))

            # Arc traces
            for arc in self.pcb.arc_traces:
                if not self.is_layer_visible(arc.layer):
                    continue
                color = get_layer_color(arc.layer, 200)
                w = max(1, int(arc.width_mm * self._zoom))
                p.setPen(QPen(color, w))
                # Draw as line segments
                pts = []
                pts.append(arc.start)
                if arc.mid.x != 0 or arc.mid.y != 0:
                    pts.append(arc.mid)
                pts.append(arc.end)
                for k in range(len(pts) - 1):
                    x1, y1 = self._to_screen(pts[k].x, pts[k].y)
                    x2, y2 = self._to_screen(pts[k+1].x, pts[k+1].y)
                    p.drawLine(int(x1), int(y1), int(x2), int(y2))

        def _draw_vias(self, p):
            for via in self.pcb.vias:
                # Show via if either of its layers is visible
                if not (self.is_layer_visible(via.layers[0]) or
                        self.is_layer_visible(via.layers[1])):
                    continue
                sx, sy = self._to_screen(via.pos.x, via.pos.y)
                r_outer = max(2, int(via.diam_mm / 2 * self._zoom))
                r_drill = max(1, int(via.drill_mm / 2 * self._zoom))
                p.setPen(QPen(QColor(200, 200, 200, 200), 1))
                p.setBrush(QBrush(QColor(180, 180, 180, 100)))
                p.drawEllipse(QPointF(sx, sy), r_outer, r_outer)
                p.setBrush(QBrush(QColor(30, 33, 39)))
                p.drawEllipse(QPointF(sx, sy), r_drill, r_drill)

        def _draw_pads(self, p):
            """Draw component pads with proper shapes."""
            for comp in self.pcb.components:
                for pad in comp.pads:
                    # Check if any pad layer is visible
                    visible = any(self.is_layer_visible(l) for l in pad.layers)
                    if not visible and pad.layers:
                        # Check wildcard layers
                        if not any(l in ('*.Cu', 'F&B.Cu') for l in pad.layers):
                            continue

                    sx, sy = self._to_screen(pad.pos.x, pad.pos.y)
                    pw = max(2, int(pad.w * self._zoom))
                    ph = max(2, int(pad.h * self._zoom))

                    # Color by pad type
                    if pad.pad_type == 'thru_hole':
                        fill = QColor(200, 200, 100, 120)
                        border = QColor(200, 200, 100, 200)
                    elif pad.pad_type == 'np_thru_hole':
                        fill = QColor(150, 150, 150, 80)
                        border = QColor(150, 150, 150, 150)
                    else:
                        layer = pad.layers[0] if pad.layers else comp.layer
                        fill = get_layer_color(layer, 100)
                        border = get_layer_color(layer, 200)

                    p.setPen(QPen(border, 1))
                    p.setBrush(QBrush(fill))

                    if pad.shape == 'circle':
                        r = max(pw, ph) // 2
                        p.drawEllipse(QPointF(sx, sy), r, r)
                    elif pad.shape == 'oval':
                        p.drawEllipse(int(sx - pw/2), int(sy - ph/2), pw, ph)
                    else:
                        p.drawRect(int(sx - pw/2), int(sy - ph/2), pw, ph)

                    # Draw drill hole
                    if pad.drill > 0:
                        dr = max(1, int(pad.drill / 2 * self._zoom))
                        p.setPen(QPen(QColor(30, 33, 39), 1))
                        p.setBrush(QBrush(QColor(30, 33, 39)))
                        p.drawEllipse(QPointF(sx, sy), dr, dr)

        def _draw_components(self, p):
            for comp in self.pcb.components:
                if not self.is_layer_visible(comp.layer):
                    continue
                bb = comp.bbox()
                x1, y1 = self._to_screen(bb[0].x, bb[0].y)
                x2, y2 = self._to_screen(bb[1].x, bb[1].y)
                w, h = x2 - x1, y2 - y1

                is_sel = comp.ref == self._selected_ref
                has_power = self.cfg and self.cfg.get_power(comp.ref) > 0
                fill = (QColor(0, 150, 136, 100) if is_sel else
                        (QColor(255, 167, 38, 80) if has_power else
                         QColor(160, 160, 170, 30)))
                border = QColor(0, 200, 180) if is_sel else QColor(160, 160, 170, 80)
                p.setBrush(QBrush(fill))
                p.setPen(QPen(border, 2 if is_sel else 1))
                p.drawRect(int(x1), int(y1), int(w), int(h))

                if w > 20 and h > 10:
                    p.setPen(QPen(QColor(220, 225, 232) if is_sel else QColor(180, 185, 190)))
                    p.setFont(QFont("sans-serif", max(7, min(10, int(min(w, h) / 4)))))
                    p.drawText(QRectF(x1, y1, w, h), Qt.AlignCenter, comp.ref)

        def _draw_mounting(self, p):
            for hole in self.pcb.mounting_holes:
                sx, sy = self._to_screen(hole.x, hole.y)
                r = max(4, int(1.6 * self._zoom))
                is_thermal = (self.cfg and any(
                    abs(mp.x_mm - hole.x) < 2 and abs(mp.y_mm - hole.y) < 2
                    and mp.fixed_temp_c is not None
                    for mp in self.cfg.mounting))
                color = QColor(0, 200, 180) if is_thermal else QColor(180, 180, 190)
                p.setPen(QPen(color, 2))
                p.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 40)))
                p.drawEllipse(QPointF(sx, sy), r, r)

        def _draw_thermal(self, p):
            if not self.mesh or self.result is None or len(self.result.temps) == 0:
                return
            mesh = self.mesh
            grid = self.result.grid_2d(mesh.nx, mesh.ny, mesh.nz, 0)
            tmin, tmax = self.result.min_t, self.result.max_t
            trange = max(tmax - tmin, 0.1)

            for iy in range(mesh.ny):
                for ix in range(mesh.nx):
                    frac = max(0, min(1, (grid[iy, ix] - tmin) / trange))
                    # Blue → Cyan → Green → Yellow → Red
                    if frac < 0.25:
                        f = frac / 0.25
                        r, g, b = 0, int(100 * f), int(255 * (1 - f * 0.3))
                    elif frac < 0.5:
                        f = (frac - 0.25) / 0.25
                        r, g, b = 0, int(100 + 155 * f), int(178 * (1 - f))
                    elif frac < 0.75:
                        f = (frac - 0.5) / 0.25
                        r, g, b = int(255 * f), 255, 0
                    else:
                        f = (frac - 0.75) / 0.25
                        r, g, b = 255, int(255 * (1 - f)), 0

                    node = mesh.nodes[mesh.idx(ix, iy, 0)]
                    half_dx = node.dx / 2
                    half_dy = node.dy / 2
                    x1, y1 = self._to_screen(node.x - half_dx, node.y - half_dy)
                    x2, y2 = self._to_screen(node.x + half_dx, node.y + half_dy)
                    p.fillRect(int(x1), int(y1),
                               max(1, int(x2 - x1)), max(1, int(y2 - y1)),
                               QColor(r, g, b, 160))

    # ── Solver Worker ──────────────────────────────────────────────
    class SolverWorker(QThread):
        progress = pyqtSignal(int, str)
        finished = pyqtSignal(object, object)
        error = pyqtSignal(str)

        def __init__(self, pcb, cfg):
            super().__init__()
            self.pcb = pcb
            self.cfg = cfg

        def run(self):
            try:
                gen = MeshGen(self.pcb, self.cfg)
                mesh = gen.generate(cb=lambda p, m: self.progress.emit(p // 2, m))
                solver = ThermalSolver()
                self.progress.emit(50, f"Solving ({solver.backend})...")
                if self.cfg.sim.mode == "transient":
                    result = solver.solve_transient(
                        mesh, self.cfg.sim,
                        cb=lambda p, m: self.progress.emit(50 + p // 2, m))
                else:
                    result = solver.solve_steady(
                        mesh, self.cfg.sim,
                        cb=lambda p, m: self.progress.emit(50 + p // 2, m))
                if result.error:
                    self.error.emit(result.error)
                else:
                    self.finished.emit(result, mesh)
            except Exception as e:
                self.error.emit(f"{e}\n{traceback.format_exc()}")

    # ── Dialogs ────────────────────────────────────────────────────
    class PowerDialog(QDialog):
        def __init__(self, parent, ref, value, current_power):
            super().__init__(parent)
            self.setWindowTitle(f"Set Power — {ref}")
            self.setMinimumWidth(320)
            self.setStyleSheet(STYLE)
            layout = QVBoxLayout()
            layout.addWidget(QLabel(f"<b>{ref}</b> ({value})"))
            form = QFormLayout()
            self.spin = QDoubleSpinBox()
            self.spin.setRange(0, 1000)
            self.spin.setDecimals(4)
            self.spin.setSuffix(" W")
            self.spin.setValue(current_power)
            self.spin.setFocus()
            form.addRow("Power dissipation:", self.spin)
            layout.addLayout(form)
            btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            btns.accepted.connect(self.accept)
            btns.rejected.connect(self.reject)
            layout.addWidget(btns)
            self.setLayout(layout)

        @property
        def power(self):
            return self.spin.value()

    class HeatsinkEditDialog(QDialog):
        def __init__(self, parent, hs=None, materials=None):
            super().__init__(parent)
            self.setWindowTitle("Edit Heatsink" if hs else "Add Heatsink")
            self.setMinimumWidth(400)
            self.setStyleSheet(STYLE)
            layout = QVBoxLayout()
            form = QFormLayout()
            self.id_edit = QLineEdit(hs.hs_id if hs else "HS1")
            form.addRow("Heatsink ID:", self.id_edit)
            self.mat_combo = QComboBox()
            self.mat_combo.addItems(materials or list(HEATSINK_MATERIALS.keys()))
            if hs:
                self.mat_combo.setCurrentText(hs.material)
            form.addRow("Material:", self.mat_combo)
            self.thick_spin = QDoubleSpinBox()
            self.thick_spin.setRange(0.1, 50)
            self.thick_spin.setDecimals(1)
            self.thick_spin.setSuffix(" mm")
            self.thick_spin.setValue(hs.thickness_mm if hs else 3.0)
            form.addRow("Thickness:", self.thick_spin)
            self.emiss_spin = QDoubleSpinBox()
            self.emiss_spin.setRange(0, 1)
            self.emiss_spin.setDecimals(2)
            mat_data = HEATSINK_MATERIALS.get(self.mat_combo.currentText(), {})
            self.emiss_spin.setValue(hs.emissivity if hs and hs.emissivity is not None
                                    else mat_data.get('emiss', 0.85))
            form.addRow("Emissivity:", self.emiss_spin)
            self.poly_edit = QTextEdit()
            self.poly_edit.setPlaceholderText("Enter polygon points:\nx1, y1\nx2, y2\n...")
            self.poly_edit.setMaximumHeight(120)
            if hs and hs.polygon:
                self.poly_edit.setPlainText(
                    "\n".join(f"{p[0]:.2f}, {p[1]:.2f}" for p in hs.polygon))
            form.addRow("Polygon (mm):", self.poly_edit)
            layout.addLayout(form)
            btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            btns.accepted.connect(self.accept)
            btns.rejected.connect(self.reject)
            layout.addWidget(btns)
            self.setLayout(layout)

        def get_heatsink(self):
            polygon = []
            for line in self.poly_edit.toPlainText().strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                try:
                    parts = line.split(',')
                    polygon.append((float(parts[0].strip()), float(parts[1].strip())))
                except (ValueError, IndexError):
                    pass
            return HeatsinkCfg(
                hs_id=self.id_edit.text(),
                material=self.mat_combo.currentText(),
                polygon=polygon,
                thickness_mm=self.thick_spin.value(),
                emissivity=self.emiss_spin.value())

    class MountingDialog(QDialog):
        def __init__(self, parent, mp=None):
            super().__init__(parent)
            self.setWindowTitle("Mounting Point")
            self.setMinimumWidth(320)
            self.setStyleSheet(STYLE)
            layout = QVBoxLayout()
            form = QFormLayout()
            self.x_spin = QDoubleSpinBox()
            self.x_spin.setRange(-1000, 1000)
            self.x_spin.setDecimals(2)
            self.x_spin.setSuffix(" mm")
            self.x_spin.setValue(mp.x_mm if mp else 0)
            form.addRow("X:", self.x_spin)
            self.y_spin = QDoubleSpinBox()
            self.y_spin.setRange(-1000, 1000)
            self.y_spin.setDecimals(2)
            self.y_spin.setSuffix(" mm")
            self.y_spin.setValue(mp.y_mm if mp else 0)
            form.addRow("Y:", self.y_spin)
            self.temp_check = QCheckBox("Fixed temperature")
            self.temp_check.setChecked(mp.fixed_temp_c is not None if mp else False)
            form.addRow(self.temp_check)
            self.temp_spin = QDoubleSpinBox()
            self.temp_spin.setRange(-200, 500)
            self.temp_spin.setDecimals(1)
            self.temp_spin.setSuffix(" °C")
            self.temp_spin.setValue(mp.fixed_temp_c if mp and mp.fixed_temp_c is not None else 25.0)
            form.addRow("Temperature:", self.temp_spin)
            layout.addLayout(form)
            btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            btns.accepted.connect(self.accept)
            btns.rejected.connect(self.reject)
            layout.addWidget(btns)
            self.setLayout(layout)

    # ── Main Window ────────────────────────────────────────────────
    class MainWindow(QMainWindow):
        def __init__(self, pcb_path=""):
            super().__init__()
            self.setWindowTitle("TVAC Thermal Analyzer v4")
            self.setMinimumSize(1280, 800)
            self.setStyleSheet(STYLE)
            self.pcb = None
            self.cfg = Config()
            self.result = None
            self.mesh = None
            self.worker = None
            self._pcb_path = pcb_path
            self._build_ui()
            if pcb_path and os.path.isfile(pcb_path):
                QTimer.singleShot(100, lambda: self._load_pcb(pcb_path))

        def _build_ui(self):
            # Menu
            menu = self.menuBar()
            file_menu = menu.addMenu("&File")
            act_open = file_menu.addAction("&Open PCB...")
            act_open.setShortcut("Ctrl+O")
            act_open.triggered.connect(self._on_open)
            act_save = file_menu.addAction("&Save Config")
            act_save.setShortcut("Ctrl+S")
            act_save.triggered.connect(self._on_save_config)
            act_load_cfg = file_menu.addAction("&Load Config...")
            act_load_cfg.triggered.connect(self._on_load_config)
            file_menu.addSeparator()
            act_quit = file_menu.addAction("&Quit")
            act_quit.setShortcut("Ctrl+Q")
            act_quit.triggered.connect(self.close)

            # Main splitter
            splitter = QSplitter(Qt.Horizontal)

            # LEFT: tabs
            left = QWidget()
            left_layout = QVBoxLayout(left)
            left_layout.setContentsMargins(0, 0, 0, 0)
            title_w = QWidget()
            title_l = QVBoxLayout(title_w)
            title_l.setContentsMargins(12, 8, 12, 4)
            lbl = QLabel("TVAC Thermal Analyzer v4")
            lbl.setObjectName("title")
            title_l.addWidget(lbl)
            self.lbl_board = QLabel("No board loaded")
            self.lbl_board.setObjectName("subtitle")
            title_l.addWidget(self.lbl_board)
            left_layout.addWidget(title_w)
            self.tabs = QTabWidget()
            left_layout.addWidget(self.tabs)
            self._build_components_tab()
            self._build_heatsink_tab()
            self._build_mounting_tab()
            self._build_current_tab()
            self._build_sim_tab()
            splitter.addWidget(left)

            # RIGHT: PCB view
            right = QWidget()
            right_layout = QVBoxLayout(right)
            right_layout.setContentsMargins(0, 0, 0, 0)
            hdr = QWidget()
            hdr_l = QHBoxLayout(hdr)
            hdr_l.setContentsMargins(12, 8, 12, 4)
            lbl_pcb = QLabel("PCB Preview")
            lbl_pcb.setObjectName("title")
            hdr_l.addWidget(lbl_pcb)
            hdr_l.addStretch()
            self.lbl_info = QLabel("")
            self.lbl_info.setObjectName("subtitle")
            hdr_l.addWidget(self.lbl_info)
            right_layout.addWidget(hdr)
            self.pcb_view = PCBView()
            self.pcb_view.componentClicked.connect(self._on_comp_clicked)
            right_layout.addWidget(self.pcb_view)

            # Layer toggles
            toggle_bar = QWidget()
            tl = QHBoxLayout(toggle_bar)
            tl.setContentsMargins(12, 4, 12, 4)
            self._layer_cbs = {}
            for name, attr, default in [
                ("Grid", "show_grid", True),
                ("Outline", "show_outline", True),
                ("Top Cu", "show_top", True),
                ("Bottom Cu", "show_bottom", True),
                ("Inner Cu", "show_inner", True),
                ("Traces", "show_traces", True),
                ("Vias", "show_vias", True),
                ("Heatsinks", "show_heatsinks", True),
                ("Mounting", "show_mounting", True),
                ("Thermal", "show_thermal", False),
            ]:
                cb = QCheckBox(name)
                cb.setChecked(default)
                cb.toggled.connect(
                    lambda v, a=attr: (setattr(self.pcb_view, a, v), self.pcb_view.update())
                )
                tl.addWidget(cb)
                self._layer_cbs[attr] = cb
            tl.addStretch()
            right_layout.addWidget(toggle_bar)
            splitter.addWidget(right)
            splitter.setSizes([440, 840])

            # Bottom bar
            bottom = QWidget()
            bl = QHBoxLayout(bottom)
            bl.setContentsMargins(12, 6, 12, 6)
            self.status_label = QLabel("● Ready")
            self.status_label.setStyleSheet("color: #66bb6a;")
            bl.addWidget(self.status_label)
            bl.addStretch()
            self.progress_bar = QProgressBar()
            self.progress_bar.setMaximumWidth(200)
            self.progress_bar.setVisible(False)
            bl.addWidget(self.progress_bar)
            btn_save = QPushButton("💾 Save Config")
            btn_save.clicked.connect(self._on_save_config)
            bl.addWidget(btn_save)
            self.btn_run = QPushButton("▶  Run Simulation")
            self.btn_run.setObjectName("runBtn")
            self.btn_run.clicked.connect(self._on_run)
            bl.addWidget(self.btn_run)

            central = QWidget()
            ml = QVBoxLayout(central)
            ml.setContentsMargins(0, 0, 0, 0)
            ml.setSpacing(0)
            ml.addWidget(splitter, 1)
            ml.addWidget(bottom)
            self.setCentralWidget(central)

        # ── Tab: Components ────────────────────────────────────────
        def _build_components_tab(self):
            w = QWidget()
            layout = QVBoxLayout(w)
            layout.addWidget(QLabel("Set power dissipation for heat-generating components"))
            sr = QHBoxLayout()
            self.comp_search = QLineEdit()
            self.comp_search.setPlaceholderText("Search components...")
            self.comp_search.textChanged.connect(self._filter_comps)
            sr.addWidget(self.comp_search)
            for pf in ["U*", "Q*", "R*", "All"]:
                btn = QPushButton(pf)
                btn.setMaximumWidth(45)
                p = pf.replace("*", "") if pf != "All" else ""
                btn.clicked.connect(lambda _, pf=p: self.comp_search.setText(pf))
                sr.addWidget(btn)
            layout.addLayout(sr)
            self.comp_table = QTableWidget()
            self.comp_table.setColumnCount(5)
            self.comp_table.setHorizontalHeaderLabels(
                ["Reference", "Value", "Footprint", "Power (W)", "Source"]
            )
            self.comp_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.comp_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.comp_table.setAlternatingRowColors(True)
            self.comp_table.cellDoubleClicked.connect(self._on_comp_dblclick)
            self.comp_table.itemSelectionChanged.connect(self._on_comp_table_select)
            layout.addWidget(self.comp_table)
            ar = QHBoxLayout()
            be = QPushButton("✎ Edit Power")
            be.clicked.connect(self._on_edit_power)
            ar.addWidget(be)
            bc = QPushButton("✕ Clear")
            bc.clicked.connect(self._on_clear_power)
            ar.addWidget(bc)
            ar.addStretch()
            self.lbl_total = QLabel("Total: 0.000 W")
            self.lbl_total.setStyleSheet("font-weight:bold;color:#dce1e8;")
            ar.addWidget(self.lbl_total)
            layout.addLayout(ar)
            self.tabs.addTab(w, "Components")

        # ── Tab: Heatsinks ─────────────────────────────────────────
        def _build_heatsink_tab(self):
            w = QWidget()
            layout = QVBoxLayout(w)
            layout.addWidget(QLabel("Configure heatsink polygons with material properties"))
            self.hs_table = QTableWidget()
            self.hs_table.setColumnCount(5)
            self.hs_table.setHorizontalHeaderLabels(
                ["ID", "Material", "Area (mm²)", "Thickness", "Emissivity"]
            )
            self.hs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.hs_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.hs_table.setAlternatingRowColors(True)
            layout.addWidget(self.hs_table)
            ar = QHBoxLayout()
            ba = QPushButton("+ Add Heatsink")
            ba.clicked.connect(self._on_add_heatsink)
            ar.addWidget(ba)
            be = QPushButton("✎ Edit")
            be.clicked.connect(self._on_edit_heatsink)
            ar.addWidget(be)
            bd = QPushButton("✕ Remove")
            bd.clicked.connect(self._on_del_heatsink)
            ar.addWidget(bd)
            ar.addStretch()
            layout.addLayout(ar)
            self.tabs.addTab(w, "Heatsinks")

        # ── Tab: Mounting ──────────────────────────────────────────
        def _build_mounting_tab(self):
            w = QWidget()
            layout = QVBoxLayout(w)
            layout.addWidget(QLabel("Define thermal boundary conditions at mounting locations"))
            br = QHBoxLayout()
            bd = QPushButton("Auto-Detect Holes")
            bd.clicked.connect(self._on_detect_mounting)
            br.addWidget(bd)
            ba = QPushButton("+ Add Manual")
            ba.clicked.connect(self._on_add_mounting)
            br.addWidget(ba)
            br.addStretch()
            layout.addLayout(br)
            self.mount_table = QTableWidget()
            self.mount_table.setColumnCount(4)
            self.mount_table.setHorizontalHeaderLabels(
                ["X (mm)", "Y (mm)", "Diameter", "Fixed T (°C)"]
            )
            self.mount_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.mount_table.setAlternatingRowColors(True)
            layout.addWidget(self.mount_table)
            ar = QHBoxLayout()
            be = QPushButton("✎ Set Temp")
            be.clicked.connect(self._on_edit_mounting)
            ar.addWidget(be)
            bd2 = QPushButton("✕ Remove")
            bd2.clicked.connect(self._on_del_mounting)
            ar.addWidget(bd2)
            ar.addStretch()
            layout.addLayout(ar)
            self.tabs.addTab(w, "Mounting")

        # ── Tab: Current Path ──────────────────────────────────────
        def _build_current_tab(self):
            w = QWidget()
            layout = QVBoxLayout(w)
            layout.addWidget(QLabel("Define current flow paths for Joule heating (I²R) analysis"))
            grp = QGroupBox("Add Current Path")
            gf = QFormLayout()
            self.cp_source = QComboBox()
            self.cp_source.addItem("(select net)")
            gf.addRow("Source Net (current in):", self.cp_source)
            self.cp_sink = QComboBox()
            self.cp_sink.addItem("(select net)")
            gf.addRow("Sink Net (current out):", self.cp_sink)
            hr = QHBoxLayout()
            self.cp_current = QDoubleSpinBox()
            self.cp_current.setRange(0, 100)
            self.cp_current.setDecimals(3)
            self.cp_current.setSuffix(" A")
            self.cp_current.setValue(1.0)
            hr.addWidget(self.cp_current)
            bcp = QPushButton("Add Path")
            bcp.clicked.connect(self._on_add_current)
            hr.addWidget(bcp)
            gf.addRow("Current:", hr)
            grp.setLayout(gf)
            layout.addWidget(grp)
            self.current_table = QTableWidget()
            self.current_table.setColumnCount(4)
            self.current_table.setHorizontalHeaderLabels(
                ["ID", "Source Net", "Sink Net", "Current (A)"]
            )
            self.current_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.current_table.setAlternatingRowColors(True)
            layout.addWidget(self.current_table)
            ar = QHBoxLayout()
            be = QPushButton("✎ Edit")
            be.clicked.connect(self._on_edit_current)
            ar.addWidget(be)
            bd = QPushButton("✕ Remove")
            bd.clicked.connect(self._on_del_current)
            ar.addWidget(bd)
            ar.addStretch()
            self.lbl_total_i = QLabel("Total: 0.000 A")
            self.lbl_total_i.setStyleSheet("font-weight:bold;color:#dce1e8;")
            ar.addWidget(self.lbl_total_i)
            layout.addLayout(ar)
            self.tabs.addTab(w, "Current Path")

        # ── Tab: Simulation ────────────────────────────────────────
        def _build_sim_tab(self):
            w = QWidget()
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            inner = QWidget()
            layout = QVBoxLayout(inner)

            # Heat source mode
            grp1 = QGroupBox("Heat Source Mode")
            g1l = QVBoxLayout()
            self.rb_comp_power = QRadioButton("Component Power Dissipation")
            self.rb_comp_power.setChecked(True)
            self.rb_current_inj = QRadioButton("Current Injection (I²R Joule Heating)")
            g1l.addWidget(self.rb_comp_power)
            g1l.addWidget(self.rb_current_inj)
            grp1.setLayout(g1l)
            layout.addWidget(grp1)

            # Analysis type
            grp2 = QGroupBox("Analysis Type")
            g2l = QVBoxLayout()
            self.rb_steady = QRadioButton("Steady State")
            self.rb_steady.setChecked(True)
            self.rb_transient = QRadioButton("Transient")
            g2l.addWidget(self.rb_steady)
            g2l.addWidget(self.rb_transient)
            grp2.setLayout(g2l)
            layout.addWidget(grp2)

            # Environment
            grp3 = QGroupBox("Environment")
            form = QFormLayout()
            self.sim_ambient = QDoubleSpinBox()
            self.sim_ambient.setRange(-200, 500)
            self.sim_ambient.setValue(25.0)
            self.sim_ambient.setSuffix(" °C")
            form.addRow("Ambient Temp:", self.sim_ambient)
            self.sim_chamber = QDoubleSpinBox()
            self.sim_chamber.setRange(-200, 500)
            self.sim_chamber.setValue(25.0)
            self.sim_chamber.setSuffix(" °C")
            form.addRow("Chamber Wall:", self.sim_chamber)
            self.sim_initial = QDoubleSpinBox()
            self.sim_initial.setRange(-200, 500)
            self.sim_initial.setValue(25.0)
            self.sim_initial.setSuffix(" °C")
            form.addRow("Initial Board:", self.sim_initial)
            self.sim_radiation = QCheckBox("Include Radiation")
            self.sim_radiation.setChecked(True)
            form.addRow(self.sim_radiation)
            grp3.setLayout(form)
            layout.addWidget(grp3)

            # Mesh
            grp4 = QGroupBox("Mesh")
            form2 = QFormLayout()
            self.sim_res = QDoubleSpinBox()
            self.sim_res.setRange(0.1, 10.0)
            self.sim_res.setValue(1.0)
            self.sim_res.setDecimals(1)
            self.sim_res.setSuffix(" mm")
            form2.addRow("Base Resolution:", self.sim_res)
            self.sim_adaptive = QCheckBox("Adaptive Mesh Refinement")
            self.sim_adaptive.setChecked(True)
            form2.addRow(self.sim_adaptive)
            self.sim_min_cell = QDoubleSpinBox()
            self.sim_min_cell.setRange(0.05, 2.0)
            self.sim_min_cell.setValue(0.15)
            self.sim_min_cell.setDecimals(2)
            self.sim_min_cell.setSuffix(" mm")
            form2.addRow("Min Cell (adaptive):", self.sim_min_cell)
            self.sim_max_cell = QDoubleSpinBox()
            self.sim_max_cell.setRange(0.5, 20.0)
            self.sim_max_cell.setValue(3.0)
            self.sim_max_cell.setDecimals(1)
            self.sim_max_cell.setSuffix(" mm")
            form2.addRow("Max Cell (adaptive):", self.sim_max_cell)
            grp4.setLayout(form2)
            layout.addWidget(grp4)

            # Transient
            grp5 = QGroupBox("Transient Settings")
            form3 = QFormLayout()
            self.sim_duration = QDoubleSpinBox()
            self.sim_duration.setRange(1, 100000)
            self.sim_duration.setValue(600)
            self.sim_duration.setSuffix(" s")
            form3.addRow("Duration:", self.sim_duration)
            self.sim_dt = QDoubleSpinBox()
            self.sim_dt.setRange(0.01, 100)
            self.sim_dt.setValue(1.0)
            self.sim_dt.setSuffix(" s")
            form3.addRow("Timestep:", self.sim_dt)
            grp5.setLayout(form3)
            layout.addWidget(grp5)

            # Solver info
            grp6 = QGroupBox("Solver Info")
            form4 = QFormLayout()
            solver = ThermalSolver()
            self.lbl_backend = QLabel(solver.backend)
            self.lbl_backend.setStyleSheet("font-weight:bold;color:#009688;")
            form4.addRow("Backend:", self.lbl_backend)
            grp6.setLayout(form4)
            layout.addWidget(grp6)

            layout.addStretch()
            scroll.setWidget(inner)
            outer_layout = QVBoxLayout()
            outer_layout.setContentsMargins(0, 0, 0, 0)
            outer_w = QWidget()
            outer_layout.addWidget(scroll)
            outer_w.setLayout(outer_layout)
            self.tabs.addTab(outer_w, "Simulation")

        # ── File operations ────────────────────────────────────────
        def _on_open(self):
            path, _ = QFileDialog.getOpenFileName(
                self, "Open KiCad PCB", "",
                "KiCad PCB (*.kicad_pcb);;All Files (*)"
            )
            if path:
                self._load_pcb(path)

        def _load_pcb(self, path):
            try:
                self.status_label.setText("● Loading PCB...")
                self.status_label.setStyleSheet("color:#ffa726;")
                QApplication.processEvents()
                self.pcb = KiCadPCBReader().read(path)
                self._pcb_path = path
                cfg_path = Path(path).with_suffix('.tvac_config.json')
                if cfg_path.exists():
                    try:
                        self.cfg = Config.load(str(cfg_path))
                    except Exception:
                        self.cfg = Config()
                else:
                    self.cfg = Config()
                # Import POWER_DISSIPATION fields from PCB
                for comp in self.pcb.components:
                    if comp.power_w > 0:
                        self.cfg.set_power(comp.ref, comp.power_w, "pcb_field")
                # Auto-detect mounting holes
                for hp in self.pcb.mounting_holes:
                    if not any(
                        abs(mp.x_mm - hp.x) < 2 and abs(mp.y_mm - hp.y) < 2
                        for mp in self.cfg.mounting
                    ):
                        self.cfg.mounting.append(
                            MountingCfg(
                                mp_id=f"H{len(self.cfg.mounting)+1}",
                                x_mm=hp.x, y_mm=hp.y,
                            )
                        )
                n_cu = len(self.pcb.copper_layers)
                n_via = len(self.pcb.vias)
                n_zone = len(self.pcb.zones)
                self.lbl_board.setText(f"Board: {Path(path).stem}")
                self.lbl_info.setText(
                    f"{len(self.pcb.components)} comps · "
                    f"{len(self.pcb.traces)} traces · "
                    f"{n_via} vias · "
                    f"{n_zone} zones · "
                    f"{n_cu} Cu layers"
                )
                self.pcb_view.set_data(self.pcb, self.cfg)
                self._refresh_all()
                self._populate_net_combos()
                self.status_label.setText("● Ready")
                self.status_label.setStyleSheet("color:#66bb6a;")
            except Exception as e:
                self.status_label.setText("● Error loading PCB")
                self.status_label.setStyleSheet("color:#ef5350;")
                QMessageBox.critical(self, "Error", f"Failed to load PCB:\n{e}")

        def _refresh_all(self):
            self._populate_comp_table()
            self._populate_hs_table()
            self._populate_mount_table()
            self._populate_current_table()

        def _populate_net_combos(self):
            nets = sorted(set(self.pcb.nets.values()) - {""}) if self.pcb else []
            for combo in [self.cp_source, self.cp_sink]:
                combo.clear()
                combo.addItem("(select net)")
                combo.addItems(nets)
            for i in range(self.cp_sink.count()):
                if 'GND' in self.cp_sink.itemText(i).upper():
                    self.cp_sink.setCurrentIndex(i)
                    break

        # ── Component table ────────────────────────────────────────
        def _populate_comp_table(self, ft=""):
            if not self.pcb:
                return
            self.comp_table.setRowCount(0)
            ft = ft.lower()
            for comp in self.pcb.components:
                if ft and ft not in comp.ref.lower():
                    continue
                row = self.comp_table.rowCount()
                self.comp_table.insertRow(row)
                self.comp_table.setItem(row, 0, QTableWidgetItem(comp.ref))
                self.comp_table.setItem(row, 1, QTableWidgetItem(comp.value))
                fp_short = (
                    comp.footprint.split(':')[-1][:25]
                    if ':' in comp.footprint
                    else comp.footprint[:25]
                )
                self.comp_table.setItem(row, 2, QTableWidgetItem(fp_short))
                pw = self.cfg.get_power(comp.ref)
                item = QTableWidgetItem(f"{pw:.4f}" if pw > 0 else "–")
                if pw > 0:
                    item.setBackground(QColor(255, 167, 38, 50))
                self.comp_table.setItem(row, 3, item)
                src = next(
                    (cp.source for cp in self.cfg.comp_power if cp.ref == comp.ref), ""
                )
                self.comp_table.setItem(row, 4, QTableWidgetItem(src))
            self.lbl_total.setText(f"Total: {self.cfg.total_power():.3f} W")

        def _filter_comps(self, text):
            self._populate_comp_table(text)

        def _on_comp_table_select(self):
            rows = self.comp_table.selectionModel().selectedRows()
            if rows:
                self.pcb_view.select(self.comp_table.item(rows[0].row(), 0).text())

        def _on_comp_clicked(self, ref):
            for row in range(self.comp_table.rowCount()):
                if self.comp_table.item(row, 0).text() == ref:
                    self.comp_table.selectRow(row)
                    break

        def _on_comp_dblclick(self, row, col):
            ref = self.comp_table.item(row, 0).text()
            self._edit_power_for(ref)

        def _on_edit_power(self):
            rows = self.comp_table.selectionModel().selectedRows()
            if rows:
                self._edit_power_for(
                    self.comp_table.item(rows[0].row(), 0).text()
                )

        def _edit_power_for(self, ref):
            comp = next((c for c in self.pcb.components if c.ref == ref), None)
            if not comp:
                return
            dlg = PowerDialog(self, ref, comp.value, self.cfg.get_power(ref))
            if dlg.exec_() == QDialog.Accepted:
                self.cfg.set_power(ref, dlg.power, "manual")
                self._populate_comp_table(self.comp_search.text())
                self.pcb_view.update()

        def _on_clear_power(self):
            rows = self.comp_table.selectionModel().selectedRows()
            if rows:
                ref = self.comp_table.item(rows[0].row(), 0).text()
                self.cfg.comp_power = [
                    cp for cp in self.cfg.comp_power if cp.ref != ref
                ]
                self._populate_comp_table(self.comp_search.text())
                self.pcb_view.update()

        # ── Heatsink table ─────────────────────────────────────────
        def _populate_hs_table(self):
            self.hs_table.setRowCount(0)
            for hs in self.cfg.heatsinks:
                row = self.hs_table.rowCount()
                self.hs_table.insertRow(row)
                self.hs_table.setItem(row, 0, QTableWidgetItem(hs.hs_id))
                self.hs_table.setItem(row, 1, QTableWidgetItem(hs.material))
                area = self._polygon_area(hs.polygon)
                self.hs_table.setItem(row, 2, QTableWidgetItem(f"{area:.1f}"))
                self.hs_table.setItem(
                    row, 3, QTableWidgetItem(f"{hs.thickness_mm:.1f} mm")
                )
                e = (
                    hs.emissivity
                    if hs.emissivity is not None
                    else HEATSINK_MATERIALS.get(hs.material, {}).get('emiss', 0.85)
                )
                self.hs_table.setItem(row, 4, QTableWidgetItem(f"{e:.2f}"))

        @staticmethod
        def _polygon_area(pts):
            if len(pts) < 3:
                return 0.0
            a = sum(
                pts[i][0] * pts[(i + 1) % len(pts)][1]
                - pts[(i + 1) % len(pts)][0] * pts[i][1]
                for i in range(len(pts))
            )
            return abs(a) / 2.0

        def _on_add_heatsink(self):
            hs_id = f"HS{len(self.cfg.heatsinks)+1}"
            dlg = HeatsinkEditDialog(self, HeatsinkCfg(hs_id=hs_id))
            if dlg.exec_() == QDialog.Accepted:
                self.cfg.heatsinks.append(dlg.get_heatsink())
                self._populate_hs_table()
                self.pcb_view.update()

        def _on_edit_heatsink(self):
            rows = self.hs_table.selectionModel().selectedRows()
            if not rows:
                return
            idx = rows[0].row()
            if idx >= len(self.cfg.heatsinks):
                return
            dlg = HeatsinkEditDialog(self, self.cfg.heatsinks[idx])
            if dlg.exec_() == QDialog.Accepted:
                self.cfg.heatsinks[idx] = dlg.get_heatsink()
                self._populate_hs_table()
                self.pcb_view.update()

        def _on_del_heatsink(self):
            rows = self.hs_table.selectionModel().selectedRows()
            if rows and rows[0].row() < len(self.cfg.heatsinks):
                del self.cfg.heatsinks[rows[0].row()]
                self._populate_hs_table()
                self.pcb_view.update()

        # ── Mounting table ─────────────────────────────────────────
        def _populate_mount_table(self):
            self.mount_table.setRowCount(0)
            for mp in self.cfg.mounting:
                row = self.mount_table.rowCount()
                self.mount_table.insertRow(row)
                self.mount_table.setItem(
                    row, 0, QTableWidgetItem(f"{mp.x_mm:.1f}")
                )
                self.mount_table.setItem(
                    row, 1, QTableWidgetItem(f"{mp.y_mm:.1f}")
                )
                self.mount_table.setItem(
                    row, 2, QTableWidgetItem(f"{mp.diameter_mm:.1f}")
                )
                t = (
                    f"{mp.fixed_temp_c:.1f}"
                    if mp.fixed_temp_c is not None
                    else "–"
                )
                item = QTableWidgetItem(t)
                if mp.fixed_temp_c is not None:
                    item.setBackground(QColor(102, 187, 106, 50))
                self.mount_table.setItem(row, 3, item)

        def _on_detect_mounting(self):
            if not self.pcb:
                return
            count = 0
            for hp in self.pcb.mounting_holes:
                if not any(
                    abs(mp.x_mm - hp.x) < 2 and abs(mp.y_mm - hp.y) < 2
                    for mp in self.cfg.mounting
                ):
                    self.cfg.mounting.append(
                        MountingCfg(
                            mp_id=f"H{len(self.cfg.mounting)+1}",
                            x_mm=hp.x, y_mm=hp.y,
                        )
                    )
                    count += 1
            self._populate_mount_table()
            self.pcb_view.update()
            QMessageBox.information(
                self, "Detect",
                f"Found {len(self.pcb.mounting_holes)} hole(s), added {count} new"
            )

        def _on_add_mounting(self):
            dlg = MountingDialog(self)
            if dlg.exec_() == QDialog.Accepted:
                temp = (
                    dlg.temp_spin.value() if dlg.temp_check.isChecked() else None
                )
                self.cfg.mounting.append(
                    MountingCfg(
                        mp_id=f"MP{len(self.cfg.mounting)+1}",
                        x_mm=dlg.x_spin.value(),
                        y_mm=dlg.y_spin.value(),
                        fixed_temp_c=temp,
                    )
                )
                self._populate_mount_table()
                self.pcb_view.update()

        def _on_edit_mounting(self):
            rows = self.mount_table.selectionModel().selectedRows()
            if not rows:
                return
            idx = rows[0].row()
            if idx >= len(self.cfg.mounting):
                return
            mp = self.cfg.mounting[idx]
            dlg = MountingDialog(self, mp)
            if dlg.exec_() == QDialog.Accepted:
                mp.x_mm = dlg.x_spin.value()
                mp.y_mm = dlg.y_spin.value()
                mp.fixed_temp_c = (
                    dlg.temp_spin.value() if dlg.temp_check.isChecked() else None
                )
                self._populate_mount_table()
                self.pcb_view.update()

        def _on_del_mounting(self):
            rows = self.mount_table.selectionModel().selectedRows()
            if rows and rows[0].row() < len(self.cfg.mounting):
                del self.cfg.mounting[rows[0].row()]
                self._populate_mount_table()
                self.pcb_view.update()

        # ── Current path table ─────────────────────────────────────
        def _populate_current_table(self):
            self.current_table.setRowCount(0)
            total = 0.0
            for cp in self.cfg.current_paths:
                row = self.current_table.rowCount()
                self.current_table.insertRow(row)
                self.current_table.setItem(row, 0, QTableWidgetItem(cp.path_id))
                self.current_table.setItem(row, 1, QTableWidgetItem(cp.source_net))
                self.current_table.setItem(row, 2, QTableWidgetItem(cp.sink_net))
                self.current_table.setItem(
                    row, 3, QTableWidgetItem(f"{cp.current_a:.3f}")
                )
                total += cp.current_a
            self.lbl_total_i.setText(f"Total: {total:.3f} A")

        def _on_add_current(self):
            src = self.cp_source.currentText()
            sink = self.cp_sink.currentText()
            if src == "(select net)" or sink == "(select net)":
                QMessageBox.warning(self, "Error", "Select source and sink nets")
                return
            if src == sink:
                QMessageBox.warning(self, "Error", "Source and sink must differ")
                return
            ids = {cp.path_id for cp in self.cfg.current_paths}
            i = 1
            while f"I{i}" in ids:
                i += 1
            self.cfg.current_paths.append(
                CurrentPathCfg(
                    path_id=f"I{i}",
                    source_net=src,
                    sink_net=sink,
                    current_a=self.cp_current.value(),
                    description=f"{src} → {sink}",
                )
            )
            self._populate_current_table()

        def _on_edit_current(self):
            rows = self.current_table.selectionModel().selectedRows()
            if not rows:
                return
            idx = rows[0].row()
            if idx >= len(self.cfg.current_paths):
                return
            cp = self.cfg.current_paths[idx]
            val, ok = QInputDialog.getDouble(
                self, "Edit Current",
                f"Current (A) for {cp.source_net} → {cp.sink_net}:",
                cp.current_a, 0, 100, 3,
            )
            if ok:
                cp.current_a = val
                self._populate_current_table()

        def _on_del_current(self):
            rows = self.current_table.selectionModel().selectedRows()
            if rows and rows[0].row() < len(self.cfg.current_paths):
                del self.cfg.current_paths[rows[0].row()]
                self._populate_current_table()

        # ── Config ─────────────────────────────────────────────────
        def _on_save_config(self):
            if self._pcb_path:
                cfg_path = str(
                    Path(self._pcb_path).with_suffix('.tvac_config.json')
                )
            else:
                cfg_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Config", "", "JSON (*.json)"
                )
                if not cfg_path:
                    return
            self._sync_sim_cfg()
            self.cfg.save(cfg_path)
            self.status_label.setText("● Config saved")
            self.status_label.setStyleSheet("color:#66bb6a;")

        def _on_load_config(self):
            path, _ = QFileDialog.getOpenFileName(
                self, "Load Config", "", "JSON (*.json)"
            )
            if path:
                try:
                    self.cfg = Config.load(path)
                    self._refresh_all()
                    self.pcb_view.cfg = self.cfg
                    self.pcb_view.update()
                    self.status_label.setText("● Config loaded")
                    self.status_label.setStyleSheet("color:#66bb6a;")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed:\n{e}")

        def _sync_sim_cfg(self):
            s = self.cfg.sim
            s.mode = (
                "transient" if self.rb_transient.isChecked() else "steady_state"
            )
            s.heat_source_mode = (
                "current_injection"
                if self.rb_current_inj.isChecked()
                else "component_power"
            )
            s.resolution_mm = self.sim_res.value()
            s.include_radiation = self.sim_radiation.isChecked()
            s.use_adaptive_mesh = self.sim_adaptive.isChecked()
            s.min_cell_mm = self.sim_min_cell.value()
            s.max_cell_mm = self.sim_max_cell.value()
            s.ambient_temp_c = self.sim_ambient.value()
            s.chamber_wall_temp_c = self.sim_chamber.value()
            s.initial_temp_c = self.sim_initial.value()
            s.duration_s = self.sim_duration.value()
            s.timestep_s = self.sim_dt.value()

        # ── Run simulation ─────────────────────────────────────────
        def _on_run(self):
            if not self.pcb:
                QMessageBox.warning(self, "No PCB", "Load a .kicad_pcb file first.")
                return
            self._sync_sim_cfg()
            mode = self.cfg.sim.heat_source_mode
            if mode == "component_power" and self.cfg.total_power() <= 0:
                QMessageBox.warning(
                    self, "No Heat Sources",
                    "No power dissipation defined.\n"
                    "Set power for components, or switch to Current Injection mode.",
                )
                return
            if mode == "current_injection" and not self.cfg.current_paths:
                QMessageBox.warning(
                    self, "No Current Paths",
                    "No current paths defined.\n"
                    "Define current paths, or switch to Component Power mode.",
                )
                return
            self.btn_run.setEnabled(False)
            self.btn_run.setText("Running...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.status_label.setText("● Running simulation...")
            self.status_label.setStyleSheet("color:#ffa726;")
            QApplication.processEvents()
            self.worker = SolverWorker(self.pcb, self.cfg)
            self.worker.progress.connect(self._on_progress)
            self.worker.finished.connect(self._on_sim_done)
            self.worker.error.connect(self._on_sim_error)
            self.worker.start()

        def _on_progress(self, pct, msg):
            self.progress_bar.setValue(pct)
            self.status_label.setText(f"● {msg} ({pct}%)")
            QApplication.processEvents()

        def _on_sim_done(self, result, mesh):
            self.result = result
            self.mesh = mesh
            self.btn_run.setEnabled(True)
            self.btn_run.setText("▶  Run Simulation")
            self.progress_bar.setVisible(False)
            self.pcb_view.set_thermal(result, mesh)
            self._layer_cbs["show_thermal"].setChecked(True)

            # Build result summary
            ebal = ""
            if hasattr(result, 'energy_balance') and result.energy_balance:
                eb = result.energy_balance
                ebal = (
                    f"\n\nEnergy Balance:\n"
                    f"  Input:      {eb.get('energy_in', 0):.4f} W\n"
                    f"  Radiation:  {eb.get('energy_rad', 0):.4f} W\n"
                    f"  Conduction: {eb.get('energy_cond', 0):.4f} W\n"
                    f"  Imbalance:  {eb.get('imbalance_frac', 0)*100:.2f}%"
                )

            self.status_label.setText(
                f"● Done: {result.min_t:.1f}°C – {result.max_t:.1f}°C  "
                f"(avg {result.avg_t:.1f}°C, {result.time_s:.2f}s)"
            )
            self.status_label.setStyleSheet("color:#66bb6a;")
            QMessageBox.information(
                self, "Simulation Complete",
                f"Solver: {ThermalSolver().backend}\n"
                f"Mesh: {len(mesh.nodes)} nodes "
                f"({mesh.nx}×{mesh.ny}×{mesh.nz})\n\n"
                f"Min: {result.min_t:.2f} °C\n"
                f"Max: {result.max_t:.2f} °C\n"
                f"Avg: {result.avg_t:.2f} °C\n"
                f"Time: {result.time_s:.2f} s\n"
                f"Converged: {'Yes' if result.converged else 'No'}"
                f"{ebal}",
            )

        def _on_sim_error(self, msg):
            self.btn_run.setEnabled(True)
            self.btn_run.setText("▶  Run Simulation")
            self.progress_bar.setVisible(False)
            self.status_label.setText("● Simulation failed")
            self.status_label.setStyleSheet("color:#ef5350;")
            QMessageBox.critical(self, "Error", msg)


# ═══════════════════════════════════════════════════════════════════════
# §11  Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    if not HAS_QT:
        print("Error: PyQt5 is required.\n  pip install PyQt5")
        sys.exit(1)
    app = QApplication(sys.argv)
    app.setApplicationName("TVAC Thermal Analyzer")
    pcb_path = sys.argv[1] if len(sys.argv) > 1 else ""
    win = MainWindow(pcb_path)
    win.show()
    if not pcb_path:
        QTimer.singleShot(200, win._on_open)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()