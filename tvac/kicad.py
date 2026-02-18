from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .sexpr import parse, find, find_all, is_list
from .geometry import PCBData, Outline, Pt, Component, Pad, Trace, ArcTrace, Via, Zone


def _as_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _val(node, tag: str, default=None):
    n = find(node, tag)
    if n and len(n) >= 2:
        return n[1]
    return default


def _float(node, tag: str, default=0.0) -> float:
    v = _val(node, tag, None)
    return _as_float(v, default)


def _layers_list(root) -> List[str]:
    layers_node = find(root, "layers")
    if not layers_node:
        return ["F.Cu", "B.Cu"]
    cu = []
    for c in layers_node[1:]:
        if is_list(c) and len(c) >= 3:
            # (0 "F.Cu" signal) OR (31 "B.Cu" signal)
            name = c[1]
            ltype = c[2]
            if isinstance(ltype, str) and ltype in ("signal", "power", "mixed"):
                cu.append(str(name))
    # If KiCad provides internal in order, keep it; otherwise ensure top/bottom exist
    if "F.Cu" not in cu:
        cu.insert(0, "F.Cu")
    if "B.Cu" not in cu:
        cu.append("B.Cu")
    return cu


def _extract_outline(root) -> Outline:
    out = Outline()
    pts: List[Pt] = []
    for gr in find_all(root, "gr_line"):
        if _val(gr, "layer", "") != "Edge.Cuts":
            continue
        s = find(gr, "start")
        e = find(gr, "end")
        if s and e and len(s) >= 3 and len(e) >= 3:
            pts.append(Pt(_as_float(s[1]), _as_float(s[2])))
            pts.append(Pt(_as_float(e[1]), _as_float(e[2])))
    # KiCad outlines can also be in gr_arc / gr_rect etc; keep bounding box fallback
    if pts:
        out.outline = pts
        out.min_x = min(p.x for p in pts)
        out.max_x = max(p.x for p in pts)
        out.min_y = min(p.y for p in pts)
        out.max_y = max(p.y for p in pts)
    return out


def _read_nets(root) -> Dict[int, str]:
    nets: Dict[int, str] = {}
    for n in find_all(root, "net"):
        if len(n) >= 3:
            try:
                nets[int(n[1])] = str(n[2])
            except Exception:
                continue
    return nets


def _tf_footprint_point(local_x: float, local_y: float, fp_rot_deg: float, is_back: bool) -> Tuple[float, float]:
    """
    KiCad footprint transform (local pad coords -> footprint coords), with flip handling.

    Empirically consistent with KiCad's "flip to back":
    - mirror local X
    - reverse rotation sign
    """
    if is_back:
        local_x = -local_x
        fp_rot_deg = -fp_rot_deg
    th = math.radians(fp_rot_deg)
    x = local_x * math.cos(th) - local_y * math.sin(th)
    y = local_x * math.sin(th) + local_y * math.cos(th)
    return x, y


def _read_footprints(root, nets: Dict[int, str]) -> List[Component]:
    comps: List[Component] = []
    for fp in find_all(root, "footprint"):
        c = Component()
        if len(fp) > 1 and isinstance(fp[1], str):
            c.footprint = fp[1]
        # layer first: affects transforms
        c.layer = str(_val(fp, "layer", "F.Cu"))
        is_back = c.layer.startswith("B.")
        # position + rotation
        at = find(fp, "at")
        if at and len(at) >= 3:
            c.pos = Pt(_as_float(at[1]), _as_float(at[2]))
            if len(at) >= 4:
                c.rotation = _as_float(at[3], 0.0)

        # KiCad 7-9: property nodes
        for prop in find_all(fp, "property"):
            if len(prop) >= 3:
                key = str(prop[1])
                val = str(prop[2])
                if key == "Reference":
                    c.ref = val
                elif key == "Value":
                    c.value = val
                elif key.upper() in ("POWER_DISSIPATION", "POWER", "PDISS", "P_DISS"):
                    try:
                        c.power_w = float(val)
                    except Exception:
                        pass

        # fallback older fp_text
        if not c.ref:
            for ft in find_all(fp, "fp_text"):
                if len(ft) >= 3:
                    if ft[1] == "reference":
                        c.ref = str(ft[2])
                    elif ft[1] == "value" and not c.value:
                        c.value = str(ft[2])

        # pads
        for pad_node in find_all(fp, "pad"):
            pad = Pad()
            if len(pad_node) >= 2:
                pad.name = str(pad_node[1])
            if len(pad_node) >= 3:
                pad.pad_type = str(pad_node[2])
            if len(pad_node) >= 4:
                pad.shape = str(pad_node[3])
            pad.ref = c.ref

            at_p = find(pad_node, "at")
            lx = ly = 0.0
            pad_local_rot = 0.0
            if at_p and len(at_p) >= 3:
                lx, ly = _as_float(at_p[1]), _as_float(at_p[2])
                if len(at_p) >= 4:
                    pad_local_rot = _as_float(at_p[3], 0.0)

            dx, dy = _tf_footprint_point(lx, ly, c.rotation, is_back)
            pad.pos = Pt(c.pos.x + dx, c.pos.y + dy)

            # absolute rotation on board (used later if we rasterize rotated pads)
            # Flip affects sign of local pad rotation as well.
            pad.rotation = (c.rotation + (-pad_local_rot if is_back else pad_local_rot))

            sz = find(pad_node, "size")
            if sz and len(sz) >= 3:
                pad.w, pad.h = _as_float(sz[1]), _as_float(sz[2])
            dr = find(pad_node, "drill")
            if dr and len(dr) >= 2:
                pad.drill = _as_float(dr[1])

            net_node = find(pad_node, "net")
            if net_node and len(net_node) >= 2:
                try:
                    pad.net_code = int(net_node[1])
                    pad.net_name = nets.get(pad.net_code, "")
                except Exception:
                    pass

            layers_node = find(pad_node, "layers")
            if layers_node and len(layers_node) >= 2:
                pad.layers = [str(l) for l in layers_node[1:] if isinstance(l, str)]

            c.pads.append(pad)

        if c.ref:
            comps.append(c)
    return comps


def _read_segments(root, nets: Dict[int, str]) -> List[Trace]:
    traces: List[Trace] = []
    for seg in find_all(root, "segment"):
        s = find(seg, "start")
        e = find(seg, "end")
        if not (s and e and len(s) >= 3 and len(e) >= 3):
            continue
        t = Trace(
            start=Pt(_as_float(s[1]), _as_float(s[2])),
            end=Pt(_as_float(e[1]), _as_float(e[2])),
            width_mm=_float(seg, "width", 0.25),
            layer=str(_val(seg, "layer", "")),
        )
        net_node = find(seg, "net")
        if net_node and len(net_node) >= 2:
            try:
                t.net_code = int(net_node[1])
                t.net_name = nets.get(t.net_code, "")
            except Exception:
                pass
        traces.append(t)
    return traces


def _read_arcs(root, nets: Dict[int, str]) -> List[ArcTrace]:
    arcs: List[ArcTrace] = []
    for arc in find_all(root, "arc"):
        # trace arcs have net
        net_node = find(arc, "net")
        if not net_node:
            continue
        s = find(arc, "start")
        m = find(arc, "mid")
        e = find(arc, "end")
        if not (s and m and e and len(s) >= 3 and len(m) >= 3 and len(e) >= 3):
            continue
        a = ArcTrace(
            start=Pt(_as_float(s[1]), _as_float(s[2])),
            mid=Pt(_as_float(m[1]), _as_float(m[2])),
            end=Pt(_as_float(e[1]), _as_float(e[2])),
            width_mm=_float(arc, "width", 0.25),
            layer=str(_val(arc, "layer", "")),
        )
        try:
            a.net_code = int(net_node[1])
            a.net_name = nets.get(a.net_code, "")
        except Exception:
            pass
        arcs.append(a)
    return arcs


def _read_vias(root, nets: Dict[int, str]) -> List[Via]:
    vias: List[Via] = []
    for v in find_all(root, "via"):
        at = find(v, "at")
        if not (at and len(at) >= 3):
            continue
        via = Via(pos=Pt(_as_float(at[1]), _as_float(at[2])))
        via.diam_mm = _float(v, "size", 0.6)
        via.drill_mm = _float(v, "drill", 0.3)
        via.via_type = str(_val(v, "type", "through"))
        net_node = find(v, "net")
        if net_node and len(net_node) >= 2:
            try:
                via.net_code = int(net_node[1])
                via.net_name = nets.get(via.net_code, "")
            except Exception:
                pass
        layers_node = find(v, "layers")
        if layers_node and len(layers_node) >= 3:
            via.layers = (str(layers_node[1]), str(layers_node[2]))
        vias.append(via)
    return vias


def _read_zones(root, nets: Dict[int, str]) -> List[Zone]:
    zones: List[Zone] = []
    for z in find_all(root, "zone"):
        zone = Zone()
        zone.layer = str(_val(z, "layer", ""))  # single layer zones common
        if not zone.layer:
            ln = find(z, "layers")
            if ln and len(ln) >= 2:
                zone.layer = str(ln[1])
        net_node = find(z, "net")
        if net_node and len(net_node) >= 2:
            try:
                zone.net_code = int(net_node[1])
            except Exception:
                pass
        nn = find(z, "net_name")
        zone.net_name = str(nn[1]) if nn and len(nn) >= 2 else nets.get(zone.net_code, "")

        poly = find(z, "polygon")
        if poly:
            pts = []
            pts_node = find(poly, "pts")
            if pts_node:
                for xy in pts_node[1:]:
                    if is_list(xy) and xy and xy[0] == "xy" and len(xy) >= 3:
                        pts.append(Pt(_as_float(xy[1]), _as_float(xy[2])))
            zone.polygon = pts
        if zone.layer and zone.polygon:
            zones.append(zone)
    return zones


def read_pcb(path: str) -> PCBData:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        txt = f.read()
    tree = parse(txt)
    if not (isinstance(tree, list) and tree and tree[0] == "kicad_pcb"):
        raise ValueError("Not a valid .kicad_pcb file")
    pcb = PCBData()
    pcb.nets = _read_nets(tree)
    pcb.copper_layers = _layers_list(tree)
    pcb.outline = _extract_outline(tree)
    pcb.components = _read_footprints(tree, pcb.nets)
    pcb.traces = _read_segments(tree, pcb.nets)
    pcb.arc_traces = _read_arcs(tree, pcb.nets)
    pcb.vias = _read_vias(tree, pcb.nets)
    pcb.zones = _read_zones(tree, pcb.nets)

    # bounds fallback if outline absent
    if pcb.outline.outline:
        pass
    elif pcb.components:
        xs = [c.pos.x for c in pcb.components]
        ys = [c.pos.y for c in pcb.components]
        m = 15.0
        pcb.outline.min_x = min(xs) - m
        pcb.outline.max_x = max(xs) + m
        pcb.outline.min_y = min(ys) - m
        pcb.outline.max_y = max(ys) + m
    return pcb
