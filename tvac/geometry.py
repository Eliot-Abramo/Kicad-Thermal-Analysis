from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Pt:
    x: float = 0.0
    y: float = 0.0


@dataclass
class Pad:
    name: str = ""
    pad_type: str = "smd"
    shape: str = "rect"
    ref: str = ""
    pos: Pt = field(default_factory=Pt)
    w: float = 0.0
    h: float = 0.0
    drill: float = 0.0
    rotation: float = 0.0  # degrees, absolute on board
    layers: List[str] = field(default_factory=list)
    net_code: int = 0
    net_name: str = ""


@dataclass
class Component:
    ref: str = ""
    value: str = ""
    footprint: str = ""
    pos: Pt = field(default_factory=Pt)
    rotation: float = 0.0
    layer: str = "F.Cu"
    power_w: float = 0.0
    pads: List[Pad] = field(default_factory=list)

    def bbox(self) -> Tuple[Pt, Pt]:
        # Simple bbox from pads (more accurate than constant size)
        if not self.pads:
            m = 1.5
            return (Pt(self.pos.x - m, self.pos.y - m), Pt(self.pos.x + m, self.pos.y + m))
        xs = [p.pos.x for p in self.pads]
        ys = [p.pos.y for p in self.pads]
        m = 0.5
        return (Pt(min(xs) - m, min(ys) - m), Pt(max(xs) + m, max(ys) + m))


@dataclass
class Trace:
    start: Pt = field(default_factory=Pt)
    end: Pt = field(default_factory=Pt)
    width_mm: float = 0.25
    layer: str = ""
    net_code: int = 0
    net_name: str = ""


@dataclass
class ArcTrace:
    start: Pt = field(default_factory=Pt)
    mid: Pt = field(default_factory=Pt)
    end: Pt = field(default_factory=Pt)
    width_mm: float = 0.25
    layer: str = ""
    net_code: int = 0
    net_name: str = ""


@dataclass
class Via:
    pos: Pt = field(default_factory=Pt)
    diam_mm: float = 0.6
    drill_mm: float = 0.3
    via_type: str = "through"
    layers: Tuple[str, str] = ("F.Cu", "B.Cu")
    net_code: int = 0
    net_name: str = ""


@dataclass
class Zone:
    layer: str = ""
    net_code: int = 0
    net_name: str = ""
    polygon: List[Pt] = field(default_factory=list)


@dataclass
class Outline:
    outline: List[Pt] = field(default_factory=list)
    min_x: float = 0.0
    max_x: float = 0.0
    min_y: float = 0.0
    max_y: float = 0.0


@dataclass
class PCBData:
    nets: dict = field(default_factory=dict)
    copper_layers: List[str] = field(default_factory=list)
    outline: Outline = field(default_factory=Outline)
    components: List[Component] = field(default_factory=list)
    traces: List[Trace] = field(default_factory=list)
    arc_traces: List[ArcTrace] = field(default_factory=list)
    vias: List[Via] = field(default_factory=list)
    zones: List[Zone] = field(default_factory=list)
    mounting_holes: List[Component] = field(default_factory=list)

    # thickness info (optional)
    copper_thickness_um: dict = field(default_factory=dict)
    dielectric_thickness_um: dict = field(default_factory=dict)
