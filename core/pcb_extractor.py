"""
TVAC Thermal Analyzer - PCB Extractor
=====================================
Extract PCB geometry from KiCad board - ONLY actual PCB components.

No schematic inference, no hallucinated components - only what's
physically present on the PCB.

Author: Space Electronics Thermal Analysis Tool
Version: 2.0.0
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

try:
    import pcbnew
    HAS_PCBNEW = True
except ImportError:
    HAS_PCBNEW = False


@dataclass
class Point2D:
    """2D point in mm."""
    x: float = 0.0
    y: float = 0.0


@dataclass
class Pad:
    """Component pad."""
    position: Point2D = field(default_factory=Point2D)
    width: float = 0.0
    height: float = 0.0
    shape: str = "rect"
    layers: List[str] = field(default_factory=list)
    drill: float = 0.0


@dataclass
class Component:
    """PCB component from footprint."""
    reference: str = ""
    value: str = ""
    footprint: str = ""
    position: Point2D = field(default_factory=Point2D)
    rotation: float = 0.0
    layer: str = "F.Cu"
    pads: List[Pad] = field(default_factory=list)
    power_dissipation_w: float = 0.0
    
    def get_bounding_box(self) -> Tuple[Point2D, Point2D]:
        """Get bounding box (min, max corners)."""
        if not self.pads:
            # Estimate from footprint name
            size = 2.0  # Default 2mm
            return (
                Point2D(self.position.x - size/2, self.position.y - size/2),
                Point2D(self.position.x + size/2, self.position.y + size/2)
            )
        
        min_x = min(p.position.x - p.width/2 for p in self.pads)
        max_x = max(p.position.x + p.width/2 for p in self.pads)
        min_y = min(p.position.y - p.height/2 for p in self.pads)
        max_y = max(p.position.y + p.height/2 for p in self.pads)
        
        return (Point2D(min_x, min_y), Point2D(max_x, max_y))


@dataclass
class TraceSegment:
    """PCB trace segment."""
    start: Point2D = field(default_factory=Point2D)
    end: Point2D = field(default_factory=Point2D)
    width: float = 0.0
    layer: str = ""
    net: str = ""


@dataclass
class Via:
    """Through or blind via."""
    position: Point2D = field(default_factory=Point2D)
    drill: float = 0.0
    diameter: float = 0.0
    net: str = ""
    via_type: str = "through"


@dataclass
class CopperPour:
    """Copper zone/pour."""
    outline: List[Point2D] = field(default_factory=list)
    layer: str = ""
    net: str = ""
    priority: int = 0


@dataclass
class MountingHole:
    """Board mounting hole."""
    hole_id: str = ""
    position: Point2D = field(default_factory=Point2D)
    drill_mm: float = 0.0
    is_plated: bool = False


@dataclass
class BoardOutline:
    """PCB board outline."""
    outline: List[Point2D] = field(default_factory=list)
    min_x: float = 0.0
    max_x: float = 100.0
    min_y: float = 0.0
    max_y: float = 100.0
    
    def get_area_mm2(self) -> float:
        """Calculate board area using shoelace formula."""
        if len(self.outline) < 3:
            return (self.max_x - self.min_x) * (self.max_y - self.min_y)
        
        area = 0.0
        n = len(self.outline)
        for i in range(n):
            j = (i + 1) % n
            area += self.outline[i].x * self.outline[j].y
            area -= self.outline[j].x * self.outline[i].y
        return abs(area) / 2.0


@dataclass
class UserLayerShape:
    """Shape on a User layer (for heatsinks/mounting)."""
    shape_id: str = ""
    layer: str = ""
    layer_id: int = 0
    shape_type: str = "polygon"
    points: List[Point2D] = field(default_factory=list)
    radius: float = 0.0
    
    def get_area_mm2(self) -> float:
        if self.shape_type == "circle":
            return 3.14159 * self.radius * self.radius
        
        if len(self.points) < 3:
            return 0.0
        
        area = 0.0
        n = len(self.points)
        for i in range(n):
            j = (i + 1) % n
            area += self.points[i].x * self.points[j].y
            area -= self.points[j].x * self.points[i].y
        return abs(area) / 2.0
    
    def get_centroid(self) -> Point2D:
        if not self.points:
            return Point2D()
        cx = sum(p.x for p in self.points) / len(self.points)
        cy = sum(p.y for p in self.points) / len(self.points)
        return Point2D(cx, cy)


@dataclass
class PCBData:
    """Complete extracted PCB data."""
    board_outline: BoardOutline = field(default_factory=BoardOutline)
    components: List[Component] = field(default_factory=list)
    traces: List[TraceSegment] = field(default_factory=list)
    vias: List[Via] = field(default_factory=list)
    copper_pours: List[CopperPour] = field(default_factory=list)
    mounting_holes: List[MountingHole] = field(default_factory=list)
    nets: Dict[int, str] = field(default_factory=dict)
    user_shapes: Dict[str, List[UserLayerShape]] = field(default_factory=dict)


class PCBExtractor:
    """Extract PCB data from KiCad board."""
    
    # KiCad units: 1nm internally, we use mm
    SCALE = 1e-6  # nm to mm
    
    def __init__(self, board):
        self.board = board
    
    def extract_all(self) -> PCBData:
        """Extract all PCB data."""
        data = PCBData()
        
        if not self.board:
            return data
        
        data.board_outline = self._extract_outline()
        data.nets = self._extract_nets()
        data.components = self._extract_components()
        data.traces = self._extract_traces()
        data.vias = self._extract_vias()
        data.copper_pours = self._extract_zones()
        data.mounting_holes = self._extract_mounting_holes()
        data.user_shapes = self._extract_user_shapes()
        
        return data
    
    def _extract_outline(self) -> BoardOutline:
        """Extract board outline from Edge.Cuts layer."""
        outline = BoardOutline()
        points = []
        
        if not HAS_PCBNEW:
            return outline
        
        for drawing in self.board.GetDrawings():
            layer = drawing.GetLayerName()
            if layer != "Edge.Cuts":
                continue
            
            if hasattr(drawing, 'GetStart') and hasattr(drawing, 'GetEnd'):
                start = drawing.GetStart()
                end = drawing.GetEnd()
                points.append(Point2D(start.x * self.SCALE, start.y * self.SCALE))
                points.append(Point2D(end.x * self.SCALE, end.y * self.SCALE))
        
        if points:
            outline.outline = points
            outline.min_x = min(p.x for p in points)
            outline.max_x = max(p.x for p in points)
            outline.min_y = min(p.y for p in points)
            outline.max_y = max(p.y for p in points)
        else:
            # Fallback: estimate from board bounding box
            bbox = self.board.GetBoundingBox()
            outline.min_x = bbox.GetLeft() * self.SCALE
            outline.max_x = bbox.GetRight() * self.SCALE
            outline.min_y = bbox.GetTop() * self.SCALE
            outline.max_y = bbox.GetBottom() * self.SCALE
        
        return outline
    
    def _extract_nets(self) -> Dict[int, str]:
        """Extract net code to name mapping."""
        nets = {}
        
        if not HAS_PCBNEW:
            return nets
        
        netinfo = self.board.GetNetInfo()
        for net in netinfo.NetsByNetcode():
            nets[net] = netinfo.GetNetItem(net).GetNetname()
        
        return nets
    
    def _extract_components(self) -> List[Component]:
        """Extract ONLY actual PCB footprints - no schematic inference."""
        components = []
        
        if not HAS_PCBNEW:
            return components
        
        for fp in self.board.GetFootprints():
            ref = fp.GetReference()
            
            # Skip invalid references
            if not ref or ref == "*" or ref.startswith("*"):
                continue
            
            pos = fp.GetPosition()
            
            comp = Component(
                reference=ref,
                value=fp.GetValue(),
                footprint=str(fp.GetFPID().GetUniStringLibItemName()),
                position=Point2D(pos.x * self.SCALE, pos.y * self.SCALE),
                rotation=fp.GetOrientationDegrees(),
                layer="F.Cu" if fp.GetLayer() == 0 else "B.Cu"
            )
            
            # Extract POWER_DISSIPATION from footprint fields
            comp.power_dissipation_w = self._extract_power_field(fp)
            
            # Extract pads
            for pad in fp.Pads():
                pad_pos = pad.GetPosition()
                pad_size = pad.GetSize()
                
                pad_obj = Pad(
                    position=Point2D(pad_pos.x * self.SCALE, pad_pos.y * self.SCALE),
                    width=pad_size.x * self.SCALE,
                    height=pad_size.y * self.SCALE,
                    shape=self._pad_shape_name(pad.GetShape()),
                    drill=pad.GetDrillSize().x * self.SCALE if pad.GetDrillSize().x > 0 else 0
                )
                comp.pads.append(pad_obj)
            
            components.append(comp)
        
        return components
    
    def _extract_power_field(self, fp) -> float:
        """Extract power dissipation from footprint fields."""
        power = 0.0
        
        # Field names to check (case-insensitive)
        power_field_names = [
            'POWER_DISSIPATION', 'Power_Dissipation', 'power_dissipation',
            'POWER', 'Power', 'power',
            'PWR', 'Pwr', 'pwr',
            'THERMAL_POWER', 'Thermal_Power', 'thermal_power',
            'P_DISS', 'P_diss', 'p_diss',
            'PDISS', 'Pdiss', 'pdiss',
        ]
        
        try:
            # Try GetFieldByName (KiCad 7+)
            if hasattr(fp, 'GetFieldByName'):
                for name in power_field_names:
                    try:
                        field = fp.GetFieldByName(name)
                        if field:
                            power = self._parse_power_value(field.GetText())
                            if power > 0:
                                return power
                    except Exception:
                        pass
            
            # Try GetProperties (older API)
            if hasattr(fp, 'GetProperties'):
                props = fp.GetProperties()
                for name in power_field_names:
                    if name in props:
                        power = self._parse_power_value(props[name])
                        if power > 0:
                            return power
            
            # Try iterating fields directly
            if hasattr(fp, 'GetFields'):
                for field in fp.GetFields():
                    field_name = field.GetName() if hasattr(field, 'GetName') else ""
                    if field_name.upper() in [n.upper() for n in power_field_names]:
                        power = self._parse_power_value(field.GetText())
                        if power > 0:
                            return power
            
            # Try GetField with index (KiCad 6 style)
            if hasattr(fp, 'GetField'):
                for i in range(10):  # Check first 10 fields
                    try:
                        field = fp.GetField(i)
                        if field:
                            field_name = field.GetName() if hasattr(field, 'GetName') else ""
                            if field_name.upper() in [n.upper() for n in power_field_names]:
                                power = self._parse_power_value(field.GetText())
                                if power > 0:
                                    return power
                    except Exception:
                        break
                        
        except Exception:
            pass
        
        return power
    
    def _parse_power_value(self, text: str) -> float:
        """Parse power value from text (handles W, mW, uW suffixes)."""
        if not text:
            return 0.0
        
        text = text.strip().upper()
        
        try:
            # Handle different suffixes
            if text.endswith('MW'):
                return float(text[:-2]) * 1e-3
            elif text.endswith('UW') or text.endswith('µW'):
                return float(text[:-2]) * 1e-6
            elif text.endswith('W'):
                return float(text[:-1])
            else:
                return float(text)
        except ValueError:
            return 0.0
    
    def _pad_shape_name(self, shape_id: int) -> str:
        """Convert pad shape ID to name."""
        shapes = {0: "circle", 1: "rect", 2: "oval", 3: "trapezoid", 4: "roundrect", 5: "custom"}
        return shapes.get(shape_id, "rect")
    
    def _extract_traces(self) -> List[TraceSegment]:
        """Extract all trace segments."""
        traces = []
        
        if not HAS_PCBNEW:
            return traces
        
        for track in self.board.GetTracks():
            if track.GetClass() == "PCB_TRACK":
                start = track.GetStart()
                end = track.GetEnd()
                
                trace = TraceSegment(
                    start=Point2D(start.x * self.SCALE, start.y * self.SCALE),
                    end=Point2D(end.x * self.SCALE, end.y * self.SCALE),
                    width=track.GetWidth() * self.SCALE,
                    layer=track.GetLayerName(),
                    net=track.GetNetname()
                )
                traces.append(trace)
        
        return traces
    
    def _extract_vias(self) -> List[Via]:
        """Extract all vias."""
        vias = []
        
        if not HAS_PCBNEW:
            return vias
        
        for track in self.board.GetTracks():
            if track.GetClass() == "PCB_VIA":
                pos = track.GetPosition()
                
                via = Via(
                    position=Point2D(pos.x * self.SCALE, pos.y * self.SCALE),
                    drill=track.GetDrillValue() * self.SCALE,
                    diameter=track.GetWidth() * self.SCALE,
                    net=track.GetNetname()
                )
                vias.append(via)
        
        return vias
    
    def _extract_zones(self) -> List[CopperPour]:
        """Extract copper zones/pours."""
        pours = []
        
        if not HAS_PCBNEW:
            return pours
        
        for zone in self.board.Zones():
            outline_points = []
            
            try:
                outline = zone.Outline()
                if outline:
                    for i in range(outline.FullPointCount()):
                        pt = outline.GetPoint(i)
                        outline_points.append(Point2D(pt.x * self.SCALE, pt.y * self.SCALE))
            except Exception:
                pass
            
            pour = CopperPour(
                outline=outline_points,
                layer=zone.GetLayerName(),
                net=zone.GetNetname(),
                priority=zone.GetAssignedPriority()
            )
            pours.append(pour)
        
        return pours
    
    def _extract_mounting_holes(self) -> List[MountingHole]:
        """Detect mounting holes from footprints."""
        holes = []
        hole_count = 0
        
        if not HAS_PCBNEW:
            return holes
        
        for fp in self.board.GetFootprints():
            fp_name = str(fp.GetFPID().GetUniStringLibItemName()).lower()
            
            # Detect mounting hole footprints
            if "mountinghole" in fp_name or "mounting_hole" in fp_name:
                pos = fp.GetPosition()
                
                # Get drill size from pads
                drill = 3.2  # Default
                for pad in fp.Pads():
                    if pad.GetDrillSize().x > 0:
                        drill = pad.GetDrillSize().x * self.SCALE
                        break
                
                hole_count += 1
                hole = MountingHole(
                    hole_id=f"MH{hole_count}",
                    position=Point2D(pos.x * self.SCALE, pos.y * self.SCALE),
                    drill_mm=drill,
                    is_plated=any(pad.GetAttribute() != 1 for pad in fp.Pads())  # NPTH = 1
                )
                holes.append(hole)
        
        return holes
    
    def _extract_user_shapes(self) -> Dict[str, List[UserLayerShape]]:
        """Extract shapes from User layers (User.1 through User.9)."""
        shapes = {}
        
        if not HAS_PCBNEW:
            return shapes
        
        shape_count = 0
        
        # Get User layer IDs
        user_layer_ids = {}
        try:
            for i in range(1, 10):
                layer_name = f"User.{i}"
                layer_id = self.board.GetLayerID(layer_name)
                if layer_id >= 0:
                    user_layer_ids[layer_id] = layer_name
        except Exception:
            pass
        
        # Extract from board drawings
        for drawing in self.board.GetDrawings():
            try:
                layer_name = drawing.GetLayerName()
                layer_id = drawing.GetLayer()
                
                # Check if User layer by name or ID
                is_user_layer = layer_name.startswith("User.")
                if not is_user_layer and layer_id in user_layer_ids:
                    layer_name = user_layer_ids[layer_id]
                    is_user_layer = True
                
                if not is_user_layer:
                    continue
                
                if layer_name not in shapes:
                    shapes[layer_name] = []
                
                shape_count += 1
                
                # Handle different shape types
                class_name = drawing.GetClass()
                
                if class_name == "PCB_SHAPE":
                    shape = self._extract_pcb_shape(drawing, layer_name, shape_count)
                    if shape and shape.points:
                        shapes[layer_name].append(shape)
                        
            except Exception as e:
                # Log but continue
                pass
        
        # Also check zones on User layers (some users put heatsink shapes as zones)
        try:
            for zone in self.board.Zones():
                layer_name = zone.GetLayerName()
                
                if not layer_name.startswith("User."):
                    continue
                
                if layer_name not in shapes:
                    shapes[layer_name] = []
                
                shape_count += 1
                points = self._extract_zone_outline(zone)
                
                if points:
                    shape = UserLayerShape(
                        shape_id=f"Zone_{shape_count}",
                        layer=layer_name,
                        shape_type="polygon",
                        points=points
                    )
                    shapes[layer_name].append(shape)
        except Exception:
            pass
        
        return shapes
    
    def _extract_pcb_shape(self, drawing, layer_name: str, shape_count: int) -> Optional[UserLayerShape]:
        """Extract a PCB_SHAPE object."""
        points = []
        radius = 0.0
        shape_type = "polygon"
        
        try:
            # Get shape type - handle both old and new API
            if hasattr(drawing, 'GetShape'):
                shape_enum = drawing.GetShape()
            else:
                return None
            
            # Shape types: 0=Segment, 1=Rect, 2=Arc, 3=Circle, 4=Polygon, 5=Bezier
            # KiCad 7+: S_SEGMENT=0, S_RECT=1, S_ARC=2, S_CIRCLE=3, S_POLYGON=4
            
            if shape_enum == 4:  # Polygon
                try:
                    poly = drawing.GetPolyShape()
                    if poly and poly.OutlineCount() > 0:
                        outline = poly.Outline(0)
                        for i in range(outline.PointCount()):
                            pt = outline.GetPoint(i)
                            points.append(Point2D(pt.x * self.SCALE, pt.y * self.SCALE))
                except Exception:
                    pass
            
            elif shape_enum == 1:  # Rectangle
                try:
                    start = drawing.GetStart()
                    end = drawing.GetEnd()
                    points = [
                        Point2D(start.x * self.SCALE, start.y * self.SCALE),
                        Point2D(end.x * self.SCALE, start.y * self.SCALE),
                        Point2D(end.x * self.SCALE, end.y * self.SCALE),
                        Point2D(start.x * self.SCALE, end.y * self.SCALE),
                    ]
                except Exception:
                    pass
            
            elif shape_enum == 3:  # Circle
                try:
                    center = drawing.GetCenter() if hasattr(drawing, 'GetCenter') else drawing.GetStart()
                    radius = drawing.GetRadius() * self.SCALE if hasattr(drawing, 'GetRadius') else 5.0
                    points = [Point2D(center.x * self.SCALE, center.y * self.SCALE)]
                    shape_type = "circle"
                except Exception:
                    pass
            
            elif shape_enum == 0:  # Segment/Line - create thin rectangle
                try:
                    start = drawing.GetStart()
                    end = drawing.GetEnd()
                    width = drawing.GetWidth() * self.SCALE if hasattr(drawing, 'GetWidth') else 0.5
                    
                    # Create rectangle around line
                    import math
                    dx = end.x - start.x
                    dy = end.y - start.y
                    length = math.sqrt(dx*dx + dy*dy)
                    if length > 0:
                        nx = -dy / length * width / 2 * self.SCALE
                        ny = dx / length * width / 2 * self.SCALE
                        
                        sx, sy = start.x * self.SCALE, start.y * self.SCALE
                        ex, ey = end.x * self.SCALE, end.y * self.SCALE
                        
                        points = [
                            Point2D(sx + nx, sy + ny),
                            Point2D(ex + nx, ey + ny),
                            Point2D(ex - nx, ey - ny),
                            Point2D(sx - nx, sy - ny),
                        ]
                except Exception:
                    pass
            
        except Exception:
            pass
        
        if points:
            return UserLayerShape(
                shape_id=f"Shape_{shape_count}",
                layer=layer_name,
                shape_type=shape_type,
                points=points,
                radius=radius
            )
        return None
    
    def _extract_zone_outline(self, zone) -> List[Point2D]:
        """Extract outline from a zone."""
        points = []
        try:
            outline = zone.Outline()
            if outline:
                for i in range(outline.FullPointCount()):
                    pt = outline.GetPoint(i)
                    points.append(Point2D(pt.x * self.SCALE, pt.y * self.SCALE))
        except Exception:
            pass
        return points


__all__ = [
    'PCBExtractor', 'PCBData', 'Component', 'Point2D',
    'TraceSegment', 'Via', 'CopperPour', 'Pad',
    'MountingHole', 'BoardOutline', 'UserLayerShape'
]
