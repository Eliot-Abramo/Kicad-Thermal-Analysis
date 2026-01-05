"""
TVAC Thermal Analyzer - PCB Geometry Extractor
==============================================
Extracts geometry data from KiCad PCB for thermal analysis.

Author: Space Electronics Thermal Analysis Tool
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum
import math
import re


@dataclass
class Point2D:
    """2D point representation."""
    x: float  # mm
    y: float  # mm
    
    def distance_to(self, other: 'Point2D') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __hash__(self):
        return hash((round(self.x, 6), round(self.y, 6)))
    
    def __eq__(self, other):
        if not isinstance(other, Point2D):
            return False
        return abs(self.x - other.x) < 1e-6 and abs(self.y - other.y) < 1e-6


@dataclass
class Point3D:
    """3D point representation."""
    x: float  # mm
    y: float  # mm
    z: float  # mm
    
    def distance_to(self, other: 'Point3D') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)


@dataclass
class TraceSegment:
    """A single trace segment on the PCB."""
    segment_id: str
    net_name: str
    net_code: int
    layer: str
    start: Point2D
    end: Point2D
    width_mm: float
    
    # Computed properties
    length_mm: float = field(init=False)
    resistance_ohm: float = 0.0  # Computed based on material and dimensions
    current_a: float = 0.0  # Computed from network analysis
    power_w: float = 0.0  # I²R heating
    
    def __post_init__(self):
        self.length_mm = self.start.distance_to(self.end)
    
    def compute_resistance(self, copper_thickness_m: float, resistivity_ohm_m: float) -> float:
        """Compute electrical resistance of segment."""
        if self.length_mm <= 0 or self.width_mm <= 0:
            return 0.0
        length_m = self.length_mm / 1000
        width_m = self.width_mm / 1000
        cross_section = width_m * copper_thickness_m
        if cross_section > 0:
            self.resistance_ohm = resistivity_ohm_m * length_m / cross_section
        return self.resistance_ohm
    
    def compute_power(self):
        """Compute power dissipation from current."""
        self.power_w = self.current_a ** 2 * self.resistance_ohm


@dataclass
class Via:
    """A via connecting PCB layers."""
    via_id: str
    net_name: str
    net_code: int
    position: Point2D
    drill_mm: float
    size_mm: float  # Pad size
    layers: Tuple[str, str]  # (start_layer, end_layer)
    via_type: str = "through"  # "through", "blind", "buried"
    
    # Thermal properties
    thermal_conductance: float = 0.0  # W/K
    current_a: float = 0.0
    
    def compute_thermal_conductance(self, copper_thickness_m: float, 
                                   plating_thickness_m: float,
                                   via_length_m: float,
                                   thermal_conductivity: float) -> float:
        """Compute thermal conductance through the via barrel."""
        # Via barrel is a hollow cylinder
        outer_radius = self.drill_mm / 2000 + plating_thickness_m
        inner_radius = self.drill_mm / 2000
        area = math.pi * (outer_radius**2 - inner_radius**2)
        
        if via_length_m > 0:
            self.thermal_conductance = thermal_conductivity * area / via_length_m
        return self.thermal_conductance


@dataclass
class CopperPour:
    """A copper pour/zone on a layer."""
    zone_id: str
    net_name: str
    net_code: int
    layer: str
    outline: List[Point2D]  # Polygon outline
    holes: List[List[Point2D]] = field(default_factory=list)  # Internal cutouts
    fill_percent: float = 100.0  # Approximation of fill
    
    def contains_point(self, point: Point2D) -> bool:
        """Check if point is inside the pour using ray casting."""
        if not self.outline:
            return False
        
        n = len(self.outline)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = self.outline[i].x, self.outline[i].y
            xj, yj = self.outline[j].x, self.outline[j].y
            
            if ((yi > point.y) != (yj > point.y)) and \
               (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        # Check if in any hole
        if inside:
            for hole in self.holes:
                if self._point_in_polygon(point, hole):
                    return False
        
        return inside
    
    def _point_in_polygon(self, point: Point2D, polygon: List[Point2D]) -> bool:
        """Helper to check point in polygon."""
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i].x, polygon[i].y
            xj, yj = polygon[j].x, polygon[j].y
            if ((yi > point.y) != (yj > point.y)) and \
               (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    def get_area_mm2(self) -> float:
        """Calculate area of the pour using shoelace formula."""
        if len(self.outline) < 3:
            return 0.0
        
        area = 0.0
        n = len(self.outline)
        for i in range(n):
            j = (i + 1) % n
            area += self.outline[i].x * self.outline[j].y
            area -= self.outline[j].x * self.outline[i].y
        area = abs(area) / 2.0
        
        # Subtract holes
        for hole in self.holes:
            hole_area = 0.0
            m = len(hole)
            for i in range(m):
                j = (i + 1) % m
                hole_area += hole[i].x * hole[j].y
                hole_area -= hole[j].x * hole[i].y
            area -= abs(hole_area) / 2.0
        
        return area * (self.fill_percent / 100.0)


@dataclass
class Pad:
    """A component pad."""
    pad_id: str
    parent_reference: str  # Component reference
    net_name: str
    net_code: int
    position: Point2D
    size: Tuple[float, float]  # (width, height) mm
    shape: str  # "rect", "circle", "oval", "roundrect", "custom"
    layers: List[str]
    drill_mm: Optional[float] = None  # For through-hole pads
    
    def get_area_mm2(self) -> float:
        """Calculate pad area."""
        w, h = self.size
        if self.shape == "circle":
            return math.pi * (w / 2) ** 2
        elif self.shape == "oval":
            # Approximation for oval/obround
            return math.pi * (min(w, h) / 2) ** 2 + (abs(w - h) * min(w, h))
        else:
            # Rectangle or roundrect
            return w * h


@dataclass
class Component:
    """A component on the PCB."""
    reference: str
    value: str
    footprint: str
    position: Point2D
    rotation: float  # degrees
    layer: str  # "F.Cu" or "B.Cu"
    pads: List[Pad] = field(default_factory=list)
    
    # Thermal properties from schematic or database
    power_dissipation_w: float = 0.0
    theta_ja: Optional[float] = None
    theta_jc: Optional[float] = None
    thermal_mass: Optional[float] = None
    
    def get_bounding_box(self) -> Tuple[Point2D, Point2D]:
        """Get component bounding box from pads."""
        if not self.pads:
            return (self.position, self.position)
        
        min_x = min(p.position.x - p.size[0]/2 for p in self.pads)
        max_x = max(p.position.x + p.size[0]/2 for p in self.pads)
        min_y = min(p.position.y - p.size[1]/2 for p in self.pads)
        max_y = max(p.position.y + p.size[1]/2 for p in self.pads)
        
        return (Point2D(min_x, min_y), Point2D(max_x, max_y))
    
    def get_footprint_area_mm2(self) -> float:
        """Get total footprint area."""
        bbox = self.get_bounding_box()
        return (bbox[1].x - bbox[0].x) * (bbox[1].y - bbox[0].y)


@dataclass
class MountingHole:
    """A mounting hole on the PCB."""
    hole_id: str
    position: Point2D
    drill_mm: float
    pad_size_mm: Optional[float] = None  # If plated
    is_plated: bool = False


@dataclass
class BoardOutline:
    """PCB board outline."""
    outline: List[Point2D] = field(default_factory=list)
    width_mm: float = 0.0
    height_mm: float = 0.0
    area_mm2: float = 0.0
    
    def compute_dimensions(self):
        """Compute board dimensions from outline."""
        if not self.outline:
            return
        
        xs = [p.x for p in self.outline]
        ys = [p.y for p in self.outline]
        
        self.width_mm = max(xs) - min(xs)
        self.height_mm = max(ys) - min(ys)
        
        # Area using shoelace
        n = len(self.outline)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.outline[i].x * self.outline[j].y
            area -= self.outline[j].x * self.outline[i].y
        self.area_mm2 = abs(area) / 2.0


@dataclass
class UserLayerShape:
    """Shape on a user layer (for heatsinks, mounting areas, etc.)."""
    shape_id: str
    layer: str
    shape_type: str  # "polygon", "rect", "circle"
    points: List[Point2D]  # For polygon/rect: vertices, for circle: [center]
    radius_mm: Optional[float] = None  # For circle
    
    def get_area_mm2(self) -> float:
        """Calculate shape area."""
        if self.shape_type == "circle" and self.radius_mm:
            return math.pi * self.radius_mm ** 2
        elif self.shape_type == "rect" and len(self.points) >= 2:
            p1, p2 = self.points[0], self.points[1]
            return abs(p2.x - p1.x) * abs(p2.y - p1.y)
        elif self.shape_type == "polygon" and len(self.points) >= 3:
            # Shoelace formula
            n = len(self.points)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += self.points[i].x * self.points[j].y
                area -= self.points[j].x * self.points[i].y
            return abs(area) / 2.0
        return 0.0


@dataclass
class PCBData:
    """Complete extracted PCB data for thermal analysis."""
    # Board info
    board_outline: BoardOutline = field(default_factory=BoardOutline)
    layer_names: List[str] = field(default_factory=list)
    copper_layers: List[str] = field(default_factory=list)
    
    # Geometry
    traces: List[TraceSegment] = field(default_factory=list)
    vias: List[Via] = field(default_factory=list)
    copper_pours: List[CopperPour] = field(default_factory=list)
    components: List[Component] = field(default_factory=list)
    mounting_holes: List[MountingHole] = field(default_factory=list)
    
    # User layer shapes
    user_shapes: Dict[str, List[UserLayerShape]] = field(default_factory=dict)
    
    # Net information
    net_names: Dict[int, str] = field(default_factory=dict)
    net_classes: Dict[str, Dict] = field(default_factory=dict)
    
    # Statistics
    total_copper_area_mm2: float = 0.0
    total_trace_length_mm: float = 0.0
    via_count: int = 0
    component_count: int = 0


class PCBExtractor:
    """Extracts PCB geometry from KiCad pcbnew board."""
    
    # Internal units conversion (KiCad uses nm internally in many cases)
    NM_TO_MM = 1e-6
    IU_TO_MM = 1e-6  # Internal units, may vary
    
    def __init__(self, board=None):
        """Initialize extractor with optional board reference."""
        self.board = board
        self.pcb_data = PCBData()
        self._segment_counter = 0
        self._via_counter = 0
        self._zone_counter = 0
        self._shape_counter = 0
        
    def set_board(self, board):
        """Set the pcbnew board object."""
        self.board = board
    
    def _iu_to_mm(self, value) -> float:
        """Convert KiCad internal units to mm."""
        # KiCad 6+ uses nm, but API might return IU
        # Check if pcbnew is available and use its conversion
        try:
            import pcbnew
            return pcbnew.ToMM(value)
        except:
            # Fallback: assume internal units are nm
            return float(value) * self.NM_TO_MM
    
    def _get_point(self, kicad_point) -> Point2D:
        """Convert KiCad point to Point2D."""
        try:
            import pcbnew
            return Point2D(
                pcbnew.ToMM(kicad_point.x),
                pcbnew.ToMM(kicad_point.y)
            )
        except:
            return Point2D(
                float(kicad_point.x) * self.NM_TO_MM,
                float(kicad_point.y) * self.NM_TO_MM
            )
    
    def extract_all(self) -> PCBData:
        """Extract all PCB data for thermal analysis."""
        if self.board is None:
            raise ValueError("No board loaded")
        
        self.pcb_data = PCBData()
        
        self._extract_board_outline()
        self._extract_layers()
        self._extract_nets()
        self._extract_traces()
        self._extract_vias()
        self._extract_copper_pours()
        self._extract_components()
        self._extract_mounting_holes()
        self._extract_user_layer_shapes()
        self._compute_statistics()
        
        return self.pcb_data
    
    def _extract_board_outline(self):
        """Extract board outline from Edge.Cuts layer."""
        try:
            import pcbnew
            
            outline_points = []
            
            for drawing in self.board.GetDrawings():
                if drawing.GetLayer() == pcbnew.Edge_Cuts:
                    if drawing.GetClass() == "PCB_SHAPE":
                        shape = drawing
                        shape_type = shape.GetShape()
                        
                        if shape_type == pcbnew.SHAPE_T_SEGMENT:
                            start = self._get_point(shape.GetStart())
                            end = self._get_point(shape.GetEnd())
                            if start not in outline_points:
                                outline_points.append(start)
                            if end not in outline_points:
                                outline_points.append(end)
                        elif shape_type == pcbnew.SHAPE_T_RECT:
                            # Rectangle
                            pos = self._get_point(shape.GetPosition())
                            w = self._iu_to_mm(shape.GetWidth())
                            h = self._iu_to_mm(shape.GetHeight())
                            outline_points.extend([
                                Point2D(pos.x, pos.y),
                                Point2D(pos.x + w, pos.y),
                                Point2D(pos.x + w, pos.y + h),
                                Point2D(pos.x, pos.y + h),
                            ])
                        elif shape_type == pcbnew.SHAPE_T_POLY:
                            # Polygon
                            poly = shape.GetPolyShape()
                            for i in range(poly.OutlineCount()):
                                outline = poly.Outline(i)
                                for j in range(outline.PointCount()):
                                    pt = outline.GetPoint(j)
                                    outline_points.append(Point2D(
                                        pcbnew.ToMM(pt.x),
                                        pcbnew.ToMM(pt.y)
                                    ))
            
            # Sort points to form closed polygon (simplified - assumes convex)
            if outline_points:
                # Use convex hull or simple sorting for now
                self.pcb_data.board_outline.outline = self._sort_outline_points(outline_points)
                self.pcb_data.board_outline.compute_dimensions()
        except Exception as e:
            print(f"Error extracting board outline: {e}")
    
    def _sort_outline_points(self, points: List[Point2D]) -> List[Point2D]:
        """Sort outline points to form a proper polygon."""
        if len(points) < 3:
            return points
        
        # Find centroid
        cx = sum(p.x for p in points) / len(points)
        cy = sum(p.y for p in points) / len(points)
        
        # Sort by angle from centroid
        def angle_key(p):
            return math.atan2(p.y - cy, p.x - cx)
        
        return sorted(points, key=angle_key)
    
    def _extract_layers(self):
        """Extract layer information."""
        try:
            import pcbnew
            
            self.pcb_data.layer_names = []
            self.pcb_data.copper_layers = []
            
            # Standard copper layers
            copper_layer_ids = [
                pcbnew.F_Cu, pcbnew.B_Cu,
                pcbnew.In1_Cu, pcbnew.In2_Cu,
                pcbnew.In3_Cu, pcbnew.In4_Cu,
                pcbnew.In5_Cu, pcbnew.In6_Cu,
            ]
            
            for layer_id in copper_layer_ids:
                if self.board.IsLayerEnabled(layer_id):
                    name = self.board.GetLayerName(layer_id)
                    self.pcb_data.layer_names.append(name)
                    self.pcb_data.copper_layers.append(name)
        except Exception as e:
            print(f"Error extracting layers: {e}")
            # Fallback to standard 4-layer
            self.pcb_data.copper_layers = ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]
    
    def _extract_nets(self):
        """Extract net information."""
        try:
            import pcbnew
            
            netinfo = self.board.GetNetInfo()
            for net in netinfo.NetsByNetcode().values():
                code = net.GetNetCode()
                name = net.GetNetname()
                self.pcb_data.net_names[code] = name
        except Exception as e:
            print(f"Error extracting nets: {e}")
    
    def _extract_traces(self):
        """Extract all trace segments."""
        try:
            import pcbnew
            
            for track in self.board.GetTracks():
                if track.GetClass() == "PCB_TRACK":
                    self._segment_counter += 1
                    seg = TraceSegment(
                        segment_id=f"T{self._segment_counter}",
                        net_name=track.GetNetname(),
                        net_code=track.GetNetCode(),
                        layer=self.board.GetLayerName(track.GetLayer()),
                        start=self._get_point(track.GetStart()),
                        end=self._get_point(track.GetEnd()),
                        width_mm=self._iu_to_mm(track.GetWidth())
                    )
                    self.pcb_data.traces.append(seg)
        except Exception as e:
            print(f"Error extracting traces: {e}")
    
    def _extract_vias(self):
        """Extract all vias."""
        try:
            import pcbnew
            
            for track in self.board.GetTracks():
                if track.GetClass() == "PCB_VIA":
                    self._via_counter += 1
                    via = track
                    
                    # Determine via type
                    via_type_enum = via.GetViaType()
                    if via_type_enum == pcbnew.VIATYPE_THROUGH:
                        via_type = "through"
                    elif via_type_enum == pcbnew.VIATYPE_BLIND_BURIED:
                        via_type = "blind"
                    elif via_type_enum == pcbnew.VIATYPE_MICROVIA:
                        via_type = "microvia"
                    else:
                        via_type = "through"
                    
                    # Get layer span
                    top_layer = self.board.GetLayerName(via.TopLayer())
                    bottom_layer = self.board.GetLayerName(via.BottomLayer())
                    
                    v = Via(
                        via_id=f"V{self._via_counter}",
                        net_name=via.GetNetname(),
                        net_code=via.GetNetCode(),
                        position=self._get_point(via.GetPosition()),
                        drill_mm=self._iu_to_mm(via.GetDrillValue()),
                        size_mm=self._iu_to_mm(via.GetWidth()),
                        layers=(top_layer, bottom_layer),
                        via_type=via_type
                    )
                    self.pcb_data.vias.append(v)
        except Exception as e:
            print(f"Error extracting vias: {e}")
    
    def _extract_copper_pours(self):
        """Extract copper pour zones."""
        try:
            import pcbnew
            
            for zone in self.board.Zones():
                self._zone_counter += 1
                
                layer = self.board.GetLayerName(zone.GetLayer())
                if layer not in self.pcb_data.copper_layers:
                    continue
                
                # Get outline
                outline_points = []
                poly = zone.Outline()
                
                for i in range(poly.OutlineCount()):
                    outline = poly.Outline(i)
                    pts = []
                    for j in range(outline.PointCount()):
                        pt = outline.GetPoint(j)
                        pts.append(Point2D(
                            pcbnew.ToMM(pt.x),
                            pcbnew.ToMM(pt.y)
                        ))
                    if i == 0:
                        outline_points = pts
                    # Holes are additional outlines (not implemented fully)
                
                pour = CopperPour(
                    zone_id=f"Z{self._zone_counter}",
                    net_name=zone.GetNetname(),
                    net_code=zone.GetNetCode(),
                    layer=layer,
                    outline=outline_points
                )
                self.pcb_data.copper_pours.append(pour)
        except Exception as e:
            print(f"Error extracting copper pours: {e}")
    
    def _extract_components(self):
        """Extract component footprints and pads."""
        try:
            import pcbnew
            
            for fp in self.board.GetFootprints():
                ref = fp.GetReference()
                
                comp = Component(
                    reference=ref,
                    value=fp.GetValue(),
                    footprint=fp.GetFPIDAsString(),
                    position=self._get_point(fp.GetPosition()),
                    rotation=fp.GetOrientationDegrees(),
                    layer="F.Cu" if fp.GetLayer() == pcbnew.F_Cu else "B.Cu"
                )
                
                # Try to get power dissipation from schematic field
                try:
                    # Look for POWER_DISSIPATION field
                    for field_idx in range(fp.GetFieldCount()):
                        field = fp.GetField(field_idx)
                        if field.GetName().upper() == "POWER_DISSIPATION":
                            try:
                                # Parse value (handle units like "0.5W", "100mW")
                                val_str = field.GetText().strip()
                                comp.power_dissipation_w = self._parse_power_value(val_str)
                            except:
                                pass
                except:
                    pass
                
                # Extract pads
                for pad in fp.Pads():
                    pad_layers = []
                    layer_set = pad.GetLayerSet()
                    for layer_id in [pcbnew.F_Cu, pcbnew.B_Cu, pcbnew.In1_Cu, pcbnew.In2_Cu]:
                        if layer_set.Contains(layer_id):
                            pad_layers.append(self.board.GetLayerName(layer_id))
                    
                    pad_size = pad.GetSize()
                    shape_enum = pad.GetShape()
                    if shape_enum == pcbnew.PAD_SHAPE_CIRCLE:
                        shape = "circle"
                    elif shape_enum == pcbnew.PAD_SHAPE_OVAL:
                        shape = "oval"
                    elif shape_enum == pcbnew.PAD_SHAPE_ROUNDRECT:
                        shape = "roundrect"
                    else:
                        shape = "rect"
                    
                    drill = None
                    if pad.GetDrillSize().x > 0:
                        drill = self._iu_to_mm(pad.GetDrillSize().x)
                    
                    p = Pad(
                        pad_id=pad.GetNumber(),
                        parent_reference=ref,
                        net_name=pad.GetNetname(),
                        net_code=pad.GetNetCode(),
                        position=self._get_point(pad.GetPosition()),
                        size=(self._iu_to_mm(pad_size.x), self._iu_to_mm(pad_size.y)),
                        shape=shape,
                        layers=pad_layers,
                        drill_mm=drill
                    )
                    comp.pads.append(p)
                
                self.pcb_data.components.append(comp)
        except Exception as e:
            print(f"Error extracting components: {e}")
    
    def _parse_power_value(self, value_str: str) -> float:
        """Parse power value with unit suffix."""
        value_str = value_str.upper().strip()
        
        # Extract numeric part and unit
        match = re.match(r'([\d.]+)\s*([A-Z]*)', value_str)
        if not match:
            return 0.0
        
        number = float(match.group(1))
        unit = match.group(2)
        
        multipliers = {
            'W': 1.0,
            'MW': 1e-3,
            'UW': 1e-6,
            'KW': 1e3,
        }
        
        return number * multipliers.get(unit, 1.0)
    
    def _extract_mounting_holes(self):
        """Extract mounting holes."""
        try:
            import pcbnew
            
            hole_counter = 0
            for fp in self.board.GetFootprints():
                # Check if it's a mounting hole footprint
                fp_name = fp.GetFPIDAsString().upper()
                if 'MOUNTINGHOLE' in fp_name or 'MOUNTING_HOLE' in fp_name or 'MTG' in fp_name:
                    hole_counter += 1
                    
                    # Get drill size from first pad or footprint
                    drill = 0.0
                    pad_size = None
                    is_plated = False
                    
                    for pad in fp.Pads():
                        if pad.GetDrillSize().x > 0:
                            drill = self._iu_to_mm(pad.GetDrillSize().x)
                            pad_size = self._iu_to_mm(pad.GetSize().x)
                            is_plated = len(pad.GetLayerSet().CuStack()) > 0
                            break
                    
                    hole = MountingHole(
                        hole_id=f"MH{hole_counter}",
                        position=self._get_point(fp.GetPosition()),
                        drill_mm=drill,
                        pad_size_mm=pad_size,
                        is_plated=is_plated
                    )
                    self.pcb_data.mounting_holes.append(hole)
        except Exception as e:
            print(f"Error extracting mounting holes: {e}")
    
    def _extract_user_layer_shapes(self):
        """Extract shapes from User layers (for heatsinks, etc.)."""
        try:
            import pcbnew
            
            user_layers = [
                (pcbnew.User_1, "User.1"),
                (pcbnew.User_2, "User.2"),
            ]
            
            for layer_id, layer_name in user_layers:
                self.pcb_data.user_shapes[layer_name] = []
                
                for drawing in self.board.GetDrawings():
                    if drawing.GetLayer() == layer_id:
                        self._shape_counter += 1
                        
                        if drawing.GetClass() == "PCB_SHAPE":
                            shape = drawing
                            shape_type = shape.GetShape()
                            
                            if shape_type == pcbnew.SHAPE_T_POLY:
                                poly = shape.GetPolyShape()
                                points = []
                                if poly.OutlineCount() > 0:
                                    outline = poly.Outline(0)
                                    for j in range(outline.PointCount()):
                                        pt = outline.GetPoint(j)
                                        points.append(Point2D(
                                            pcbnew.ToMM(pt.x),
                                            pcbnew.ToMM(pt.y)
                                        ))
                                
                                user_shape = UserLayerShape(
                                    shape_id=f"US{self._shape_counter}",
                                    layer=layer_name,
                                    shape_type="polygon",
                                    points=points
                                )
                                self.pcb_data.user_shapes[layer_name].append(user_shape)
                            
                            elif shape_type == pcbnew.SHAPE_T_RECT:
                                pos = self._get_point(shape.GetPosition())
                                w = self._iu_to_mm(shape.GetWidth())
                                h = self._iu_to_mm(shape.GetHeight())
                                
                                user_shape = UserLayerShape(
                                    shape_id=f"US{self._shape_counter}",
                                    layer=layer_name,
                                    shape_type="rect",
                                    points=[pos, Point2D(pos.x + w, pos.y + h)]
                                )
                                self.pcb_data.user_shapes[layer_name].append(user_shape)
                            
                            elif shape_type == pcbnew.SHAPE_T_CIRCLE:
                                center = self._get_point(shape.GetCenter())
                                radius = self._iu_to_mm(shape.GetRadius())
                                
                                user_shape = UserLayerShape(
                                    shape_id=f"US{self._shape_counter}",
                                    layer=layer_name,
                                    shape_type="circle",
                                    points=[center],
                                    radius_mm=radius
                                )
                                self.pcb_data.user_shapes[layer_name].append(user_shape)
        except Exception as e:
            print(f"Error extracting user layer shapes: {e}")
    
    def _compute_statistics(self):
        """Compute aggregate statistics."""
        # Total trace length
        self.pcb_data.total_trace_length_mm = sum(t.length_mm for t in self.pcb_data.traces)
        
        # Via count
        self.pcb_data.via_count = len(self.pcb_data.vias)
        
        # Component count  
        self.pcb_data.component_count = len(self.pcb_data.components)
        
        # Copper area (approximate from pours)
        self.pcb_data.total_copper_area_mm2 = sum(z.get_area_mm2() for z in self.pcb_data.copper_pours)


__all__ = [
    'Point2D',
    'Point3D',
    'TraceSegment',
    'Via',
    'CopperPour',
    'Pad',
    'Component',
    'MountingHole',
    'BoardOutline',
    'UserLayerShape',
    'PCBData',
    'PCBExtractor',
]
