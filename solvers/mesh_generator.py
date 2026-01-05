"""
TVAC Thermal Analyzer - Mesh Generator
=====================================
Generates adaptive thermal mesh from PCB geometry.

Author: Space Electronics Thermal Analysis Tool
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import math
from enum import Enum

from ..core.pcb_extractor import PCBData, Point2D, Point3D, TraceSegment, Via, CopperPour


class NodeType(Enum):
    """Types of mesh nodes for thermal analysis."""
    SUBSTRATE = 0      # PCB substrate (FR4, etc.)
    COPPER = 1         # Copper (traces, pours, pads)
    VIA = 2           # Via barrel
    COMPONENT = 3     # Component body
    PAD = 4           # Component pad
    AIR = 5           # Empty space (for radiation)
    HEATSINK = 6      # Heatsink
    MOUNTING = 7      # Mounting point
    BOUNDARY = 8      # Boundary condition


@dataclass
class MeshNode:
    """A single node in the thermal mesh."""
    node_id: int
    position: Point3D
    node_type: NodeType
    layer_index: int  # 0 = top, increasing downward
    
    # Thermal properties (will be set based on type and location)
    thermal_conductivity: float = 0.0  # W/(m·K)
    specific_heat: float = 0.0  # J/(kg·K)
    density: float = 0.0  # kg/m³
    emissivity: float = 0.0
    
    # Volume element
    volume_m3: float = 0.0
    area_m2: float = 0.0  # Surface area for radiation
    
    # Heat source
    heat_generation_w: float = 0.0  # Internal heat generation (Joule heating, component power)
    
    # State
    temperature_c: float = 25.0  # Current temperature
    
    # References
    net_code: int = -1  # If part of electrical network
    component_ref: str = ""  # If associated with component
    trace_id: str = ""  # If on a trace
    
    # Boundary conditions
    is_fixed_temp: bool = False
    fixed_temp_c: float = 0.0
    
    # Connections to other nodes (for sparse matrix building)
    neighbors: List[int] = field(default_factory=list)
    conductances: List[float] = field(default_factory=list)  # Thermal conductance to each neighbor


@dataclass
class MeshElement:
    """A mesh element (tetrahedron for 3D, triangle for 2D)."""
    element_id: int
    node_ids: List[int]  # 4 nodes for tetrahedron, 3 for triangle
    volume_m3: float
    centroid: Point3D
    
    # Material properties (averaged from nodes or set directly)
    thermal_conductivity: float = 0.0
    density: float = 0.0
    specific_heat: float = 0.0


@dataclass
class ThermalMesh:
    """Complete thermal mesh for simulation."""
    nodes: List[MeshNode] = field(default_factory=list)
    elements: List[MeshElement] = field(default_factory=list)
    
    # Dimensions
    nx: int = 0  # Nodes in X
    ny: int = 0  # Nodes in Y  
    nz: int = 0  # Nodes in Z (layers)
    
    # Grid spacing
    dx_mm: float = 0.0
    dy_mm: float = 0.0
    dz_mm: List[float] = field(default_factory=list)  # Per-layer thickness
    
    # Bounds
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    
    # Index helpers
    node_index_map: Dict[Tuple[int, int, int], int] = field(default_factory=dict)
    
    # Arrays for fast solver access (populated after mesh generation)
    temperature_array: Optional[np.ndarray] = None
    conductivity_array: Optional[np.ndarray] = None
    heat_source_array: Optional[np.ndarray] = None
    
    def get_node(self, ix: int, iy: int, iz: int) -> Optional[MeshNode]:
        """Get node by grid indices."""
        idx = self.node_index_map.get((ix, iy, iz))
        if idx is not None and 0 <= idx < len(self.nodes):
            return self.nodes[idx]
        return None
    
    def get_node_at_position(self, x: float, y: float, z: float) -> Optional[MeshNode]:
        """Get nearest node to a position."""
        ix = int((x - self.x_min) / self.dx_mm)
        iy = int((y - self.y_min) / self.dy_mm)
        
        # Find Z index
        iz = 0
        z_pos = 0.0
        for i, dz in enumerate(self.dz_mm):
            if z_pos + dz > z:
                iz = i
                break
            z_pos += dz
            iz = i
        
        return self.get_node(ix, iy, iz)
    
    def to_numpy_arrays(self):
        """Convert mesh data to NumPy arrays for solver."""
        n = len(self.nodes)
        
        self.temperature_array = np.array([node.temperature_c for node in self.nodes], dtype=np.float64)
        self.conductivity_array = np.array([node.thermal_conductivity for node in self.nodes], dtype=np.float64)
        self.heat_source_array = np.array([node.heat_generation_w for node in self.nodes], dtype=np.float64)
        
        return self.temperature_array, self.conductivity_array, self.heat_source_array


class MeshGenerator:
    """Generates thermal mesh from PCB data."""
    
    def __init__(self, pcb_data: PCBData, resolution_mm: float = 0.5,
                 use_adaptive: bool = True, adaptive_factor: float = 2.0):
        """Initialize mesh generator."""
        self.pcb_data = pcb_data
        self.base_resolution = resolution_mm
        self.use_adaptive = use_adaptive
        self.adaptive_factor = adaptive_factor
        
        self.mesh = ThermalMesh()
        self._node_counter = 0
        
    def generate(self, stackup_config: 'PCBStackupConfig', simulation_3d: bool = True) -> ThermalMesh:
        """Generate the thermal mesh."""
        self.mesh = ThermalMesh()
        self._node_counter = 0
        
        # Calculate grid dimensions
        self._calculate_grid_dimensions(stackup_config)
        
        # Generate nodes
        if simulation_3d:
            self._generate_3d_nodes(stackup_config)
        else:
            self._generate_2d_nodes(stackup_config)
        
        # Assign material properties based on PCB geometry
        self._assign_copper_properties(stackup_config)
        self._assign_via_properties(stackup_config)
        self._assign_component_properties()
        
        # Build neighbor connections
        self._build_neighbor_connections(simulation_3d)
        
        # Calculate volumes and conductances
        self._calculate_volumes_and_conductances()
        
        # Convert to NumPy arrays
        self.mesh.to_numpy_arrays()
        
        return self.mesh
    
    def _calculate_grid_dimensions(self, stackup_config):
        """Calculate grid dimensions based on board size and resolution."""
        outline = self.pcb_data.board_outline
        
        # Use board dimensions or estimate from components
        if outline.width_mm > 0 and outline.height_mm > 0:
            width = outline.width_mm
            height = outline.height_mm
            
            # Get bounding box
            if outline.outline:
                xs = [p.x for p in outline.outline]
                ys = [p.y for p in outline.outline]
                self.mesh.x_min = min(xs)
                self.mesh.x_max = max(xs)
                self.mesh.y_min = min(ys)
                self.mesh.y_max = max(ys)
            else:
                self.mesh.x_min = 0
                self.mesh.x_max = width
                self.mesh.y_min = 0
                self.mesh.y_max = height
        else:
            # Estimate from traces and components
            all_x = []
            all_y = []
            for t in self.pcb_data.traces:
                all_x.extend([t.start.x, t.end.x])
                all_y.extend([t.start.y, t.end.y])
            for c in self.pcb_data.components:
                all_x.append(c.position.x)
                all_y.append(c.position.y)
            
            if all_x and all_y:
                margin = 5.0  # mm margin
                self.mesh.x_min = min(all_x) - margin
                self.mesh.x_max = max(all_x) + margin
                self.mesh.y_min = min(all_y) - margin
                self.mesh.y_max = max(all_y) + margin
            else:
                # Default 80x80mm
                self.mesh.x_min = 0
                self.mesh.x_max = 80
                self.mesh.y_min = 0
                self.mesh.y_max = 80
        
        width = self.mesh.x_max - self.mesh.x_min
        height = self.mesh.y_max - self.mesh.y_min
        
        # Grid spacing
        self.mesh.dx_mm = self.base_resolution
        self.mesh.dy_mm = self.base_resolution
        
        self.mesh.nx = max(2, int(math.ceil(width / self.mesh.dx_mm)) + 1)
        self.mesh.ny = max(2, int(math.ceil(height / self.mesh.dy_mm)) + 1)
        
        # Z layers based on stackup
        self._calculate_z_layers(stackup_config)
    
    def _calculate_z_layers(self, stackup_config):
        """Calculate Z layer positions from stackup."""
        # Build layer structure
        # Each copper layer + dielectric between them
        
        n_copper = stackup_config.layer_count
        self.mesh.dz_mm = []
        
        # Simplified: assume copper and dielectric alternate
        copper_thickness = list(stackup_config.copper_thickness_um.values())
        dielectric_thickness = stackup_config.dielectric_thickness_um
        
        for i in range(n_copper):
            # Copper layer
            if i < len(copper_thickness):
                self.mesh.dz_mm.append(copper_thickness[i] / 1000.0)  # Convert µm to mm
            else:
                self.mesh.dz_mm.append(0.035)  # Default 1oz copper
            
            # Dielectric (except after last layer)
            if i < n_copper - 1 and i < len(dielectric_thickness):
                self.mesh.dz_mm.append(dielectric_thickness[i] / 1000.0)
            elif i < n_copper - 1:
                self.mesh.dz_mm.append(0.2)  # Default 200µm
        
        self.mesh.nz = len(self.mesh.dz_mm)
    
    def _generate_3d_nodes(self, stackup_config):
        """Generate 3D mesh nodes."""
        # Pre-compute refinement map if using adaptive meshing
        refinement_map = None
        if self.use_adaptive:
            refinement_map = self._compute_refinement_map()
        
        z_position = 0.0
        
        for iz, dz in enumerate(self.mesh.dz_mm):
            for iy in range(self.mesh.ny):
                y = self.mesh.y_min + iy * self.mesh.dy_mm
                
                for ix in range(self.mesh.nx):
                    x = self.mesh.x_min + ix * self.mesh.dx_mm
                    
                    # Determine node type based on layer (copper vs dielectric)
                    # Even iz = copper, odd iz = dielectric (simplified)
                    if iz % 2 == 0:
                        node_type = NodeType.SUBSTRATE  # Will be overwritten if copper present
                    else:
                        node_type = NodeType.SUBSTRATE
                    
                    node = MeshNode(
                        node_id=self._node_counter,
                        position=Point3D(x, y, z_position + dz/2),
                        node_type=node_type,
                        layer_index=iz
                    )
                    
                    # Set default substrate properties
                    self._set_substrate_properties(node, stackup_config)
                    
                    self.mesh.nodes.append(node)
                    self.mesh.node_index_map[(ix, iy, iz)] = self._node_counter
                    self._node_counter += 1
            
            z_position += dz
    
    def _generate_2d_nodes(self, stackup_config):
        """Generate 2D mesh nodes (single layer approximation)."""
        for iy in range(self.mesh.ny):
            y = self.mesh.y_min + iy * self.mesh.dy_mm
            
            for ix in range(self.mesh.nx):
                x = self.mesh.x_min + ix * self.mesh.dx_mm
                
                node = MeshNode(
                    node_id=self._node_counter,
                    position=Point3D(x, y, stackup_config.total_thickness_mm / 2),
                    node_type=NodeType.SUBSTRATE,
                    layer_index=0
                )
                
                self._set_substrate_properties(node, stackup_config)
                
                self.mesh.nodes.append(node)
                self.mesh.node_index_map[(ix, iy, 0)] = self._node_counter
                self._node_counter += 1
        
        self.mesh.nz = 1
        self.mesh.dz_mm = [stackup_config.total_thickness_mm]
    
    def _compute_refinement_map(self) -> np.ndarray:
        """Compute areas needing mesh refinement."""
        # Create a 2D map of refinement factors
        ref_map = np.ones((self.mesh.ny, self.mesh.nx), dtype=np.float32)
        
        # Refine near traces
        for trace in self.pcb_data.traces:
            self._mark_refinement_line(ref_map, trace.start, trace.end, trace.width_mm)
        
        # Refine near vias
        for via in self.pcb_data.vias:
            ix = int((via.position.x - self.mesh.x_min) / self.mesh.dx_mm)
            iy = int((via.position.y - self.mesh.y_min) / self.mesh.dy_mm)
            if 0 <= ix < self.mesh.nx and 0 <= iy < self.mesh.ny:
                ref_map[iy, ix] = self.adaptive_factor
        
        # Refine near components
        for comp in self.pcb_data.components:
            ix = int((comp.position.x - self.mesh.x_min) / self.mesh.dx_mm)
            iy = int((comp.position.y - self.mesh.y_min) / self.mesh.dy_mm)
            if 0 <= ix < self.mesh.nx and 0 <= iy < self.mesh.ny:
                # Mark region around component
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        nix = ix + dx
                        niy = iy + dy
                        if 0 <= nix < self.mesh.nx and 0 <= niy < self.mesh.ny:
                            ref_map[niy, nix] = max(ref_map[niy, nix], self.adaptive_factor)
        
        return ref_map
    
    def _mark_refinement_line(self, ref_map: np.ndarray, start: Point2D, end: Point2D, width: float):
        """Mark refinement along a line (trace)."""
        length = start.distance_to(end)
        if length < 0.001:
            return
        
        steps = max(2, int(length / self.mesh.dx_mm))
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0
            x = start.x + t * (end.x - start.x)
            y = start.y + t * (end.y - start.y)
            
            ix = int((x - self.mesh.x_min) / self.mesh.dx_mm)
            iy = int((y - self.mesh.y_min) / self.mesh.dy_mm)
            
            if 0 <= ix < self.mesh.nx and 0 <= iy < self.mesh.ny:
                ref_map[iy, ix] = self.adaptive_factor
    
    def _set_substrate_properties(self, node: MeshNode, stackup_config):
        """Set material properties for substrate node."""
        from .constants import MaterialsDatabase
        
        material_key = stackup_config.substrate_material
        material = MaterialsDatabase.PCB_SUBSTRATES.get(
            material_key,
            MaterialsDatabase.PCB_SUBSTRATES['FR4']
        )
        
        node.thermal_conductivity = material.thermal_conductivity
        node.specific_heat = material.specific_heat
        node.density = material.density
        node.emissivity = material.emissivity
    
    def _assign_copper_properties(self, stackup_config):
        """Assign copper properties to nodes on copper features."""
        from .constants import MaterialsDatabase
        
        copper = MaterialsDatabase.CONDUCTORS['COPPER']
        
        # Map layer names to Z indices
        copper_layers = self.pcb_data.copper_layers
        layer_to_z = {}
        z_idx = 0
        for i, layer in enumerate(copper_layers):
            layer_to_z[layer] = z_idx
            z_idx += 2  # Skip dielectric layers
        
        # Process traces
        for trace in self.pcb_data.traces:
            if trace.layer not in layer_to_z:
                continue
            
            iz = layer_to_z[trace.layer]
            self._mark_line_as_copper(trace.start, trace.end, trace.width_mm, iz, copper, trace.segment_id)
        
        # Process copper pours
        for pour in self.pcb_data.copper_pours:
            if pour.layer not in layer_to_z:
                continue
            
            iz = layer_to_z[pour.layer]
            self._mark_polygon_as_copper(pour.outline, iz, copper)
        
        # Process pads
        for comp in self.pcb_data.components:
            for pad in comp.pads:
                for layer in pad.layers:
                    if layer in layer_to_z:
                        iz = layer_to_z[layer]
                        self._mark_pad_as_copper(pad, iz, copper)
    
    def _mark_line_as_copper(self, start: Point2D, end: Point2D, width: float, 
                             iz: int, copper_material, trace_id: str):
        """Mark nodes along a trace as copper."""
        length = start.distance_to(end)
        if length < 0.001:
            return
        
        steps = max(2, int(length / (self.mesh.dx_mm / 2)))
        half_width = width / 2
        
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0
            cx = start.x + t * (end.x - start.x)
            cy = start.y + t * (end.y - start.y)
            
            # Find all nodes within trace width
            ix_min = int((cx - half_width - self.mesh.x_min) / self.mesh.dx_mm)
            ix_max = int((cx + half_width - self.mesh.x_min) / self.mesh.dx_mm) + 1
            iy_min = int((cy - half_width - self.mesh.y_min) / self.mesh.dy_mm)
            iy_max = int((cy + half_width - self.mesh.y_min) / self.mesh.dy_mm) + 1
            
            for iy in range(max(0, iy_min), min(self.mesh.ny, iy_max)):
                for ix in range(max(0, ix_min), min(self.mesh.nx, ix_max)):
                    if iz < self.mesh.nz:
                        idx = self.mesh.node_index_map.get((ix, iy, iz))
                        if idx is not None:
                            node = self.mesh.nodes[idx]
                            node.node_type = NodeType.COPPER
                            node.thermal_conductivity = copper_material.thermal_conductivity
                            node.specific_heat = copper_material.specific_heat
                            node.density = copper_material.density
                            node.emissivity = copper_material.emissivity
                            node.trace_id = trace_id
    
    def _mark_polygon_as_copper(self, outline: List[Point2D], iz: int, copper_material):
        """Mark nodes inside a polygon as copper."""
        if not outline or len(outline) < 3:
            return
        
        # Get bounding box
        xs = [p.x for p in outline]
        ys = [p.y for p in outline]
        
        ix_min = int((min(xs) - self.mesh.x_min) / self.mesh.dx_mm)
        ix_max = int((max(xs) - self.mesh.x_min) / self.mesh.dx_mm) + 1
        iy_min = int((min(ys) - self.mesh.y_min) / self.mesh.dy_mm)
        iy_max = int((max(ys) - self.mesh.y_min) / self.mesh.dy_mm) + 1
        
        for iy in range(max(0, iy_min), min(self.mesh.ny, iy_max)):
            for ix in range(max(0, ix_min), min(self.mesh.nx, ix_max)):
                x = self.mesh.x_min + ix * self.mesh.dx_mm
                y = self.mesh.y_min + iy * self.mesh.dy_mm
                
                if self._point_in_polygon(Point2D(x, y), outline):
                    if iz < self.mesh.nz:
                        idx = self.mesh.node_index_map.get((ix, iy, iz))
                        if idx is not None:
                            node = self.mesh.nodes[idx]
                            node.node_type = NodeType.COPPER
                            node.thermal_conductivity = copper_material.thermal_conductivity
                            node.specific_heat = copper_material.specific_heat
                            node.density = copper_material.density
                            node.emissivity = copper_material.emissivity
    
    def _mark_pad_as_copper(self, pad, iz: int, copper_material):
        """Mark nodes at pad location as copper."""
        w, h = pad.size
        cx, cy = pad.position.x, pad.position.y
        
        ix_min = int((cx - w/2 - self.mesh.x_min) / self.mesh.dx_mm)
        ix_max = int((cx + w/2 - self.mesh.x_min) / self.mesh.dx_mm) + 1
        iy_min = int((cy - h/2 - self.mesh.y_min) / self.mesh.dy_mm)
        iy_max = int((cy + h/2 - self.mesh.y_min) / self.mesh.dy_mm) + 1
        
        for iy in range(max(0, iy_min), min(self.mesh.ny, iy_max)):
            for ix in range(max(0, ix_min), min(self.mesh.nx, ix_max)):
                if iz < self.mesh.nz:
                    idx = self.mesh.node_index_map.get((ix, iy, iz))
                    if idx is not None:
                        node = self.mesh.nodes[idx]
                        node.node_type = NodeType.PAD
                        node.thermal_conductivity = copper_material.thermal_conductivity
                        node.specific_heat = copper_material.specific_heat
                        node.density = copper_material.density
                        node.emissivity = copper_material.emissivity
    
    def _point_in_polygon(self, point: Point2D, polygon: List[Point2D]) -> bool:
        """Ray casting algorithm for point in polygon."""
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
    
    def _assign_via_properties(self, stackup_config):
        """Assign via thermal properties."""
        from .constants import MaterialsDatabase
        
        copper = MaterialsDatabase.CONDUCTORS['COPPER']
        
        for via in self.pcb_data.vias:
            ix = int((via.position.x - self.mesh.x_min) / self.mesh.dx_mm)
            iy = int((via.position.y - self.mesh.y_min) / self.mesh.dy_mm)
            
            # Mark via through all layers it spans
            # For through-hole vias, this is all layers
            for iz in range(self.mesh.nz):
                idx = self.mesh.node_index_map.get((ix, iy, iz))
                if idx is not None:
                    node = self.mesh.nodes[idx]
                    node.node_type = NodeType.VIA
                    # Via has higher Z-conductivity due to copper barrel
                    node.thermal_conductivity = copper.thermal_conductivity
                    node.specific_heat = copper.specific_heat
                    node.density = copper.density
    
    def _assign_component_properties(self):
        """Assign component thermal properties."""
        from .constants import ComponentThermalDatabase
        
        for comp in self.pcb_data.components:
            # Find component location
            ix = int((comp.position.x - self.mesh.x_min) / self.mesh.dx_mm)
            iy = int((comp.position.y - self.mesh.y_min) / self.mesh.dy_mm)
            
            # Get layer (top or bottom)
            iz = 0 if comp.layer == "F.Cu" else (self.mesh.nz - 1 if self.mesh.nz > 1 else 0)
            
            idx = self.mesh.node_index_map.get((ix, iy, iz))
            if idx is not None:
                node = self.mesh.nodes[idx]
                node.node_type = NodeType.COMPONENT
                node.component_ref = comp.reference
                
                # Apply power dissipation
                if comp.power_dissipation_w > 0:
                    # Distribute power across component footprint
                    bbox = comp.get_bounding_box()
                    area = comp.get_footprint_area_mm2()
                    
                    if area > 0:
                        power_per_mm2 = comp.power_dissipation_w / area
                        
                        # Find all nodes in component area
                        ix_min = int((bbox[0].x - self.mesh.x_min) / self.mesh.dx_mm)
                        ix_max = int((bbox[1].x - self.mesh.x_min) / self.mesh.dx_mm) + 1
                        iy_min = int((bbox[0].y - self.mesh.y_min) / self.mesh.dy_mm)
                        iy_max = int((bbox[1].y - self.mesh.y_min) / self.mesh.dy_mm) + 1
                        
                        node_area = self.mesh.dx_mm * self.mesh.dy_mm
                        power_per_node = power_per_mm2 * node_area
                        
                        for niy in range(max(0, iy_min), min(self.mesh.ny, iy_max)):
                            for nix in range(max(0, ix_min), min(self.mesh.nx, ix_max)):
                                nidx = self.mesh.node_index_map.get((nix, niy, iz))
                                if nidx is not None:
                                    self.mesh.nodes[nidx].heat_generation_w += power_per_node
    
    def _build_neighbor_connections(self, is_3d: bool):
        """Build neighbor connectivity for each node."""
        for ix in range(self.mesh.nx):
            for iy in range(self.mesh.ny):
                for iz in range(self.mesh.nz):
                    idx = self.mesh.node_index_map.get((ix, iy, iz))
                    if idx is None:
                        continue
                    
                    node = self.mesh.nodes[idx]
                    node.neighbors = []
                    
                    # 6-connectivity for 3D, 4-connectivity for 2D
                    neighbors_to_check = [
                        (ix-1, iy, iz),
                        (ix+1, iy, iz),
                        (ix, iy-1, iz),
                        (ix, iy+1, iz),
                    ]
                    
                    if is_3d:
                        neighbors_to_check.extend([
                            (ix, iy, iz-1),
                            (ix, iy, iz+1),
                        ])
                    
                    for nix, niy, niz in neighbors_to_check:
                        if 0 <= nix < self.mesh.nx and 0 <= niy < self.mesh.ny and 0 <= niz < self.mesh.nz:
                            nidx = self.mesh.node_index_map.get((nix, niy, niz))
                            if nidx is not None:
                                node.neighbors.append(nidx)
    
    def _calculate_volumes_and_conductances(self):
        """Calculate volume elements and thermal conductances."""
        for node in self.mesh.nodes:
            # Volume
            dx = self.mesh.dx_mm / 1000  # Convert to m
            dy = self.mesh.dy_mm / 1000
            dz = self.mesh.dz_mm[node.layer_index] / 1000 if node.layer_index < len(self.mesh.dz_mm) else 0.001
            
            node.volume_m3 = dx * dy * dz
            
            # Surface area (for radiation - top/bottom surfaces)
            node.area_m2 = dx * dy
            
            # Calculate conductances to neighbors
            node.conductances = []
            for neighbor_idx in node.neighbors:
                neighbor = self.mesh.nodes[neighbor_idx]
                
                # Determine direction and use harmonic mean of conductivities
                k1 = node.thermal_conductivity
                k2 = neighbor.thermal_conductivity
                
                # Harmonic mean
                if k1 > 0 and k2 > 0:
                    k_eff = 2 * k1 * k2 / (k1 + k2)
                else:
                    k_eff = max(k1, k2)
                
                # Calculate conductance based on direction
                # G = k * A / L
                # Determine direction
                dx_dir = abs(node.position.x - neighbor.position.x) > 0.001
                dy_dir = abs(node.position.y - neighbor.position.y) > 0.001
                dz_dir = abs(node.position.z - neighbor.position.z) > 0.001
                
                if dx_dir:
                    area = dy * dz
                    length = dx
                elif dy_dir:
                    area = dx * dz
                    length = dy
                else:  # dz_dir
                    area = dx * dy
                    length = (dz + self.mesh.dz_mm[neighbor.layer_index] / 1000) / 2 if neighbor.layer_index < len(self.mesh.dz_mm) else dz
                
                if length > 0:
                    conductance = k_eff * area / length
                else:
                    conductance = 0.0
                
                node.conductances.append(conductance)
    
    def apply_heat_sources(self, trace_power: Dict[str, float], component_power: Dict[str, float]):
        """Apply heat sources from computed trace and component power."""
        # Apply trace Joule heating
        for node in self.mesh.nodes:
            if node.trace_id and node.trace_id in trace_power:
                node.heat_generation_w = trace_power[node.trace_id]
        
        # Apply component power (already partially done in _assign_component_properties)
        for ref, power in component_power.items():
            for node in self.mesh.nodes:
                if node.component_ref == ref and node.node_type == NodeType.COMPONENT:
                    # Distribute power - this is simplified
                    node.heat_generation_w = power
        
        # Update numpy arrays
        if self.mesh.heat_source_array is not None:
            self.mesh.heat_source_array = np.array([node.heat_generation_w for node in self.mesh.nodes])
    
    def apply_boundary_conditions(self, mounting_points: List, fixed_temps: Dict[str, float]):
        """Apply thermal boundary conditions."""
        for mp in mounting_points:
            ix = int((mp.x_mm - self.mesh.x_min) / self.mesh.dx_mm)
            iy = int((mp.y_mm - self.mesh.y_min) / self.mesh.dy_mm)
            
            if mp.interface_temp_c is not None:
                # Fixed temperature boundary
                for iz in range(self.mesh.nz):
                    idx = self.mesh.node_index_map.get((ix, iy, iz))
                    if idx is not None:
                        node = self.mesh.nodes[idx]
                        node.is_fixed_temp = True
                        node.fixed_temp_c = mp.interface_temp_c
                        node.node_type = NodeType.MOUNTING


__all__ = [
    'NodeType',
    'MeshNode',
    'MeshElement',
    'ThermalMesh',
    'MeshGenerator',
]
