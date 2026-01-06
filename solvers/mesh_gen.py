"""
TVAC Thermal Analyzer - Mesh Generator
======================================
Generate thermal mesh from PCB data.

Features:
- Adaptive mesh refinement near heat sources
- Material property assignment
- Conductance calculation
- Boundary condition application

Author: Space Electronics Thermal Analysis Tool
Version: 2.0.0
"""

import math
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from ..core.pcb_extractor import PCBData, Point2D
from ..core.config import ThermalAnalysisConfig
from ..core.constants import MaterialsDatabase, SimulationDefaults
from .thermal_solver import ThermalNode, ThermalMesh


@dataclass
class MeshSettings:
    """Mesh generation settings."""
    base_resolution_mm: float = 0.5
    adaptive_refinement: bool = True
    refinement_factor: float = 2.0
    min_resolution_mm: float = 0.1
    max_resolution_mm: float = 2.0
    heat_source_refinement_radius_mm: float = 5.0


class MeshGenerator:
    """Generate thermal mesh from PCB data."""
    
    def __init__(self, pcb_data: PCBData, config: ThermalAnalysisConfig):
        self.pcb_data = pcb_data
        self.config = config
        self.settings = MeshSettings(
            base_resolution_mm=config.simulation.resolution_mm,
            adaptive_refinement=config.simulation.use_adaptive_mesh,
        )
    
    def generate(self, progress_callback: Optional[Callable] = None) -> ThermalMesh:
        """Generate complete thermal mesh."""
        
        mesh = ThermalMesh()
        
        if progress_callback:
            progress_callback(5, "Computing bounds...")
        
        # Get board bounds
        bounds = self._compute_bounds()
        mesh.board_min_x = bounds[0]
        mesh.board_max_x = bounds[1]
        mesh.board_min_y = bounds[2]
        mesh.board_max_y = bounds[3]
        
        if progress_callback:
            progress_callback(10, "Generating grid...")
        
        # Generate grid coordinates
        x_coords, y_coords = self._generate_grid_coordinates(bounds)
        z_coords = self._generate_z_coordinates()
        
        mesh.nx = len(x_coords)
        mesh.ny = len(y_coords)
        mesh.nz = len(z_coords)
        
        if mesh.nx * mesh.ny * mesh.nz == 0:
            return mesh
        
        # Calculate cell sizes
        mesh.dx = (bounds[1] - bounds[0]) / max(1, mesh.nx - 1) * 1e-3  # Convert to m
        mesh.dy = (bounds[3] - bounds[2]) / max(1, mesh.ny - 1) * 1e-3
        mesh.dz = sum(l.thickness_um for l in self.config.stackup.layers) / mesh.nz * 1e-6
        
        if progress_callback:
            progress_callback(20, "Creating nodes...")
        
        # Create nodes
        self._create_nodes(mesh, x_coords, y_coords, z_coords)
        
        if progress_callback:
            progress_callback(40, "Assigning materials...")
        
        # Assign material properties
        self._assign_material_properties(mesh)
        
        if progress_callback:
            progress_callback(60, "Adding heat sources...")
        
        # Add heat sources
        self._add_heat_sources(mesh)
        
        if progress_callback:
            progress_callback(75, "Calculating conductances...")
        
        # Calculate conductances
        self._calculate_conductances(mesh)
        
        if progress_callback:
            progress_callback(90, "Applying boundary conditions...")
        
        # Apply boundary conditions
        self._apply_boundary_conditions(mesh)
        
        return mesh
    
    def _compute_bounds(self) -> Tuple[float, float, float, float]:
        """Compute board bounds from PCB data."""
        outline = self.pcb_data.board_outline
        
        if outline.outline:
            min_x = min(p.x for p in outline.outline)
            max_x = max(p.x for p in outline.outline)
            min_y = min(p.y for p in outline.outline)
            max_y = max(p.y for p in outline.outline)
        else:
            min_x = outline.min_x
            max_x = outline.max_x
            min_y = outline.min_y
            max_y = outline.max_y
        
        # Fallback if bounds are invalid
        if max_x - min_x < 1 or max_y - min_y < 1:
            # Try to compute from components
            if self.pcb_data.components:
                xs = [c.position.x for c in self.pcb_data.components]
                ys = [c.position.y for c in self.pcb_data.components]
                margin = 10.0
                min_x = min(xs) - margin
                max_x = max(xs) + margin
                min_y = min(ys) - margin
                max_y = max(ys) + margin
            else:
                min_x, max_x = 0, 100
                min_y, max_y = 0, 100
        
        return (min_x, max_x, min_y, max_y)
    
    def _generate_grid_coordinates(self, bounds: Tuple[float, float, float, float]) -> Tuple[List[float], List[float]]:
        """Generate X and Y grid coordinates."""
        min_x, max_x, min_y, max_y = bounds
        
        # Get heat source locations for refinement
        heat_sources = []
        for cp in self.config.component_power:
            if cp.power_w > 0:
                for comp in self.pcb_data.components:
                    if comp.reference == cp.reference:
                        heat_sources.append((comp.position.x, comp.position.y, cp.power_w))
                        break
        
        x_coords = self._generate_1d_grid(min_x, max_x, heat_sources, axis=0)
        y_coords = self._generate_1d_grid(min_y, max_y, heat_sources, axis=1)
        
        return x_coords, y_coords
    
    def _generate_1d_grid(self, start: float, end: float, 
                         heat_sources: List[Tuple[float, float, float]],
                         axis: int) -> List[float]:
        """Generate 1D grid with optional refinement."""
        res = self.settings.base_resolution_mm
        
        if not self.settings.adaptive_refinement or not heat_sources:
            # Uniform grid
            n = max(2, int((end - start) / res) + 1)
            return [start + i * (end - start) / (n - 1) for i in range(n)]
        
        # Adaptive grid - finer near heat sources
        coords = set()
        coords.add(start)
        coords.add(end)
        
        # Add points along length
        pos = start
        while pos < end:
            # Check distance to nearest heat source
            min_dist = float('inf')
            for hs in heat_sources:
                hs_pos = hs[axis]
                dist = abs(pos - hs_pos)
                min_dist = min(min_dist, dist)
            
            # Adjust spacing based on distance
            if min_dist < self.settings.heat_source_refinement_radius_mm:
                local_res = self.settings.min_resolution_mm
            else:
                t = min(1.0, min_dist / (self.settings.heat_source_refinement_radius_mm * 3))
                local_res = self.settings.min_resolution_mm + t * (
                    self.settings.max_resolution_mm - self.settings.min_resolution_mm
                )
            
            coords.add(pos)
            pos += local_res
        
        return sorted(coords)
    
    def _generate_z_coordinates(self) -> List[float]:
        """Generate Z coordinates from stackup."""
        coords = [0.0]
        z = 0.0
        
        for layer in self.config.stackup.layers:
            z += layer.thickness_um * 1e-3  # Convert to mm
            coords.append(z)
        
        # Simplify to reasonable number of layers
        if len(coords) > 10:
            # Keep first, last, and evenly spaced middle
            n = 7
            step = len(coords) // (n - 1)
            coords = [coords[i * step] for i in range(n - 1)] + [coords[-1]]
        
        return coords
    
    def _create_nodes(self, mesh: ThermalMesh, 
                     x_coords: List[float], y_coords: List[float], z_coords: List[float]):
        """Create mesh nodes."""
        node_id = 0
        
        for iz, z in enumerate(z_coords):
            for iy, y in enumerate(y_coords):
                for ix, x in enumerate(x_coords):
                    # Calculate cell size
                    dx = self._get_cell_size_x(x_coords, ix) * 1e-3  # Convert to m
                    dy = self._get_cell_size_y(y_coords, iy) * 1e-3
                    dz = mesh.dz
                    
                    node = ThermalNode(
                        node_id=node_id,
                        x=x,
                        y=y,
                        z=z,
                        layer_idx=iz,
                        volume=dx * dy * dz,
                        surface_area=dx * dy if iz == 0 or iz == len(z_coords) - 1 else 0,
                    )
                    
                    mesh.nodes.append(node)
                    node_id += 1
    
    def _get_cell_size_x(self, coords: List[float], idx: int) -> float:
        """Get cell size in X direction."""
        if len(coords) < 2:
            return 1.0
        if idx == 0:
            return coords[1] - coords[0]
        if idx >= len(coords) - 1:
            return coords[-1] - coords[-2]
        return (coords[idx + 1] - coords[idx - 1]) / 2
    
    def _get_cell_size_y(self, coords: List[float], idx: int) -> float:
        """Get cell size in Y direction."""
        return self._get_cell_size_x(coords, idx)
    
    def _assign_material_properties(self, mesh: ThermalMesh):
        """Assign material properties to nodes - optimized for speed."""
        try:
            # Get default materials
            substrate_name = self.config.stackup.substrate_material
            substrate = MaterialsDatabase.PCB_SUBSTRATES.get(
                substrate_name, 
                MaterialsDatabase.PCB_SUBSTRATES.get('FR4')
            )
            
            if substrate is None:
                # Fallback to FR4 properties
                substrate = type('obj', (object,), {
                    'thermal_conductivity': 0.29,
                    'specific_heat': 1100,
                    'density': 1850,
                    'emissivity': 0.9
                })()
            
            copper = MaterialsDatabase.CONDUCTORS.get('COPPER')
            if copper is None:
                copper = type('obj', (object,), {
                    'thermal_conductivity': 385.0,
                    'specific_heat': 385,
                    'density': 8960,
                    'emissivity': 0.03
                })()
            
            # Assign default substrate properties to all nodes
            # (Skip expensive per-node copper check for speed)
            for node in mesh.nodes:
                node.k = substrate.thermal_conductivity
                node.cp = substrate.specific_heat
                node.rho = substrate.density
                node.emissivity = getattr(substrate, 'emissivity', 0.9)
            
            # Apply heatsink properties (only for top layer nodes in heatsink regions)
            self._apply_heatsink_properties(mesh)
            
        except Exception as e:
            # If anything fails, just use FR4 defaults
            for node in mesh.nodes:
                node.k = 0.29
                node.cp = 1100
                node.rho = 1850
                node.emissivity = 0.9
    
    def _check_copper_at_location(self, x: float, y: float, layer_idx: int) -> bool:
        """Check if there's copper at the given location."""
        # Check traces
        for trace in self.pcb_data.traces:
            # Simple distance check to trace line
            dx = trace.end.x - trace.start.x
            dy = trace.end.y - trace.start.y
            length = math.sqrt(dx*dx + dy*dy)
            
            if length < 0.01:
                continue
            
            # Project point onto line
            t = max(0, min(1, (
                (x - trace.start.x) * dx + (y - trace.start.y) * dy
            ) / (length * length)))
            
            px = trace.start.x + t * dx
            py = trace.start.y + t * dy
            
            dist = math.sqrt((x - px)**2 + (y - py)**2)
            if dist < trace.width / 2:
                return True
        
        # Check copper pours
        for pour in self.pcb_data.copper_pours:
            if self._point_in_polygon(x, y, pour.outline):
                return True
        
        # Check pads
        for comp in self.pcb_data.components:
            for pad in comp.pads:
                if self._point_in_pad(x, y, pad):
                    return True
        
        return False
    
    def _point_in_polygon(self, x: float, y: float, polygon: List[Point2D]) -> bool:
        """Check if point is inside polygon using ray casting."""
        if len(polygon) < 3:
            return False
        
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i].x, polygon[i].y
            xj, yj = polygon[j].x, polygon[j].y
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            
            j = i
        
        return inside
    
    def _point_in_pad(self, x: float, y: float, pad) -> bool:
        """Check if point is inside pad."""
        dx = abs(x - pad.position.x)
        dy = abs(y - pad.position.y)
        
        if pad.shape == "circle":
            return dx*dx + dy*dy <= (pad.width/2)**2
        else:  # Rectangle or similar
            return dx <= pad.width/2 and dy <= pad.height/2
    
    def _apply_heatsink_properties(self, mesh: ThermalMesh):
        """Apply heatsink material properties."""
        try:
            for hs in self.config.heatsinks:
                if not hs.polygon_points or len(hs.polygon_points) < 3:
                    continue
                
                material = MaterialsDatabase.HEATSINK_MATERIALS.get(hs.material)
                if material is None:
                    material = MaterialsDatabase.HEATSINK_MATERIALS.get('ALUMINUM_6061')
                if material is None:
                    # Fallback
                    material = type('obj', (object,), {
                        'thermal_conductivity': 167.0,
                        'emissivity': 0.04
                    })()
                
                # Find nodes within heatsink polygon
                points = [Point2D(p[0], p[1]) for p in hs.polygon_points]
                
                for node in mesh.nodes:
                    if node.layer_idx == 0:  # Top layer
                        if self._point_in_polygon(node.x, node.y, points):
                            node.k = material.thermal_conductivity
                            node.emissivity = hs.emissivity_override if hs.emissivity_override is not None else getattr(material, 'emissivity', 0.04)
        except Exception as e:
            # Log but don't crash
            pass
    
    def _add_heat_sources(self, mesh: ThermalMesh):
        """Add heat sources based on simulation mode."""
        if self.config.simulation.heat_source_mode == "current_injection":
            self._add_current_heat_sources(mesh)
        else:
            self._add_component_heat_sources(mesh)
    
    def _add_component_heat_sources(self, mesh: ThermalMesh):
        """Add heat sources from component power dissipation."""
        for cp in self.config.component_power:
            if cp.power_w <= 0:
                continue
            
            # Find component
            comp = None
            for c in self.pcb_data.components:
                if c.reference == cp.reference:
                    comp = c
                    break
            
            if not comp:
                continue
            
            # Get component bounds
            bbox = comp.get_bounding_box()
            min_x, min_y = bbox[0].x, bbox[0].y
            max_x, max_y = bbox[1].x, bbox[1].y
            
            # Find nodes within component footprint
            nodes_in_footprint = []
            for node in mesh.nodes:
                if node.layer_idx == 0:  # Top layer
                    if min_x <= node.x <= max_x and min_y <= node.y <= max_y:
                        nodes_in_footprint.append(node)
            
            # Distribute power among nodes
            if nodes_in_footprint:
                power_per_node = cp.power_w / len(nodes_in_footprint)
                for node in nodes_in_footprint:
                    node.heat_source += power_per_node
    
    def _add_current_heat_sources(self, mesh: ThermalMesh):
        """Add heat sources from current injection (I²R Joule heating).
        
        This simplified model:
        1. Calculates total current flow from injection points
        2. Distributes heat along trace paths (simplified uniform model)
        3. Uses copper resistivity and trace geometry for I²R calculation
        """
        from ..core.constants import PhysicalConstants
        
        # Get current injection points
        injection_points = self.config.current_injection_points
        if not injection_points:
            return
        
        # Calculate total current (should balance for valid simulation)
        total_current_in = sum(cp.current_a for cp in injection_points if cp.current_a > 0)
        total_current_out = sum(abs(cp.current_a) for cp in injection_points if cp.current_a < 0)
        
        # Use average for heat calculation
        avg_current = (total_current_in + total_current_out) / 2
        if avg_current <= 0:
            return
        
        # Get copper properties
        copper_resistivity = PhysicalConstants.COPPER_RESISTIVITY  # Ω·m
        
        # Calculate total trace resistance and heat
        # This is a simplified model - assumes current flows uniformly through traces
        total_heat_w = 0.0
        trace_segments = []
        
        for trace in self.pcb_data.traces:
            # Calculate trace length
            dx = trace.end.x - trace.start.x
            dy = trace.end.y - trace.start.y
            length_m = math.sqrt(dx*dx + dy*dy) * 1e-3  # mm to m
            
            if length_m < 1e-6:
                continue
            
            # Get copper thickness (assume 35um = 1oz)
            copper_thickness_m = 35e-6
            
            # Trace cross-section area
            width_m = trace.width * 1e-3  # mm to m
            area_m2 = width_m * copper_thickness_m
            
            if area_m2 < 1e-12:
                continue
            
            # Trace resistance R = ρL/A
            resistance = copper_resistivity * length_m / area_m2
            
            # Power P = I²R (assuming current flows through this trace)
            # This is simplified - real current distribution would require network analysis
            power = avg_current * avg_current * resistance
            
            trace_segments.append({
                'start': (trace.start.x, trace.start.y),
                'end': (trace.end.x, trace.end.y),
                'power': power,
                'length': length_m * 1000  # back to mm
            })
            total_heat_w += power
        
        # Distribute heat to mesh nodes along traces
        for segment in trace_segments:
            if segment['power'] <= 0:
                continue
            
            # Find nodes near this trace segment
            start_x, start_y = segment['start']
            end_x, end_y = segment['end']
            
            # Simple approach: find nodes within trace width of the trace line
            trace_nodes = []
            for node in mesh.nodes:
                if node.layer_idx == 0:  # Top layer
                    # Distance from point to line segment
                    dist = self._point_to_segment_distance(
                        node.x, node.y, start_x, start_y, end_x, end_y
                    )
                    if dist < 1.0:  # Within 1mm of trace
                        trace_nodes.append(node)
            
            # Distribute power to trace nodes
            if trace_nodes:
                power_per_node = segment['power'] / len(trace_nodes)
                for node in trace_nodes:
                    node.heat_source += power_per_node
        
        # Also add heat at injection points (contact resistance)
        contact_resistance = 1e-3  # 1 mΩ contact resistance estimate
        for cp in injection_points:
            contact_heat = cp.current_a * cp.current_a * contact_resistance
            
            # Find nearest node to injection point
            min_dist = float('inf')
            nearest_node = None
            for node in mesh.nodes:
                if node.layer_idx == 0:
                    dist = math.sqrt((node.x - cp.x_mm)**2 + (node.y - cp.y_mm)**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_node = node
            
            if nearest_node:
                nearest_node.heat_source += abs(contact_heat)
    
    def _point_to_segment_distance(self, px: float, py: float,
                                    x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate distance from point to line segment."""
        dx = x2 - x1
        dy = y2 - y1
        
        length_sq = dx*dx + dy*dy
        if length_sq < 1e-10:
            return math.sqrt((px - x1)**2 + (py - y1)**2)
        
        t = max(0, min(1, ((px - x1)*dx + (py - y1)*dy) / length_sq))
        
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)
    
    def _calculate_conductances(self, mesh: ThermalMesh):
        """Calculate thermal conductances between nodes."""
        nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
        
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    idx = mesh.get_node_index(ix, iy, iz)
                    if idx >= len(mesh.nodes):
                        continue
                    
                    node = mesh.nodes[idx]
                    
                    # X neighbors
                    if ix > 0:
                        neighbor_idx = mesh.get_node_index(ix - 1, iy, iz)
                        self._add_conductance(mesh, node, neighbor_idx, 'x')
                    if ix < nx - 1:
                        neighbor_idx = mesh.get_node_index(ix + 1, iy, iz)
                        self._add_conductance(mesh, node, neighbor_idx, 'x')
                    
                    # Y neighbors
                    if iy > 0:
                        neighbor_idx = mesh.get_node_index(ix, iy - 1, iz)
                        self._add_conductance(mesh, node, neighbor_idx, 'y')
                    if iy < ny - 1:
                        neighbor_idx = mesh.get_node_index(ix, iy + 1, iz)
                        self._add_conductance(mesh, node, neighbor_idx, 'y')
                    
                    # Z neighbors
                    if iz > 0:
                        neighbor_idx = mesh.get_node_index(ix, iy, iz - 1)
                        self._add_conductance(mesh, node, neighbor_idx, 'z')
                    if iz < nz - 1:
                        neighbor_idx = mesh.get_node_index(ix, iy, iz + 1)
                        self._add_conductance(mesh, node, neighbor_idx, 'z')
    
    def _add_conductance(self, mesh: ThermalMesh, node: ThermalNode, 
                        neighbor_idx: int, direction: str):
        """Add conductance between node and neighbor."""
        if neighbor_idx < 0 or neighbor_idx >= len(mesh.nodes):
            return
        
        neighbor = mesh.nodes[neighbor_idx]
        
        # Calculate distance
        if direction == 'x':
            dist = abs(neighbor.x - node.x) * 1e-3  # Convert to m
            area = mesh.dy * mesh.dz
        elif direction == 'y':
            dist = abs(neighbor.y - node.y) * 1e-3
            area = mesh.dx * mesh.dz
        else:  # z
            dist = abs(neighbor.z - node.z) * 1e-3
            area = mesh.dx * mesh.dy
        
        if dist < 1e-9:
            dist = 1e-6  # Minimum distance
        
        # Harmonic mean of conductivities
        k_eff = 2 * node.k * neighbor.k / (node.k + neighbor.k) if (node.k + neighbor.k) > 0 else 0
        
        # Conductance: G = k * A / L
        G = k_eff * area / dist
        
        node.neighbors[neighbor_idx] = G
    
    def _apply_boundary_conditions(self, mesh: ThermalMesh):
        """Apply boundary conditions from mounting points."""
        for mp in self.config.mounting_points:
            if mp.fixed_temp_c is not None:
                # Find nearest node
                nearest_node = self._find_nearest_node(mesh, mp.x_mm, mp.y_mm)
                if nearest_node:
                    nearest_node.is_fixed_temp = True
                    nearest_node.fixed_temp = mp.fixed_temp_c
    
    def _find_nearest_node(self, mesh: ThermalMesh, x: float, y: float) -> Optional[ThermalNode]:
        """Find nearest node to given coordinates."""
        min_dist = float('inf')
        nearest = None
        
        for node in mesh.nodes:
            if node.layer_idx == 0:  # Top layer
                dist = math.sqrt((node.x - x)**2 + (node.y - y)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest = node
        
        return nearest


__all__ = ['MeshGenerator', 'MeshSettings']
