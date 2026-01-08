"""
TVAC Thermal Analyzer - Current Distribution Solver
===================================================
Solves for current distribution in PCB copper network using nodal analysis.

Author: Space Electronics Thermal Analysis Tool
Version: 1.0.0
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import math

from ..core.pcb_extractor import (
    PCBData, Point2D, TraceSegment, Via, CopperPour, Pad, Component
)
try:
    from ..core.config import CurrentInjectionPoint, TraceCurrentOverride
except Exception:  # pragma: no cover
    from ..core.config import CurrentInjectionPoint
    from dataclasses import dataclass

    @dataclass
    class TraceCurrentOverride:
        """Fallback legacy override dataclass (kept for backward compatibility)."""
        trace_id: str = ""
        current_a: float = 0.0
        direction: int = 1

from ..core.constants import MaterialsDatabase, PhysicalConstants
from ..utils.logger import get_logger, timed_function


@dataclass
class ElectricalNode:
    """Node in the electrical network."""
    node_id: int
    position: Point2D
    layer: str
    net_code: int
    net_name: str
    
    # Connected elements
    connected_segments: List[str] = field(default_factory=list)
    connected_vias: List[str] = field(default_factory=list)
    
    # Solution
    voltage: float = 0.0
    is_reference: bool = False  # Ground reference
    current_injection: float = 0.0  # External current source/sink


@dataclass
class ElectricalSegment:
    """Segment in the electrical network (trace or via barrel)."""
    segment_id: str
    node1_id: int
    node2_id: int
    resistance_ohm: float
    
    # Trace properties
    length_m: float = 0.0
    width_m: float = 0.0
    thickness_m: float = 0.0
    layer: str = ""
    
    # Solution
    current_a: float = 0.0
    power_w: float = 0.0
    voltage_drop: float = 0.0
    
    # Original trace reference
    trace_ref: Optional[TraceSegment] = None


@dataclass
class CurrentDistributionResult:
    """Results of current distribution analysis."""
    # Per-segment results
    segment_currents: Dict[str, float] = field(default_factory=dict)  # segment_id -> current (A)
    segment_power: Dict[str, float] = field(default_factory=dict)     # segment_id -> power (W)
    segment_voltage_drop: Dict[str, float] = field(default_factory=dict)
    
    # Per-node results
    node_voltages: Dict[int, float] = field(default_factory=dict)     # node_id -> voltage (V)
    
    # Summary
    total_power_w: float = 0.0
    max_current_a: float = 0.0
    max_current_segment: str = ""
    max_temp_rise_estimate: float = 0.0
    
    # Trace-level results (aggregated from segments)
    trace_currents: Dict[str, float] = field(default_factory=dict)    # trace_id -> current
    trace_power: Dict[str, float] = field(default_factory=dict)       # trace_id -> power
    
    # Warnings
    warnings: List[str] = field(default_factory=list)


class CurrentDistributionSolver:
    """
    Solves for current distribution in PCB copper network.
    
    Uses Modified Nodal Analysis (MNA):
    - Builds conductance matrix G from trace/via resistances
    - Applies current sources/sinks at injection points
    - Solves G·V = I for node voltages
    - Computes currents from voltage differences
    """
    
    def __init__(self, pcb_data: PCBData):
        """Initialize solver with PCB data."""
        self.pcb_data = pcb_data
        self.logger = get_logger()
        
        # Network model
        self.nodes: Dict[int, ElectricalNode] = {}
        self.segments: Dict[str, ElectricalSegment] = {}
        
        # Mapping
        self.position_to_node: Dict[Tuple[float, float, str], int] = {}
        self.trace_to_segments: Dict[str, List[str]] = defaultdict(list)
        
        # Solver matrices
        self.G_matrix: Optional[sparse.csr_matrix] = None
        self.I_vector: Optional[np.ndarray] = None
        
        self._node_counter = 0
        self._segment_counter = 0
    
    @timed_function("current_network_build")
    def build_network(self, copper_thickness: Dict[str, float],
                     include_ac_effects: bool = False,
                     frequency_hz: float = 0.0,
                     via_plating_thickness_m: float = 25e-6,
                     net_code_filter: Optional[int] = None):
        """
        Build electrical network from PCB geometry.
        
        Args:
            copper_thickness: Dict mapping layer name to thickness in meters
            include_ac_effects: Whether to include skin effect
            frequency_hz: Frequency for skin effect calculation
        """
        self.logger.info("Building electrical network from PCB geometry")
        
        self.nodes.clear()
        self.segments.clear()
        self.position_to_node.clear()
        self.trace_to_segments.clear()
        self._node_counter = 0
        self._segment_counter = 0
        
        copper = MaterialsDatabase.CONDUCTORS['COPPER']
        
        # Process each net separately
        nets_processed = set()
        
        # Build network from traces
        for trace in self.pcb_data.traces:
            if net_code_filter is not None and getattr(trace, 'net_code', None) != net_code_filter:
                continue
            net_code = trace.net_code
            layer = trace.layer
            
            # Get copper thickness for this layer
            thickness = copper_thickness.get(layer, 35e-6)  # Default 1oz
            
            # Calculate resistivity (with temperature coefficient if needed)
            resistivity = copper.electrical_resistivity
            
            # Include skin effect if requested
            if include_ac_effects and frequency_hz > 0:
                skin_depth = self._calculate_skin_depth(resistivity, frequency_hz)
                effective_thickness = min(thickness, 2 * skin_depth)
            else:
                effective_thickness = thickness
            
            # Create/get nodes for start and end points
            node1_id = self._get_or_create_node(
                trace.start, layer, trace.net_code, trace.net_name
            )
            node2_id = self._get_or_create_node(
                trace.end, layer, trace.net_code, trace.net_name
            )
            
            # Calculate resistance
            length_m = trace.length_mm / 1000
            width_m = trace.width_mm / 1000
            
            if width_m > 0 and effective_thickness > 0 and length_m > 0:
                resistance = resistivity * length_m / (width_m * effective_thickness)
            else:
                resistance = 1e-6  # Very small resistance to avoid division issues
            
            # Create segment
            self._segment_counter += 1
            seg_id = f"SEG_{self._segment_counter}"
            
            segment = ElectricalSegment(
                segment_id=seg_id,
                node1_id=node1_id,
                node2_id=node2_id,
                resistance_ohm=resistance,
                length_m=length_m,
                width_m=width_m,
                thickness_m=effective_thickness,
                layer=layer,
                trace_ref=trace
            )
            
            self.segments[seg_id] = segment
            self.trace_to_segments[trace.segment_id].append(seg_id)
            
            # Update node connections
            self.nodes[node1_id].connected_segments.append(seg_id)
            self.nodes[node2_id].connected_segments.append(seg_id)
            
            nets_processed.add(net_code)
        
        # Add vias as connections between layers
        self._add_via_connections(copper_thickness, copper, via_plating_thickness_m, net_code_filter)
        
        self.logger.info(f"Network built: {len(self.nodes)} nodes, {len(self.segments)} segments")
    
    def _get_or_create_node(self, position: Point2D, layer: str, 
                           net_code: int, net_name: str) -> int:
        """Get existing node or create new one at position."""
        # Round position to avoid floating point comparison issues
        key = (round(position.x, 4), round(position.y, 4), layer, int(net_code))
        
        if key in self.position_to_node:
            return self.position_to_node[key]
        
        # Create new node
        node = ElectricalNode(
            node_id=self._node_counter,
            position=position,
            layer=layer,
            net_code=net_code,
            net_name=net_name
        )
        
        self.nodes[self._node_counter] = node
        self.position_to_node[key] = self._node_counter
        self._node_counter += 1
        
        return node.node_id
    
    def _add_via_connections(self, copper_thickness: Dict[str, float], copper_material,
                             via_plating_thickness_m: float = 25e-6,
                             net_code_filter: Optional[int] = None):
        """Add electrical connections through vias."""
        for via in self.pcb_data.vias:
            if net_code_filter is not None and getattr(via, 'net_code', None) != net_code_filter:
                continue
            # Find nodes at via position on different layers
            via_nodes = []
            
            for layer in self.pcb_data.copper_layers:
                key = (round(via.position.x, 4), round(via.position.y, 4), layer, int(getattr(via, 'net_code', 0)))
                if key in self.position_to_node:
                    via_nodes.append((layer, self.position_to_node[key]))
            
            # If no existing nodes, create them
            if len(via_nodes) < 2:
                for layer in self.pcb_data.copper_layers:
                    key = (round(via.position.x, 4), round(via.position.y, 4), layer, int(getattr(via, 'net_code', 0)))
                    if key not in self.position_to_node:
                        node_id = self._get_or_create_node(
                            via.position, layer, via.net_code, via.net_name
                        )
                        via_nodes.append((layer, node_id))
            
            # Connect adjacent layers through via barrel
            via_nodes.sort(key=lambda x: self.pcb_data.copper_layers.index(x[0]) 
                          if x[0] in self.pcb_data.copper_layers else 0)
            
            for i in range(len(via_nodes) - 1):
                layer1, node1_id = via_nodes[i]
                layer2, node2_id = via_nodes[i + 1]
                
                # Calculate via barrel resistance
                # Resistance of hollow cylinder
                plating_thickness = float(via_plating_thickness_m)  # Via barrel copper plating thickness
                outer_radius = via.drill_mm / 2000 + plating_thickness
                inner_radius = via.drill_mm / 2000
                
                # Estimate via length between layers
                layer1_idx = self.pcb_data.copper_layers.index(layer1) if layer1 in self.pcb_data.copper_layers else 0
                layer2_idx = self.pcb_data.copper_layers.index(layer2) if layer2 in self.pcb_data.copper_layers else 0
                
                # Approximate via length
                board_thickness_m = float(getattr(self.pcb_data, 'board_thickness_mm', 1.6)) / 1000.0
                n_gaps = max(1, len(getattr(self.pcb_data, 'copper_layers', [])) - 1)
                gap_m = board_thickness_m / n_gaps
                via_length = gap_m * abs(layer2_idx - layer1_idx)
                
                cross_section = math.pi * (outer_radius**2 - inner_radius**2)
                
                if cross_section > 0 and via_length > 0:
                    via_resistance = copper_material.electrical_resistivity * via_length / cross_section
                else:
                    via_resistance = 1e-6
                
                # Create segment for via
                self._segment_counter += 1
                seg_id = f"VIA_{via.via_id}_{i}"
                
                segment = ElectricalSegment(
                    segment_id=seg_id,
                    node1_id=node1_id,
                    node2_id=node2_id,
                    resistance_ohm=via_resistance,
                    length_m=via_length,
                    layer=f"{layer1}-{layer2}"
                )
                
                self.segments[seg_id] = segment
                self.nodes[node1_id].connected_segments.append(seg_id)
                self.nodes[node1_id].connected_vias.append(via.via_id)
                self.nodes[node2_id].connected_segments.append(seg_id)
                self.nodes[node2_id].connected_vias.append(via.via_id)
    
    def _calculate_skin_depth(self, resistivity: float, frequency: float) -> float:
        """Calculate skin depth for AC resistance correction."""
        mu_0 = 4 * math.pi * 1e-7  # Permeability of free space
        mu_r = 1.0  # Relative permeability of copper
        
        if frequency <= 0:
            return float('inf')
        
        return math.sqrt(resistivity / (math.pi * frequency * mu_0 * mu_r))
    
    @timed_function("current_matrix_build")
    def _build_conductance_matrix(self):
        """Build the conductance matrix G for nodal analysis."""
        n_nodes = len(self.nodes)
        
        if n_nodes == 0:
            self.logger.warning("No nodes in network")
            return
        
        # Use sparse matrix for efficiency
        rows = []
        cols = []
        data = []
        
        for seg_id, segment in self.segments.items():
            if segment.resistance_ohm <= 0:
                continue
            
            g = 1.0 / segment.resistance_ohm  # Conductance
            n1 = segment.node1_id
            n2 = segment.node2_id
            
            # Stamp conductance into matrix
            # G[n1,n1] += g
            # G[n2,n2] += g
            # G[n1,n2] -= g
            # G[n2,n1] -= g
            
            rows.extend([n1, n2, n1, n2])
            cols.extend([n1, n2, n2, n1])
            data.extend([g, g, -g, -g])
        
        self.G_matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_nodes, n_nodes),
            dtype=np.float64
        )
    
    def apply_current_sources(self, injection_points: List[CurrentInjectionPoint]):
        """Apply current injection/extraction at specified points."""
        n_nodes = len(self.nodes)
        self.I_vector = np.zeros(n_nodes, dtype=np.float64)
        
        for point in injection_points:
            # Find nearest node to injection point
            best_node = None
            best_distance = float('inf')
            
            for node_id, node in self.nodes.items():
                if node.net_name == point.net_name or point.net_name == "":
                    dist = math.sqrt(
                        (node.position.x - point.x_mm)**2 + 
                        (node.position.y - point.y_mm)**2
                    )
                    # Also check layer if specified
                    if point.layer and node.layer != point.layer:
                        continue
                    
                    if dist < best_distance:
                        best_distance = dist
                        best_node = node_id
            
            if best_node is not None:
                self.nodes[best_node].current_injection = point.current_a
                self.I_vector[best_node] = point.current_a
                self.logger.debug(f"Applied {point.current_a}A at node {best_node} ({point.description})")
            else:
                self.logger.warning(f"Could not find node for injection point: {point.description}")
    
    def _set_reference_node(self):
        """Set a reference (ground) node for the solver."""
        # Find a node with negative current (sink) to use as reference
        ref_node = None
        
        for node_id, node in self.nodes.items():
            if node.current_injection < 0:
                ref_node = node_id
                break
        
        # If no sink found, use first node
        if ref_node is None and self.nodes:
            ref_node = next(iter(self.nodes.keys()))
        
        if ref_node is not None:
            self.nodes[ref_node].is_reference = True
            self.nodes[ref_node].voltage = 0.0
        
        return ref_node
    
    @timed_function("current_solve")
    def solve(self, injection_points: List[CurrentInjectionPoint],
             overrides: List[TraceCurrentOverride] = None) -> CurrentDistributionResult:
        """
        Solve for current distribution in the network.
        
        Args:
            injection_points: List of current injection/extraction points
            overrides: Optional manual current overrides for specific segments
        
        Returns:
            CurrentDistributionResult with currents and power for each segment
        """
        result = CurrentDistributionResult()
        
        if not self.nodes or not self.segments:
            self.logger.warning("Empty network - cannot solve")
            return result
        
        self.logger.info("Solving current distribution...")
        
        # Build conductance matrix
        self._build_conductance_matrix()
        
        # Apply current sources
        self.apply_current_sources(injection_points)
        
        # Check current balance
        total_current = sum(self.I_vector)
        if abs(total_current) > 1e-9:
            self.logger.warning(f"Current not balanced: {total_current}A net injection")
            result.warnings.append(f"Current not balanced: {total_current:.6f}A net injection")
        
        # Set reference node
        ref_node = self._set_reference_node()
        
        # Solve G·V = I
        # Modify matrix to fix reference node voltage
        if ref_node is not None:
            G_modified = self.G_matrix.tolil()
            G_modified[ref_node, :] = 0
            G_modified[ref_node, ref_node] = 1
            self.I_vector[ref_node] = 0
            G_modified = G_modified.tocsr()
        else:
            G_modified = self.G_matrix
        
        try:
            # Use sparse solver
            voltages = sparse_linalg.spsolve(G_modified, self.I_vector)
            
            # Store voltages
            for node_id, node in self.nodes.items():
                node.voltage = voltages[node_id]
                result.node_voltages[node_id] = voltages[node_id]
            
        except Exception as e:
            self.logger.error(f"Solver failed: {e}")
            result.warnings.append(f"Solver failed: {e}")
            return result
        
        # Calculate currents from voltage differences
        for seg_id, segment in self.segments.items():
            v1 = self.nodes[segment.node1_id].voltage
            v2 = self.nodes[segment.node2_id].voltage
            
            segment.voltage_drop = v1 - v2
            
            if segment.resistance_ohm > 0:
                segment.current_a = segment.voltage_drop / segment.resistance_ohm
            else:
                segment.current_a = 0.0
            
            segment.power_w = segment.current_a ** 2 * segment.resistance_ohm
            
            result.segment_currents[seg_id] = segment.current_a
            result.segment_power[seg_id] = segment.power_w
            result.segment_voltage_drop[seg_id] = segment.voltage_drop
        
        # Apply manual overrides if specified
        if overrides:
            self._apply_overrides(overrides, result)
        
        # Aggregate to trace level
        self._aggregate_to_traces(result)
        
        # Compute summary statistics
        self._compute_summary(result)
        
        self.logger.info(f"Solution complete: Total power = {result.total_power_w:.3f}W, "
                        f"Max current = {result.max_current_a:.3f}A")
        
        return result
    
    def _apply_overrides(self, overrides: List[TraceCurrentOverride], 
                        result: CurrentDistributionResult):
        """Apply manual current overrides."""
        for override in overrides:
            # Find segment nearest to override location
            best_seg = None
            best_distance = float('inf')
            
            for seg_id, segment in self.segments.items():
                if segment.layer != override.layer:
                    continue
                
                # Check if segment is near override line
                if segment.trace_ref:
                    trace = segment.trace_ref
                    mid_x = (trace.start.x + trace.end.x) / 2
                    mid_y = (trace.start.y + trace.end.y) / 2
                    
                    override_mid_x = (override.start_x + override.end_x) / 2
                    override_mid_y = (override.start_y + override.end_y) / 2
                    
                    dist = math.sqrt((mid_x - override_mid_x)**2 + (mid_y - override_mid_y)**2)
                    
                    if dist < best_distance:
                        best_distance = dist
                        best_seg = seg_id
            
            if best_seg and best_distance < 1.0:  # Within 1mm
                old_current = self.segments[best_seg].current_a
                self.segments[best_seg].current_a = override.current_a
                self.segments[best_seg].power_w = override.current_a ** 2 * self.segments[best_seg].resistance_ohm
                
                result.segment_currents[best_seg] = override.current_a
                result.segment_power[best_seg] = self.segments[best_seg].power_w
                
                self.logger.debug(f"Override applied to {best_seg}: {old_current:.3f}A -> {override.current_a:.3f}A")
    
    def _aggregate_to_traces(self, result: CurrentDistributionResult):
        """Aggregate segment results to trace level."""
        for trace_id, seg_ids in self.trace_to_segments.items():
            if seg_ids:
                # Use average current magnitude
                currents = [abs(result.segment_currents.get(sid, 0)) for sid in seg_ids]
                powers = [result.segment_power.get(sid, 0) for sid in seg_ids]
                
                result.trace_currents[trace_id] = sum(currents) / len(currents) if currents else 0
                result.trace_power[trace_id] = sum(powers)
    
    def _compute_summary(self, result: CurrentDistributionResult):
        """Compute summary statistics."""
        if result.segment_currents:
            result.max_current_a = max(abs(i) for i in result.segment_currents.values())
            result.max_current_segment = max(
                result.segment_currents.keys(),
                key=lambda k: abs(result.segment_currents[k])
            )
        
        result.total_power_w = sum(result.segment_power.values())
        
        # Estimate max temperature rise (rough approximation)
        # Using simplified trace heating formula
        if result.max_current_a > 0:
            max_seg = self.segments.get(result.max_current_segment)
            if max_seg:
                # ΔT ≈ I² × R × θ (simplified)
                # Using rough thermal resistance estimate
                theta_estimate = 50  # °C/W rough estimate for PCB trace
                result.max_temp_rise_estimate = max_seg.power_w * theta_estimate
    
    def get_segment_by_position(self, x: float, y: float, layer: str) -> Optional[str]:
        """Find segment nearest to a position."""
        best_seg = None
        best_distance = float('inf')
        
        for seg_id, segment in self.segments.items():
            if segment.layer != layer:
                continue
            
            if segment.trace_ref:
                trace = segment.trace_ref
                # Distance to line segment
                dist = self._point_to_segment_distance(
                    x, y,
                    trace.start.x, trace.start.y,
                    trace.end.x, trace.end.y
                )
                
                if dist < best_distance:
                    best_distance = dist
                    best_seg = seg_id
        
        return best_seg if best_distance < 2.0 else None  # Within 2mm
    
    def _point_to_segment_distance(self, px: float, py: float,
                                   x1: float, y1: float,
                                   x2: float, y2: float) -> float:
        """Calculate distance from point to line segment."""
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)
        
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)


__all__ = [
    'ElectricalNode',
    'ElectricalSegment',
    'CurrentDistributionResult',
    'CurrentDistributionSolver',
]
