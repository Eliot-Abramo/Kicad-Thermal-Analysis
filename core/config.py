"""
TVAC Thermal Analyzer - Configuration Management
================================================
Handles persistent configuration, project settings, and user preferences.

Author: Space Electronics Thermal Analysis Tool
Version: 1.0.0
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import hashlib


@dataclass
class CurrentInjectionPoint:
    """Defines a current injection/extraction point."""
    point_id: str
    net_name: str
    x_mm: float
    y_mm: float
    layer: str
    current_a: float  # Positive = injection, Negative = extraction
    description: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CurrentInjectionPoint':
        return cls(**data)


@dataclass
class TraceCurrentOverride:
    """Manual current override for a specific trace segment."""
    segment_id: str
    net_name: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    layer: str
    current_a: float
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TraceCurrentOverride':
        return cls(**data)


@dataclass
class ComponentPowerConfig:
    """Power dissipation configuration for a component."""
    reference: str  # Component reference (e.g., U1, R5)
    power_w: float  # Power dissipation in Watts
    source: str = "manual"  # "schematic", "manual", "database"
    theta_ja_override: Optional[float] = None
    theta_jc_override: Optional[float] = None
    thermal_mass_override: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ComponentPowerConfig':
        return cls(**data)


@dataclass
class MountingPoint:
    """Thermal mounting point definition."""
    point_id: str
    x_mm: float
    y_mm: float
    hole_diameter_mm: float
    contact_type: str  # "conductive", "isolative"
    interface_material: str  # Key from MaterialsDatabase.THERMAL_INTERFACE
    interface_temp_c: Optional[float] = None  # Fixed temperature if set
    contact_area_mm2: Optional[float] = None  # Override calculated area
    thermal_resistance_k_per_w: Optional[float] = None  # Override calculated
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MountingPoint':
        return cls(**data)


@dataclass
class HeatsinkConfig:
    """Heatsink configuration from User1 layer shapes."""
    heatsink_id: str
    material: str  # Key from MaterialsDatabase.HEATSINK_MATERIALS
    polygon_points: List[Tuple[float, float]]  # (x, y) in mm
    thickness_mm: float = 3.0
    fin_count: int = 0  # For finned heatsinks
    fin_height_mm: float = 0.0
    fin_spacing_mm: float = 0.0
    connection_type: str = "bolted_paste"  # "bolted_paste", "bonded", "floating"
    emissivity_override: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'HeatsinkConfig':
        return cls(**data)


@dataclass 
class PCBStackupConfig:
    """PCB stackup configuration."""
    total_thickness_mm: float = 1.6
    layer_count: int = 4
    substrate_material: str = "FR4"
    copper_thickness_um: Dict[str, float] = field(default_factory=lambda: {
        "F.Cu": 35.0,
        "In1.Cu": 35.0,
        "In2.Cu": 35.0,
        "B.Cu": 35.0
    })
    dielectric_thickness_um: List[float] = field(default_factory=lambda: [200.0, 1000.0, 200.0])
    surface_finish: str = "HASL"
    solder_mask_color: str = "GREEN"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PCBStackupConfig':
        return cls(**data)


@dataclass
class SimulationParameters:
    """Simulation control parameters."""
    # Grid settings
    resolution_mm: float = 0.5
    use_adaptive_mesh: bool = True
    adaptive_refinement_factor: float = 2.0
    
    # Simulation type
    simulation_3d: bool = True
    include_radiation: bool = True
    include_ac_effects: bool = False
    ac_frequency_hz: float = 1e6
    
    # Time settings
    simulation_mode: str = "transient"  # "transient", "steady_state"
    duration_s: float = 600.0  # 10 minutes default
    timestep_s: float = 0.5
    output_interval_s: float = 1.0  # How often to save frames
    
    # Boundary conditions
    ambient_temp_c: float = 25.0
    chamber_wall_temp_c: float = 25.0
    initial_board_temp_c: float = 25.0
    
    # Solver settings
    steady_state_tolerance: float = 0.01
    max_iterations: int = 10000
    convergence_criterion: float = 1e-6
    
    # Performance
    num_threads: int = 4
    use_gpu: bool = False
    memory_limit_mb: int = 4096
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SimulationParameters':
        return cls(**data)


@dataclass
class ThermalAnalysisConfig:
    """Complete thermal analysis configuration for a PCB project."""
    # Metadata
    config_version: str = "1.0.0"
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_date: str = field(default_factory=lambda: datetime.now().isoformat())
    pcb_file_hash: str = ""
    
    # PCB stackup
    stackup: PCBStackupConfig = field(default_factory=PCBStackupConfig)
    
    # Current definition
    current_injection_points: List[CurrentInjectionPoint] = field(default_factory=list)
    trace_current_overrides: List[TraceCurrentOverride] = field(default_factory=list)
    
    # Component power
    component_power: List[ComponentPowerConfig] = field(default_factory=list)
    
    # Thermal interface
    mounting_points: List[MountingPoint] = field(default_factory=list)
    heatsinks: List[HeatsinkConfig] = field(default_factory=list)
    
    # Simulation parameters
    simulation: SimulationParameters = field(default_factory=SimulationParameters)
    
    # Custom materials (user-added)
    custom_materials: Dict[str, Dict] = field(default_factory=dict)
    
    # Custom component thermal data
    custom_component_thermal: Dict[str, Dict] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Serialize entire config to dictionary."""
        return {
            'config_version': self.config_version,
            'created_date': self.created_date,
            'modified_date': self.modified_date,
            'pcb_file_hash': self.pcb_file_hash,
            'stackup': self.stackup.to_dict(),
            'current_injection_points': [p.to_dict() for p in self.current_injection_points],
            'trace_current_overrides': [t.to_dict() for t in self.trace_current_overrides],
            'component_power': [c.to_dict() for c in self.component_power],
            'mounting_points': [m.to_dict() for m in self.mounting_points],
            'heatsinks': [h.to_dict() for h in self.heatsinks],
            'simulation': self.simulation.to_dict(),
            'custom_materials': self.custom_materials,
            'custom_component_thermal': self.custom_component_thermal,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ThermalAnalysisConfig':
        """Deserialize from dictionary."""
        config = cls()
        config.config_version = data.get('config_version', '1.0.0')
        config.created_date = data.get('created_date', datetime.now().isoformat())
        config.modified_date = data.get('modified_date', datetime.now().isoformat())
        config.pcb_file_hash = data.get('pcb_file_hash', '')
        
        if 'stackup' in data:
            config.stackup = PCBStackupConfig.from_dict(data['stackup'])
        
        config.current_injection_points = [
            CurrentInjectionPoint.from_dict(p) 
            for p in data.get('current_injection_points', [])
        ]
        config.trace_current_overrides = [
            TraceCurrentOverride.from_dict(t) 
            for t in data.get('trace_current_overrides', [])
        ]
        config.component_power = [
            ComponentPowerConfig.from_dict(c) 
            for c in data.get('component_power', [])
        ]
        config.mounting_points = [
            MountingPoint.from_dict(m) 
            for m in data.get('mounting_points', [])
        ]
        config.heatsinks = [
            HeatsinkConfig.from_dict(h) 
            for h in data.get('heatsinks', [])
        ]
        
        if 'simulation' in data:
            config.simulation = SimulationParameters.from_dict(data['simulation'])
        
        config.custom_materials = data.get('custom_materials', {})
        config.custom_component_thermal = data.get('custom_component_thermal', {})
        
        return config
    
    def update_modified_date(self):
        """Update the modification timestamp."""
        self.modified_date = datetime.now().isoformat()


class ConfigManager:
    """Manages loading and saving of thermal analysis configurations."""
    
    CONFIG_FILENAME = "tvac_thermal_config.json"
    
    def __init__(self, pcb_path: Optional[str] = None):
        """Initialize config manager for a PCB project."""
        self.pcb_path = pcb_path
        self.config_path = None
        self.config = ThermalAnalysisConfig()
        
        if pcb_path:
            self._set_config_path(pcb_path)
    
    def _set_config_path(self, pcb_path: str):
        """Set config file path based on PCB file location."""
        pcb_dir = os.path.dirname(os.path.abspath(pcb_path))
        pcb_name = os.path.splitext(os.path.basename(pcb_path))[0]
        self.config_path = os.path.join(pcb_dir, f"{pcb_name}_{self.CONFIG_FILENAME}")
    
    def _compute_pcb_hash(self, pcb_path: str) -> str:
        """Compute hash of PCB file for change detection."""
        try:
            with open(pcb_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def load(self) -> bool:
        """Load configuration from file. Returns True if successful."""
        if not self.config_path or not os.path.exists(self.config_path):
            return False
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.config = ThermalAnalysisConfig.from_dict(data)
            return True
        except Exception as e:
            print(f"Error loading config: {e}")
            return False
    
    def save(self) -> bool:
        """Save configuration to file. Returns True if successful."""
        if not self.config_path:
            return False
        
        try:
            self.config.update_modified_date()
            if self.pcb_path:
                self.config.pcb_file_hash = self._compute_pcb_hash(self.pcb_path)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def has_pcb_changed(self) -> bool:
        """Check if PCB file has changed since config was saved."""
        if not self.pcb_path or not self.config.pcb_file_hash:
            return True
        current_hash = self._compute_pcb_hash(self.pcb_path)
        return current_hash != self.config.pcb_file_hash
    
    def get_config(self) -> ThermalAnalysisConfig:
        """Get current configuration."""
        return self.config
    
    def set_config(self, config: ThermalAnalysisConfig):
        """Set configuration."""
        self.config = config
    
    # Convenience methods for common operations
    def add_current_point(self, point: CurrentInjectionPoint):
        """Add a current injection point."""
        # Remove existing point with same ID
        self.config.current_injection_points = [
            p for p in self.config.current_injection_points 
            if p.point_id != point.point_id
        ]
        self.config.current_injection_points.append(point)
    
    def remove_current_point(self, point_id: str):
        """Remove a current injection point."""
        self.config.current_injection_points = [
            p for p in self.config.current_injection_points 
            if p.point_id != point_id
        ]
    
    def set_component_power(self, reference: str, power_w: float, source: str = "manual"):
        """Set power dissipation for a component."""
        # Remove existing entry
        self.config.component_power = [
            c for c in self.config.component_power 
            if c.reference != reference
        ]
        self.config.component_power.append(
            ComponentPowerConfig(reference=reference, power_w=power_w, source=source)
        )
    
    def get_component_power(self, reference: str) -> Optional[float]:
        """Get power dissipation for a component."""
        for c in self.config.component_power:
            if c.reference == reference:
                return c.power_w
        return None
    
    def add_mounting_point(self, point: MountingPoint):
        """Add a mounting point."""
        self.config.mounting_points = [
            p for p in self.config.mounting_points 
            if p.point_id != point.point_id
        ]
        self.config.mounting_points.append(point)
    
    def add_heatsink(self, heatsink: HeatsinkConfig):
        """Add a heatsink."""
        self.config.heatsinks = [
            h for h in self.config.heatsinks 
            if h.heatsink_id != heatsink.heatsink_id
        ]
        self.config.heatsinks.append(heatsink)
    
    def export_config_summary(self) -> str:
        """Generate a human-readable summary of the configuration."""
        lines = [
            "=" * 60,
            "TVAC Thermal Analysis Configuration Summary",
            "=" * 60,
            f"Created: {self.config.created_date}",
            f"Modified: {self.config.modified_date}",
            "",
            "--- PCB Stackup ---",
            f"  Layers: {self.config.stackup.layer_count}",
            f"  Thickness: {self.config.stackup.total_thickness_mm} mm",
            f"  Substrate: {self.config.stackup.substrate_material}",
            f"  Surface Finish: {self.config.stackup.surface_finish}",
            "",
            "--- Current Injection Points ---",
        ]
        
        for p in self.config.current_injection_points:
            lines.append(f"  {p.point_id}: {p.current_a:+.3f} A on {p.net_name} ({p.description})")
        
        lines.extend([
            "",
            "--- Component Power Dissipation ---",
        ])
        
        for c in self.config.component_power:
            lines.append(f"  {c.reference}: {c.power_w:.3f} W ({c.source})")
        
        lines.extend([
            "",
            "--- Mounting Points ---",
        ])
        
        for m in self.config.mounting_points:
            lines.append(f"  {m.point_id}: ({m.x_mm:.1f}, {m.y_mm:.1f}) mm, {m.contact_type}")
        
        lines.extend([
            "",
            "--- Simulation Parameters ---",
            f"  Resolution: {self.config.simulation.resolution_mm} mm",
            f"  Mode: {self.config.simulation.simulation_mode}",
            f"  Duration: {self.config.simulation.duration_s} s",
            f"  3D Simulation: {self.config.simulation.simulation_3d}",
            f"  Include Radiation: {self.config.simulation.include_radiation}",
            f"  Ambient Temp: {self.config.simulation.ambient_temp_c} °C",
            f"  Chamber Wall Temp: {self.config.simulation.chamber_wall_temp_c} °C",
            "=" * 60,
        ])
        
        return "\n".join(lines)


__all__ = [
    'CurrentInjectionPoint',
    'TraceCurrentOverride',
    'ComponentPowerConfig',
    'MountingPoint',
    'HeatsinkConfig',
    'PCBStackupConfig',
    'SimulationParameters',
    'ThermalAnalysisConfig',
    'ConfigManager',
]
