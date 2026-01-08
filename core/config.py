"""
TVAC Thermal Analyzer - Configuration Management
=================================================
Configuration data structures and JSON serialization.

Author: Space Electronics Thermal Analysis Tool
Version: 2.0.0
"""

import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime


@dataclass
class CurrentPath:
    """Current flow path for Joule heating analysis.

    This is evolving from a net-to-net concept toward an unambiguous terminal-based definition.
    - Legacy: source_net/sink_net
    - New: source_ref/source_pad and sink_ref/sink_pad (KiCad footprint reference + pad name)
    """
    path_id: str = ""

    # New (preferred): explicit terminals (REF:PAD)
    source_ref: str = ""   # e.g. "J1"
    source_pad: str = ""   # e.g. "1"
    sink_ref: str = ""     # e.g. "U3"
    sink_pad: str = ""     # e.g. "5"

    # Legacy (kept for backward compatibility)
    source_net: str = ""   # Net where current enters (e.g., VIN, +12V)
    sink_net: str = ""     # Net where current exits (e.g., GND, PGND)

    current_a: float = 1.0   # Current magnitude in Amps
    description: str = ""


# Keep for backwards compatibility but mark as legacy
@dataclass
class CurrentInjectionPoint:
    """Current injection point for Joule heating (legacy)."""
    point_id: str = ""
    net_name: str = ""
    x_mm: float = 0.0
    y_mm: float = 0.0
    layer: str = "F.Cu"
    current_a: float = 0.0
    description: str = ""

@dataclass
class TraceCurrentOverride:
    """Per-trace current override (legacy helper).

    Some current workflows want to force a specific current through a specific segment.
    Kept to preserve compatibility with older solver code paths.
    """
    trace_id: str = ""
    current_a: float = 0.0
    direction: int = 1  # +1 or -1



@dataclass
class ComponentPowerConfig:
    """Power configuration for a component."""
    reference: str = ""
    power_w: float = 0.0
    source: str = "manual"  # manual, database, calculated
    theta_ja_override: Optional[float] = None
    theta_jc_override: Optional[float] = None


@dataclass
class MountingPointConfig:
    """Thermal mounting point configuration."""
    point_id: str = ""
    x_mm: float = 0.0
    y_mm: float = 0.0
    diameter_mm: float = 3.2
    contact_type: str = "conductive"  # conductive, isolative
    interface_material: str = "direct"
    fixed_temp_c: Optional[float] = None
    thermal_resistance: float = 0.0  # K/W


@dataclass
class HeatsinkConfig:
    """Heatsink configuration."""
    heatsink_id: str = ""
    material: str = "ALUMINUM_6061"
    polygon_points: List[Tuple[float, float]] = field(default_factory=list)
    thickness_mm: float = 3.0
    fin_count: int = 0
    fin_height_mm: float = 0.0
    fin_spacing_mm: float = 0.0
    emissivity_override: Optional[float] = None


@dataclass
class StackupLayerConfig:
    """Single layer in board stackup."""
    name: str = ""
    layer_type: str = "dielectric"  # copper, dielectric
    thickness_um: float = 35.0
    material: str = "FR4"


@dataclass
class StackupConfig:
    """Complete board stackup."""
    total_thickness_mm: float = 1.6
    substrate_material: str = "FR4"
    surface_finish: str = "HASL"
    solder_mask_color: str = "green"
    layers: List[StackupLayerConfig] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.layers:
            # Default 4-layer stackup
            self.layers = [
                StackupLayerConfig("F.Cu", "copper", 35, "Copper"),
                StackupLayerConfig("Prepreg1", "dielectric", 200, "FR4"),
                StackupLayerConfig("In1.Cu", "copper", 35, "Copper"),
                StackupLayerConfig("Core", "dielectric", 1000, "FR4"),
                StackupLayerConfig("In2.Cu", "copper", 35, "Copper"),
                StackupLayerConfig("Prepreg2", "dielectric", 200, "FR4"),
                StackupLayerConfig("B.Cu", "copper", 35, "Copper"),
            ]
    
    def get_copper_layers(self) -> List[StackupLayerConfig]:
        return [l for l in self.layers if l.layer_type == "copper"]
    
    def get_layer_thickness_um(self, name: str) -> float:
        for layer in self.layers:
            if layer.name == name:
                return layer.thickness_um
        return 35.0


@dataclass
class SimulationConfig:
    """Simulation parameters."""
    resolution_mm: float = 0.5
    use_adaptive_mesh: bool = True
    adaptive_factor: float = 2.0
    simulation_3d: bool = True
    include_radiation: bool = True
    include_ac_effects: bool = False
    mode: str = "steady_state"  # steady_state, transient
    heat_source_mode: str = "component_power"  # component_power, current_injection
    duration_s: float = 600.0
    timestep_s: float = 0.5
    output_interval_s: float = 1.0
    ambient_temp_c: float = 25.0
    chamber_wall_temp_c: float = 25.0
    initial_board_temp_c: float = 25.0
    convergence: float = 1e-6
    max_iterations: int = 10000
    use_gpu: bool = False
    num_threads: int = 0  # 0 = auto

    # --- Electrical / mounting parameters (industrial defaults) ---
    mounting_box_temp_c: float = 25.0  # Mounting fixture temperature (box / cold plate)
    copper_resistivity_ohm_m: float = 1.724e-8  # Copper resistivity at 20°C
    copper_tempco_per_c: float = 0.00393  # Copper temperature coefficient (1/°C)
    via_plating_thickness_um: float = 25.0  # Via barrel copper plating thickness
    plane_grid_mm: float = 2.0  # Resistive-plane discretization grid size

    # --- Validation / debug outputs ---
    validation_mode: bool = False
    validation_top_n: int = 20


@dataclass
class LayerMappingConfig:
    """Layer mapping for heatsinks and mounting."""
    heatsink_layer: str = "User.1"
    mounting_layer: str = "User.2"


@dataclass
class ThermalAnalysisConfig:
    """Complete thermal analysis configuration."""
    version: str = "2.0.0"
    created: str = ""
    modified: str = ""
    
    stackup: StackupConfig = field(default_factory=StackupConfig)
    layer_mapping: LayerMappingConfig = field(default_factory=LayerMappingConfig)
    current_paths: List[CurrentPath] = field(default_factory=list)  # New simplified model
    current_injection_points: List[CurrentInjectionPoint] = field(default_factory=list)  # Legacy
    component_power: List[ComponentPowerConfig] = field(default_factory=list)
    mounting_points: List[MountingPointConfig] = field(default_factory=list)
    heatsinks: List[HeatsinkConfig] = field(default_factory=list)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    
    def __post_init__(self):
        if not self.created:
            self.created = datetime.now().isoformat()
        self.modified = datetime.now().isoformat()
    
    def get_component_power(self, reference: str) -> Optional[float]:
        """Get power for a component."""
        for cp in self.component_power:
            if cp.reference == reference:
                return cp.power_w
        return None
    
    def set_component_power(self, reference: str, power: float, source: str = "manual"):
        """Set power for a component."""
        for cp in self.component_power:
            if cp.reference == reference:
                cp.power_w = power
                cp.source = source
                return
        
        self.component_power.append(ComponentPowerConfig(
            reference=reference, power_w=power, source=source
        ))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        self.modified = datetime.now().isoformat()
        
        def convert(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: convert(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            else:
                return obj
        
        return convert(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThermalAnalysisConfig':
        """Create from dictionary."""
        config = cls()

        def _filter_kwargs(dc_type, d: Dict[str, Any]) -> Dict[str, Any]:
            """Filter dict keys to those accepted by the dataclass constructor."""
            allowed = getattr(dc_type, '__dataclass_fields__', {}).keys()
            return {k: v for k, v in (d or {}).items() if k in allowed}

        
        if 'version' in data:
            config.version = data['version']
        if 'created' in data:
            config.created = data['created']
        
        # Stackup
        if 'stackup' in data:
            s = data['stackup']
            config.stackup = StackupConfig(
                total_thickness_mm=s.get('total_thickness_mm', 1.6),
                substrate_material=s.get('substrate_material', 'FR4'),
                surface_finish=s.get('surface_finish', 'HASL'),
                solder_mask_color=s.get('solder_mask_color', 'green'),
                layers=[StackupLayerConfig(**l) for l in s.get('layers', [])]
            )
        
        # Layer mapping
        if 'layer_mapping' in data:
            config.layer_mapping = LayerMappingConfig(**_filter_kwargs(LayerMappingConfig, data['layer_mapping']))
        
        # Component power
        if 'component_power' in data:
            config.component_power = [ComponentPowerConfig(**_filter_kwargs(ComponentPowerConfig, cp)) for cp in data['component_power']]
        
        # Mounting points
        if 'mounting_points' in data:
            config.mounting_points = [MountingPointConfig(**_filter_kwargs(MountingPointConfig, mp)) for mp in data['mounting_points']]
        
        # Heatsinks
        if 'heatsinks' in data:
            config.heatsinks = [HeatsinkConfig(**_filter_kwargs(HeatsinkConfig, hs)) for hs in data['heatsinks']]
        
        # Current injection
        if 'current_injection_points' in data:
            config.current_injection_points = [CurrentInjectionPoint(**_filter_kwargs(CurrentInjectionPoint, p)) for p in data['current_injection_points']]
        
        # Current paths (new simplified model)
        if 'current_paths' in data:
            config.current_paths = [CurrentPath(**_filter_kwargs(CurrentPath, p)) for p in data['current_paths']]

        # Simulation
        if 'simulation' in data:
            config.simulation = SimulationConfig(**_filter_kwargs(SimulationConfig, data['simulation']))
        
        return config


class ConfigManager:
    """Manages loading and saving configuration."""
    
    def __init__(self, pcb_path: str = ""):
        self.pcb_path = Path(pcb_path) if pcb_path else None
        self.config: Optional[ThermalAnalysisConfig] = None
        self._config_path: Optional[Path] = None
        
        if self.pcb_path and self.pcb_path.exists():
            self._config_path = self.pcb_path.with_name(
                self.pcb_path.stem + "_tvac_thermal_config.json"
            )
    
    def get_config(self) -> ThermalAnalysisConfig:
        """Get configuration, loading from file if exists."""
        if self.config:
            return self.config
        
        if self._config_path and self._config_path.exists():
            try:
                with open(self._config_path, 'r') as f:
                    data = json.load(f)
                self.config = ThermalAnalysisConfig.from_dict(data)
            except Exception as e:
                print(f"Failed to load config: {e}")
                self.config = ThermalAnalysisConfig()
        else:
            self.config = ThermalAnalysisConfig()
        
        return self.config
    
    def save(self) -> bool:
        """Save configuration to file."""
        if not self.config:
            return False
        
        if not self._config_path:
            return False
        
        try:
            data = self.config.to_dict()
            with open(self._config_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Failed to save config: {e}")
            return False
    
    def export(self, path: str) -> bool:
        """Export configuration to specified path."""
        if not self.config:
            return False
        
        try:
            data = self.config.to_dict()
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception:
            return False
    
    def import_config(self, path: str) -> bool:
        """Import configuration from file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.config = ThermalAnalysisConfig.from_dict(data)
            return True
        except Exception:
            return False


__all__ = [
    'ThermalAnalysisConfig', 'ConfigManager',
    'ComponentPowerConfig', 'HeatsinkConfig', 'MountingPointConfig',
    'StackupConfig', 'StackupLayerConfig', 'SimulationConfig',
    'LayerMappingConfig', 'CurrentPath', 'CurrentInjectionPoint', 'TraceCurrentOverride',
]
