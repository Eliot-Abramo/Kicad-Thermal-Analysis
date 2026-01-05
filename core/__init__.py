"""
TVAC Thermal Analyzer - Core Module
===================================
Core data structures, configuration, and PCB extraction.
"""

from .constants import (
    PhysicalConstants,
    ThermalMaterial,
    MaterialsDatabase,
    ComponentThermalData,
    ComponentThermalDatabase,
    PCBLayerDefaults,
    SimulationDefaults,
)

from .config import (
    CurrentInjectionPoint,
    TraceCurrentOverride,
    ComponentPowerConfig,
    MountingPoint,
    HeatsinkConfig,
    PCBStackupConfig,
    SimulationParameters,
    ThermalAnalysisConfig,
    ConfigManager,
)

from .pcb_extractor import (
    Point2D,
    Point3D,
    TraceSegment,
    Via,
    CopperPour,
    Pad,
    Component,
    MountingHole,
    BoardOutline,
    UserLayerShape,
    PCBData,
    PCBExtractor,
)

__all__ = [
    # Constants
    'PhysicalConstants',
    'ThermalMaterial',
    'MaterialsDatabase',
    'ComponentThermalData',
    'ComponentThermalDatabase',
    'PCBLayerDefaults',
    'SimulationDefaults',
    # Config
    'CurrentInjectionPoint',
    'TraceCurrentOverride',
    'ComponentPowerConfig',
    'MountingPoint',
    'HeatsinkConfig',
    'PCBStackupConfig',
    'SimulationParameters',
    'ThermalAnalysisConfig',
    'ConfigManager',
    # PCB Extractor
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
