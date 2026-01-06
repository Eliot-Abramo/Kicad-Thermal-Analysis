"""
TVAC Thermal Analyzer - Core Module
===================================
Core data structures, configuration, and PCB extraction.
"""

from .pcb_extractor import (
    PCBExtractor, PCBData, Component, Point2D,
    TraceSegment, Via, CopperPour, Pad, 
    MountingHole, BoardOutline, UserLayerShape
)

from .config import (
    ThermalAnalysisConfig, ConfigManager,
    ComponentPowerConfig, HeatsinkConfig, MountingPointConfig,
    StackupConfig, SimulationConfig, LayerMappingConfig
)

from .constants import (
    PhysicalConstants, MaterialProperties, MaterialsDatabase,
    PackageThermalData, ComponentThermalDatabase, SimulationDefaults
)

__all__ = [
    # PCB Extraction
    'PCBExtractor', 'PCBData', 'Component', 'Point2D',
    'TraceSegment', 'Via', 'CopperPour', 'Pad',
    'MountingHole', 'BoardOutline', 'UserLayerShape',
    
    # Configuration
    'ThermalAnalysisConfig', 'ConfigManager',
    'ComponentPowerConfig', 'HeatsinkConfig', 'MountingPointConfig',
    'StackupConfig', 'SimulationConfig', 'LayerMappingConfig',
    
    # Constants & Materials
    'PhysicalConstants', 'MaterialProperties', 'MaterialsDatabase',
    'PackageThermalData', 'ComponentThermalDatabase', 'SimulationDefaults',
]
