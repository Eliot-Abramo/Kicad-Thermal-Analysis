"""
TVAC Thermal Analyzer for KiCad
===============================

A professional thermal simulation tool for PCB analysis under thermal-vacuum
(TVAC) conditions. Designed for space electronics and high-reliability applications.

Features:
- 3D thermal finite element analysis
- Current distribution and Joule heating calculation
- Radiation heat transfer modeling (vacuum conditions)
- Real-time heat map visualization
- Professional PDF report generation
- Persistent configuration with project

Author: Space Electronics Thermal Analysis Tool
Version: 1.0.0
License: MIT
KiCad Version: 9.0+
"""

__version__ = "1.0.0"
__author__ = "Space Electronics Thermal Analysis Tool"
__license__ = "MIT"
__kicad_version__ = "9.0"

# Version info
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'stable',
    'build': '20250105',
}

def get_version_string() -> str:
    """Get formatted version string."""
    return f"{VERSION_INFO['major']}.{VERSION_INFO['minor']}.{VERSION_INFO['patch']}"


# Import submodules for convenient access
from . import core
from . import solvers
from . import utils
from . import ui

# Convenience imports
from .core import (
    ThermalAnalysisConfig,
    ConfigManager,
    PCBExtractor,
    PCBData,
    MaterialsDatabase,
    ComponentThermalDatabase,
)

from .solvers import (
    ThermalSimulationResult,
    ThermalResults,
    ThermalSolver,
    ThermalAnalysisEngine,
    CurrentDistributionSolver,
    CurrentDistributionResult,
    MeshGenerator,
    ThermalMesh,
    run_simulation,
)

from .utils import (
    get_logger,
    initialize_logger,
)

from .ui import (
    TVACThermalAnalyzerDialog,
    HeatMapOverlay,
)


__all__ = [
    # Version
    '__version__',
    '__author__',
    '__license__',
    '__kicad_version__',
    'VERSION_INFO',
    'get_version_string',
    # Submodules
    'core',
    'solvers',
    'utils',
    'ui',
    # Core
    'ThermalAnalysisConfig',
    'ConfigManager',
    'PCBExtractor',
    'PCBData',
    'MaterialsDatabase',
    'ComponentThermalDatabase',
    # Solvers
    'ThermalSimulationResult',
    'ThermalResults',
    'ThermalSolver',
    'ThermalAnalysisEngine',
    'CurrentDistributionSolver',
    'CurrentDistributionResult',
    'MeshGenerator',
    'ThermalMesh',
    'run_simulation',
    # Utils
    'get_logger',
    'initialize_logger',
    # UI
    'TVACThermalAnalyzerDialog',
    'HeatMapOverlay',
]
