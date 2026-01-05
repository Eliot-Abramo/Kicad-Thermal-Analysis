"""
TVAC Thermal Analyzer - UI Module
=================================
User interface components for KiCad integration.
"""

from .main_dialog import (
    TVACThermalAnalyzerDialog,
    ProgressDialog,
)

from .heat_map import (
    ColorMap,
    ThermalColorMapper,
    HeatMapRenderer,
    HeatMapConfig,
)

# Alias for compatibility
HeatMapOverlay = HeatMapRenderer

__all__ = [
    'TVACThermalAnalyzerDialog',
    'ProgressDialog',
    'ColorMap',
    'ThermalColorMapper',
    'HeatMapRenderer',
    'HeatMapConfig',
    'HeatMapOverlay',
]
