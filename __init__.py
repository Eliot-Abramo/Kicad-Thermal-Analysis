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
    "major": 1,
    "minor": 0,
    "patch": 0,
    "release": "stable",
    "build": "20250105",
}

def get_version_string() -> str:
    """Get formatted version string."""
    return f"{VERSION_INFO['major']}.{VERSION_INFO['minor']}.{VERSION_INFO['patch']}"


def _register_with_kicad() -> None:
    """Register the action plugin when running inside KiCad."""
    try:
        import pcbnew  # only available inside KiCad's PCB Editor runtime
    except Exception:
        return

    try:
        from .tvac_thermal_analyzer_plugin import TVACThermalAnalyzerPlugin
        TVACThermalAnalyzerPlugin().register()
    except Exception as e:
        # Never raise during discovery: KiCad will silently skip the plugin.
        try:
            print(f"TVAC Thermal Analyzer: registration failed: {e}")
        except Exception:
            pass


_register_with_kicad()
del _register_with_kicad

#
# IMPORTANT (KiCad plugin discovery):
# KiCad imports Python *packages* found in the plugins folder at startup.
# That means THIS FILE runs during plugin discovery.
#
# Keep imports here lightweight: do NOT import subpackages that pull optional
# dependencies (numpy/scipy/reportlab/etc.), or KiCad may fail to import the
# package and the plugin will not appear.
#
# NOTE: Plugin registration is handled by tvac_thermal_analyzer_plugin.py
# Do NOT register here to avoid duplicate menu entries.
#

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "__kicad_version__",
    "VERSION_INFO",
    "get_version_string",
]
