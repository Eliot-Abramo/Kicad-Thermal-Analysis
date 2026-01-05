"""
TVAC Thermal Analyzer - KiCad Plugin Entry Point
================================================

This is the main entry point for the KiCad Action Plugin.
Place this file and the tvac_thermal_analyzer folder in your KiCad plugins directory:
  - Windows: %APPDATA%/kicad/9.0/scripting/plugins/
  - Linux: ~/.local/share/kicad/9.0/scripting/plugins/
  - macOS: ~/Library/Preferences/kicad/9.0/scripting/plugins/

Author: Space Electronics Thermal Analysis Tool
Version: 1.0.0
License: MIT
"""

import os
import sys
import site
import traceback

# Add plugin directory to path for imports
plugin_dir = os.path.dirname(os.path.abspath(__file__))
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

try:
    import pcbnew
    import wx
    HAS_KICAD = True
except ImportError:
    HAS_KICAD = False
    print("Warning: KiCad pcbnew module not available. Running in standalone mode.")


class TVACThermalAnalyzerPlugin(pcbnew.ActionPlugin if HAS_KICAD else object):
    """
    KiCad Action Plugin for TVAC Thermal Analysis.
    
    This plugin provides comprehensive thermal simulation capabilities
    for PCBs under thermal-vacuum conditions.
    """
    
    def defaults(self):
        """Set plugin defaults."""
        self.name = "TVAC Thermal Analyzer"
        self.category = "Thermal Analysis"
        self.description = (
            "Simulate PCB thermal behavior under TVAC conditions.\n"
            "Features: 3D thermal FEM, current distribution, radiation modeling, "
            "heat map visualization, and professional PDF reports."
        )
        self.show_toolbar_button = True
        self.icon_file_name = os.path.join(
            os.path.dirname(__file__),
            "resources",
            "icon.png"
        )
        
        # Check if icon exists, use default if not
        if not os.path.exists(self.icon_file_name):
            self.icon_file_name = ""
    
    def Run(self):
        """Execute the plugin."""

        # KiCad sometimes doesn't include user site-packages on sys.path
        try:
            user_site = site.getusersitepackages()
            if user_site and user_site not in sys.path:
                site.addsitedir(user_site)
        except Exception:
            pass

        # Dependency check with actionable error message
        try:
            import scipy  # noqa: F401
        except Exception as e:
            msg = (
                "SciPy is not available in KiCad's Python environment.\n\n"
                f"KiCad Python: {sys.executable}\n"
                f"Version: {sys.version}\n\n"
                "Install SciPy into this interpreter, e.g.:\n"
                f"  \"{sys.executable}\" -m pip install scipy\n\n"
                f"Original error: {e}"
            )
            wx.MessageBox(msg, "TVAC Thermal Analyzer Error", wx.OK | wx.ICON_ERROR)
            return

        try:
            # Initialize logger
            from tvac_thermal_analyzer.utils import initialize_logger
            logger = initialize_logger()
            logger.info("TVAC Thermal Analyzer plugin started")
            
            # Get the current board
            board = pcbnew.GetBoard()
            
            if board is None:
                wx.MessageBox(
                    "No PCB is currently open.\n"
                    "Please open a PCB file first.",
                    "TVAC Thermal Analyzer",
                    wx.OK | wx.ICON_ERROR
                )
                return
            
            # Import and show the main dialog
            from tvac_thermal_analyzer.ui import TVACThermalAnalyzerDialog
            
            # IMPORTANT: parent to the PCB Editor frame, not the Project Manager.
            parent = None
            try:
                # In KiCad 9+, use FindWindowByName to get the PCB Editor specifically
                for w in wx.GetTopLevelWindows():
                    if w and w.IsShown():
                        title = w.GetTitle()
                        # PCB Editor window title contains the .kicad_pcb filename
                        if '.kicad_pcb' in title or 'PCB Editor' in title:
                            parent = w
                            break
            except Exception:
                pass

            if parent is None:
                # Fallback: choose a shown top-level window (better than index 0)
                tops = [w for w in wx.GetTopLevelWindows() if w and w.IsShown()]
                parent = tops[0] if tops else None

            # Create and show dialog
            dialog = TVACThermalAnalyzerDialog(parent, board)
            dialog.Show()
            # dialog.Destroy()
            
            logger.info("TVAC Thermal Analyzer plugin closed")
            
        except Exception as e:
            error_msg = f"Plugin error: {str(e)}\n\n{traceback.format_exc()}"
            wx.MessageBox(
                error_msg,
                "TVAC Thermal Analyzer Error",
                wx.OK | wx.ICON_ERROR
            )


# # Register the plugin with KiCad
# if HAS_KICAD:
#     TVACThermalAnalyzerPlugin().register()


# =============================================================================
# STANDALONE TESTING
# =============================================================================

def run_standalone():
    """Run the plugin in standalone mode for testing."""
    print("=" * 60)
    print("TVAC Thermal Analyzer - Standalone Test Mode")
    print("=" * 60)
    
    # Ensure path is set up
    import sys
    import os
    plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if plugin_dir not in sys.path:
        sys.path.insert(0, plugin_dir)
    
    # Import modules
    from tvac_thermal_analyzer import (
        get_version_string,
        MaterialsDatabase,
        ComponentThermalDatabase,
        ThermalAnalysisConfig,
    )
    from tvac_thermal_analyzer.core.config import SimulationParameters
    
    print(f"\nVersion: {get_version_string()}")
    
    # Test materials database
    print("\n--- Materials Database ---")
    print(f"PCB Substrates: {len(MaterialsDatabase.PCB_SUBSTRATES)}")
    print(f"Conductors: {len(MaterialsDatabase.CONDUCTORS)}")
    print(f"Heatsink Materials: {len(MaterialsDatabase.HEATSINK_MATERIALS)}")
    
    # Show FR4 properties
    fr4 = MaterialsDatabase.PCB_SUBSTRATES['FR4']
    print(f"\nFR4 Properties:")
    print(f"  Thermal Conductivity: {fr4.thermal_conductivity} W/(m·K)")
    print(f"  Specific Heat: {fr4.specific_heat} J/(kg·K)")
    print(f"  Density: {fr4.density} kg/m³")
    print(f"  Emissivity: {fr4.emissivity}")
    
    # Test component database
    print("\n--- Component Thermal Database ---")
    all_packages = ComponentThermalDatabase.get_all_packages()
    print(f"Total packages: {len(all_packages)}")
    
    # Show QFN-32 properties
    qfn32 = all_packages.get('QFN-32')
    if qfn32:
        print(f"\nQFN-32 Properties:")
        print(f"  θja: {qfn32.theta_ja} °C/W")
        print(f"  θjc: {qfn32.theta_jc} °C/W")
        print(f"  Thermal Mass: {qfn32.thermal_mass} J/K")
    
    # Test configuration
    print("\n--- Default Configuration ---")
    config = ThermalAnalysisConfig()
    print(f"Resolution: {config.simulation.resolution_mm} mm")
    print(f"Mode: {config.simulation.simulation_mode}")
    print(f"Duration: {config.simulation.duration_s} s")
    print(f"3D Simulation: {config.simulation.simulation_3d}")
    print(f"Radiation: {config.simulation.include_radiation}")
    
    print("\n" + "=" * 60)
    print("Standalone test completed successfully!")
    print("=" * 60)


# Run standalone test if executed directly
if __name__ == "__main__":
    run_standalone()
