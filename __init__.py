"""
TVAC Thermal Analyzer - KiCad Plugin Entry Point
================================================
Industrial-grade thermal analysis for space electronics PCBs.

Features:
- TVAC chamber simulation (radiation-dominated)
- Interactive BOM-style PCB visualization
- Component power management
- Heatsink and mounting point configuration
- Transient and steady-state analysis
- Professional reporting

Installation:
1. Copy the tvac_thermal_analyzer folder to your KiCad plugins directory
2. Restart KiCad or reload plugins
3. Access via Tools > External Plugins > TVAC Thermal Analyzer

Author: Space Electronics Thermal Analysis Tool
Version: 2.0.0
License: MIT
"""

import os
import sys
from pathlib import Path

# Ensure package is in path
plugin_dir = Path(__file__).parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

try:
    import pcbnew
    HAS_PCBNEW = True
except ImportError:
    HAS_PCBNEW = False


class TVACThermalAnalyzerPlugin(pcbnew.ActionPlugin if HAS_PCBNEW else object):
    """KiCad Action Plugin for TVAC Thermal Analysis."""
    
    def defaults(self):
        self.name = "TVAC Thermal Analyzer"
        self.category = "Thermal Analysis"
        self.description = "Industrial-grade thermal analysis for space electronics PCBs under TVAC conditions"
        self.show_toolbar_button = True
        self.icon_file_name = str(plugin_dir / "resources" / "icon.png")
        
        # Fallback if icon doesn't exist
        if not Path(self.icon_file_name).exists():
            self.icon_file_name = ""
    
    def Run(self):
        """Execute the plugin."""
        # Deferred imports to avoid loading issues
        import wx
        
        try:
            from .ui.main_dialog import MainDialog
            from .core.pcb_extractor import PCBExtractor
        except ImportError as e:
            wx.MessageBox(
                f"Failed to load TVAC Thermal Analyzer:\n{e}\n\n"
                "Please ensure all dependencies are installed.",
                "Plugin Error",
                wx.ICON_ERROR
            )
            return
        
        # Get the current board
        board = pcbnew.GetBoard()
        
        if board is None:
            wx.MessageBox(
                "No PCB is currently open.\n\n"
                "Please open a PCB file before running the thermal analyzer.",
                "No Board",
                wx.ICON_WARNING
            )
            return
        
        # Check for components
        footprints = board.GetFootprints()
        if not footprints or len(footprints) == 0:
            wx.MessageBox(
                "The current PCB has no components.\n\n"
                "Please ensure your PCB has components placed before running thermal analysis.",
                "No Components",
                wx.ICON_WARNING
            )
            return
        
        # Get parent window
        parent = None
        try:
            parent = wx.GetTopLevelWindows()[0] if wx.GetTopLevelWindows() else None
        except Exception:
            pass
        
        # Launch the main dialog (modeless for PCB interaction)
        try:
            dialog = MainDialog(parent, board)
            dialog.Show()
        except Exception as e:
            import traceback
            wx.MessageBox(
                f"Error launching TVAC Thermal Analyzer:\n{e}\n\n"
                f"Details:\n{traceback.format_exc()}",
                "Launch Error",
                wx.ICON_ERROR
            )


# Register the plugin with KiCad
if HAS_PCBNEW:
    TVACThermalAnalyzerPlugin().register()


def run_standalone():
    """Run the analyzer in standalone mode for testing."""
    import wx
    
    app = wx.App()
    
    # Create a mock board or load one
    print("TVAC Thermal Analyzer - Standalone Mode")
    print("This mode is for UI testing without KiCad.")
    print()
    
    # Try to load a board if path provided
    board = None
    if len(sys.argv) > 1:
        pcb_path = sys.argv[1]
        if os.path.exists(pcb_path):
            try:
                board = pcbnew.LoadBoard(pcb_path)
                print(f"Loaded: {pcb_path}")
            except Exception as e:
                print(f"Failed to load board: {e}")
    
    if board is None:
        print("No board loaded. Creating demo mode...")
        # Create empty board for demo
        try:
            board = pcbnew.BOARD()
        except Exception:
            print("Cannot create board without KiCad. Please run from within KiCad.")
            return
    
    from ui.main_dialog import MainDialog
    
    dialog = MainDialog(None, board)
    dialog.Show()
    
    app.MainLoop()


# Allow running as standalone script
if __name__ == "__main__":
    run_standalone()
