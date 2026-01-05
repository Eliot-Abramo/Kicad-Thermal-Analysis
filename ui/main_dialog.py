"""
TVAC Thermal Analyzer - Main UI Dialog
======================================
Main user interface for thermal analysis configuration and execution.

Author: Space Electronics Thermal Analysis Tool
Version: 1.0.0
"""

import os
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass

try:
    import wx
    import wx.adv
    import wx.lib.scrolledpanel as scrolled
    import wx.lib.agw.floatspin as fs
    HAS_WX = True
except ImportError:
    HAS_WX = False

try:
    import pcbnew
    HAS_PCBNEW = True
except ImportError:
    HAS_PCBNEW = False

from ..core.config import (
    ThermalAnalysisConfig, ConfigManager, SimulationParameters,
    CurrentInjectionPoint, ComponentPowerConfig, MountingPoint,
    PCBStackupConfig, HeatsinkConfig
)
from ..core.constants import (
    MaterialsDatabase, ComponentThermalDatabase, SimulationDefaults
)
from ..core.pcb_extractor import PCBExtractor, PCBData
from ..core.schematic_importer import find_schematic_for_pcb, extract_component_powers_from_schematic
from ..solvers.thermal_solver import ThermalAnalysisEngine, ThermalSimulationResult
from ..utils.logger import get_logger, initialize_logger


# UI Constants
DIALOG_WIDTH = 900
DIALOG_HEIGHT = 700
PANEL_PADDING = 10
LABEL_WIDTH = 150

# Dark theme colors (Altium-inspired)
COLORS = {
    'bg_dark': '#1e1e1e',
    'bg_panel': '#252526',
    'bg_input': '#3c3c3c',
    'fg_text': '#cccccc',
    'fg_label': '#9cdcfe',
    'accent': '#0e639c',
    'accent_hover': '#1177bb',
    'success': '#4ec9b0',
    'warning': '#dcdcaa',
    'error': '#f14c4c',
    'border': '#3c3c3c',
}


def hex_to_wx_color(hex_color: str) -> 'wx.Colour':
    """Convert hex color string to wx.Colour."""
    if not HAS_WX:
        return None
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return wx.Colour(r, g, b)


class StyledPanel(wx.Panel if HAS_WX else object):
    """Base panel with dark theme styling."""
    
    def __init__(self, parent, *args, **kwargs):
        if not HAS_WX:
            return
        super().__init__(parent, *args, **kwargs)
        self.SetBackgroundColour(hex_to_wx_color(COLORS['bg_panel']))
        self.SetForegroundColour(hex_to_wx_color(COLORS['fg_text']))


class StyledButton(wx.Button if HAS_WX else object):
    """Styled button with hover effects."""
    
    def __init__(self, parent, label, *args, **kwargs):
        if not HAS_WX:
            return
        super().__init__(parent, label=label, *args, **kwargs)
        self.SetBackgroundColour(hex_to_wx_color(COLORS['accent']))
        self.SetForegroundColour(hex_to_wx_color('#ffffff'))
        
        self.Bind(wx.EVT_ENTER_WINDOW, self._on_hover)
        self.Bind(wx.EVT_LEAVE_WINDOW, self._on_leave)
    
    def _on_hover(self, event):
        self.SetBackgroundColour(hex_to_wx_color(COLORS['accent_hover']))
        self.Refresh()
        event.Skip()
    
    def _on_leave(self, event):
        self.SetBackgroundColour(hex_to_wx_color(COLORS['accent']))
        self.Refresh()
        event.Skip()


class StyledTextCtrl(wx.TextCtrl if HAS_WX else object):
    """Styled text control."""
    
    def __init__(self, parent, *args, **kwargs):
        if not HAS_WX:
            return
        super().__init__(parent, *args, **kwargs)
        self.SetBackgroundColour(hex_to_wx_color(COLORS['bg_input']))
        self.SetForegroundColour(hex_to_wx_color(COLORS['fg_text']))


class StyledChoice(wx.Choice if HAS_WX else object):
    """Styled dropdown choice control."""
    
    def __init__(self, parent, *args, **kwargs):
        if not HAS_WX:
            return
        super().__init__(parent, *args, **kwargs)
        self.SetBackgroundColour(hex_to_wx_color(COLORS['bg_input']))
        self.SetForegroundColour(hex_to_wx_color(COLORS['fg_text']))


class SimulationSettingsPanel(StyledPanel):
    """Panel for simulation parameter configuration."""
    
    def __init__(self, parent, config: ThermalAnalysisConfig):
        super().__init__(parent)
        self.config = config
        self._create_controls()
        self._create_layout()
        self._bind_events()
        self._load_from_config()
    
    def _create_controls(self):
        """Create all controls."""
        # Resolution
        self.lbl_resolution = wx.StaticText(self, label="Grid Resolution (mm):")
        self.spin_resolution = wx.SpinCtrlDouble(
            self, min=0.1, max=2.0, initial=0.5, inc=0.1
        )
        
        # Simulation mode
        self.lbl_mode = wx.StaticText(self, label="Simulation Mode:")
        self.choice_mode = StyledChoice(
            self, choices=['Transient', 'Steady State']
        )
        
        # Duration
        self.lbl_duration = wx.StaticText(self, label="Duration:")
        self.choice_duration = StyledChoice(
            self, choices=[
                '1 minute', '5 minutes', '10 minutes', '30 minutes',
                '1 hour', '3 hours', '5 hours', 'Custom'
            ]
        )
        self.spin_custom_duration = wx.SpinCtrl(
            self, min=1, max=36000, initial=600
        )
        self.spin_custom_duration.Hide()
        
        # Time step
        self.lbl_timestep = wx.StaticText(self, label="Time Step (s):")
        self.spin_timestep = wx.SpinCtrlDouble(
            self, min=0.01, max=10.0, initial=0.5, inc=0.1
        )
        
        # 3D simulation
        self.chk_3d = wx.CheckBox(self, label="3D Thermal Simulation")
        self.chk_3d.SetValue(True)
        
        # Radiation
        self.chk_radiation = wx.CheckBox(self, label="Include Radiation (TVAC)")
        self.chk_radiation.SetValue(True)
        
        # AC effects
        self.chk_ac = wx.CheckBox(self, label="Include AC/Skin Effect")
        self.lbl_frequency = wx.StaticText(self, label="Frequency (Hz):")
        self.spin_frequency = wx.SpinCtrl(
            self, min=1, max=10000000000, initial=1000000
        )
        
        # Temperatures
        self.lbl_ambient = wx.StaticText(self, label="Ambient Temp (°C):")
        self.spin_ambient = wx.SpinCtrlDouble(
            self, min=-200, max=300, initial=25, inc=1
        )
        
        self.lbl_chamber = wx.StaticText(self, label="Chamber Wall Temp (°C):")
        self.spin_chamber = wx.SpinCtrlDouble(
            self, min=-200, max=300, initial=25, inc=1
        )
        
        self.lbl_initial = wx.StaticText(self, label="Initial Board Temp (°C):")
        self.spin_initial = wx.SpinCtrlDouble(
            self, min=-200, max=300, initial=25, inc=1
        )
        
        # Apply styling
        for ctrl in [self.lbl_resolution, self.lbl_mode, self.lbl_duration,
                    self.lbl_timestep, self.lbl_frequency, self.lbl_ambient,
                    self.lbl_chamber, self.lbl_initial]:
            ctrl.SetForegroundColour(hex_to_wx_color(COLORS['fg_label']))
    
    def _create_layout(self):
        """Create panel layout."""
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(self, label="Simulation Settings")
        title.SetFont(title.GetFont().Bold())
        title.SetForegroundColour(hex_to_wx_color(COLORS['success']))
        main_sizer.Add(title, 0, wx.ALL, 5)
        
        grid = wx.FlexGridSizer(rows=0, cols=2, vgap=8, hgap=10)
        grid.AddGrowableCol(1)
        
        # Add controls to grid
        grid.Add(self.lbl_resolution, 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.spin_resolution, 0, wx.EXPAND)
        
        grid.Add(self.lbl_mode, 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.choice_mode, 0, wx.EXPAND)
        
        grid.Add(self.lbl_duration, 0, wx.ALIGN_CENTER_VERTICAL)
        duration_sizer = wx.BoxSizer(wx.HORIZONTAL)
        duration_sizer.Add(self.choice_duration, 1, wx.EXPAND)
        duration_sizer.Add(self.spin_custom_duration, 1, wx.EXPAND | wx.LEFT, 5)
        grid.Add(duration_sizer, 0, wx.EXPAND)
        
        grid.Add(self.lbl_timestep, 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.spin_timestep, 0, wx.EXPAND)
        
        grid.Add(wx.StaticText(self), 0)  # Spacer
        grid.Add(self.chk_3d, 0)
        
        grid.Add(wx.StaticText(self), 0)
        grid.Add(self.chk_radiation, 0)
        
        grid.Add(wx.StaticText(self), 0)
        ac_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ac_sizer.Add(self.chk_ac, 0)
        ac_sizer.Add(self.lbl_frequency, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 20)
        ac_sizer.Add(self.spin_frequency, 0, wx.LEFT, 5)
        grid.Add(ac_sizer, 0)
        
        grid.Add(self.lbl_ambient, 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.spin_ambient, 0, wx.EXPAND)
        
        grid.Add(self.lbl_chamber, 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.spin_chamber, 0, wx.EXPAND)
        
        grid.Add(self.lbl_initial, 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.spin_initial, 0, wx.EXPAND)
        
        main_sizer.Add(grid, 1, wx.EXPAND | wx.ALL, 10)
        self.SetSizer(main_sizer)
    
    def _bind_events(self):
        """Bind event handlers."""
        self.choice_duration.Bind(wx.EVT_CHOICE, self._on_duration_change)
        self.chk_ac.Bind(wx.EVT_CHECKBOX, self._on_ac_toggle)
    
    def _on_duration_change(self, event):
        """Handle duration preset change."""
        selection = self.choice_duration.GetStringSelection()
        if selection == 'Custom':
            self.spin_custom_duration.Show()
        else:
            self.spin_custom_duration.Hide()
        self.Layout()
    
    def _on_ac_toggle(self, event):
        """Handle AC checkbox toggle."""
        enabled = self.chk_ac.GetValue()
        self.spin_frequency.Enable(enabled)
        self.lbl_frequency.Enable(enabled)
    
    def _load_from_config(self):
        """Load values from config."""
        sim = self.config.simulation
        self.spin_resolution.SetValue(sim.resolution_mm)
        self.choice_mode.SetSelection(0 if sim.simulation_mode == 'transient' else 1)
        self.spin_timestep.SetValue(sim.timestep_s)
        self.chk_3d.SetValue(sim.simulation_3d)
        self.chk_radiation.SetValue(sim.include_radiation)
        self.chk_ac.SetValue(sim.include_ac_effects)
        self.spin_frequency.SetValue(int(sim.ac_frequency_hz))
        self.spin_ambient.SetValue(sim.ambient_temp_c)
        self.spin_chamber.SetValue(sim.chamber_wall_temp_c)
        self.spin_initial.SetValue(sim.initial_board_temp_c)
    
    def save_to_config(self):
        """Save values to config."""
        sim = self.config.simulation
        sim.resolution_mm = self.spin_resolution.GetValue()
        sim.simulation_mode = 'transient' if self.choice_mode.GetSelection() == 0 else 'steady_state'
        sim.timestep_s = self.spin_timestep.GetValue()
        sim.simulation_3d = self.chk_3d.GetValue()
        sim.include_radiation = self.chk_radiation.GetValue()
        sim.include_ac_effects = self.chk_ac.GetValue()
        sim.ac_frequency_hz = float(self.spin_frequency.GetValue())
        sim.ambient_temp_c = self.spin_ambient.GetValue()
        sim.chamber_wall_temp_c = self.spin_chamber.GetValue()
        sim.initial_board_temp_c = self.spin_initial.GetValue()
        
        # Duration
        duration_map = {
            '1 minute': 60, '5 minutes': 300, '10 minutes': 600,
            '30 minutes': 1800, '1 hour': 3600, '3 hours': 10800, '5 hours': 18000
        }
        selection = self.choice_duration.GetStringSelection()
        if selection == 'Custom':
            sim.duration_s = float(self.spin_custom_duration.GetValue())
        else:
            sim.duration_s = float(duration_map.get(selection, 600))


class CurrentInjectionPanel(StyledPanel):
    """Panel for defining current injection points."""
    
    def __init__(self, parent, config: ThermalAnalysisConfig, pcb_data: Optional[PCBData] = None, board=None):
        super().__init__(parent)
        self.config = config
        self.pcb_data = pcb_data
        self.board = board  # KiCad board object for picking
        self._create_controls()
        self._create_layout()
        self._bind_events()
        self._load_from_config()
    
    def _create_controls(self):
        """Create controls."""
        # List of injection points
        self.list_ctrl = wx.ListCtrl(
            self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL
        )
        self.list_ctrl.InsertColumn(0, "ID", width=60)
        self.list_ctrl.InsertColumn(1, "Net", width=80)
        self.list_ctrl.InsertColumn(2, "Layer", width=60)
        self.list_ctrl.InsertColumn(3, "X (mm)", width=60)
        self.list_ctrl.InsertColumn(4, "Y (mm)", width=60)
        self.list_ctrl.InsertColumn(5, "Current (A)", width=80)
        self.list_ctrl.InsertColumn(6, "Description", width=120)
        
        self.list_ctrl.SetBackgroundColour(hex_to_wx_color(COLORS['bg_input']))
        self.list_ctrl.SetForegroundColour(hex_to_wx_color(COLORS['fg_text']))
        
        # Input fields
        self.txt_net = StyledTextCtrl(self, size=(100, -1))
        self.choice_layer = StyledChoice(self, choices=['F.Cu', 'B.Cu', 'In1.Cu', 'In2.Cu'])
        self.spin_x = wx.SpinCtrlDouble(self, min=-500, max=500, initial=0, inc=0.1)
        self.spin_y = wx.SpinCtrlDouble(self, min=-500, max=500, initial=0, inc=0.1)
        self.spin_current = wx.SpinCtrlDouble(self, min=-100, max=100, initial=0, inc=0.1)
        self.txt_desc = StyledTextCtrl(self, size=(150, -1))
        
        # Buttons
        self.btn_add = StyledButton(self, "Add")
        self.btn_remove = StyledButton(self, "Remove")
        self.btn_pick = StyledButton(self, "Pick from PCB")
    
    def _create_layout(self):
        """Create layout."""
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        title = wx.StaticText(self, label="Current Injection Points")
        title.SetFont(title.GetFont().Bold())
        title.SetForegroundColour(hex_to_wx_color(COLORS['success']))
        main_sizer.Add(title, 0, wx.ALL, 5)
        
        # List
        main_sizer.Add(self.list_ctrl, 1, wx.EXPAND | wx.ALL, 5)
        
        # Input row
        input_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        input_sizer.Add(wx.StaticText(self, label="Net:"), 0, wx.ALIGN_CENTER_VERTICAL)
        input_sizer.Add(self.txt_net, 0, wx.LEFT, 3)
        input_sizer.Add(wx.StaticText(self, label="Layer:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        input_sizer.Add(self.choice_layer, 0, wx.LEFT, 3)
        input_sizer.Add(wx.StaticText(self, label="X:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        input_sizer.Add(self.spin_x, 0, wx.LEFT, 3)
        input_sizer.Add(wx.StaticText(self, label="Y:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)
        input_sizer.Add(self.spin_y, 0, wx.LEFT, 3)
        input_sizer.Add(wx.StaticText(self, label="I(A):"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        input_sizer.Add(self.spin_current, 0, wx.LEFT, 3)
        
        main_sizer.Add(input_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Description row
        desc_sizer = wx.BoxSizer(wx.HORIZONTAL)
        desc_sizer.Add(wx.StaticText(self, label="Description:"), 0, wx.ALIGN_CENTER_VERTICAL)
        desc_sizer.Add(self.txt_desc, 1, wx.EXPAND | wx.LEFT, 5)
        main_sizer.Add(desc_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Button row
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.Add(self.btn_add, 0, wx.RIGHT, 5)
        btn_sizer.Add(self.btn_remove, 0, wx.RIGHT, 5)
        btn_sizer.Add(self.btn_pick, 0)
        main_sizer.Add(btn_sizer, 0, wx.ALL, 5)
        
        # Info text
        info = wx.StaticText(
            self, 
            label="Use positive current for injection, negative for extraction. Total must sum to 0."
        )
        info.SetForegroundColour(hex_to_wx_color(COLORS['warning']))
        main_sizer.Add(info, 0, wx.ALL, 5)
        
        self.SetSizer(main_sizer)
    
    def _bind_events(self):
        """Bind events."""
        self.btn_add.Bind(wx.EVT_BUTTON, self._on_add)
        self.btn_remove.Bind(wx.EVT_BUTTON, self._on_remove)
        self.btn_pick.Bind(wx.EVT_BUTTON, self._on_pick)
    
    def _on_add(self, event):
        """Add new injection point."""
        point_id = f"IP{len(self.config.current_injection_points) + 1}"
        point = CurrentInjectionPoint(
            point_id=point_id,
            net_name=self.txt_net.GetValue(),
            x_mm=self.spin_x.GetValue(),
            y_mm=self.spin_y.GetValue(),
            layer=self.choice_layer.GetStringSelection(),
            current_a=self.spin_current.GetValue(),
            description=self.txt_desc.GetValue()
        )
        self.config.current_injection_points.append(point)
        self._refresh_list()
    
    def _on_remove(self, event):
        """Remove selected injection point."""
        idx = self.list_ctrl.GetFirstSelected()
        if idx >= 0 and idx < len(self.config.current_injection_points):
            del self.config.current_injection_points[idx]
            self._refresh_list()
    
    def _on_pick(self, event):
        """Pick location from PCB selection."""
        if not HAS_PCBNEW or self.board is None:
            wx.MessageBox(
                "No PCB board available.\n"
                "Open a PCB in the editor first.",
                "Pick from PCB",
                wx.OK | wx.ICON_WARNING
            )
            return
        
        try:
            import pcbnew
            
            # Get the current selection from the PCB editor
            # KiCad 9+: use pcbnew.GetCurrentSelection() or iterate
            try:
                # Try KiCad 9+ method first
                selection = pcbnew.GetCurrentSelection()
            except AttributeError:
                # Fallback: iterate through all items looking for selected ones
                selection = []
                for track in self.board.GetTracks():
                    if track.IsSelected():
                        selection.append(track)
                for fp in self.board.GetFootprints():
                    if fp.IsSelected():
                        selection.append(fp)
                    for pad in fp.Pads():
                        if pad.IsSelected():
                            selection.append(pad)
                for zone in self.board.Zones():
                    if zone.IsSelected():
                        selection.append(zone)
            
            # Handle both list and iterator types
            if hasattr(selection, 'GetCount'):
                count = selection.GetCount()
                items = [selection.GetItem(i) for i in range(count)] if count > 0 else []
            else:
                items = list(selection) if selection else []
            
            if not items:
                wx.MessageBox(
                    "No item selected in the PCB editor.\n"
                    "Select a pad, track, via, or zone first,\n"
                    "then click 'Pick from PCB'.",
                    "Pick from PCB",
                    wx.OK | wx.ICON_INFORMATION
                )
                return
            
            # Get the first selected item
            item = items[0]
            
            net_name = ""
            layer = ""
            x_mm = 0.0
            y_mm = 0.0
            description = ""
            
            item_class = item.GetClass()
            
            if item_class == "PCB_PAD":
                pad = item
                pos = pad.GetPosition()
                x_mm = pcbnew.ToMM(pos.x)
                y_mm = pcbnew.ToMM(pos.y)
                net_name = pad.GetNetname()
                # Get first copper layer
                layer_set = pad.GetLayerSet()
                for layer_id in [pcbnew.F_Cu, pcbnew.B_Cu, pcbnew.In1_Cu, pcbnew.In2_Cu]:
                    if layer_set.Contains(layer_id):
                        layer = self.board.GetLayerName(layer_id)
                        break
                description = f"Pad {pad.GetNumber()} of {pad.GetParent().GetReference()}"
                
            elif item_class == "PCB_TRACK":
                track = item
                # Use midpoint of track
                start = track.GetStart()
                end = track.GetEnd()
                x_mm = pcbnew.ToMM((start.x + end.x) / 2)
                y_mm = pcbnew.ToMM((start.y + end.y) / 2)
                net_name = track.GetNetname()
                layer = self.board.GetLayerName(track.GetLayer())
                description = f"Track on {net_name}"
                
            elif item_class == "PCB_VIA":
                via = item
                pos = via.GetPosition()
                x_mm = pcbnew.ToMM(pos.x)
                y_mm = pcbnew.ToMM(pos.y)
                net_name = via.GetNetname()
                layer = self.board.GetLayerName(via.TopLayer())
                description = f"Via on {net_name}"
                
            elif item_class == "ZONE":
                zone = item
                # Use zone centroid approximation
                bbox = zone.GetBoundingBox()
                x_mm = pcbnew.ToMM((bbox.GetLeft() + bbox.GetRight()) / 2)
                y_mm = pcbnew.ToMM((bbox.GetTop() + bbox.GetBottom()) / 2)
                net_name = zone.GetNetname()
                layer = self.board.GetLayerName(zone.GetLayer())
                description = f"Zone on {net_name}"
                
            else:
                wx.MessageBox(
                    f"Cannot pick from item type: {item_class}\n"
                    "Please select a pad, track, via, or zone.",
                    "Pick from PCB",
                    wx.OK | wx.ICON_WARNING
                )
                return
            
            # Populate the input fields
            self.txt_net.SetValue(net_name)
            self.spin_x.SetValue(x_mm)
            self.spin_y.SetValue(y_mm)
            self.txt_desc.SetValue(description)
            
            # Set layer in dropdown
            layer_idx = self.choice_layer.FindString(layer)
            if layer_idx != wx.NOT_FOUND:
                self.choice_layer.SetSelection(layer_idx)
            
            wx.MessageBox(
                f"Picked from PCB:\n"
                f"  Net: {net_name}\n"
                f"  Layer: {layer}\n"
                f"  Position: ({x_mm:.2f}, {y_mm:.2f}) mm\n\n"
                "Set the current value and click 'Add' to create the injection point.",
                "Pick from PCB",
                wx.OK | wx.ICON_INFORMATION
            )
            
        except Exception as e:
            wx.MessageBox(
                f"Error picking from PCB:\n{e}",
                "Pick from PCB",
                wx.OK | wx.ICON_ERROR
            )
    
    def _load_from_config(self):
        """Load from config."""
        self._refresh_list()
    
    def _refresh_list(self):
        """Refresh the list control."""
        self.list_ctrl.DeleteAllItems()
        for i, point in enumerate(self.config.current_injection_points):
            idx = self.list_ctrl.InsertItem(i, point.point_id)
            self.list_ctrl.SetItem(idx, 1, point.net_name)
            self.list_ctrl.SetItem(idx, 2, point.layer)
            self.list_ctrl.SetItem(idx, 3, f"{point.x_mm:.1f}")
            self.list_ctrl.SetItem(idx, 4, f"{point.y_mm:.1f}")
            self.list_ctrl.SetItem(idx, 5, f"{point.current_a:+.3f}")
            self.list_ctrl.SetItem(idx, 6, point.description)


class ComponentPowerPanel(StyledPanel):
    """Panel for component power dissipation configuration."""
    
    def __init__(self, parent, config: ThermalAnalysisConfig, pcb_data: Optional[PCBData] = None, pcb_path: Optional[str] = None):
        super().__init__(parent)
        self.config = config
        self.pcb_data = pcb_data
        self.pcb_path = pcb_path
        self._create_controls()
        self._create_layout()
        self._bind_events()
        self._load_from_config()
    
    def _create_controls(self):
        """Create controls."""
        self.list_ctrl = wx.ListCtrl(
            self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL
        )
        self.list_ctrl.InsertColumn(0, "Reference", width=80)
        self.list_ctrl.InsertColumn(1, "Power (W)", width=80)
        self.list_ctrl.InsertColumn(2, "Source", width=80)
        self.list_ctrl.InsertColumn(3, "θja (°C/W)", width=80)
        
        self.list_ctrl.SetBackgroundColour(hex_to_wx_color(COLORS['bg_input']))
        self.list_ctrl.SetForegroundColour(hex_to_wx_color(COLORS['fg_text']))
        
        # Input fields
        self.txt_ref = StyledTextCtrl(self, size=(80, -1))
        self.spin_power = wx.SpinCtrlDouble(self, min=0, max=1000, initial=0, inc=0.01)
        
        # Buttons
        self.btn_add = StyledButton(self, "Add/Update")
        self.btn_remove = StyledButton(self, "Remove")
        self.btn_import = StyledButton(self, "Import from Schematic")
        self.btn_auto = StyledButton(self, "Auto-detect from PCB")
    
    def _create_layout(self):
        """Create layout."""
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        title = wx.StaticText(self, label="Component Power Dissipation")
        title.SetFont(title.GetFont().Bold())
        title.SetForegroundColour(hex_to_wx_color(COLORS['success']))
        main_sizer.Add(title, 0, wx.ALL, 5)
        
        # List
        main_sizer.Add(self.list_ctrl, 1, wx.EXPAND | wx.ALL, 5)
        
        # Input row
        input_sizer = wx.BoxSizer(wx.HORIZONTAL)
        input_sizer.Add(wx.StaticText(self, label="Reference:"), 0, wx.ALIGN_CENTER_VERTICAL)
        input_sizer.Add(self.txt_ref, 0, wx.LEFT, 5)
        input_sizer.Add(wx.StaticText(self, label="Power (W):"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 15)
        input_sizer.Add(self.spin_power, 0, wx.LEFT, 5)
        input_sizer.Add(self.btn_add, 0, wx.LEFT, 15)
        input_sizer.Add(self.btn_remove, 0, wx.LEFT, 5)
        main_sizer.Add(input_sizer, 0, wx.ALL, 5)
        
        # Import buttons
        import_sizer = wx.BoxSizer(wx.HORIZONTAL)
        import_sizer.Add(self.btn_import, 0, wx.RIGHT, 5)
        import_sizer.Add(self.btn_auto, 0)
        main_sizer.Add(import_sizer, 0, wx.ALL, 5)
        
        self.SetSizer(main_sizer)
    
    def _bind_events(self):
        """Bind events."""
        self.btn_add.Bind(wx.EVT_BUTTON, self._on_add)
        self.btn_remove.Bind(wx.EVT_BUTTON, self._on_remove)
        self.btn_import.Bind(wx.EVT_BUTTON, self._on_import)
        self.btn_auto.Bind(wx.EVT_BUTTON, self._on_auto)
    
    def _on_add(self, event):
        """Add or update component power."""
        ref = self.txt_ref.GetValue().strip()
        if not ref:
            return
        
        power = self.spin_power.GetValue()
        
        # Update or add
        for comp in self.config.component_power:
            if comp.reference == ref:
                comp.power_w = power
                comp.source = "manual"
                self._refresh_list()
                return
        
        self.config.component_power.append(ComponentPowerConfig(
            reference=ref,
            power_w=power,
            source="manual"
        ))
        self._refresh_list()
    
    def _on_remove(self, event):
        """Remove selected component."""
        idx = self.list_ctrl.GetFirstSelected()
        if idx >= 0 and idx < len(self.config.component_power):
            del self.config.component_power[idx]
            self._refresh_list()
    
    def _on_import(self, event):
        """Import power from the .kicad_sch POWER_DISSIPATION properties."""

        if not self.pcb_path:
            wx.MessageBox(
                "Can't locate the PCB file path.\n"
                "Save/open the PCB (*.kicad_pcb) and try again.",
                "Import Error",
                wx.OK | wx.ICON_WARNING
            )
            return

        sch_path = find_schematic_for_pcb(self.pcb_path)
        if not sch_path:
            wx.MessageBox(
                "Couldn't locate a schematic (*.kicad_sch) next to this PCB.\n"
                "Expected something like <project>.kicad_sch in the same folder.",
                "Import Error",
                wx.OK | wx.ICON_WARNING
            )
            return

        try:
            powers = extract_component_powers_from_schematic(sch_path)
        except Exception as e:
            wx.MessageBox(
                f"Failed to parse schematic:\n{e}",
                "Import Error",
                wx.OK | wx.ICON_ERROR
            )
            return

        valid_refs = None
        if self.pcb_data:
            valid_refs = {c.reference for c in self.pcb_data.components}

        count = 0
        for ref, power_w in powers.items():
            if power_w <= 0:
                continue
            if valid_refs is not None and ref not in valid_refs:
                continue

            # Don't overwrite existing entries (especially manual)
            if any(cfg.reference == ref for cfg in self.config.component_power):
                continue

            self.config.component_power.append(ComponentPowerConfig(
                reference=ref,
                power_w=power_w,
                source="schematic"
            ))
            count += 1

        self._refresh_list()
        wx.MessageBox(f"Imported {count} components from schematic.", "Import Complete")

    def _on_auto(self, event):
        """Auto-detect components from PCB."""
        if self.pcb_data:
            count = 0
            for comp in self.pcb_data.components:
                exists = False
                for cfg in self.config.component_power:
                    if cfg.reference == comp.reference:
                        exists = True
                        break
                
                if not exists:
                    self.config.component_power.append(ComponentPowerConfig(
                        reference=comp.reference,
                        power_w=0.0,  # Will need manual entry
                        source="auto-detected"
                    ))
                    count += 1
            
            self._refresh_list()
            wx.MessageBox(f"Added {count} components. Please set power values.", "Auto-detect Complete")
        else:
            wx.MessageBox("No PCB data available.", "Auto-detect Error", wx.ICON_WARNING)
    
    def _load_from_config(self):
        """Load from config."""
        self._refresh_list()
    
    def _refresh_list(self):
        """Refresh list."""
        self.list_ctrl.DeleteAllItems()
        for i, comp in enumerate(self.config.component_power):
            idx = self.list_ctrl.InsertItem(i, comp.reference)
            self.list_ctrl.SetItem(idx, 1, f"{comp.power_w:.3f}")
            self.list_ctrl.SetItem(idx, 2, comp.source)
            theta = comp.theta_ja_override or "-"
            self.list_ctrl.SetItem(idx, 3, str(theta))


class ProgressDialog(wx.Dialog if HAS_WX else object):
    """Progress dialog for simulation."""
    
    def __init__(self, parent, title="Thermal Simulation"):
        if not HAS_WX:
            return
        super().__init__(parent, title=title, size=(400, 200),
                        style=wx.DEFAULT_DIALOG_STYLE)
        
        self.SetBackgroundColour(hex_to_wx_color(COLORS['bg_panel']))
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.lbl_status = wx.StaticText(self, label="Initializing...")
        self.lbl_status.SetForegroundColour(hex_to_wx_color(COLORS['fg_text']))
        sizer.Add(self.lbl_status, 0, wx.ALL | wx.EXPAND, 10)
        
        self.gauge = wx.Gauge(self, range=100)
        sizer.Add(self.gauge, 0, wx.ALL | wx.EXPAND, 10)
        
        self.lbl_detail = wx.StaticText(self, label="")
        self.lbl_detail.SetForegroundColour(hex_to_wx_color(COLORS['fg_label']))
        sizer.Add(self.lbl_detail, 0, wx.ALL | wx.EXPAND, 10)
        
        self.btn_cancel = StyledButton(self, "Cancel")
        self.btn_cancel.Bind(wx.EVT_BUTTON, self._on_cancel)
        sizer.Add(self.btn_cancel, 0, wx.ALL | wx.ALIGN_CENTER, 10)
        
        self.SetSizer(sizer)
        self.Centre()
        
        self._cancelled = False
    
    def update_progress(self, progress: float, message: str):
        """Update progress."""
        wx.CallAfter(self._do_update, progress, message)
    
    def _do_update(self, progress: float, message: str):
        """Actually update UI (must be called from main thread)."""
        self.gauge.SetValue(int(progress * 100))
        self.lbl_detail.SetLabel(message)
    
    def _on_cancel(self, event):
        """Handle cancel."""
        self._cancelled = True
        self.EndModal(wx.ID_CANCEL)
    
    @property
    def cancelled(self) -> bool:
        return self._cancelled


class StackupPanel(StyledPanel):
    """Panel for PCB stackup configuration."""
    
    def __init__(self, parent, config: ThermalAnalysisConfig, pcb_data: Optional[PCBData] = None):
        super().__init__(parent)
        self.config = config
        self.pcb_data = pcb_data
        self._create_controls()
        self._create_layout()
        self._load_from_config()
    
    def _create_controls(self):
        """Create controls."""
        # Stackup list
        self.list_ctrl = wx.ListCtrl(
            self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL
        )
        self.list_ctrl.InsertColumn(0, "Layer", width=100)
        self.list_ctrl.InsertColumn(1, "Type", width=80)
        self.list_ctrl.InsertColumn(2, "Thickness (µm)", width=100)
        self.list_ctrl.InsertColumn(3, "Material", width=120)
        
        self.list_ctrl.SetBackgroundColour(hex_to_wx_color(COLORS['bg_input']))
        self.list_ctrl.SetForegroundColour(hex_to_wx_color(COLORS['fg_text']))
        
        # Substrate material dropdown
        self.lbl_substrate = wx.StaticText(self, label="Substrate Material:")
        self.lbl_substrate.SetForegroundColour(hex_to_wx_color(COLORS['fg_label']))
        
        substrate_choices = ['FR4', 'FR4_HIGH_TG', 'POLYIMIDE', 'ROGERS_4350B', 'ROGERS_4003C', 
                           'CERAMIC_ALUMINA', 'CERAMIC_ALN']
        self.choice_substrate = StyledChoice(self, choices=substrate_choices)
        
        # Surface finish dropdown
        self.lbl_finish = wx.StaticText(self, label="Surface Finish:")
        self.lbl_finish.SetForegroundColour(hex_to_wx_color(COLORS['fg_label']))
        
        finish_choices = ['HASL', 'ENIG', 'OSP', 'IMMERSION_SILVER', 'IMMERSION_TIN', 'BARE_COPPER']
        self.choice_finish = StyledChoice(self, choices=finish_choices)
        
        # Solder mask color
        self.lbl_mask = wx.StaticText(self, label="Solder Mask:")
        self.lbl_mask.SetForegroundColour(hex_to_wx_color(COLORS['fg_label']))
        
        mask_choices = ['GREEN', 'BLACK', 'WHITE', 'RED', 'BLUE', 'YELLOW', 'MATTE_BLACK']
        self.choice_mask = StyledChoice(self, choices=mask_choices)
        
        # Total thickness
        self.lbl_thickness = wx.StaticText(self, label="Total Thickness (mm):")
        self.lbl_thickness.SetForegroundColour(hex_to_wx_color(COLORS['fg_label']))
        self.spin_thickness = wx.SpinCtrlDouble(self, min=0.2, max=5.0, initial=1.6, inc=0.1)
        
        # Apply button
        self.btn_apply = StyledButton(self, "Apply Changes")
        self.btn_apply.Bind(wx.EVT_BUTTON, self._on_apply)
    
    def _create_layout(self):
        """Create layout."""
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        title = wx.StaticText(self, label="PCB Stackup Configuration")
        title.SetFont(title.GetFont().Bold())
        title.SetForegroundColour(hex_to_wx_color(COLORS['success']))
        main_sizer.Add(title, 0, wx.ALL, 5)
        
        # Stackup list
        main_sizer.Add(self.list_ctrl, 1, wx.EXPAND | wx.ALL, 5)
        
        # Options grid
        grid = wx.FlexGridSizer(rows=0, cols=2, vgap=8, hgap=10)
        grid.AddGrowableCol(1)
        
        grid.Add(self.lbl_substrate, 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.choice_substrate, 0, wx.EXPAND)
        
        grid.Add(self.lbl_finish, 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.choice_finish, 0, wx.EXPAND)
        
        grid.Add(self.lbl_mask, 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.choice_mask, 0, wx.EXPAND)
        
        grid.Add(self.lbl_thickness, 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.spin_thickness, 0, wx.EXPAND)
        
        main_sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 10)
        main_sizer.Add(self.btn_apply, 0, wx.ALL | wx.ALIGN_RIGHT, 10)
        
        self.SetSizer(main_sizer)
    
    def _load_from_config(self):
        """Load from config."""
        stackup = self.config.stackup
        
        # Populate list
        self.list_ctrl.DeleteAllItems()
        row = 0
        
        for layer, thickness in stackup.copper_thickness_um.items():
            idx = self.list_ctrl.InsertItem(row, layer)
            self.list_ctrl.SetItem(idx, 1, "Copper")
            self.list_ctrl.SetItem(idx, 2, f"{thickness:.1f}")
            self.list_ctrl.SetItem(idx, 3, "Copper")
            row += 1
        
        for i, thickness in enumerate(stackup.dielectric_thickness_um):
            idx = self.list_ctrl.InsertItem(row, f"Dielectric {i+1}")
            self.list_ctrl.SetItem(idx, 1, "Dielectric")
            self.list_ctrl.SetItem(idx, 2, f"{thickness:.1f}")
            self.list_ctrl.SetItem(idx, 3, stackup.substrate_material)
            row += 1
        
        # Set dropdowns
        substrate_idx = self.choice_substrate.FindString(stackup.substrate_material)
        if substrate_idx != wx.NOT_FOUND:
            self.choice_substrate.SetSelection(substrate_idx)
        
        finish_idx = self.choice_finish.FindString(stackup.surface_finish)
        if finish_idx != wx.NOT_FOUND:
            self.choice_finish.SetSelection(finish_idx)
        
        mask_idx = self.choice_mask.FindString(stackup.solder_mask_color)
        if mask_idx != wx.NOT_FOUND:
            self.choice_mask.SetSelection(mask_idx)
        
        self.spin_thickness.SetValue(stackup.total_thickness_mm)
    
    def _on_apply(self, event):
        """Apply changes to config."""
        self.config.stackup.substrate_material = self.choice_substrate.GetStringSelection()
        self.config.stackup.surface_finish = self.choice_finish.GetStringSelection()
        self.config.stackup.solder_mask_color = self.choice_mask.GetStringSelection()
        self.config.stackup.total_thickness_mm = self.spin_thickness.GetValue()
        
        wx.MessageBox("Stackup changes applied.", "Stackup", wx.OK | wx.ICON_INFORMATION)


class MountingPanel(StyledPanel):
    """Panel for mounting points and heatsink configuration."""
    
    def __init__(self, parent, config: ThermalAnalysisConfig, pcb_data: Optional[PCBData] = None):
        super().__init__(parent)
        self.config = config
        self.pcb_data = pcb_data
        self._create_controls()
        self._create_layout()
        self._bind_events()
        self._load_from_config()
    
    def _create_controls(self):
        """Create controls."""
        # Mounting points list
        self.mounting_list = wx.ListCtrl(
            self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL
        )
        self.mounting_list.InsertColumn(0, "ID", width=60)
        self.mounting_list.InsertColumn(1, "X (mm)", width=60)
        self.mounting_list.InsertColumn(2, "Y (mm)", width=60)
        self.mounting_list.InsertColumn(3, "Diameter", width=70)
        self.mounting_list.InsertColumn(4, "Type", width=80)
        self.mounting_list.InsertColumn(5, "Interface", width=100)
        self.mounting_list.InsertColumn(6, "Temp (°C)", width=70)
        
        self.mounting_list.SetBackgroundColour(hex_to_wx_color(COLORS['bg_input']))
        self.mounting_list.SetForegroundColour(hex_to_wx_color(COLORS['fg_text']))
        
        # Input fields for mounting points
        self.spin_mp_x = wx.SpinCtrlDouble(self, min=-500, max=500, initial=0, inc=0.1)
        self.spin_mp_y = wx.SpinCtrlDouble(self, min=-500, max=500, initial=0, inc=0.1)
        self.spin_mp_dia = wx.SpinCtrlDouble(self, min=0.5, max=20, initial=3.2, inc=0.1)
        
        contact_choices = ['conductive', 'isolative']
        self.choice_contact = StyledChoice(self, choices=contact_choices)
        
        interface_choices = ['THERMAL_PASTE_STANDARD', 'THERMAL_PASTE_HIGH', 'THERMAL_PAD', 
                            'THERMAL_ADHESIVE', 'INDIUM_FOIL']
        self.choice_interface = StyledChoice(self, choices=interface_choices)
        
        self.spin_mp_temp = wx.SpinCtrlDouble(self, min=-200, max=300, initial=25, inc=1)
        self.chk_fixed_temp = wx.CheckBox(self, label="Fixed Temp")
        
        # Buttons
        self.btn_add_mp = StyledButton(self, "Add Mounting Point")
        self.btn_remove_mp = StyledButton(self, "Remove")
        self.btn_detect_holes = StyledButton(self, "Detect from PCB")
        
        # Heatsinks list
        self.heatsink_list = wx.ListCtrl(
            self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL
        )
        self.heatsink_list.InsertColumn(0, "ID", width=60)
        self.heatsink_list.InsertColumn(1, "Material", width=100)
        self.heatsink_list.InsertColumn(2, "Thickness", width=70)
        self.heatsink_list.InsertColumn(3, "Fins", width=50)
        self.heatsink_list.InsertColumn(4, "Connection", width=100)
        
        self.heatsink_list.SetBackgroundColour(hex_to_wx_color(COLORS['bg_input']))
        self.heatsink_list.SetForegroundColour(hex_to_wx_color(COLORS['fg_text']))
        
        self.btn_add_hs = StyledButton(self, "Add Heatsink")
        self.btn_remove_hs = StyledButton(self, "Remove")
    
    def _create_layout(self):
        """Create layout."""
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Mounting points section
        mp_title = wx.StaticText(self, label="Mounting Points (Thermal Interfaces)")
        mp_title.SetFont(mp_title.GetFont().Bold())
        mp_title.SetForegroundColour(hex_to_wx_color(COLORS['success']))
        main_sizer.Add(mp_title, 0, wx.ALL, 5)
        
        main_sizer.Add(self.mounting_list, 1, wx.EXPAND | wx.ALL, 5)
        
        # Input row for mounting points
        mp_input = wx.BoxSizer(wx.HORIZONTAL)
        mp_input.Add(wx.StaticText(self, label="X:"), 0, wx.ALIGN_CENTER_VERTICAL)
        mp_input.Add(self.spin_mp_x, 0, wx.LEFT, 3)
        mp_input.Add(wx.StaticText(self, label="Y:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
        mp_input.Add(self.spin_mp_y, 0, wx.LEFT, 3)
        mp_input.Add(wx.StaticText(self, label="Dia:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
        mp_input.Add(self.spin_mp_dia, 0, wx.LEFT, 3)
        mp_input.Add(self.choice_contact, 0, wx.LEFT, 8)
        mp_input.Add(self.choice_interface, 0, wx.LEFT, 5)
        main_sizer.Add(mp_input, 0, wx.EXPAND | wx.ALL, 5)
        
        mp_input2 = wx.BoxSizer(wx.HORIZONTAL)
        mp_input2.Add(self.chk_fixed_temp, 0, wx.ALIGN_CENTER_VERTICAL)
        mp_input2.Add(self.spin_mp_temp, 0, wx.LEFT, 5)
        mp_input2.Add(self.btn_add_mp, 0, wx.LEFT, 20)
        mp_input2.Add(self.btn_remove_mp, 0, wx.LEFT, 5)
        mp_input2.Add(self.btn_detect_holes, 0, wx.LEFT, 10)
        main_sizer.Add(mp_input2, 0, wx.ALL, 5)
        
        # Heatsinks section
        hs_title = wx.StaticText(self, label="Heatsinks (from User.1 layer)")
        hs_title.SetFont(hs_title.GetFont().Bold())
        hs_title.SetForegroundColour(hex_to_wx_color(COLORS['success']))
        main_sizer.Add(hs_title, 0, wx.ALL | wx.TOP, 10)
        
        main_sizer.Add(self.heatsink_list, 1, wx.EXPAND | wx.ALL, 5)
        
        hs_btns = wx.BoxSizer(wx.HORIZONTAL)
        hs_btns.Add(self.btn_add_hs, 0)
        hs_btns.Add(self.btn_remove_hs, 0, wx.LEFT, 5)
        main_sizer.Add(hs_btns, 0, wx.ALL, 5)
        
        # Info text
        info = wx.StaticText(
            self, 
            label="Tip: Draw heatsink outlines on User.1 layer in KiCad. They will be auto-detected."
        )
        info.SetForegroundColour(hex_to_wx_color(COLORS['warning']))
        main_sizer.Add(info, 0, wx.ALL, 5)
        
        self.SetSizer(main_sizer)
    
    def _bind_events(self):
        """Bind events."""
        self.btn_add_mp.Bind(wx.EVT_BUTTON, self._on_add_mounting)
        self.btn_remove_mp.Bind(wx.EVT_BUTTON, self._on_remove_mounting)
        self.btn_detect_holes.Bind(wx.EVT_BUTTON, self._on_detect_holes)
        self.btn_add_hs.Bind(wx.EVT_BUTTON, self._on_add_heatsink)
        self.btn_remove_hs.Bind(wx.EVT_BUTTON, self._on_remove_heatsink)
    
    def _load_from_config(self):
        """Load from config."""
        self._refresh_mounting_list()
        self._refresh_heatsink_list()
    
    def _refresh_mounting_list(self):
        """Refresh mounting points list."""
        self.mounting_list.DeleteAllItems()
        for i, mp in enumerate(self.config.mounting_points):
            idx = self.mounting_list.InsertItem(i, mp.point_id)
            self.mounting_list.SetItem(idx, 1, f"{mp.x_mm:.1f}")
            self.mounting_list.SetItem(idx, 2, f"{mp.y_mm:.1f}")
            self.mounting_list.SetItem(idx, 3, f"{mp.hole_diameter_mm:.1f}")
            self.mounting_list.SetItem(idx, 4, mp.contact_type)
            self.mounting_list.SetItem(idx, 5, mp.interface_material)
            temp_str = f"{mp.interface_temp_c:.1f}" if mp.interface_temp_c is not None else "-"
            self.mounting_list.SetItem(idx, 6, temp_str)
    
    def _refresh_heatsink_list(self):
        """Refresh heatsinks list."""
        self.heatsink_list.DeleteAllItems()
        for i, hs in enumerate(self.config.heatsinks):
            idx = self.heatsink_list.InsertItem(i, hs.heatsink_id)
            self.heatsink_list.SetItem(idx, 1, hs.material)
            self.heatsink_list.SetItem(idx, 2, f"{hs.thickness_mm:.1f}")
            self.heatsink_list.SetItem(idx, 3, str(hs.fin_count))
            self.heatsink_list.SetItem(idx, 4, hs.connection_type)
    
    def _on_add_mounting(self, event):
        """Add mounting point."""
        from ..core.config import MountingPoint
        
        point_id = f"MP{len(self.config.mounting_points) + 1}"
        temp = self.spin_mp_temp.GetValue() if self.chk_fixed_temp.GetValue() else None
        
        mp = MountingPoint(
            point_id=point_id,
            x_mm=self.spin_mp_x.GetValue(),
            y_mm=self.spin_mp_y.GetValue(),
            hole_diameter_mm=self.spin_mp_dia.GetValue(),
            contact_type=self.choice_contact.GetStringSelection(),
            interface_material=self.choice_interface.GetStringSelection(),
            interface_temp_c=temp
        )
        self.config.mounting_points.append(mp)
        self._refresh_mounting_list()
    
    def _on_remove_mounting(self, event):
        """Remove selected mounting point."""
        idx = self.mounting_list.GetFirstSelected()
        if idx >= 0 and idx < len(self.config.mounting_points):
            del self.config.mounting_points[idx]
            self._refresh_mounting_list()
    
    def _on_detect_holes(self, event):
        """Detect mounting holes from PCB."""
        if not self.pcb_data or not self.pcb_data.mounting_holes:
            wx.MessageBox(
                "No mounting holes found in PCB.\n"
                "Add mounting hole footprints to your design.",
                "Detect Mounting Holes",
                wx.OK | wx.ICON_INFORMATION
            )
            return
        
        from ..core.config import MountingPoint
        
        count = 0
        for hole in self.pcb_data.mounting_holes:
            # Skip if already exists at this location
            exists = any(
                abs(mp.x_mm - hole.position.x) < 0.5 and abs(mp.y_mm - hole.position.y) < 0.5
                for mp in self.config.mounting_points
            )
            if exists:
                continue
            
            mp = MountingPoint(
                point_id=hole.hole_id,
                x_mm=hole.position.x,
                y_mm=hole.position.y,
                hole_diameter_mm=hole.drill_mm,
                contact_type="conductive" if hole.is_plated else "isolative",
                interface_material="THERMAL_PASTE_STANDARD"
            )
            self.config.mounting_points.append(mp)
            count += 1
        
        self._refresh_mounting_list()
        wx.MessageBox(f"Added {count} mounting points from PCB.", "Detect Complete")
    
    def _on_add_heatsink(self, event):
        """Add heatsink (placeholder for now)."""
        wx.MessageBox(
            "To add a heatsink:\n"
            "1. Draw a polygon on User.1 layer in KiCad\n"
            "2. Run 'Detect from PCB' (coming soon)\n\n"
            "Heatsinks define areas with enhanced thermal conductivity.",
            "Add Heatsink",
            wx.OK | wx.ICON_INFORMATION
        )
    
    def _on_remove_heatsink(self, event):
        """Remove selected heatsink."""
        idx = self.heatsink_list.GetFirstSelected()
        if idx >= 0 and idx < len(self.config.heatsinks):
            del self.config.heatsinks[idx]
            self._refresh_heatsink_list()


class TVACThermalAnalyzerDialog(wx.Dialog if HAS_WX else object):
    """Main dialog for TVAC Thermal Analyzer."""
    
    def __init__(self, parent, board=None):
        if not HAS_WX:
            return
        
        super().__init__(
            parent, 
            title="TVAC Thermal Analyzer",
            size=(DIALOG_WIDTH, DIALOG_HEIGHT),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER
        )
        
        self.board = board
        self.logger = get_logger()
        
        # Resolve PCB path early (also used for config + schematic import)
        self.pcb_path = None
        if board is not None:
            try:
                self.pcb_path = board.GetFileName()
            except Exception:
                self.pcb_path = None

        # Initialize config (persist next to the PCB)
        self.config_manager = ConfigManager(self.pcb_path) if self.pcb_path else ConfigManager()
        if self.pcb_path:
            self.config_manager.load()
        self.config = self.config_manager.get_config()

        # Extract PCB data if board available
        self.pcb_data = None
        if board is not None:
            try:
                extractor = PCBExtractor(board)
                self.pcb_data = extractor.extract_all()
                self.logger.info(f"PCB data extracted: {self.pcb_data.component_count} components")
            except Exception as e:
                self.logger.error(f"Failed to extract PCB data: {e}")
                wx.MessageBox(
                    f"Failed to extract PCB data.\n\n{e}\n\n"
                    "Make sure you're running this from the PCB Editor with a board open.",
                    "TVAC Thermal Analyzer",
                    wx.OK | wx.ICON_WARNING    
                )    
        # Results
        self.thermal_result: Optional[ThermalSimulationResult] = None
        
        self._create_ui()
        self._bind_events()
        self._apply_theme()
    
    def _create_ui(self):
        """Create the main UI."""
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Notebook for tabs
        self.notebook = wx.Notebook(self)
        
        # Tab 1: Simulation Settings
        self.settings_panel = SimulationSettingsPanel(self.notebook, self.config)
        self.notebook.AddPage(self.settings_panel, "Simulation")
        
        # Tab 2: Current Injection
        self.current_panel = CurrentInjectionPanel(self.notebook, self.config, self.pcb_data, board=self.board)
        self.notebook.AddPage(self.current_panel, "Current")
        
        # Tab 3: Component Power
        self.power_panel = ComponentPowerPanel(self.notebook, self.config, self.pcb_data, pcb_path=self.pcb_path)
        self.notebook.AddPage(self.power_panel, "Components")
        
        # Tab 4: Stackup
        self.stackup_panel = StackupPanel(self.notebook, self.config, self.pcb_data)
        self.notebook.AddPage(self.stackup_panel, "Stackup")
        
        # Tab 5: Mounting/Heatsinks
        self.mounting_panel = MountingPanel(self.notebook, self.config, self.pcb_data)
        self.notebook.AddPage(self.mounting_panel, "Mounting")
        
        main_sizer.Add(self.notebook, 1, wx.EXPAND | wx.ALL, 5)
        
        # Bottom button bar
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.btn_run = StyledButton(self, "Run Simulation")
        self.btn_run.SetMinSize((150, 35))
        
        self.btn_view_results = StyledButton(self, "View Results")
        self.btn_view_results.Enable(False)
        
        self.btn_export = StyledButton(self, "Export Report")
        self.btn_export.Enable(False)
        
        self.btn_save = StyledButton(self, "Save Config")
        self.btn_close = StyledButton(self, "Close")
        
        btn_sizer.Add(self.btn_run, 0, wx.RIGHT, 10)
        btn_sizer.Add(self.btn_view_results, 0, wx.RIGHT, 10)
        btn_sizer.Add(self.btn_export, 0, wx.RIGHT, 10)
        btn_sizer.AddStretchSpacer()
        btn_sizer.Add(self.btn_save, 0, wx.RIGHT, 5)
        btn_sizer.Add(self.btn_close, 0)
        
        main_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Status bar
        self.status_bar = wx.StaticText(self, label="Ready")
        self.status_bar.SetForegroundColour(hex_to_wx_color(COLORS['fg_label']))
        main_sizer.Add(self.status_bar, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        
        self.SetSizer(main_sizer)
    
    def _bind_events(self):
        """Bind events."""
        self.btn_run.Bind(wx.EVT_BUTTON, self._on_run)
        self.btn_view_results.Bind(wx.EVT_BUTTON, self._on_view_results)
        self.btn_export.Bind(wx.EVT_BUTTON, self._on_export)
        self.btn_save.Bind(wx.EVT_BUTTON, self._on_save)
        self.btn_close.Bind(wx.EVT_BUTTON, self._on_close)
        self.Bind(wx.EVT_CLOSE, self._on_window_close)
    
    def _apply_theme(self):
        """Apply dark theme."""
        self.SetBackgroundColour(hex_to_wx_color(COLORS['bg_dark']))
        self.SetForegroundColour(hex_to_wx_color(COLORS['fg_text']))
    
    def _on_run(self, event):
        """Run thermal simulation."""
        # Save settings to config
        self.settings_panel.save_to_config()
        
        # Validate
        if not self.pcb_data:
            wx.MessageBox(
                "No PCB data available. Please open a PCB first.",
                "Error",
                wx.OK | wx.ICON_ERROR
            )
            return
        
        # Check current balance
        total_current = sum(p.current_a for p in self.config.current_injection_points)
        if abs(total_current) > 0.001:
            result = wx.MessageBox(
                f"Current injection is not balanced (net: {total_current:.3f}A).\n"
                "Continue anyway?",
                "Warning",
                wx.YES_NO | wx.ICON_WARNING
            )
            if result != wx.YES:
                return
        
        # Show progress dialog
        progress_dlg = ProgressDialog(self, "Running Thermal Simulation")
        
        # Run simulation in background thread
        def run_simulation():
            try:
                engine = ThermalAnalysisEngine()
                engine.set_progress_callback(progress_dlg.update_progress)
                self.thermal_result = engine.run_analysis(self.pcb_data, self.config)
                wx.CallAfter(self._on_simulation_complete, True)
            except Exception as e:
                self.logger.exception(f"Simulation failed: {e}")
                wx.CallAfter(self._on_simulation_complete, False, str(e))
        
        thread = threading.Thread(target=run_simulation)
        thread.start()
        
        result = progress_dlg.ShowModal()
        if result == wx.ID_CANCEL:
            # TODO: Implement cancellation
            pass
        
        progress_dlg.Destroy()
    
    def _on_simulation_complete(self, success: bool, error_msg: str = ""):
        """Handle simulation completion."""
        if success:
            self.btn_view_results.Enable(True)
            self.btn_export.Enable(True)
            
            # Show summary
            if self.thermal_result and self.thermal_result.frames:
                final = self.thermal_result.frames[-1]
                self.status_bar.SetLabel(
                    f"Complete: Tmax={final.max_temp:.1f}°C, "
                    f"Tmin={final.min_temp:.1f}°C, "
                    f"Tavg={final.avg_temp:.1f}°C"
                )
                
                wx.MessageBox(
                    f"Simulation Complete!\n\n"
                    f"Maximum Temperature: {final.max_temp:.1f}°C\n"
                    f"Minimum Temperature: {final.min_temp:.1f}°C\n"
                    f"Average Temperature: {final.avg_temp:.1f}°C\n"
                    f"Time: {self.thermal_result.total_simulation_time:.1f}s",
                    "Simulation Complete",
                    wx.OK | wx.ICON_INFORMATION
                )
        else:
            self.status_bar.SetLabel(f"Simulation failed: {error_msg}")
            wx.MessageBox(f"Simulation failed:\n{error_msg}", "Error", wx.OK | wx.ICON_ERROR)
    
    def _on_view_results(self, event):
        """View simulation results."""
        if self.thermal_result:
            # TODO: Open results viewer with heatmap
            wx.MessageBox(
                "Results viewer will display the temperature heatmap overlay.\n"
                "This feature is being finalized.",
                "View Results",
                wx.OK | wx.ICON_INFORMATION
            )
    
    def _on_export(self, event):
        """Export report."""
        if not self.thermal_result:
            return
        
        # Ask for save location
        dlg = wx.FileDialog(
            self,
            "Save Report",
            wildcard="PDF files (*.pdf)|*.pdf",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        )
        
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            try:
                from ..utils.report_generator import ThermalReportGenerator, ReportConfig
                
                report_config = ReportConfig(
                    title="TVAC Thermal Analysis Report",
                    project_name=self.config.stackup.substrate_material,
                    author="TVAC Thermal Analyzer",
                )
                
                generator = ThermalReportGenerator(config=report_config)
                
                # Get mesh from thermal analysis engine if available
                mesh = None
                current_result = None
                
                success = generator.generate_report(
                    output_path=path,
                    thermal_results=self.thermal_result,
                    current_results=current_result,
                    mesh=mesh,
                    analysis_config=self.config,
                    pcb_data=self.pcb_data
                )
                
                if success:
                    wx.MessageBox(f"Report saved to:\n{path}", "Export Complete")
                else:
                    wx.MessageBox("Report generation failed.", "Error", wx.ICON_ERROR)
                
            except Exception as e:
                self.logger.exception(f"Report generation failed: {e}")
                wx.MessageBox(f"Failed to generate report:\n{e}", "Error", wx.ICON_ERROR)
        
        dlg.Destroy()
    
    def _on_save(self, event):
        """Save configuration."""
        self.settings_panel.save_to_config()
        
        if self.config_manager.save():
            wx.MessageBox("Configuration saved.", "Save Complete")
        else:
            wx.MessageBox("Failed to save configuration.", "Error", wx.ICON_ERROR)
    
    def _on_close(self, event):
        """Close dialog via button click."""
        self.Close()
    
    def _on_window_close(self, event):
        """Handle window close event (X button or Close() call)."""
        # Save config before closing
        try:
            self.settings_panel.save_to_config()
            self.config_manager.save()
        except Exception:
            pass
        
        # Destroy the dialog
        self.Destroy()


# Import datetime for report generation
from datetime import datetime


__all__ = [
    'TVACThermalAnalyzerDialog',
    'ProgressDialog',
]
