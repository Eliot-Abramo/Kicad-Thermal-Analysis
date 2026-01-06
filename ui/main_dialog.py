"""
TVAC Thermal Analyzer - Main Dialog
===================================
Main application dialog with Interactive BOM-style PCB visualization.

Author: Space Electronics Thermal Analysis Tool
Version: 2.0.0
"""

import wx
import wx.lib.agw.aui as aui
from typing import Optional, List, Dict
from pathlib import Path

try:
    import pcbnew
    HAS_PCBNEW = True
except ImportError:
    HAS_PCBNEW = False

from .design_system import (
    Colors, Spacing, Fonts,
    BaseFrame, IconButton, AccentButton,
    InfoBanner, SectionHeader, StatusIndicator,
    SearchBox, ProgressDialog, TabPanel
)
from .pcb_visualization import (
    PCBVisualizationPanel, LayerTogglePanel, VisualizationLayer
)

from ..core.pcb_extractor import PCBExtractor, PCBData
from ..core.config import (
    ThermalAnalysisConfig, ConfigManager,
    ComponentPowerConfig, HeatsinkConfig, MountingPointConfig
)
from ..core.constants import MaterialsDatabase, ComponentThermalDatabase


class ComponentPowerPanel(TabPanel):
    """Panel for configuring component power dissipation."""
    
    def __init__(self, parent, config: ThermalAnalysisConfig, 
                 pcb_data: PCBData, viz_panel: PCBVisualizationPanel):
        super().__init__(parent)
        self.config = config
        self.pcb_data = pcb_data
        self.viz_panel = viz_panel
        
        self._build_ui()
        self._import_power_from_pcb()
        self._populate_list()
    
    def _build_ui(self):
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Header
        header = SectionHeader(self, "Component Power", 
                              "Set power dissipation for heat-generating components")
        sizer.Add(header, 0, wx.ALL | wx.EXPAND, Spacing.MD)
        
        # Search and filter
        filter_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.search = SearchBox(self, "Search components...")
        self.search.Bind(wx.EVT_TEXT, self._on_search)
        filter_sizer.Add(self.search, 0, wx.RIGHT, Spacing.MD)
        
        # Quick filters
        btn_u = wx.Button(self, label="U*", size=(40, 26))
        btn_u.SetToolTip("Show ICs only")
        btn_u.Bind(wx.EVT_BUTTON, lambda e: self._filter_prefix("U"))
        filter_sizer.Add(btn_u, 0, wx.RIGHT, Spacing.XS)
        
        btn_q = wx.Button(self, label="Q*", size=(40, 26))
        btn_q.SetToolTip("Show transistors only")
        btn_q.Bind(wx.EVT_BUTTON, lambda e: self._filter_prefix("Q"))
        filter_sizer.Add(btn_q, 0, wx.RIGHT, Spacing.XS)
        
        btn_r = wx.Button(self, label="R*", size=(40, 26))
        btn_r.SetToolTip("Show resistors only")
        btn_r.Bind(wx.EVT_BUTTON, lambda e: self._filter_prefix("R"))
        filter_sizer.Add(btn_r, 0, wx.RIGHT, Spacing.XS)
        
        btn_all = wx.Button(self, label="All", size=(40, 26))
        btn_all.Bind(wx.EVT_BUTTON, lambda e: self._filter_prefix(""))
        filter_sizer.Add(btn_all, 0)
        
        sizer.Add(filter_sizer, 0, wx.ALL | wx.EXPAND, Spacing.MD)
        
        # Component list
        self.list = wx.ListCtrl(self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL | wx.BORDER_SIMPLE)
        self.list.SetBackgroundColour(Colors.INPUT_BG)
        self.list.SetFont(Fonts.body())
        
        self.list.InsertColumn(0, "Reference", width=80)
        self.list.InsertColumn(1, "Value", width=100)
        self.list.InsertColumn(2, "Footprint", width=120)
        self.list.InsertColumn(3, "Power (W)", width=80)
        self.list.InsertColumn(4, "Source", width=80)
        
        self.list.Bind(wx.EVT_LIST_ITEM_SELECTED, self._on_select)
        self.list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self._on_edit)
        
        sizer.Add(self.list, 1, wx.ALL | wx.EXPAND, Spacing.MD)
        
        # Action buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        btn_edit = IconButton(self, "Edit Power", icon='edit', size=(100, 32))
        btn_edit.Bind(wx.EVT_BUTTON, self._on_edit)
        btn_sizer.Add(btn_edit, 0, wx.RIGHT, Spacing.SM)
        
        btn_clear = IconButton(self, "Clear", icon='delete', size=(70, 32))
        btn_clear.Bind(wx.EVT_BUTTON, self._on_clear)
        btn_sizer.Add(btn_clear, 0, wx.RIGHT, Spacing.SM)
        
        btn_sizer.AddStretchSpacer()
        
        self.total_label = wx.StaticText(self, label="Total: 0.000 W")
        self.total_label.SetFont(Fonts.bold())
        btn_sizer.Add(self.total_label, 0, wx.ALIGN_CENTER_VERTICAL)
        
        sizer.Add(btn_sizer, 0, wx.ALL | wx.EXPAND, Spacing.MD)
        
        self.SetSizer(sizer)
    
    def _import_power_from_pcb(self):
        """Import POWER_DISSIPATION from PCB component fields."""
        imported_count = 0
        
        for comp in self.pcb_data.components:
            if comp.power_dissipation_w > 0:
                # Check if not already set
                existing = any(cp.reference == comp.reference for cp in self.config.component_power)
                if not existing:
                    self.config.set_component_power(
                        comp.reference, comp.power_dissipation_w, "schematic_field"
                    )
                    imported_count += 1
        
        if imported_count > 0:
            print(f"Imported power from {imported_count} component(s)")
    
    def _populate_list(self, filter_text: str = ""):
        """Populate component list."""
        self.list.DeleteAllItems()
        
        filter_text = filter_text.lower()
        
        for comp in self.pcb_data.components:
            if filter_text and filter_text not in comp.reference.lower():
                continue
            
            # Get power config
            power = 0.0
            source = ""
            for cp in self.config.component_power:
                if cp.reference == comp.reference:
                    power = cp.power_w
                    source = cp.source
                    break
            
            idx = self.list.InsertItem(self.list.GetItemCount(), comp.reference)
            self.list.SetItem(idx, 1, comp.value)
            self.list.SetItem(idx, 2, comp.footprint[:20])
            self.list.SetItem(idx, 3, f"{power:.4f}" if power > 0 else "—")
            self.list.SetItem(idx, 4, source)
            
            # Highlight if has power
            if power > 0:
                self.list.SetItemBackgroundColour(idx, Colors.WARNING_BG)
        
        self._update_total()
    
    def _update_total(self):
        """Update total power display."""
        total = sum(cp.power_w for cp in self.config.component_power)
        self.total_label.SetLabel(f"Total: {total:.3f} W")
    
    def _filter_prefix(self, prefix: str):
        """Filter by reference prefix."""
        self.search.text.SetValue(prefix)
        self._populate_list(prefix)
    
    def _on_search(self, event):
        self._populate_list(self.search.GetValue())
    
    def _on_select(self, event):
        """Highlight selected component on PCB."""
        idx = event.GetIndex()
        if idx >= 0:
            ref = self.list.GetItemText(idx, 0)
            self.viz_panel.clear_selection()
            self.viz_panel.select_component(ref)
            self.viz_panel.Refresh()
    
    def _on_edit(self, event):
        """Edit component power."""
        idx = self.list.GetFirstSelected()
        if idx < 0:
            return
        
        ref = self.list.GetItemText(idx, 0)
        
        # Get current power
        current = 0.0
        for cp in self.config.component_power:
            if cp.reference == ref:
                current = cp.power_w
                break
        
        dlg = wx.TextEntryDialog(
            self, f"Power dissipation for {ref} (Watts):",
            "Edit Component Power", f"{current:.4f}"
        )
        
        if dlg.ShowModal() == wx.ID_OK:
            try:
                power = float(dlg.GetValue())
                self.config.set_component_power(ref, power, "manual")
                self._populate_list(self.search.GetValue())
            except ValueError:
                wx.MessageBox("Invalid power value", "Error", wx.ICON_ERROR)
        
        dlg.Destroy()
    
    def _on_clear(self, event):
        """Clear power for selected component."""
        idx = self.list.GetFirstSelected()
        if idx < 0:
            return
        
        ref = self.list.GetItemText(idx, 0)
        
        # Remove from config
        self.config.component_power = [
            cp for cp in self.config.component_power if cp.reference != ref
        ]
        self._populate_list(self.search.GetValue())


class HeatsinkPanel(TabPanel):
    """Panel for configuring heatsinks from user layers."""
    
    def __init__(self, parent, config: ThermalAnalysisConfig,
                 pcb_data: PCBData, viz_panel: PCBVisualizationPanel):
        super().__init__(parent)
        self.config = config
        self.pcb_data = pcb_data
        self.viz_panel = viz_panel
        
        self._build_ui()
        self._populate_list()
    
    def _build_ui(self):
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        header = SectionHeader(self, "Heatsinks",
                              "Import heatsink polygons from User layers")
        sizer.Add(header, 0, wx.ALL | wx.EXPAND, Spacing.MD)
        
        # Info about detected shapes
        detected_layers = list(self.pcb_data.user_shapes.keys())
        detected_count = sum(len(shapes) for shapes in self.pcb_data.user_shapes.values())
        
        if detected_count > 0:
            info = InfoBanner(self, f"Found {detected_count} shape(s) on layers: {', '.join(detected_layers)}", 
                             style='success')
        else:
            info = InfoBanner(self, "No shapes found on User layers. Draw rectangles/polygons on User.1", 
                             style='warning')
        sizer.Add(info, 0, wx.ALL | wx.EXPAND, Spacing.SM)
        
        # Layer selection
        layer_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        layer_sizer.Add(wx.StaticText(self, label="Source Layer:"), 
                       0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, Spacing.SM)
        
        available_layers = list(self.pcb_data.user_shapes.keys())
        if not available_layers:
            available_layers = ["User.1", "User.2", "User.3"]
        
        self.layer_choice = wx.Choice(self, choices=available_layers)
        if self.config.layer_mapping.heatsink_layer in available_layers:
            self.layer_choice.SetStringSelection(self.config.layer_mapping.heatsink_layer)
        elif available_layers:
            self.layer_choice.SetSelection(0)
        
        layer_sizer.Add(self.layer_choice, 0, wx.RIGHT, Spacing.MD)
        
        btn_refresh = IconButton(self, "Import Shapes", icon='refresh', size=(120, 32))
        btn_refresh.Bind(wx.EVT_BUTTON, self._on_refresh)
        layer_sizer.Add(btn_refresh, 0)
        
        sizer.Add(layer_sizer, 0, wx.ALL | wx.EXPAND, Spacing.MD)
        
        # Heatsink list
        self.list = wx.ListCtrl(self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL | wx.BORDER_SIMPLE)
        self.list.SetBackgroundColour(Colors.INPUT_BG)
        
        self.list.InsertColumn(0, "ID", width=100)
        self.list.InsertColumn(1, "Material", width=120)
        self.list.InsertColumn(2, "Area (mm²)", width=80)
        self.list.InsertColumn(3, "Thickness", width=80)
        self.list.InsertColumn(4, "Emissivity", width=80)
        
        self.list.Bind(wx.EVT_LIST_ITEM_SELECTED, self._on_select)
        self.list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self._on_edit)
        
        sizer.Add(self.list, 1, wx.ALL | wx.EXPAND, Spacing.MD)
        
        # Material default
        mat_sizer = wx.BoxSizer(wx.HORIZONTAL)
        mat_sizer.Add(wx.StaticText(self, label="Default Material:"),
                     0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, Spacing.SM)
        
        materials = list(MaterialsDatabase.HEATSINK_MATERIALS.keys())
        self.material_choice = wx.Choice(self, choices=materials)
        self.material_choice.SetStringSelection("ALUMINUM_6061")
        mat_sizer.Add(self.material_choice, 0)
        
        sizer.Add(mat_sizer, 0, wx.ALL | wx.EXPAND, Spacing.MD)
        
        self.SetSizer(sizer)
    
    def _populate_list(self):
        """Populate heatsink list."""
        self.list.DeleteAllItems()
        
        for hs in self.config.heatsinks:
            area = self._calculate_polygon_area(hs.polygon_points)
            
            mat = MaterialsDatabase.HEATSINK_MATERIALS.get(hs.material)
            emissivity = hs.emissivity_override or (mat.emissivity if mat else 0.9)
            
            idx = self.list.InsertItem(self.list.GetItemCount(), hs.heatsink_id)
            self.list.SetItem(idx, 1, hs.material)
            self.list.SetItem(idx, 2, f"{area:.1f}")
            self.list.SetItem(idx, 3, f"{hs.thickness_mm:.1f} mm")
            self.list.SetItem(idx, 4, f"{emissivity:.2f}")
        
        self._update_visualization()
    
    def _calculate_polygon_area(self, points: List) -> float:
        """Calculate polygon area using shoelace formula."""
        if len(points) < 3:
            return 0.0
        
        area = 0.0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        return abs(area) / 2.0
    
    def _update_visualization(self):
        """Update PCB visualization with heatsinks."""
        # Clear existing heatsinks
        self.viz_panel.heatsinks.clear()
        
        for hs in self.config.heatsinks:
            self.viz_panel.add_heatsink(hs.heatsink_id, hs.polygon_points, hs.material)
        
        self.viz_panel.Refresh()
    
    def _on_select(self, event):
        """Highlight selected heatsink on PCB."""
        idx = event.GetIndex()
        if idx >= 0:
            hs_id = self.list.GetItemText(idx, 0)
            self.viz_panel.clear_selection()
            self.viz_panel.select_heatsink(hs_id)
            self.viz_panel.Refresh()
    
    def _on_refresh(self, event):
        """Import shapes from selected layer."""
        layer = self.layer_choice.GetStringSelection()
        if not layer:
            return
        
        # Update config layer mapping
        self.config.layer_mapping.heatsink_layer = layer
        
        # Import shapes
        shapes = self.pcb_data.user_shapes.get(layer, [])
        
        # Clear existing heatsinks from this layer
        self.config.heatsinks = [
            hs for hs in self.config.heatsinks 
            if not hs.heatsink_id.startswith(f"HS_{layer}_")
        ]
        
        default_material = self.material_choice.GetStringSelection()
        
        for i, shape in enumerate(shapes):
            points = [(p.x, p.y) for p in shape.points]
            
            hs = HeatsinkConfig(
                heatsink_id=f"HS_{layer}_{i+1}",
                material=default_material,
                polygon_points=points,
                thickness_mm=3.0
            )
            self.config.heatsinks.append(hs)
        
        self._populate_list()
        
        wx.MessageBox(f"Imported {len(shapes)} shape(s) from {layer}", 
                     "Import Complete", wx.ICON_INFORMATION)
    
    def _on_edit(self, event):
        """Edit heatsink properties."""
        idx = self.list.GetFirstSelected()
        if idx < 0:
            return
        
        hs_id = self.list.GetItemText(idx, 0)
        
        for hs in self.config.heatsinks:
            if hs.heatsink_id == hs_id:
                # Simple dialog for now
                dlg = wx.TextEntryDialog(
                    self, "Thickness (mm):",
                    f"Edit {hs_id}", f"{hs.thickness_mm:.1f}"
                )
                
                if dlg.ShowModal() == wx.ID_OK:
                    try:
                        hs.thickness_mm = float(dlg.GetValue())
                        self._populate_list()
                    except ValueError:
                        pass
                
                dlg.Destroy()
                break


class MountingPanel(TabPanel):
    """Panel for configuring thermal mounting points."""
    
    def __init__(self, parent, config: ThermalAnalysisConfig,
                 pcb_data: PCBData, viz_panel: PCBVisualizationPanel):
        super().__init__(parent)
        self.config = config
        self.pcb_data = pcb_data
        self.viz_panel = viz_panel
        
        self._build_ui()
        self._populate_list()
    
    def _build_ui(self):
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        header = SectionHeader(self, "Mounting Points",
                              "Define thermal boundary conditions at mounting locations")
        sizer.Add(header, 0, wx.ALL | wx.EXPAND, Spacing.MD)
        
        # Auto-detect button
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        btn_detect = IconButton(self, "Auto-Detect Holes", icon='search', size=(140, 32))
        btn_detect.Bind(wx.EVT_BUTTON, self._on_detect)
        btn_sizer.Add(btn_detect, 0, wx.RIGHT, Spacing.SM)
        
        btn_add = IconButton(self, "Add Manual", icon='add', size=(100, 32))
        btn_add.Bind(wx.EVT_BUTTON, self._on_add)
        btn_sizer.Add(btn_add, 0)
        
        sizer.Add(btn_sizer, 0, wx.ALL | wx.EXPAND, Spacing.MD)
        
        # Mounting point list
        self.list = wx.ListCtrl(self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL | wx.BORDER_SIMPLE)
        self.list.SetBackgroundColour(Colors.INPUT_BG)
        
        self.list.InsertColumn(0, "ID", width=70)
        self.list.InsertColumn(1, "X (mm)", width=70)
        self.list.InsertColumn(2, "Y (mm)", width=70)
        self.list.InsertColumn(3, "Diameter", width=70)
        self.list.InsertColumn(4, "Fixed Temp", width=80)
        self.list.InsertColumn(5, "Interface", width=80)
        
        self.list.Bind(wx.EVT_LIST_ITEM_SELECTED, self._on_select)
        self.list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self._on_edit)
        
        sizer.Add(self.list, 1, wx.ALL | wx.EXPAND, Spacing.MD)
        
        # Action buttons
        action_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        btn_edit = IconButton(self, "Set Temp", icon='edit', size=(90, 32))
        btn_edit.Bind(wx.EVT_BUTTON, self._on_edit)
        action_sizer.Add(btn_edit, 0, wx.RIGHT, Spacing.SM)
        
        btn_remove = IconButton(self, "Remove", icon='delete', size=(80, 32))
        btn_remove.Bind(wx.EVT_BUTTON, self._on_remove)
        action_sizer.Add(btn_remove, 0)
        
        sizer.Add(action_sizer, 0, wx.ALL | wx.EXPAND, Spacing.MD)
        
        self.SetSizer(sizer)
    
    def _populate_list(self):
        """Populate mounting point list."""
        self.list.DeleteAllItems()
        
        for mp in self.config.mounting_points:
            idx = self.list.InsertItem(self.list.GetItemCount(), mp.point_id)
            self.list.SetItem(idx, 1, f"{mp.x_mm:.1f}")
            self.list.SetItem(idx, 2, f"{mp.y_mm:.1f}")
            self.list.SetItem(idx, 3, f"{mp.diameter_mm:.1f}")
            
            if mp.fixed_temp_c is not None:
                self.list.SetItem(idx, 4, f"{mp.fixed_temp_c:.1f}°C")
                self.list.SetItemBackgroundColour(idx, Colors.SUCCESS_BG)
            else:
                self.list.SetItem(idx, 4, "—")
            
            self.list.SetItem(idx, 5, mp.interface_material)
        
        self._update_visualization()
    
    def _update_visualization(self):
        """Update PCB visualization with mounting points."""
        self.viz_panel.mounting_points.clear()
        
        for mp in self.config.mounting_points:
            self.viz_panel.add_mounting_point(
                mp.point_id, mp.x_mm, mp.y_mm, mp.diameter_mm,
                is_thermal=(mp.fixed_temp_c is not None)
            )
        
        self.viz_panel.Refresh()
    
    def _on_select(self, event):
        """Highlight selected mounting point on PCB."""
        idx = event.GetIndex()
        if idx >= 0:
            mp_id = self.list.GetItemText(idx, 0)
            self.viz_panel.clear_selection()
            self.viz_panel.select_mounting_point(mp_id)
            self.viz_panel.Refresh()
    
    def _on_detect(self, event):
        """Auto-detect mounting holes from PCB."""
        # Import from PCB data
        for i, hole in enumerate(self.pcb_data.mounting_holes):
            existing = any(
                abs(mp.x_mm - hole.position.x) < 1 and abs(mp.y_mm - hole.position.y) < 1
                for mp in self.config.mounting_points
            )
            
            if not existing:
                mp = MountingPointConfig(
                    point_id=hole.hole_id or f"MP{i+1}",
                    x_mm=hole.position.x,
                    y_mm=hole.position.y,
                    diameter_mm=hole.drill_mm,
                    contact_type="conductive" if hole.is_plated else "isolative"
                )
                self.config.mounting_points.append(mp)
        
        self._populate_list()
        wx.MessageBox(f"Found {len(self.pcb_data.mounting_holes)} mounting hole(s)",
                     "Detection Complete", wx.ICON_INFORMATION)
    
    def _on_add(self, event):
        """Add manual mounting point."""
        dlg = wx.TextEntryDialog(
            self, "Enter X,Y position (mm), e.g. '10.5, 25.0':",
            "Add Mounting Point", ""
        )
        
        if dlg.ShowModal() == wx.ID_OK:
            try:
                parts = dlg.GetValue().split(',')
                x = float(parts[0].strip())
                y = float(parts[1].strip())
                
                mp_id = f"MP{len(self.config.mounting_points) + 1}"
                mp = MountingPointConfig(
                    point_id=mp_id, x_mm=x, y_mm=y, diameter_mm=3.2
                )
                self.config.mounting_points.append(mp)
                self._populate_list()
                
            except (ValueError, IndexError):
                wx.MessageBox("Invalid format. Use: X, Y", "Error", wx.ICON_ERROR)
        
        dlg.Destroy()
    
    def _on_edit(self, event):
        """Edit mounting point fixed temperature."""
        idx = self.list.GetFirstSelected()
        if idx < 0:
            return
        
        mp_id = self.list.GetItemText(idx, 0)
        
        for mp in self.config.mounting_points:
            if mp.point_id == mp_id:
                current = str(mp.fixed_temp_c) if mp.fixed_temp_c is not None else ""
                
                dlg = wx.TextEntryDialog(
                    self, "Fixed temperature (°C), leave blank for none:",
                    f"Edit {mp_id}", current
                )
                
                if dlg.ShowModal() == wx.ID_OK:
                    val = dlg.GetValue().strip()
                    if val:
                        try:
                            mp.fixed_temp_c = float(val)
                        except ValueError:
                            pass
                    else:
                        mp.fixed_temp_c = None
                    
                    self._populate_list()
                
                dlg.Destroy()
                break
    
    def _on_remove(self, event):
        """Remove mounting point."""
        idx = self.list.GetFirstSelected()
        if idx < 0:
            return
        
        mp_id = self.list.GetItemText(idx, 0)
        self.config.mounting_points = [
            mp for mp in self.config.mounting_points if mp.point_id != mp_id
        ]
        self._populate_list()


class SimulationPanel(TabPanel):
    """Panel for simulation settings."""
    
    def __init__(self, parent, config: ThermalAnalysisConfig):
        super().__init__(parent)
        self.config = config
        self._build_ui()
    
    def _build_ui(self):
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        header = SectionHeader(self, "Simulation Settings",
                              "Configure thermal simulation parameters")
        sizer.Add(header, 0, wx.ALL | wx.EXPAND, Spacing.MD)
        
        # Mode selection
        mode_box = wx.StaticBox(self, label="Analysis Mode")
        mode_sizer = wx.StaticBoxSizer(mode_box, wx.VERTICAL)
        
        self.rb_steady = wx.RadioButton(self, label="Steady State", style=wx.RB_GROUP)
        self.rb_transient = wx.RadioButton(self, label="Transient")
        
        if self.config.simulation.mode == "transient":
            self.rb_transient.SetValue(True)
        else:
            self.rb_steady.SetValue(True)
        
        mode_sizer.Add(self.rb_steady, 0, wx.ALL, Spacing.SM)
        mode_sizer.Add(self.rb_transient, 0, wx.ALL, Spacing.SM)
        
        sizer.Add(mode_sizer, 0, wx.ALL | wx.EXPAND, Spacing.MD)
        
        # Environment settings
        env_box = wx.StaticBox(self, label="Environment")
        env_sizer = wx.StaticBoxSizer(env_box, wx.VERTICAL)
        
        grid = wx.FlexGridSizer(4, 2, Spacing.SM, Spacing.MD)
        
        grid.Add(wx.StaticText(self, label="Ambient Temp (°C):"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.txt_ambient = wx.TextCtrl(self, value=str(self.config.simulation.ambient_temp_c), size=(80, -1))
        grid.Add(self.txt_ambient, 0)
        
        grid.Add(wx.StaticText(self, label="Chamber Wall (°C):"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.txt_chamber = wx.TextCtrl(self, value=str(self.config.simulation.chamber_wall_temp_c), size=(80, -1))
        grid.Add(self.txt_chamber, 0)
        
        grid.Add(wx.StaticText(self, label="Initial Board (°C):"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.txt_initial = wx.TextCtrl(self, value=str(self.config.simulation.initial_board_temp_c), size=(80, -1))
        grid.Add(self.txt_initial, 0)
        
        grid.Add(wx.StaticText(self, label=""), 0)
        self.chk_radiation = wx.CheckBox(self, label="Include Radiation")
        self.chk_radiation.SetValue(self.config.simulation.include_radiation)
        grid.Add(self.chk_radiation, 0)
        
        env_sizer.Add(grid, 0, wx.ALL, Spacing.SM)
        sizer.Add(env_sizer, 0, wx.ALL | wx.EXPAND, Spacing.MD)
        
        # Mesh settings
        mesh_box = wx.StaticBox(self, label="Mesh")
        mesh_sizer = wx.StaticBoxSizer(mesh_box, wx.VERTICAL)
        
        mesh_grid = wx.FlexGridSizer(2, 2, Spacing.SM, Spacing.MD)
        
        mesh_grid.Add(wx.StaticText(self, label="Resolution (mm):"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.txt_resolution = wx.TextCtrl(self, value=str(self.config.simulation.resolution_mm), size=(80, -1))
        mesh_grid.Add(self.txt_resolution, 0)
        
        mesh_grid.Add(wx.StaticText(self, label=""), 0)
        self.chk_adaptive = wx.CheckBox(self, label="Adaptive Refinement")
        self.chk_adaptive.SetValue(self.config.simulation.use_adaptive_mesh)
        mesh_grid.Add(self.chk_adaptive, 0)
        
        mesh_sizer.Add(mesh_grid, 0, wx.ALL, Spacing.SM)
        sizer.Add(mesh_sizer, 0, wx.ALL | wx.EXPAND, Spacing.MD)
        
        # Transient settings
        trans_box = wx.StaticBox(self, label="Transient Settings")
        trans_sizer = wx.StaticBoxSizer(trans_box, wx.VERTICAL)
        
        trans_grid = wx.FlexGridSizer(3, 2, Spacing.SM, Spacing.MD)
        
        trans_grid.Add(wx.StaticText(self, label="Duration (s):"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.txt_duration = wx.TextCtrl(self, value=str(self.config.simulation.duration_s), size=(80, -1))
        trans_grid.Add(self.txt_duration, 0)
        
        trans_grid.Add(wx.StaticText(self, label="Timestep (s):"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.txt_timestep = wx.TextCtrl(self, value=str(self.config.simulation.timestep_s), size=(80, -1))
        trans_grid.Add(self.txt_timestep, 0)
        
        trans_grid.Add(wx.StaticText(self, label="Output Interval (s):"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.txt_output = wx.TextCtrl(self, value=str(self.config.simulation.output_interval_s), size=(80, -1))
        trans_grid.Add(self.txt_output, 0)
        
        trans_sizer.Add(trans_grid, 0, wx.ALL, Spacing.SM)
        sizer.Add(trans_sizer, 0, wx.ALL | wx.EXPAND, Spacing.MD)
        
        self.SetSizer(sizer)
    
    def save_settings(self):
        """Save settings back to config."""
        sim = self.config.simulation
        
        sim.mode = "transient" if self.rb_transient.GetValue() else "steady_state"
        
        try:
            sim.ambient_temp_c = float(self.txt_ambient.GetValue())
            sim.chamber_wall_temp_c = float(self.txt_chamber.GetValue())
            sim.initial_board_temp_c = float(self.txt_initial.GetValue())
            sim.resolution_mm = float(self.txt_resolution.GetValue())
            sim.duration_s = float(self.txt_duration.GetValue())
            sim.timestep_s = float(self.txt_timestep.GetValue())
            sim.output_interval_s = float(self.txt_output.GetValue())
        except ValueError:
            pass
        
        sim.include_radiation = self.chk_radiation.GetValue()
        sim.use_adaptive_mesh = self.chk_adaptive.GetValue()


class MainDialog(BaseFrame):
    """Main TVAC Thermal Analyzer dialog."""
    
    def __init__(self, parent, board):
        super().__init__(parent, "TVAC Thermal Analyzer",
                        size=(1400, 900), min_size=(1000, 700))
        
        self.board = board
        
        # Extract PCB data
        self.pcb_data = PCBExtractor(board).extract_all()
        
        # Load/create config
        pcb_path = board.GetFileName() if board else ""
        self.config_manager = ConfigManager(pcb_path)
        self.config = self.config_manager.get_config()
        
        self._build_ui()
        self._populate_visualization()
        
        self.Bind(wx.EVT_CLOSE, self._on_close)
    
    def _build_ui(self):
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Header
        header = self._create_header()
        main_sizer.Add(header, 0, wx.EXPAND)
        
        # Main content with splitter
        splitter = wx.SplitterWindow(self, style=wx.SP_LIVE_UPDATE)
        
        # Left: Notebook with config tabs
        left_panel = wx.Panel(splitter)
        left_panel.SetBackgroundColour(Colors.PANEL_BG)
        left_sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.notebook = wx.Notebook(left_panel)
        
        # Create tabs (we need viz_panel first)
        right_panel = wx.Panel(splitter)
        right_panel.SetBackgroundColour(Colors.BACKGROUND)
        right_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # PCB Visualization
        viz_header = SectionHeader(right_panel, "PCB Preview", 
                                   "Interactive thermal visualization")
        right_sizer.Add(viz_header, 0, wx.ALL | wx.EXPAND, Spacing.MD)
        
        self.viz_panel = PCBVisualizationPanel(right_panel)
        right_sizer.Add(self.viz_panel, 1, wx.ALL | wx.EXPAND, Spacing.MD)
        
        # Layer toggles
        self.layer_toggles = LayerTogglePanel(right_panel, self.viz_panel)
        right_sizer.Add(self.layer_toggles, 0, wx.ALL | wx.EXPAND, Spacing.MD)
        
        right_panel.SetSizer(right_sizer)
        
        # Now create tabs with viz_panel reference
        self.comp_panel = ComponentPowerPanel(self.notebook, self.config, 
                                              self.pcb_data, self.viz_panel)
        self.notebook.AddPage(self.comp_panel, "Components")
        
        self.heatsink_panel = HeatsinkPanel(self.notebook, self.config,
                                            self.pcb_data, self.viz_panel)
        self.notebook.AddPage(self.heatsink_panel, "Heatsinks")
        
        self.mounting_panel = MountingPanel(self.notebook, self.config,
                                            self.pcb_data, self.viz_panel)
        self.notebook.AddPage(self.mounting_panel, "Mounting")
        
        self.sim_panel = SimulationPanel(self.notebook, self.config)
        self.notebook.AddPage(self.sim_panel, "Simulation")
        
        left_sizer.Add(self.notebook, 1, wx.EXPAND)
        left_panel.SetSizer(left_sizer)
        
        # Setup splitter
        splitter.SplitVertically(left_panel, right_panel)
        splitter.SetSashGravity(0.4)
        splitter.SetMinimumPaneSize(400)
        
        main_sizer.Add(splitter, 1, wx.EXPAND)
        
        # Footer
        footer = self._create_footer()
        main_sizer.Add(footer, 0, wx.EXPAND)
        
        self.SetSizer(main_sizer)
    
    def _create_header(self):
        """Create header panel."""
        header = wx.Panel(self)
        header.SetBackgroundColour(Colors.HEADER_BG)
        
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        title_sizer = wx.BoxSizer(wx.VERTICAL)
        title = wx.StaticText(header, label="TVAC Thermal Analyzer")
        title.SetFont(Fonts.header())
        title.SetForegroundColour(Colors.HEADER_FG)
        title_sizer.Add(title, 0, wx.BOTTOM, Spacing.XS)
        
        board_name = Path(self.board.GetFileName()).stem if self.board and self.board.GetFileName() else "Untitled"
        subtitle = wx.StaticText(header, label=f"Board: {board_name}")
        subtitle.SetFont(Fonts.body())
        subtitle.SetForegroundColour(Colors.HEADER_SUBTITLE)
        title_sizer.Add(subtitle, 0)
        
        sizer.Add(title_sizer, 1, wx.ALL, Spacing.LG)
        
        # Component count badge
        comp_count = len(self.pcb_data.components)
        badge = wx.StaticText(header, label=f"{comp_count} components")
        badge.SetFont(Fonts.body())
        badge.SetForegroundColour(Colors.HEADER_SUBTITLE)
        sizer.Add(badge, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, Spacing.LG)
        
        header.SetSizer(sizer)
        return header
    
    def _create_footer(self):
        """Create footer with action buttons."""
        footer = wx.Panel(self)
        footer.SetBackgroundColour(Colors.BACKGROUND)
        
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.status = StatusIndicator(footer)
        sizer.Add(self.status, 1, wx.ALL | wx.EXPAND, Spacing.SM)
        
        btn_save = IconButton(footer, "Save Config", icon='save', size=(110, 34))
        btn_save.Bind(wx.EVT_BUTTON, self._on_save)
        sizer.Add(btn_save, 0, wx.ALL, Spacing.SM)
        
        btn_run = AccentButton(footer, "Run Simulation", size=(130, 34))
        btn_run.Bind(wx.EVT_BUTTON, self._on_run)
        sizer.Add(btn_run, 0, wx.ALL, Spacing.SM)
        
        btn_close = wx.Button(footer, label="Close", size=(80, 34))
        btn_close.Bind(wx.EVT_BUTTON, lambda e: self.Close())
        sizer.Add(btn_close, 0, wx.ALL, Spacing.SM)
        
        footer.SetSizer(sizer)
        return footer
    
    def _populate_visualization(self):
        """Populate PCB visualization with extracted data."""
        # Set board outline
        if self.pcb_data.board_outline.outline:
            points = [(p.x, p.y) for p in self.pcb_data.board_outline.outline]
            self.viz_panel.set_outline(points)
        else:
            bo = self.pcb_data.board_outline
            self.viz_panel.set_board_bounds(bo.min_x, bo.max_x, bo.min_y, bo.max_y)
        
        # Add components
        for comp in self.pcb_data.components:
            bbox = comp.get_bounding_box()
            width = bbox[1].x - bbox[0].x
            height = bbox[1].y - bbox[0].y
            is_top = comp.layer == "F.Cu"
            
            self.viz_panel.add_component(
                comp.reference, comp.value,
                comp.position.x, comp.position.y,
                max(width, 1), max(height, 1),
                comp.rotation, is_top
            )
        
        # Add existing heatsinks from config
        for hs in self.config.heatsinks:
            self.viz_panel.add_heatsink(hs.heatsink_id, hs.polygon_points, hs.material)
        
        # Add existing mounting points from config
        for mp in self.config.mounting_points:
            self.viz_panel.add_mounting_point(
                mp.point_id, mp.x_mm, mp.y_mm, mp.diameter_mm,
                is_thermal=(mp.fixed_temp_c is not None)
            )
        
        self.viz_panel.fit_view()
        self.status.set_status("Ready", 'ok')
    
    def _on_save(self, event):
        """Save configuration."""
        self.sim_panel.save_settings()
        
        if self.config_manager.save():
            self.status.set_status("Configuration saved", 'ok')
        else:
            self.status.set_status("Failed to save config", 'error')
            wx.MessageBox("Could not save configuration file", "Error", wx.ICON_ERROR)
    
    def _on_run(self, event):
        """Run thermal simulation."""
        # Save current settings
        self.sim_panel.save_settings()
        
        # Check for power sources
        total_power = sum(cp.power_w for cp in self.config.component_power)
        if total_power <= 0:
            wx.MessageBox(
                "No power dissipation defined.\n\n"
                "Please set power values for heat-generating components.",
                "No Heat Sources", wx.ICON_WARNING
            )
            return
        
        # Run simulation
        progress = ProgressDialog(self, "Running Thermal Simulation")
        progress.Show()
        wx.Yield()
        
        self.status.set_status("Running simulation...", 'working')
        
        try:
            from ..solvers.mesh_gen import MeshGenerator
            from ..solvers.thermal_solver import ThermalSolver
            
            # Generate mesh
            progress.update(10, "Generating mesh...")
            mesh_gen = MeshGenerator(self.pcb_data, self.config)
            mesh = mesh_gen.generate(lambda p, m: progress.update(10 + p // 4, m))
            
            # Run solver
            solver = ThermalSolver()
            
            if self.config.simulation.mode == "transient":
                result = solver.solve_transient(
                    mesh,
                    duration_s=self.config.simulation.duration_s,
                    timestep_s=self.config.simulation.timestep_s,
                    initial_temp_c=self.config.simulation.initial_board_temp_c,
                    chamber_temp_c=self.config.simulation.chamber_wall_temp_c,
                    include_radiation=self.config.simulation.include_radiation,
                    output_interval_s=self.config.simulation.output_interval_s,
                    progress_callback=lambda p, m: progress.update(35 + p * 0.6, m)
                )
            else:
                result = solver.solve_steady_state(
                    mesh,
                    chamber_temp_c=self.config.simulation.chamber_wall_temp_c,
                    include_radiation=self.config.simulation.include_radiation,
                    progress_callback=lambda p, m: progress.update(35 + p * 0.6, m)
                )
            
            progress.Destroy()
            
            if result.error_message:
                self.status.set_status("Simulation failed", 'error')
                wx.MessageBox(result.error_message, "Simulation Error", wx.ICON_ERROR)
                return
            
            # Update visualization with thermal overlay
            temp_grid = result.get_temperature_grid(mesh.nx, mesh.ny, mesh.nz, layer=0)
            if temp_grid is not None:
                self.viz_panel.set_thermal_data(temp_grid, result.min_temp, result.max_temp)
                self.viz_panel.set_layer_visible(VisualizationLayer.THERMAL_OVERLAY, True)
                
                # Update layer toggle checkbox
                if VisualizationLayer.THERMAL_OVERLAY in self.layer_toggles.checkboxes:
                    self.layer_toggles.checkboxes[VisualizationLayer.THERMAL_OVERLAY].SetValue(True)
            
            self.status.set_status(
                f"Complete: {result.min_temp:.1f}°C - {result.max_temp:.1f}°C", 'ok'
            )
            
            # Show results
            wx.MessageBox(
                f"Simulation Complete\n\n"
                f"Backend: {solver.backend_name}\n"
                f"Min Temperature: {result.min_temp:.2f}°C\n"
                f"Max Temperature: {result.max_temp:.2f}°C\n"
                f"Average: {result.avg_temp:.2f}°C\n"
                f"Compute Time: {result.compute_time:.2f}s",
                "Results", wx.ICON_INFORMATION
            )
            
        except Exception as e:
            progress.Destroy()
            self.status.set_status("Simulation failed", 'error')
            import traceback
            wx.MessageBox(
                f"Simulation error:\n{e}\n\n{traceback.format_exc()}",
                "Error", wx.ICON_ERROR
            )
    
    def _on_close(self, event):
        """Handle close event."""
        self.Destroy()


__all__ = ['MainDialog']
