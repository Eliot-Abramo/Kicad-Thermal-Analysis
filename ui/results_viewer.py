"""
TVAC Thermal Analyzer - Results Viewer
=====================================
View and animate thermal simulation results.

Features:
- Animated transient playback
- Temperature history plots
- Component temperature table
- Export to PDF report
- Save thermal images

Author: Space Electronics Thermal Analysis Tool
Version: 2.0.0
"""

import wx
import wx.grid as gridlib
from typing import Optional, List, Dict, Callable
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .design_system import (
    Colors, Spacing, Fonts,
    BaseDialog, IconButton, AccentButton,
    SectionHeader, StatusIndicator, InfoBanner
)
from .pcb_visualization import PCBVisualizationPanel, VisualizationLayer

from ..core.config import ThermalAnalysisConfig
from ..core.pcb_extractor import PCBData
from ..solvers.thermal_solver import ThermalResult, ThermalMesh


class ResultsViewerDialog(BaseDialog):
    """
    Dialog for viewing thermal simulation results.
    
    Supports:
    - Static steady-state view
    - Animated transient playback
    - Temperature time plots
    - Component temperature table
    - PDF report export
    """
    
    def __init__(self, parent, 
                 config: ThermalAnalysisConfig,
                 pcb_data: PCBData,
                 result: ThermalResult,
                 mesh: ThermalMesh):
        super().__init__(parent, "Thermal Analysis Results",
                        size=(1200, 800), min_size=(900, 600))
        
        self.config = config
        self.pcb_data = pcb_data
        self.result = result
        self.mesh = mesh
        
        # Animation state
        self.is_transient = len(result.time_points) > 1
        self.current_frame = 0
        self.is_playing = False
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_timer, self.timer)
        
        self._build_ui()
        self._populate_data()
        
        self.Bind(wx.EVT_CLOSE, self._on_close)
    
    def _build_ui(self):
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Header
        header = self._create_header()
        main_sizer.Add(header, 0, wx.EXPAND)
        
        # Content splitter
        splitter = wx.SplitterWindow(self, style=wx.SP_LIVE_UPDATE)
        
        # Left panel - Visualization
        left_panel = wx.Panel(splitter)
        left_panel.SetBackgroundColour(Colors.BACKGROUND)
        left_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # PCB Visualization
        viz_header = SectionHeader(left_panel, "Thermal Map", "Temperature distribution")
        left_sizer.Add(viz_header, 0, wx.ALL | wx.EXPAND, Spacing.MD)
        
        self.viz_panel = PCBVisualizationPanel(left_panel)
        left_sizer.Add(self.viz_panel, 1, wx.ALL | wx.EXPAND, Spacing.MD)
        
        # Animation controls (for transient)
        if self.is_transient:
            anim_panel = self._create_animation_controls(left_panel)
            left_sizer.Add(anim_panel, 0, wx.ALL | wx.EXPAND, Spacing.MD)
        
        left_panel.SetSizer(left_sizer)
        
        # Right panel - Data
        right_panel = wx.Panel(splitter)
        right_panel.SetBackgroundColour(Colors.PANEL_BG)
        right_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Summary section
        summary_header = SectionHeader(right_panel, "Results Summary", "Key thermal metrics")
        right_sizer.Add(summary_header, 0, wx.ALL | wx.EXPAND, Spacing.MD)
        
        self.summary_panel = self._create_summary_panel(right_panel)
        right_sizer.Add(self.summary_panel, 0, wx.ALL | wx.EXPAND, Spacing.MD)
        
        # Component temperatures
        comp_header = SectionHeader(right_panel, "Hot Components", 
                                   "Components with highest power dissipation")
        right_sizer.Add(comp_header, 0, wx.ALL | wx.EXPAND, Spacing.MD)
        
        self.comp_list = wx.ListCtrl(right_panel, 
                                     style=wx.LC_REPORT | wx.LC_SINGLE_SEL | wx.BORDER_SIMPLE)
        self.comp_list.SetBackgroundColour(Colors.INPUT_BG)
        self.comp_list.SetFont(Fonts.body())
        
        self.comp_list.InsertColumn(0, "Rank", width=50)
        self.comp_list.InsertColumn(1, "Reference", width=80)
        self.comp_list.InsertColumn(2, "Value", width=100)
        self.comp_list.InsertColumn(3, "Power (W)", width=80)
        
        self.comp_list.Bind(wx.EVT_LIST_ITEM_SELECTED, self._on_component_select)
        right_sizer.Add(self.comp_list, 1, wx.ALL | wx.EXPAND, Spacing.MD)
        
        right_panel.SetSizer(right_sizer)
        
        # Setup splitter
        splitter.SplitVertically(left_panel, right_panel)
        splitter.SetSashGravity(0.6)
        splitter.SetMinimumPaneSize(350)
        
        main_sizer.Add(splitter, 1, wx.EXPAND)
        
        # Footer
        footer = self._create_footer()
        main_sizer.Add(footer, 0, wx.EXPAND)
        
        self.SetSizer(main_sizer)
    
    def _create_header(self):
        """Create header panel with result summary."""
        header = wx.Panel(self)
        header.SetBackgroundColour(Colors.HEADER_BG)
        
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        title_sizer = wx.BoxSizer(wx.VERTICAL)
        title = wx.StaticText(header, label="Thermal Analysis Results")
        title.SetFont(Fonts.header())
        title.SetForegroundColour(Colors.HEADER_FG)
        title_sizer.Add(title, 0, wx.BOTTOM, Spacing.XS)
        
        mode = "Transient" if self.is_transient else "Steady State"
        subtitle = wx.StaticText(header, label=f"Mode: {mode}")
        subtitle.SetFont(Fonts.body())
        subtitle.SetForegroundColour(Colors.HEADER_SUBTITLE)
        title_sizer.Add(subtitle, 0)
        
        sizer.Add(title_sizer, 1, wx.ALL, Spacing.LG)
        
        # Temperature badge
        temp_badge = wx.Panel(header)
        temp_badge.SetBackgroundColour(self._get_temp_color(self.result.max_temp))
        temp_sizer = wx.BoxSizer(wx.VERTICAL)
        
        max_label = wx.StaticText(temp_badge, label="Max Temp")
        max_label.SetFont(Fonts.small())
        max_label.SetForegroundColour(wx.WHITE)
        temp_sizer.Add(max_label, 0, wx.ALIGN_CENTER)
        
        max_value = wx.StaticText(temp_badge, label=f"{self.result.max_temp:.1f}°C")
        max_value.SetFont(Fonts.title())
        max_value.SetForegroundColour(wx.WHITE)
        temp_sizer.Add(max_value, 0, wx.ALIGN_CENTER)
        
        temp_badge.SetSizer(temp_sizer)
        sizer.Add(temp_badge, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, Spacing.MD)
        
        header.SetSizer(sizer)
        return header
    
    def _get_temp_color(self, temp: float) -> wx.Colour:
        """Get color based on temperature severity."""
        if temp > 125:
            return wx.Colour(198, 40, 40)   # Red - critical
        elif temp > 85:
            return wx.Colour(245, 124, 0)   # Orange - warning
        elif temp > 60:
            return wx.Colour(251, 192, 45)  # Yellow - caution
        else:
            return wx.Colour(67, 160, 71)   # Green - good
    
    def _create_animation_controls(self, parent):
        """Create animation playback controls."""
        panel = wx.Panel(parent)
        panel.SetBackgroundColour(Colors.BACKGROUND)
        
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Play/Pause button
        self.btn_play = IconButton(panel, "Play", icon='play', size=(80, 32))
        self.btn_play.Bind(wx.EVT_BUTTON, self._on_play_pause)
        sizer.Add(self.btn_play, 0, wx.RIGHT, Spacing.SM)
        
        # Frame slider
        self.slider = wx.Slider(panel, value=0, minValue=0, 
                               maxValue=max(0, len(self.result.time_points) - 1),
                               style=wx.SL_HORIZONTAL)
        self.slider.Bind(wx.EVT_SLIDER, self._on_slider)
        sizer.Add(self.slider, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, Spacing.SM)
        
        # Time label
        self.time_label = wx.StaticText(panel, label="t = 0.0 s")
        self.time_label.SetFont(Fonts.mono())
        sizer.Add(self.time_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, Spacing.SM)
        
        # Speed control
        sizer.Add(wx.StaticText(panel, label="Speed:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, Spacing.XS)
        self.speed_choice = wx.Choice(panel, choices=["0.5x", "1x", "2x", "4x"])
        self.speed_choice.SetSelection(1)
        sizer.Add(self.speed_choice, 0, wx.ALIGN_CENTER_VERTICAL)
        
        panel.SetSizer(sizer)
        return panel
    
    def _create_summary_panel(self, parent):
        """Create summary statistics panel."""
        panel = wx.Panel(parent)
        panel.SetBackgroundColour(Colors.INPUT_BG)
        
        sizer = wx.FlexGridSizer(4, 2, Spacing.SM, Spacing.LG)
        sizer.AddGrowableCol(1)
        
        metrics = [
            ("Minimum", f"{self.result.min_temp:.2f} °C"),
            ("Maximum", f"{self.result.max_temp:.2f} °C"),
            ("Average", f"{self.result.avg_temp:.2f} °C"),
            ("Range", f"{self.result.max_temp - self.result.min_temp:.2f} °C"),
        ]
        
        for label, value in metrics:
            lbl = wx.StaticText(panel, label=label + ":")
            lbl.SetFont(Fonts.body())
            sizer.Add(lbl, 0, wx.ALIGN_CENTER_VERTICAL)
            
            val = wx.StaticText(panel, label=value)
            val.SetFont(Fonts.bold())
            sizer.Add(val, 0, wx.ALIGN_CENTER_VERTICAL)
        
        outer = wx.BoxSizer(wx.VERTICAL)
        outer.Add(sizer, 0, wx.ALL | wx.EXPAND, Spacing.MD)
        panel.SetSizer(outer)
        
        return panel
    
    def _create_footer(self):
        """Create footer with export buttons."""
        footer = wx.Panel(self)
        footer.SetBackgroundColour(Colors.BACKGROUND)
        
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.status = StatusIndicator(footer)
        sizer.Add(self.status, 1, wx.ALL | wx.EXPAND, Spacing.SM)
        
        btn_export_img = IconButton(footer, "Save Image", icon='save', size=(100, 34))
        btn_export_img.Bind(wx.EVT_BUTTON, self._on_export_image)
        sizer.Add(btn_export_img, 0, wx.ALL, Spacing.SM)
        
        btn_export_pdf = AccentButton(footer, "Export PDF Report", size=(140, 34))
        btn_export_pdf.Bind(wx.EVT_BUTTON, self._on_export_pdf)
        sizer.Add(btn_export_pdf, 0, wx.ALL, Spacing.SM)
        
        btn_close = wx.Button(footer, label="Close", size=(80, 34))
        btn_close.Bind(wx.EVT_BUTTON, lambda e: self.Close())
        sizer.Add(btn_close, 0, wx.ALL, Spacing.SM)
        
        footer.SetSizer(sizer)
        return footer
    
    def _populate_data(self):
        """Populate visualization and data tables."""
        # Set up PCB visualization
        if self.pcb_data.board_outline.outline:
            outline_points = [(p.x, p.y) for p in self.pcb_data.board_outline.outline]
            self.viz_panel.set_outline(outline_points)
        else:
            bo = self.pcb_data.board_outline
            self.viz_panel.set_board_bounds(bo.min_x, bo.max_x, bo.min_y, bo.max_y)
        
        # Add components
        for comp in self.pcb_data.components:
            bbox = comp.get_bounding_box()
            width = bbox[1].x - bbox[0].x
            height = bbox[1].y - bbox[0].y
            is_top = comp.layer == "F.Cu"
            self.viz_panel.add_component(comp.reference, comp.value,
                                        comp.position.x, comp.position.y,
                                        width, height, comp.rotation, is_top)
        
        # Set thermal overlay
        self._update_thermal_display(0)
        self.viz_panel.set_layer_visible(VisualizationLayer.THERMAL_OVERLAY, True)
        self.viz_panel.fit_view()
        
        # Populate component list (sorted by power)
        sorted_power = sorted(self.config.component_power, 
                             key=lambda x: x.power_w, reverse=True)
        
        for rank, cp in enumerate(sorted_power[:20], 1):
            value = ""
            for comp in self.pcb_data.components:
                if comp.reference == cp.reference:
                    value = comp.value
                    break
            
            idx = self.comp_list.InsertItem(self.comp_list.GetItemCount(), str(rank))
            self.comp_list.SetItem(idx, 1, cp.reference)
            self.comp_list.SetItem(idx, 2, value[:15])
            self.comp_list.SetItem(idx, 3, f"{cp.power_w:.4f}")
        
        self.status.set_status("Results loaded", 'ok')
    
    def _update_thermal_display(self, frame_idx: int):
        """Update thermal overlay for a specific frame."""
        if not HAS_NUMPY:
            return
        
        if self.is_transient and frame_idx < len(self.result.temp_history):
            # Get temperatures for this frame
            temps = self.result.temp_history[frame_idx]
            t_min = min(temps)
            t_max = max(temps)
            
            # Create grid
            temp_grid = np.zeros((self.mesh.ny, self.mesh.nx))
            for iy in range(self.mesh.ny):
                for ix in range(self.mesh.nx):
                    idx = iy * self.mesh.nx + ix
                    if idx < len(temps):
                        temp_grid[iy, ix] = temps[idx]
            
            self.viz_panel.set_thermal_data(temp_grid, t_min, t_max)
        else:
            # Steady state - use final result
            temp_grid = self.result.get_temperature_grid(
                self.mesh.nx, self.mesh.ny, self.mesh.nz, layer=0
            )
            if temp_grid is not None:
                self.viz_panel.set_thermal_data(temp_grid, 
                                               self.result.min_temp, 
                                               self.result.max_temp)
        
        self.viz_panel.Refresh()
    
    def _on_play_pause(self, event):
        """Toggle animation playback."""
        if self.is_playing:
            self._stop_animation()
        else:
            self._start_animation()
    
    def _start_animation(self):
        """Start animation playback."""
        self.is_playing = True
        self.btn_play.SetLabel("⏸ Pause")
        
        # Get interval based on speed
        speed_map = {"0.5x": 200, "1x": 100, "2x": 50, "4x": 25}
        speed = self.speed_choice.GetStringSelection()
        interval = speed_map.get(speed, 100)
        
        self.timer.Start(interval)
    
    def _stop_animation(self):
        """Stop animation playback."""
        self.is_playing = False
        self.btn_play.SetLabel("▶ Play")
        self.timer.Stop()
    
    def _on_timer(self, event):
        """Handle animation timer tick."""
        self.current_frame += 1
        if self.current_frame >= len(self.result.time_points):
            self.current_frame = 0
        
        self.slider.SetValue(self.current_frame)
        self._update_frame(self.current_frame)
    
    def _on_slider(self, event):
        """Handle slider movement."""
        self.current_frame = self.slider.GetValue()
        self._update_frame(self.current_frame)
    
    def _update_frame(self, frame_idx: int):
        """Update display for a specific frame."""
        if frame_idx < len(self.result.time_points):
            t = self.result.time_points[frame_idx]
            self.time_label.SetLabel(f"t = {t:.1f} s")
        
        self._update_thermal_display(frame_idx)
    
    def _on_component_select(self, event):
        """Highlight selected component on PCB."""
        idx = event.GetIndex()
        if idx >= 0:
            ref = self.comp_list.GetItemText(idx, 1)
            self.viz_panel.clear_selection()
            self.viz_panel.select_component(ref)
    
    def _on_export_image(self, event):
        """Export thermal map as image."""
        dlg = wx.FileDialog(
            self, "Save Thermal Map",
            wildcard="PNG files (*.png)|*.png",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        )
        
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            
            try:
                # Get bitmap from visualization panel
                dc = wx.ClientDC(self.viz_panel)
                w, h = self.viz_panel.GetSize()
                
                bitmap = wx.Bitmap(w, h)
                mem_dc = wx.MemoryDC(bitmap)
                mem_dc.Blit(0, 0, w, h, dc, 0, 0)
                mem_dc.SelectObject(wx.NullBitmap)
                
                bitmap.SaveFile(path, wx.BITMAP_TYPE_PNG)
                self.status.set_status(f"Saved: {Path(path).name}", 'ok')
                
            except Exception as e:
                self.status.set_status("Export failed", 'error')
                wx.MessageBox(f"Failed to save image:\n{e}", "Error", wx.ICON_ERROR)
        
        dlg.Destroy()
    
    def _on_export_pdf(self, event):
        """Export results to PDF report."""
        dlg = wx.FileDialog(
            self, "Save PDF Report",
            wildcard="PDF files (*.pdf)|*.pdf",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        )
        
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            
            try:
                from ..utils.report_generator import generate_report, ReportSettings
                
                settings = ReportSettings(
                    title="TVAC Thermal Analysis Report",
                    project_name=Path(self.pcb_data.board_outline.min_x).stem if hasattr(self.pcb_data.board_outline, 'min_x') else "",
                )
                
                self.status.set_status("Generating report...", 'working')
                wx.Yield()
                
                success = generate_report(path, self.config, self.pcb_data, 
                                         self.result, self.mesh, settings)
                
                if success:
                    self.status.set_status(f"Saved: {Path(path).name}", 'ok')
                    wx.MessageBox(f"Report saved to:\n{path}", "Export Complete", 
                                 wx.ICON_INFORMATION)
                else:
                    self.status.set_status("Export failed", 'error')
                    wx.MessageBox("Failed to generate report", "Error", wx.ICON_ERROR)
                    
            except ImportError:
                self.status.set_status("reportlab not installed", 'error')
                wx.MessageBox(
                    "PDF export requires reportlab.\n\n"
                    "Install with: pip install reportlab",
                    "Missing Dependency",
                    wx.ICON_WARNING
                )
            except Exception as e:
                self.status.set_status("Export failed", 'error')
                wx.MessageBox(f"Failed to generate report:\n{e}", "Error", wx.ICON_ERROR)
        
        dlg.Destroy()
    
    def _on_close(self, event):
        """Clean up on close."""
        self._stop_animation()
        self.Destroy()


__all__ = ['ResultsViewerDialog']
