"""
TVAC Thermal Analyzer - PCB Visualization
=========================================
Interactive BOM-style PCB visualization panel.

Features:
- Pan and zoom with mouse
- Component selection and highlighting
- Thermal overlay rendering
- Layer visibility toggles

Author: Space Electronics Thermal Analysis Tool
Version: 2.0.0
"""

import wx
import math
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import IntEnum

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .design_system import Colors, Spacing, Fonts


class VisualizationLayer(IntEnum):
    """Visualization layers."""
    BOARD_OUTLINE = 0
    COMPONENTS_TOP = 1
    COMPONENTS_BOTTOM = 2
    HEATSINKS = 3
    MOUNTING_POINTS = 4
    THERMAL_OVERLAY = 5
    GRID = 6


@dataclass
class VisualComponent:
    """Visual representation of a component."""
    reference: str = ""
    value: str = ""
    x: float = 0.0
    y: float = 0.0
    width: float = 2.0
    height: float = 2.0
    rotation: float = 0.0
    is_top: bool = True
    selected: bool = False
    highlighted: bool = False


@dataclass
class VisualHeatsink:
    """Visual representation of a heatsink."""
    heatsink_id: str = ""
    points: List[Tuple[float, float]] = field(default_factory=list)
    material: str = ""
    highlighted: bool = False


@dataclass
class VisualMounting:
    """Visual representation of a mounting point."""
    point_id: str = ""
    x: float = 0.0
    y: float = 0.0
    diameter: float = 3.0
    is_thermal: bool = False
    highlighted: bool = False


class PCBVisualizationPanel(wx.Panel):
    """Interactive PCB visualization panel."""
    
    # Colors
    BG_COLOR = wx.Colour(40, 44, 52)
    BOARD_COLOR = wx.Colour(34, 90, 34)
    OUTLINE_COLOR = wx.Colour(200, 200, 200)
    COMPONENT_TOP = wx.Colour(100, 100, 100)
    COMPONENT_BOTTOM = wx.Colour(80, 80, 100)
    COMPONENT_SELECTED = wx.Colour(255, 200, 50)
    COMPONENT_HIGHLIGHTED = wx.Colour(100, 200, 255)
    HEATSINK_COLOR = wx.Colour(100, 149, 237, 150)  # Cornflower blue
    MOUNTING_COLOR = wx.Colour(144, 238, 144)  # Light green
    MOUNTING_THERMAL = wx.Colour(255, 165, 0)  # Orange
    GRID_COLOR = wx.Colour(60, 60, 60)
    
    def __init__(self, parent, size=wx.DefaultSize):
        super().__init__(parent, size=size)
        
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.SetBackgroundColour(self.BG_COLOR)
        
        # View state
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.min_scale = 0.1
        self.max_scale = 20.0
        
        # Board geometry
        self.board_outline: List[Tuple[float, float]] = []
        self.board_min_x = 0.0
        self.board_max_x = 100.0
        self.board_min_y = 0.0
        self.board_max_y = 100.0
        
        # Visual elements
        self.components: List[VisualComponent] = []
        self.heatsinks: List[VisualHeatsink] = []
        self.mounting_points: List[VisualMounting] = []
        
        # Thermal overlay
        self.thermal_data: Optional['np.ndarray'] = None
        self.thermal_min = 0.0
        self.thermal_max = 100.0
        
        # Layer visibility
        self.layer_visible = {
            VisualizationLayer.BOARD_OUTLINE: True,
            VisualizationLayer.COMPONENTS_TOP: True,
            VisualizationLayer.COMPONENTS_BOTTOM: True,
            VisualizationLayer.HEATSINKS: True,
            VisualizationLayer.MOUNTING_POINTS: True,
            VisualizationLayer.THERMAL_OVERLAY: False,
            VisualizationLayer.GRID: True,
        }
        
        # Interaction
        self.is_panning = False
        self.pan_start = None
        self.selected_component: Optional[str] = None
        
        # Callbacks
        self.on_component_selected: Optional[Callable] = None
        self.on_point_clicked: Optional[Callable] = None
        
        # Bind events
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_SIZE, self._on_size)
        self.Bind(wx.EVT_MOUSEWHEEL, self._on_wheel)
        self.Bind(wx.EVT_LEFT_DOWN, self._on_left_down)
        self.Bind(wx.EVT_LEFT_UP, self._on_left_up)
        self.Bind(wx.EVT_MOTION, self._on_motion)
        self.Bind(wx.EVT_LEFT_DCLICK, self._on_double_click)
        self.Bind(wx.EVT_RIGHT_DOWN, self._on_right_down)
    
    def set_outline(self, points: List[Tuple[float, float]]):
        """Set board outline from list of (x, y) points."""
        self.board_outline = points
        if points:
            self.board_min_x = min(p[0] for p in points)
            self.board_max_x = max(p[0] for p in points)
            self.board_min_y = min(p[1] for p in points)
            self.board_max_y = max(p[1] for p in points)
        self.Refresh()
    
    def set_board_bounds(self, min_x: float, max_x: float, min_y: float, max_y: float):
        """Set board bounds directly."""
        self.board_min_x = min_x
        self.board_max_x = max_x
        self.board_min_y = min_y
        self.board_max_y = max_y
        self.board_outline = [
            (min_x, min_y), (max_x, min_y),
            (max_x, max_y), (min_x, max_y)
        ]
        self.Refresh()
    
    def add_component(self, ref: str, value: str, x: float, y: float,
                     width: float, height: float, rotation: float = 0, is_top: bool = True):
        """Add a component to visualization."""
        comp = VisualComponent(
            reference=ref, value=value, x=x, y=y,
            width=width, height=height, rotation=rotation, is_top=is_top
        )
        self.components.append(comp)
    
    def add_heatsink(self, hs_id: str, points: List[Tuple[float, float]], material: str = ""):
        """Add a heatsink polygon."""
        hs = VisualHeatsink(heatsink_id=hs_id, points=points, material=material)
        self.heatsinks.append(hs)
    
    def add_mounting_point(self, point_id: str, x: float, y: float, 
                          diameter: float, is_thermal: bool = False):
        """Add a mounting point."""
        mp = VisualMounting(point_id=point_id, x=x, y=y, 
                          diameter=diameter, is_thermal=is_thermal)
        self.mounting_points.append(mp)
    
    def set_thermal_data(self, data: 'np.ndarray', t_min: float, t_max: float):
        """Set thermal overlay data."""
        self.thermal_data = data
        self.thermal_min = t_min
        self.thermal_max = t_max
        self.Refresh()
    
    def set_layer_visible(self, layer: VisualizationLayer, visible: bool):
        """Set layer visibility."""
        self.layer_visible[layer] = visible
        self.Refresh()
    
    def clear_selection(self):
        """Clear all selections."""
        for comp in self.components:
            comp.selected = False
            comp.highlighted = False
        self.selected_component = None
        self.Refresh()
    
    def select_component(self, reference: str):
        """Select component by reference."""
        for comp in self.components:
            comp.selected = (comp.reference == reference)
            if comp.selected:
                self.selected_component = reference
        self.Refresh()
    
    def fit_view(self):
        """Fit board in view."""
        w, h = self.GetSize()
        if w <= 0 or h <= 0:
            return
        
        board_w = self.board_max_x - self.board_min_x
        board_h = self.board_max_y - self.board_min_y
        
        if board_w <= 0 or board_h <= 0:
            return
        
        margin = 20
        scale_x = (w - 2 * margin) / board_w
        scale_y = (h - 2 * margin) / board_h
        self.scale = min(scale_x, scale_y)
        
        center_x = (self.board_min_x + self.board_max_x) / 2
        center_y = (self.board_min_y + self.board_max_y) / 2
        
        self.offset_x = w / 2 - center_x * self.scale
        self.offset_y = h / 2 - center_y * self.scale
        
        self.Refresh()
    
    def board_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert board coordinates to screen coordinates."""
        sx = int(x * self.scale + self.offset_x)
        sy = int(y * self.scale + self.offset_y)
        return sx, sy
    
    def screen_to_board(self, sx: int, sy: int) -> Tuple[float, float]:
        """Convert screen coordinates to board coordinates."""
        x = (sx - self.offset_x) / self.scale
        y = (sy - self.offset_y) / self.scale
        return x, y
    
    def _on_paint(self, event):
        """Handle paint event."""
        dc = wx.AutoBufferedPaintDC(self)
        gc = wx.GraphicsContext.Create(dc)
        
        if not gc:
            return
        
        w, h = self.GetSize()
        
        # Background
        gc.SetBrush(wx.Brush(self.BG_COLOR))
        gc.DrawRectangle(0, 0, w, h)
        
        # Grid
        if self.layer_visible[VisualizationLayer.GRID]:
            self._draw_grid(gc)
        
        # Board fill
        self._draw_board_fill(gc)
        
        # Thermal overlay (before components)
        if self.layer_visible[VisualizationLayer.THERMAL_OVERLAY] and self.thermal_data is not None:
            self._draw_thermal_overlay(gc)
        
        # Heatsinks
        if self.layer_visible[VisualizationLayer.HEATSINKS]:
            self._draw_heatsinks(gc)
        
        # Components
        if self.layer_visible[VisualizationLayer.COMPONENTS_BOTTOM]:
            self._draw_components(gc, top=False)
        if self.layer_visible[VisualizationLayer.COMPONENTS_TOP]:
            self._draw_components(gc, top=True)
        
        # Mounting points
        if self.layer_visible[VisualizationLayer.MOUNTING_POINTS]:
            self._draw_mounting_points(gc)
        
        # Board outline
        if self.layer_visible[VisualizationLayer.BOARD_OUTLINE]:
            self._draw_board_outline(gc)
        
        # Scale indicator
        self._draw_scale_indicator(gc)
    
    def _draw_grid(self, gc):
        """Draw background grid."""
        w, h = self.GetSize()
        
        # Determine grid spacing based on zoom
        base_spacing = 10.0  # mm
        if self.scale < 2:
            spacing = base_spacing * 2
        elif self.scale < 5:
            spacing = base_spacing
        else:
            spacing = base_spacing / 2
        
        gc.SetPen(wx.Pen(self.GRID_COLOR, 1))
        
        # Vertical lines
        x = math.floor(self.board_min_x / spacing) * spacing
        while x <= self.board_max_x:
            sx, _ = self.board_to_screen(x, 0)
            gc.StrokeLine(sx, 0, sx, h)
            x += spacing
        
        # Horizontal lines
        y = math.floor(self.board_min_y / spacing) * spacing
        while y <= self.board_max_y:
            _, sy = self.board_to_screen(0, y)
            gc.StrokeLine(0, sy, w, sy)
            y += spacing
    
    def _draw_board_fill(self, gc):
        """Draw board fill color."""
        if not self.board_outline:
            return
        
        path = gc.CreatePath()
        first = True
        for px, py in self.board_outline:
            sx, sy = self.board_to_screen(px, py)
            if first:
                path.MoveToPoint(sx, sy)
                first = False
            else:
                path.AddLineToPoint(sx, sy)
        path.CloseSubpath()
        
        gc.SetBrush(wx.Brush(self.BOARD_COLOR))
        gc.SetPen(wx.TRANSPARENT_PEN)
        gc.FillPath(path)
    
    def _draw_board_outline(self, gc):
        """Draw board outline."""
        if not self.board_outline:
            return
        
        gc.SetPen(wx.Pen(self.OUTLINE_COLOR, 2))
        gc.SetBrush(wx.TRANSPARENT_BRUSH)
        
        path = gc.CreatePath()
        first = True
        for px, py in self.board_outline:
            sx, sy = self.board_to_screen(px, py)
            if first:
                path.MoveToPoint(sx, sy)
                first = False
            else:
                path.AddLineToPoint(sx, sy)
        path.CloseSubpath()
        
        gc.StrokePath(path)
    
    def _draw_components(self, gc, top: bool):
        """Draw components on specified layer."""
        for comp in self.components:
            if comp.is_top != top:
                continue
            
            # Determine color
            if comp.selected:
                color = self.COMPONENT_SELECTED
            elif comp.highlighted:
                color = self.COMPONENT_HIGHLIGHTED
            else:
                color = self.COMPONENT_TOP if comp.is_top else self.COMPONENT_BOTTOM
            
            # Calculate screen coordinates
            cx, cy = self.board_to_screen(comp.x, comp.y)
            w = comp.width * self.scale
            h = comp.height * self.scale
            
            # Save state and rotate
            gc.PushState()
            gc.Translate(cx, cy)
            gc.Rotate(math.radians(comp.rotation))
            
            # Draw rectangle
            gc.SetBrush(wx.Brush(color))
            gc.SetPen(wx.Pen(wx.WHITE, 1))
            gc.DrawRectangle(-w/2, -h/2, w, h)
            
            # Draw reference text if zoomed in enough
            if self.scale > 3 and w > 15:
                gc.SetFont(wx.Font(8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL), wx.WHITE)
                tw, th, _, _ = gc.GetFullTextExtent(comp.reference)
                gc.DrawText(comp.reference, -tw/2, -th/2)
            
            gc.PopState()
    
    def _draw_heatsinks(self, gc):
        """Draw heatsink polygons."""
        for hs in self.heatsinks:
            if not hs.points:
                continue
            
            path = gc.CreatePath()
            first = True
            for px, py in hs.points:
                sx, sy = self.board_to_screen(px, py)
                if first:
                    path.MoveToPoint(sx, sy)
                    first = False
                else:
                    path.AddLineToPoint(sx, sy)
            path.CloseSubpath()
            
            color = wx.Colour(100, 149, 237, 180 if hs.highlighted else 100)
            gc.SetBrush(wx.Brush(color))
            gc.SetPen(wx.Pen(wx.Colour(100, 149, 237), 2))
            gc.FillPath(path)
            gc.StrokePath(path)
    
    def _draw_mounting_points(self, gc):
        """Draw mounting points."""
        for mp in self.mounting_points:
            cx, cy = self.board_to_screen(mp.x, mp.y)
            r = mp.diameter / 2 * self.scale
            
            color = self.MOUNTING_THERMAL if mp.is_thermal else self.MOUNTING_COLOR
            if mp.highlighted:
                color = self.COMPONENT_HIGHLIGHTED
            
            gc.SetBrush(wx.Brush(color))
            gc.SetPen(wx.Pen(wx.WHITE, 1))
            gc.DrawEllipse(cx - r, cy - r, r * 2, r * 2)
            
            # Cross marker
            gc.SetPen(wx.Pen(wx.BLACK, 1))
            gc.StrokeLine(cx - r/2, cy, cx + r/2, cy)
            gc.StrokeLine(cx, cy - r/2, cx, cy + r/2)
    
    def _draw_thermal_overlay(self, gc):
        """Draw thermal overlay."""
        if not HAS_NUMPY or self.thermal_data is None:
            return
        
        data = self.thermal_data
        h, w = data.shape
        
        # Normalize
        t_range = self.thermal_max - self.thermal_min
        if t_range < 0.1:
            t_range = 1.0
        
        # Calculate screen bounds
        sx1, sy1 = self.board_to_screen(self.board_min_x, self.board_min_y)
        sx2, sy2 = self.board_to_screen(self.board_max_x, self.board_max_y)
        
        cell_w = abs(sx2 - sx1) / w
        cell_h = abs(sy2 - sy1) / h
        
        # Draw cells (simplified for performance)
        step = max(1, int(2 / self.scale))
        
        for iy in range(0, h, step):
            for ix in range(0, w, step):
                t = data[iy, ix]
                norm = (t - self.thermal_min) / t_range
                norm = max(0, min(1, norm))
                
                # Color map: blue -> cyan -> green -> yellow -> red
                color = self._temp_to_color(norm)
                
                px = self.board_min_x + (ix + 0.5) * (self.board_max_x - self.board_min_x) / w
                py = self.board_min_y + (iy + 0.5) * (self.board_max_y - self.board_min_y) / h
                
                cx, cy = self.board_to_screen(px, py)
                
                gc.SetBrush(wx.Brush(wx.Colour(*color, 150)))
                gc.SetPen(wx.TRANSPARENT_PEN)
                gc.DrawRectangle(cx - cell_w * step / 2, cy - cell_h * step / 2,
                               cell_w * step, cell_h * step)
    
    def _temp_to_color(self, norm: float) -> Tuple[int, int, int]:
        """Convert normalized temperature to RGB color."""
        if norm < 0.25:
            t = norm / 0.25
            return (0, int(255 * t), 255)  # Blue to Cyan
        elif norm < 0.5:
            t = (norm - 0.25) / 0.25
            return (0, 255, int(255 * (1 - t)))  # Cyan to Green
        elif norm < 0.75:
            t = (norm - 0.5) / 0.25
            return (int(255 * t), 255, 0)  # Green to Yellow
        else:
            t = (norm - 0.75) / 0.25
            return (255, int(255 * (1 - t)), 0)  # Yellow to Red
    
    def _draw_scale_indicator(self, gc):
        """Draw scale indicator."""
        w, h = self.GetSize()
        
        # Determine scale bar length
        target_px = 100
        mm_per_px = 1 / self.scale
        bar_mm = round(target_px * mm_per_px)
        
        # Round to nice number
        for nice in [1, 2, 5, 10, 20, 50, 100]:
            if bar_mm <= nice:
                bar_mm = nice
                break
        
        bar_px = int(bar_mm * self.scale)
        
        x = w - bar_px - 20
        y = h - 25
        
        gc.SetPen(wx.Pen(wx.WHITE, 2))
        gc.StrokeLine(x, y, x + bar_px, y)
        gc.StrokeLine(x, y - 5, x, y + 5)
        gc.StrokeLine(x + bar_px, y - 5, x + bar_px, y + 5)
        
        gc.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL), wx.WHITE)
        label = f"{bar_mm} mm"
        tw, _, _, _ = gc.GetFullTextExtent(label)
        gc.DrawText(label, x + bar_px/2 - tw/2, y - 18)
    
    def _on_size(self, event):
        self.Refresh()
        event.Skip()
    
    def _on_wheel(self, event):
        """Handle mouse wheel for zoom."""
        mx, my = event.GetPosition()
        
        # Get board position before zoom
        bx, by = self.screen_to_board(mx, my)
        
        # Adjust scale
        if event.GetWheelRotation() > 0:
            self.scale *= 1.2
        else:
            self.scale /= 1.2
        
        self.scale = max(self.min_scale, min(self.max_scale, self.scale))
        
        # Adjust offset to keep mouse position fixed
        new_sx, new_sy = self.board_to_screen(bx, by)
        self.offset_x += mx - new_sx
        self.offset_y += my - new_sy
        
        self.Refresh()
    
    def _on_left_down(self, event):
        """Handle left mouse down."""
        self.is_panning = True
        self.pan_start = event.GetPosition()
        self.CaptureMouse()
    
    def _on_left_up(self, event):
        """Handle left mouse up."""
        if self.HasCapture():
            self.ReleaseMouse()
        
        if self.is_panning and self.pan_start:
            pos = event.GetPosition()
            dx = abs(pos.x - self.pan_start.x)
            dy = abs(pos.y - self.pan_start.y)
            
            # If minimal movement, treat as click
            if dx < 5 and dy < 5:
                self._handle_click(pos)
        
        self.is_panning = False
        self.pan_start = None
    
    def _on_motion(self, event):
        """Handle mouse motion."""
        if self.is_panning and event.Dragging() and self.pan_start:
            pos = event.GetPosition()
            self.offset_x += pos.x - self.pan_start.x
            self.offset_y += pos.y - self.pan_start.y
            self.pan_start = pos
            self.Refresh()
    
    def _on_double_click(self, event):
        """Handle double-click to fit view."""
        self.fit_view()
    
    def _on_right_down(self, event):
        """Handle right-click for context menu."""
        pass
    
    def _handle_click(self, pos):
        """Handle click to select component."""
        bx, by = self.screen_to_board(pos.x, pos.y)
        
        # Find clicked component
        for comp in reversed(self.components):
            half_w = comp.width / 2
            half_h = comp.height / 2
            
            if (comp.x - half_w <= bx <= comp.x + half_w and
                comp.y - half_h <= by <= comp.y + half_h):
                self.clear_selection()
                comp.selected = True
                self.selected_component = comp.reference
                
                if self.on_component_selected:
                    self.on_component_selected(comp.reference)
                
                self.Refresh()
                return
        
        # No component clicked
        self.clear_selection()
        
        if self.on_point_clicked:
            self.on_point_clicked(bx, by)


class LayerTogglePanel(wx.Panel):
    """Panel with layer visibility toggles."""
    
    def __init__(self, parent, viz_panel: PCBVisualizationPanel):
        super().__init__(parent)
        self.viz_panel = viz_panel
        self.SetBackgroundColour(Colors.BACKGROUND)
        
        self.checkboxes: Dict[VisualizationLayer, wx.CheckBox] = {}
        
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        layers = [
            (VisualizationLayer.GRID, "Grid"),
            (VisualizationLayer.BOARD_OUTLINE, "Outline"),
            (VisualizationLayer.COMPONENTS_TOP, "Top"),
            (VisualizationLayer.COMPONENTS_BOTTOM, "Bottom"),
            (VisualizationLayer.HEATSINKS, "Heatsinks"),
            (VisualizationLayer.MOUNTING_POINTS, "Mounting"),
            (VisualizationLayer.THERMAL_OVERLAY, "Thermal"),
        ]
        
        for layer, label in layers:
            cb = wx.CheckBox(self, label=label)
            cb.SetValue(viz_panel.layer_visible.get(layer, True))
            cb.Bind(wx.EVT_CHECKBOX, lambda e, l=layer: self._on_toggle(l, e))
            sizer.Add(cb, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, Spacing.SM)
            self.checkboxes[layer] = cb
        
        self.SetSizer(sizer)
    
    def _on_toggle(self, layer: VisualizationLayer, event):
        self.viz_panel.set_layer_visible(layer, event.IsChecked())


__all__ = [
    'PCBVisualizationPanel', 'LayerTogglePanel', 'VisualizationLayer',
    'VisualComponent', 'VisualHeatsink', 'VisualMounting',
]
