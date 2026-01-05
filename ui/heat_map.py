"""
TVAC Thermal Analyzer - Heat Map Visualization
==============================================
Real-time heat map overlay for KiCad PCB view.

Author: Space Electronics Thermal Analysis Tool
Version: 1.0.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum

try:
    import wx
    HAS_WX = True
except ImportError:
    HAS_WX = False

try:
    import pcbnew
    HAS_PCBNEW = True
except ImportError:
    HAS_PCBNEW = False


class ColorMap(Enum):
    """Available color maps for thermal visualization."""
    JET = "jet"
    HOT = "hot"
    COOL = "cool"
    THERMAL = "thermal"
    RAINBOW = "rainbow"
    VIRIDIS = "viridis"


@dataclass
class ColorStop:
    """Color stop for gradient definition."""
    position: float  # 0.0 to 1.0
    r: int
    g: int
    b: int
    a: int = 255


class ThermalColorMapper:
    """Maps temperature values to colors."""
    
    # Pre-defined color maps
    COLORMAPS = {
        ColorMap.JET: [
            ColorStop(0.0, 0, 0, 128),      # Dark blue
            ColorStop(0.1, 0, 0, 255),      # Blue
            ColorStop(0.35, 0, 255, 255),   # Cyan
            ColorStop(0.5, 0, 255, 0),      # Green
            ColorStop(0.65, 255, 255, 0),   # Yellow
            ColorStop(0.9, 255, 0, 0),      # Red
            ColorStop(1.0, 128, 0, 0),      # Dark red
        ],
        ColorMap.HOT: [
            ColorStop(0.0, 0, 0, 0),        # Black
            ColorStop(0.33, 255, 0, 0),     # Red
            ColorStop(0.66, 255, 255, 0),   # Yellow
            ColorStop(1.0, 255, 255, 255),  # White
        ],
        ColorMap.THERMAL: [
            ColorStop(0.0, 0, 0, 64),       # Dark blue
            ColorStop(0.2, 0, 64, 255),     # Blue
            ColorStop(0.4, 0, 192, 192),    # Cyan
            ColorStop(0.5, 64, 192, 64),    # Green
            ColorStop(0.6, 192, 192, 0),    # Yellow
            ColorStop(0.8, 255, 128, 0),    # Orange
            ColorStop(1.0, 255, 0, 0),      # Red
        ],
        ColorMap.VIRIDIS: [
            ColorStop(0.0, 68, 1, 84),
            ColorStop(0.25, 59, 82, 139),
            ColorStop(0.5, 33, 145, 140),
            ColorStop(0.75, 94, 201, 98),
            ColorStop(1.0, 253, 231, 37),
        ],
    }
    
    def __init__(self, colormap: ColorMap = ColorMap.JET,
                 t_min: float = 20.0, t_max: float = 100.0):
        """Initialize color mapper."""
        self.colormap = colormap
        self.t_min = t_min
        self.t_max = t_max
        self.stops = self.COLORMAPS.get(colormap, self.COLORMAPS[ColorMap.JET])
        
        # Pre-compute lookup table for speed
        self._lut_size = 256
        self._lut = self._build_lut()
    
    def _build_lut(self) -> np.ndarray:
        """Build color lookup table."""
        lut = np.zeros((self._lut_size, 4), dtype=np.uint8)
        
        for i in range(self._lut_size):
            t = i / (self._lut_size - 1)
            r, g, b, a = self._interpolate_color(t)
            lut[i] = [r, g, b, a]
        
        return lut
    
    def _interpolate_color(self, t: float) -> Tuple[int, int, int, int]:
        """Interpolate color at position t (0-1)."""
        t = max(0.0, min(1.0, t))
        
        # Find surrounding stops
        lower = self.stops[0]
        upper = self.stops[-1]
        
        for i, stop in enumerate(self.stops):
            if stop.position <= t:
                lower = stop
            if stop.position >= t:
                upper = stop
                break
        
        # Interpolate
        if upper.position == lower.position:
            frac = 0.0
        else:
            frac = (t - lower.position) / (upper.position - lower.position)
        
        r = int(lower.r + frac * (upper.r - lower.r))
        g = int(lower.g + frac * (upper.g - lower.g))
        b = int(lower.b + frac * (upper.b - lower.b))
        a = int(lower.a + frac * (upper.a - lower.a))
        
        return r, g, b, a
    
    def set_range(self, t_min: float, t_max: float):
        """Set temperature range."""
        self.t_min = t_min
        self.t_max = t_max
    
    def temperature_to_color(self, temperature: float) -> Tuple[int, int, int, int]:
        """Map temperature to RGBA color."""
        if self.t_max == self.t_min:
            t_norm = 0.5
        else:
            t_norm = (temperature - self.t_min) / (self.t_max - self.t_min)
        
        t_norm = max(0.0, min(1.0, t_norm))
        idx = int(t_norm * (self._lut_size - 1))
        
        return tuple(self._lut[idx])
    
    def temperature_array_to_image(self, temperatures: np.ndarray,
                                   alpha: float = 0.7) -> np.ndarray:
        """Convert temperature array to RGBA image."""
        # Normalize temperatures
        if self.t_max == self.t_min:
            t_norm = np.full_like(temperatures, 0.5)
        else:
            t_norm = (temperatures - self.t_min) / (self.t_max - self.t_min)
        
        t_norm = np.clip(t_norm, 0.0, 1.0)
        
        # Map to LUT indices
        indices = (t_norm * (self._lut_size - 1)).astype(np.int32)
        
        # Create image
        image = self._lut[indices].copy()
        image[:, :, 3] = int(alpha * 255)  # Set alpha
        
        return image


@dataclass
class HeatMapConfig:
    """Configuration for heat map display."""
    enabled: bool = True
    colormap: ColorMap = ColorMap.JET
    opacity: float = 0.7
    show_legend: bool = True
    show_isotherms: bool = False
    isotherm_interval: float = 10.0  # °C between isotherms
    auto_scale: bool = True
    t_min_override: float = 20.0
    t_max_override: float = 100.0
    interpolation: str = "bilinear"  # "nearest", "bilinear"


class HeatMapRenderer:
    """Renders thermal heat map overlay."""
    
    def __init__(self, config: HeatMapConfig = None):
        """Initialize renderer."""
        self.config = config or HeatMapConfig()
        self.color_mapper = ThermalColorMapper(self.config.colormap)
        
        # Current data
        self.temperature_grid: Optional[np.ndarray] = None
        self.x_min = 0.0
        self.x_max = 100.0
        self.y_min = 0.0
        self.y_max = 100.0
        
        # Current time in simulation
        self.current_time = 0.0
        
        # Cached image
        self._cached_image: Optional[np.ndarray] = None
        self._cache_valid = False
    
    def set_temperature_data(self, grid: np.ndarray,
                            x_min: float, x_max: float,
                            y_min: float, y_max: float,
                            t_min: float = None, t_max: float = None):
        """Set temperature data for rendering."""
        self.temperature_grid = grid
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        
        # Update color range
        if self.config.auto_scale:
            if t_min is None:
                t_min = float(np.min(grid))
            if t_max is None:
                t_max = float(np.max(grid))
        else:
            t_min = self.config.t_min_override
            t_max = self.config.t_max_override
        
        self.color_mapper.set_range(t_min, t_max)
        self._cache_valid = False
    
    def render_to_image(self, width: int, height: int) -> Optional[np.ndarray]:
        """Render heat map to RGBA image array."""
        if self.temperature_grid is None:
            return None
        
        # Check cache
        if self._cache_valid and self._cached_image is not None:
            if self._cached_image.shape[:2] == (height, width):
                return self._cached_image
        
        # Resize temperature grid to match output size
        from scipy import ndimage
        
        zoom_y = height / self.temperature_grid.shape[0]
        zoom_x = width / self.temperature_grid.shape[1]
        
        if self.config.interpolation == "bilinear":
            order = 1
        else:
            order = 0
        
        resized = ndimage.zoom(self.temperature_grid, (zoom_y, zoom_x), order=order)
        
        # Convert to image
        image = self.color_mapper.temperature_array_to_image(
            resized, alpha=self.config.opacity
        )
        
        self._cached_image = image
        self._cache_valid = True
        
        return image
    
    def get_temperature_at_point(self, x: float, y: float) -> Optional[float]:
        """Get temperature at a board position."""
        if self.temperature_grid is None:
            return None
        
        # Convert board coordinates to grid indices
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            return None
        
        col = int((x - self.x_min) / (self.x_max - self.x_min) * (self.temperature_grid.shape[1] - 1))
        row = int((y - self.y_min) / (self.y_max - self.y_min) * (self.temperature_grid.shape[0] - 1))
        
        col = max(0, min(col, self.temperature_grid.shape[1] - 1))
        row = max(0, min(row, self.temperature_grid.shape[0] - 1))
        
        return float(self.temperature_grid[row, col])
    
    def invalidate_cache(self):
        """Invalidate cached image."""
        self._cache_valid = False


class LegendRenderer:
    """Renders color legend for heat map."""
    
    def __init__(self, color_mapper: ThermalColorMapper):
        """Initialize legend renderer."""
        self.color_mapper = color_mapper
        self.width = 30
        self.height = 200
        self.margin = 10
        self.tick_count = 5
    
    def render_to_image(self) -> np.ndarray:
        """Render legend to RGBA image."""
        image = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        
        # Fill with gradient
        for y in range(self.height):
            # Bottom is cold (t_min), top is hot (t_max)
            t_norm = 1.0 - (y / (self.height - 1))
            temp = self.color_mapper.t_min + t_norm * (self.color_mapper.t_max - self.color_mapper.t_min)
            color = self.color_mapper.temperature_to_color(temp)
            image[y, :] = color
        
        return image
    
    def get_tick_labels(self) -> List[Tuple[float, str]]:
        """Get tick positions and labels."""
        ticks = []
        t_range = self.color_mapper.t_max - self.color_mapper.t_min
        
        for i in range(self.tick_count):
            pos = i / (self.tick_count - 1)  # 0 to 1
            temp = self.color_mapper.t_min + pos * t_range
            label = f"{temp:.1f}°C"
            ticks.append((1.0 - pos, label))  # Invert for display
        
        return ticks


class TimeSlider:
    """Virtual time slider for scrubbing through simulation results."""
    
    def __init__(self, duration: float = 0.0, num_frames: int = 0):
        """Initialize time slider."""
        self.duration = duration
        self.num_frames = num_frames
        self.current_time = 0.0
        self.current_frame = 0
        self.is_playing = False
        self.playback_speed = 1.0
        
        self.on_time_changed: Optional[Callable[[float, int], None]] = None
    
    def set_time(self, time_s: float):
        """Set current time."""
        self.current_time = max(0.0, min(time_s, self.duration))
        
        if self.num_frames > 0 and self.duration > 0:
            self.current_frame = int(self.current_time / self.duration * (self.num_frames - 1))
        
        if self.on_time_changed:
            self.on_time_changed(self.current_time, self.current_frame)
    
    def set_frame(self, frame: int):
        """Set current frame."""
        self.current_frame = max(0, min(frame, self.num_frames - 1))
        
        if self.num_frames > 1:
            self.current_time = self.current_frame / (self.num_frames - 1) * self.duration
        
        if self.on_time_changed:
            self.on_time_changed(self.current_time, self.current_frame)
    
    def step_forward(self):
        """Step one frame forward."""
        self.set_frame(self.current_frame + 1)
    
    def step_backward(self):
        """Step one frame backward."""
        self.set_frame(self.current_frame - 1)
    
    def goto_start(self):
        """Go to start."""
        self.set_frame(0)
    
    def goto_end(self):
        """Go to end."""
        self.set_frame(self.num_frames - 1)


__all__ = [
    'ColorMap',
    'ColorStop',
    'ThermalColorMapper',
    'HeatMapConfig',
    'HeatMapRenderer',
    'LegendRenderer',
    'TimeSlider',
]
