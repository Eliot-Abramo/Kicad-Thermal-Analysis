"""
TVAC Thermal Analyzer - UI Design System
========================================
Professional UI components following Multi-Board Manager style.

Author: Space Electronics Thermal Analysis Tool
Version: 2.0.0
"""

import wx
from typing import Optional


class Colors:
    """Professional light theme color palette."""
    # Backgrounds
    BACKGROUND = wx.Colour(245, 246, 247)
    PANEL_BG = wx.Colour(255, 255, 255)
    INPUT_BG = wx.Colour(250, 250, 250)
    
    # Header
    HEADER_BG = wx.Colour(38, 50, 56)
    HEADER_FG = wx.Colour(255, 255, 255)
    HEADER_SUBTITLE = wx.Colour(176, 190, 197)
    
    # Accent & Actions
    ACCENT = wx.Colour(30, 136, 229)
    ACCENT_HOVER = wx.Colour(25, 118, 210)
    
    # Text
    TEXT_PRIMARY = wx.Colour(32, 33, 36)
    TEXT_SECONDARY = wx.Colour(95, 99, 104)
    TEXT_DISABLED = wx.Colour(180, 180, 180)
    
    # Borders
    BORDER = wx.Colour(218, 220, 224)
    BORDER_FOCUS = wx.Colour(30, 136, 229)
    
    # Status
    SUCCESS = wx.Colour(67, 160, 71)
    WARNING = wx.Colour(251, 140, 0)
    ERROR = wx.Colour(229, 57, 53)
    INFO = wx.Colour(30, 136, 229)
    
    # Banners
    INFO_BG = wx.Colour(227, 242, 253)
    WARNING_BG = wx.Colour(255, 243, 224)
    ERROR_BG = wx.Colour(255, 235, 238)
    SUCCESS_BG = wx.Colour(232, 245, 233)
    
    # Selection
    SELECTED = wx.Colour(232, 240, 254)
    HOVER = wx.Colour(245, 245, 245)


class Spacing:
    """Consistent spacing values."""
    XS = 4
    SM = 8
    MD = 12
    LG = 16
    XL = 24
    XXL = 32


class Fonts:
    """Font configurations with caching."""
    _cache = {}
    
    @classmethod
    def _get(cls, key, size, family, style, weight):
        if key not in cls._cache:
            cls._cache[key] = wx.Font(size, family, style, weight)
        return cls._cache[key]
    
    @classmethod
    def header(cls):
        return cls._get('header', 15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
    
    @classmethod
    def title(cls):
        return cls._get('title', 12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
    
    @classmethod
    def subtitle(cls):
        return cls._get('subtitle', 11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
    
    @classmethod
    def body(cls):
        return cls._get('body', 10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
    
    @classmethod
    def small(cls):
        return cls._get('small', 9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
    
    @classmethod
    def mono(cls):
        return cls._get('mono', 9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
    
    @classmethod
    def bold(cls):
        return cls._get('bold', 10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)


class BaseDialog(wx.Dialog):
    """Base dialog with common functionality."""
    
    def __init__(self, parent, title, size, min_size=(400, 300), **kwargs):
        size = (max(size[0], min_size[0]), max(size[1], min_size[1]))
        style = kwargs.pop('style', wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        
        super().__init__(parent, title=title, size=size, style=style, **kwargs)
        
        self.SetMinSize(min_size)
        self.SetBackgroundColour(Colors.PANEL_BG)
        
        if parent:
            self.CentreOnParent()
        else:
            self.CentreOnScreen()
        
        self.Bind(wx.EVT_CHAR_HOOK, self._on_char)
    
    def _on_char(self, event):
        if event.GetKeyCode() == wx.WXK_ESCAPE:
            self.EndModal(wx.ID_CANCEL)
        else:
            event.Skip()


class BaseFrame(wx.Frame):
    """Base frame for modeless dialogs."""
    
    def __init__(self, parent, title, size, min_size=(400, 300), **kwargs):
        size = (max(size[0], min_size[0]), max(size[1], min_size[1]))
        style = kwargs.pop('style', wx.DEFAULT_FRAME_STYLE)
        
        super().__init__(parent, title=title, size=size, style=style, **kwargs)
        
        self.SetMinSize(min_size)
        self.SetBackgroundColour(Colors.PANEL_BG)
        
        if parent:
            self.CentreOnParent()
        else:
            self.CentreOnScreen()
        
        self.Bind(wx.EVT_CHAR_HOOK, self._on_char)
    
    def _on_char(self, event):
        if event.GetKeyCode() == wx.WXK_ESCAPE:
            self.Close()
        else:
            event.Skip()


class IconButton(wx.Button):
    """Button with Unicode icon prefix."""
    
    ICONS = {
        'add': '+', 'new': '+', 'delete': '×', 'remove': '×',
        'edit': '✎', 'save': '💾', 'open': '↗', 'refresh': '↻',
        'search': '⌕', 'check': '✓', 'play': '▶', 'pause': '⏸',
        'stop': '⏹', 'settings': '⚙', 'info': 'ℹ', 'warning': '⚠',
        'error': '✖', 'success': '✓', 'export': '↓', 'import': '↑',
    }
    
    def __init__(self, parent, label, icon=None, **kwargs):
        if icon and icon in self.ICONS:
            label = f"{self.ICONS[icon]} {label}"
        super().__init__(parent, label=label, **kwargs)


class AccentButton(wx.Button):
    """Primary action button with accent color."""
    
    def __init__(self, parent, label, **kwargs):
        super().__init__(parent, label=label, **kwargs)
        self.SetBackgroundColour(Colors.ACCENT)
        self.SetForegroundColour(wx.WHITE)


class InfoBanner(wx.Panel):
    """Information banner with icon."""
    
    def __init__(self, parent, message, style='info'):
        super().__init__(parent)
        
        configs = {
            'info': (Colors.INFO_BG, Colors.ACCENT, 'ℹ'),
            'warning': (Colors.WARNING_BG, Colors.WARNING, '⚠'),
            'error': (Colors.ERROR_BG, Colors.ERROR, '✖'),
            'success': (Colors.SUCCESS_BG, Colors.SUCCESS, '✓'),
        }
        bg, fg, icon = configs.get(style, configs['info'])
        
        self.SetBackgroundColour(bg)
        
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        icon_text = wx.StaticText(self, label=icon)
        icon_text.SetForegroundColour(fg)
        icon_text.SetFont(Fonts.title())
        sizer.Add(icon_text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, Spacing.MD)
        
        msg_text = wx.StaticText(self, label=message)
        msg_text.SetForegroundColour(Colors.TEXT_PRIMARY)
        msg_text.SetFont(Fonts.body())
        msg_text.Wrap(600)
        sizer.Add(msg_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, Spacing.MD)
        
        self.SetSizer(sizer)


class SectionHeader(wx.Panel):
    """Section header with title and optional subtitle."""
    
    def __init__(self, parent, title, subtitle=None):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.PANEL_BG)
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        title_text = wx.StaticText(self, label=title)
        title_text.SetFont(Fonts.title())
        title_text.SetForegroundColour(Colors.TEXT_PRIMARY)
        sizer.Add(title_text, 0, wx.BOTTOM, Spacing.XS)
        
        if subtitle:
            sub_text = wx.StaticText(self, label=subtitle)
            sub_text.SetFont(Fonts.small())
            sub_text.SetForegroundColour(Colors.TEXT_SECONDARY)
            sizer.Add(sub_text, 0)
        
        self.SetSizer(sizer)


class StatusIndicator(wx.Panel):
    """Status bar with icon and message."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.BACKGROUND)
        
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.icon = wx.StaticText(self, label="●")
        self.icon.SetForegroundColour(Colors.SUCCESS)
        sizer.Add(self.icon, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, Spacing.XS)
        
        self.text = wx.StaticText(self, label="Ready")
        self.text.SetFont(Fonts.small())
        self.text.SetForegroundColour(Colors.TEXT_SECONDARY)
        sizer.Add(self.text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, Spacing.XS)
        
        self.SetSizer(sizer)
    
    def set_status(self, message, status='ok'):
        colors = {
            'ok': Colors.SUCCESS, 'success': Colors.SUCCESS,
            'warning': Colors.WARNING,
            'error': Colors.ERROR,
            'working': Colors.ACCENT, 'info': Colors.INFO,
        }
        self.icon.SetForegroundColour(colors.get(status, Colors.SUCCESS))
        self.text.SetLabel(message)
        self.Refresh()


class SearchBox(wx.Panel):
    """Search/filter input box."""
    
    def __init__(self, parent, placeholder="Filter..."):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.BACKGROUND)
        
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        icon = wx.StaticText(self, label="⌕")
        icon.SetFont(Fonts.body())
        icon.SetForegroundColour(Colors.TEXT_SECONDARY)
        sizer.Add(icon, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, Spacing.SM)
        
        self.text = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER, size=(200, -1))
        self.text.SetHint(placeholder)
        self.text.SetFont(Fonts.body())
        sizer.Add(self.text, 1, wx.ALL | wx.EXPAND, Spacing.XS)
        
        self.clear_btn = wx.Button(self, label="×", size=(24, 24))
        self.clear_btn.SetToolTip("Clear")
        self.clear_btn.Bind(wx.EVT_BUTTON, self._on_clear)
        self.clear_btn.Hide()
        sizer.Add(self.clear_btn, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, Spacing.XS)
        
        self.SetSizer(sizer)
        self.text.Bind(wx.EVT_TEXT, self._on_text)
    
    def _on_text(self, event):
        self.clear_btn.Show(bool(self.text.GetValue()))
        self.Layout()
        event.Skip()
    
    def _on_clear(self, event):
        self.text.SetValue("")
        self.text.SetFocus()
    
    def GetValue(self):
        return self.text.GetValue()
    
    def Bind(self, event_type, handler):
        if event_type == wx.EVT_TEXT:
            self.text.Bind(event_type, handler)
        else:
            super().Bind(event_type, handler)


class LabeledControl(wx.Panel):
    """Control with label."""
    
    def __init__(self, parent, label, control, vertical=False):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.PANEL_BG)
        
        if vertical:
            sizer = wx.BoxSizer(wx.VERTICAL)
        else:
            sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        lbl = wx.StaticText(self, label=label)
        lbl.SetFont(Fonts.body())
        lbl.SetForegroundColour(Colors.TEXT_SECONDARY)
        
        if vertical:
            sizer.Add(lbl, 0, wx.BOTTOM, Spacing.XS)
            sizer.Add(control, 0, wx.EXPAND)
        else:
            sizer.Add(lbl, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, Spacing.SM)
            sizer.Add(control, 1)
        
        self.SetSizer(sizer)
        self.control = control


class ProgressDialog(BaseDialog):
    """Progress indicator dialog."""
    
    def __init__(self, parent, title="Working..."):
        super().__init__(parent, title, size=(450, 150), min_size=(350, 120),
                        style=wx.CAPTION | wx.STAY_ON_TOP)
        
        panel = wx.Panel(self)
        panel.SetBackgroundColour(Colors.PANEL_BG)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.label = wx.StaticText(panel, label="Initializing...")
        self.label.SetFont(Fonts.body())
        sizer.Add(self.label, 0, wx.ALL | wx.EXPAND, Spacing.LG)
        
        self.gauge = wx.Gauge(panel, range=100, size=(-1, 8))
        sizer.Add(self.gauge, 0, wx.LEFT | wx.RIGHT | wx.EXPAND, Spacing.LG)
        
        self.percent = wx.StaticText(panel, label="0%")
        self.percent.SetFont(Fonts.small())
        self.percent.SetForegroundColour(Colors.TEXT_SECONDARY)
        sizer.Add(self.percent, 0, wx.ALL, Spacing.LG)
        
        panel.SetSizer(sizer)
        
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(main_sizer)
    
    def update(self, percent: int, message: str):
        self.gauge.SetValue(min(percent, 100))
        self.label.SetLabel(message)
        self.percent.SetLabel(f"{percent}%")
        wx.Yield()


class TabPanel(wx.Panel):
    """Base class for notebook tab panels."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.PANEL_BG)


__all__ = [
    'Colors', 'Spacing', 'Fonts',
    'BaseDialog', 'BaseFrame', 'IconButton', 'AccentButton',
    'InfoBanner', 'SectionHeader', 'StatusIndicator', 'SearchBox',
    'LabeledControl', 'ProgressDialog', 'TabPanel',
]
