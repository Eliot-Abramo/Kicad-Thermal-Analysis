"""
TVAC Thermal Analyzer - UI Module
=================================
User interface components and dialogs.
"""

from .design_system import (
    Colors, Spacing, Fonts,
    BaseDialog, BaseFrame, IconButton, AccentButton,
    InfoBanner, SectionHeader, StatusIndicator, SearchBox,
    LabeledControl, ProgressDialog, TabPanel
)

from .pcb_visualization import (
    PCBVisualizationPanel, LayerTogglePanel, VisualizationLayer
)

from .main_dialog import MainDialog
from .results_viewer import ResultsViewerDialog

__all__ = [
    # Design System
    'Colors', 'Spacing', 'Fonts',
    'BaseDialog', 'BaseFrame', 'IconButton', 'AccentButton',
    'InfoBanner', 'SectionHeader', 'StatusIndicator', 'SearchBox',
    'LabeledControl', 'ProgressDialog', 'TabPanel',
    
    # Visualization
    'PCBVisualizationPanel', 'LayerTogglePanel', 'VisualizationLayer',
    
    # Dialogs
    'MainDialog',
    'ResultsViewerDialog',
]
