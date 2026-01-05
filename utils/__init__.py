"""
TVAC Thermal Analyzer - Utilities Module
========================================
Logging, report generation, and helper functions.
"""

from .logger import (
    ThermalAnalyzerLogger,
    ThermalAnalyzerFormatter,
    PerformanceTracker,
    get_logger,
    initialize_logger,
    log_function,
    timed_function,
    log_section,
    log_exception_handler,
    format_error_report,
)

# Lazy import for report generator to avoid circular imports
def get_report_generator():
    """Get the ThermalReportGenerator class (lazy import)."""
    from .report_generator import ThermalReportGenerator
    return ThermalReportGenerator

def get_report_metadata():
    """Get the ReportMetadata class (lazy import)."""
    from .report_generator import ReportMetadata
    return ReportMetadata

# For backwards compatibility, we also expose these at module level
# but they will be imported lazily
ReportMetadata = None
ThermalReportGenerator = None

def __getattr__(name):
    """Lazy attribute access for report classes."""
    if name == 'ReportMetadata':
        from .report_generator import ReportMetadata
        return ReportMetadata
    elif name == 'ThermalReportGenerator':
        from .report_generator import ThermalReportGenerator
        return ThermalReportGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Logger
    'ThermalAnalyzerLogger',
    'ThermalAnalyzerFormatter',
    'PerformanceTracker',
    'get_logger',
    'initialize_logger',
    'log_function',
    'timed_function',
    'log_section',
    'log_exception_handler',
    'format_error_report',
    # Report (lazy loaded)
    'ReportMetadata',
    'ThermalReportGenerator',
    'get_report_generator',
    'get_report_metadata',
]
