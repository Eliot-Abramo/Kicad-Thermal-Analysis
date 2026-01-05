"""
TVAC Thermal Analyzer - Logging System
======================================
Comprehensive logging with multiple levels, file output, and performance tracking.

Author: Space Electronics Thermal Analysis Tool
Version: 1.0.0
"""

import logging
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from functools import wraps
from pathlib import Path
import threading
from contextlib import contextmanager

def _safe_isatty(stream) -> bool:
    """Return True if stream looks like a tty; never raise."""
    try:
        return bool(getattr(stream, "isatty", lambda: False)())
    except Exception:
        return False

def _get_console_stream():
    """
    KiCad (GUI) can set sys.stdout/sys.stderr to None.
    Prefer the original streams if available.
    Returning None is fine: logging.StreamHandler(None) falls back to sys.stderr.
    """
    for name in ("stdout", "__stdout__", "stderr", "__stderr__"):
        s = getattr(sys, name, None)
        if s is not None:
            return s
    return None

class ThermalAnalyzerFormatter(logging.Formatter):
    """Custom formatter with color support and detailed formatting."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m',
    }
    
    def __init__(self, use_colors: bool = True, stream=None):
        super().__init__()
        if stream is None:
            stream = _get_console_stream()
        self.use_colors = bool(use_colors and _safe_isatty(stream))
    
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        level = record.levelname
        module = record.module
        func = record.funcName
        line = record.lineno
        message = record.getMessage()
        
        # Add thread info for multi-threaded operations
        thread_name = threading.current_thread().name
        if thread_name != 'MainThread':
            thread_info = f"[{thread_name}]"
        else:
            thread_info = ""
        
        if self.use_colors:
            color = self.COLORS.get(level, '')
            reset = self.COLORS['RESET']
            formatted = f"{timestamp} {color}{level:8s}{reset} {thread_info}[{module}.{func}:{line}] {message}"
        else:
            formatted = f"{timestamp} {level:8s} {thread_info}[{module}.{func}:{line}] {message}"
        
        # Add exception info if present
        if record.exc_info:
            formatted += '\n' + self.formatException(record.exc_info)
        
        return formatted


class PerformanceTracker:
    """Tracks performance metrics for simulation operations."""
    
    def __init__(self):
        self.timings: Dict[str, list] = {}
        self.memory_samples: list = []
        self.lock = threading.Lock()
    
    def record_timing(self, operation: str, duration_s: float):
        """Record timing for an operation."""
        with self.lock:
            if operation not in self.timings:
                self.timings[operation] = []
            self.timings[operation].append(duration_s)
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        with self.lock:
            if operation not in self.timings or not self.timings[operation]:
                return {'count': 0, 'total': 0, 'mean': 0, 'min': 0, 'max': 0}
            
            times = self.timings[operation]
            return {
                'count': len(times),
                'total': sum(times),
                'mean': sum(times) / len(times),
                'min': min(times),
                'max': max(times)
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {op: self.get_stats(op) for op in self.timings}
    
    def clear(self):
        """Clear all recorded data."""
        with self.lock:
            self.timings.clear()
            self.memory_samples.clear()


class ThermalAnalyzerLogger:
    """Main logger class for TVAC Thermal Analyzer."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern for global logger access."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, 
                 log_dir: Optional[str] = None,
                 log_level: int = logging.DEBUG,
                 console_level: int = logging.INFO,
                 enable_file_logging: bool = True,
                 enable_performance_tracking: bool = True):
        """Initialize the logger."""
        if self._initialized:
            return
        
        self._initialized = True
        self.log_dir = log_dir
        self.log_level = log_level
        self.console_level = console_level
        self.enable_file_logging = enable_file_logging
        
        # Create logger
        self.logger = logging.getLogger('TVACThermalAnalyzer')
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear any existing handlers
        
        # Console handler
        stream = _get_console_stream()
        console_handler = logging.StreamHandler(stream)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(ThermalAnalyzerFormatter(use_colors=True, stream=stream))
        self.logger.addHandler(console_handler)
        
        # File handler
        if enable_file_logging:
            self._setup_file_handler()
        
        # Performance tracker
        self.performance = PerformanceTracker() if enable_performance_tracking else None
        
        # Simulation state
        self.simulation_id: Optional[str] = None
        self.simulation_start_time: Optional[float] = None
    
    def _setup_file_handler(self):
        """Setup file logging handler."""
        if not self.log_dir:
            # Default to user's home directory
            self.log_dir = os.path.join(
                os.path.expanduser('~'),
                '.tvac_thermal_analyzer',
                'logs'
            )
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.log_dir, f'thermal_analyzer_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(ThermalAnalyzerFormatter(use_colors=False))
        self.logger.addHandler(file_handler)
        
        self.current_log_file = log_file
        self.logger.info(f"Log file created: {log_file}")
    
    def set_log_level(self, level: int):
        """Change log level at runtime."""
        self.logger.setLevel(level)
        self.log_level = level
    
    def set_console_level(self, level: int):
        """Change console log level at runtime."""
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(level)
        self.console_level = level
    
    # Logging methods
    def debug(self, message: str, *args, **kwargs):
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        self.logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        self.logger.exception(message, *args, **kwargs)
    
    # Simulation lifecycle logging
    def start_simulation(self, sim_id: str, params: Dict[str, Any]):
        """Log simulation start with parameters."""
        self.simulation_id = sim_id
        self.simulation_start_time = time.time()
        
        self.info("=" * 60)
        self.info(f"SIMULATION STARTED: {sim_id}")
        self.info("=" * 60)
        self.info("Parameters:")
        for key, value in params.items():
            self.info(f"  {key}: {value}")
        self.info("-" * 60)
    
    def end_simulation(self, success: bool = True, message: str = ""):
        """Log simulation end with summary."""
        if self.simulation_start_time:
            duration = time.time() - self.simulation_start_time
        else:
            duration = 0
        
        self.info("-" * 60)
        status = "COMPLETED" if success else "FAILED"
        self.info(f"SIMULATION {status}: {self.simulation_id}")
        self.info(f"Total Duration: {duration:.2f} seconds")
        
        if message:
            self.info(f"Message: {message}")
        
        # Log performance summary
        if self.performance:
            stats = self.performance.get_all_stats()
            if stats:
                self.info("Performance Summary:")
                for op, s in stats.items():
                    self.info(f"  {op}: {s['count']} calls, total={s['total']:.3f}s, mean={s['mean']:.3f}s")
        
        self.info("=" * 60)
        
        self.simulation_id = None
        self.simulation_start_time = None
        if self.performance:
            self.performance.clear()
    
    def log_progress(self, current: int, total: int, stage: str = ""):
        """Log simulation progress."""
        percent = (current / total * 100) if total > 0 else 0
        self.info(f"Progress: {current}/{total} ({percent:.1f}%) {stage}")
    
    def log_thermal_result(self, timestep: int, time_s: float, 
                          min_temp: float, max_temp: float, avg_temp: float):
        """Log thermal simulation results for a timestep."""
        self.debug(f"t={time_s:.3f}s: Tmin={min_temp:.2f}°C, Tmax={max_temp:.2f}°C, Tavg={avg_temp:.2f}°C")
    
    def log_convergence(self, iteration: int, residual: float, target: float):
        """Log solver convergence progress."""
        self.debug(f"Iteration {iteration}: residual={residual:.2e}, target={target:.2e}")
    
    def log_mesh_stats(self, nodes: int, elements: int, min_size: float, max_size: float):
        """Log mesh generation statistics."""
        self.info(f"Mesh: {nodes} nodes, {elements} elements, size range [{min_size:.3f}, {max_size:.3f}] mm")
    
    def log_component(self, ref: str, power: float, thermal_data: Dict):
        """Log component thermal setup."""
        self.debug(f"Component {ref}: {power:.3f}W, θja={thermal_data.get('theta_ja', 'N/A')}°C/W")
    
    def log_memory_usage(self):
        """Log current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            self.debug(f"Memory: RSS={mem_info.rss / 1024 / 1024:.1f}MB, VMS={mem_info.vms / 1024 / 1024:.1f}MB")
        except ImportError:
            pass


# Global logger instance
_logger: Optional[ThermalAnalyzerLogger] = None


def get_logger() -> ThermalAnalyzerLogger:
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        _logger = ThermalAnalyzerLogger()
    return _logger


def initialize_logger(log_dir: Optional[str] = None,
                     log_level: int = logging.DEBUG,
                     console_level: int = logging.INFO,
                     enable_file_logging: bool = True) -> ThermalAnalyzerLogger:
    """Initialize the global logger with custom settings."""
    global _logger
    _logger = ThermalAnalyzerLogger(
        log_dir=log_dir,
        log_level=log_level,
        console_level=console_level,
        enable_file_logging=enable_file_logging
    )
    return _logger


# Decorators for logging and performance tracking
def log_function(level: int = logging.DEBUG):
    """Decorator to log function entry and exit."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            func_name = func.__name__
            logger.logger.log(level, f"Entering {func_name}")
            try:
                result = func(*args, **kwargs)
                logger.logger.log(level, f"Exiting {func_name}")
                return result
            except Exception as e:
                logger.error(f"Exception in {func_name}: {e}")
                raise
        return wrapper
    return decorator


def timed_function(operation_name: Optional[str] = None):
    """Decorator to time function execution."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            op_name = operation_name or func.__name__
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                if logger.performance:
                    logger.performance.record_timing(op_name, duration)
                logger.debug(f"{op_name} completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start
                logger.error(f"{op_name} failed after {duration:.3f}s: {e}")
                raise
        return wrapper
    return decorator


@contextmanager
def log_section(section_name: str):
    """Context manager for logging a section of code."""
    logger = get_logger()
    logger.info(f"--- {section_name} ---")
    start = time.time()
    try:
        yield
        duration = time.time() - start
        logger.info(f"--- {section_name} completed in {duration:.3f}s ---")
    except Exception as e:
        duration = time.time() - start
        logger.error(f"--- {section_name} failed after {duration:.3f}s: {e} ---")
        raise


def log_exception_handler(func: Callable):
    """Decorator to catch and log exceptions without crashing."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.critical(f"Unhandled exception in {func.__name__}: {e}")
            logger.critical(traceback.format_exc())
            # Re-raise to allow proper error handling upstream
            raise
    return wrapper


# Utility function for safe error reporting
def format_error_report(error: Exception, context: Dict[str, Any] = None) -> str:
    """Format a comprehensive error report."""
    lines = [
        "=" * 60,
        "ERROR REPORT",
        "=" * 60,
        f"Error Type: {type(error).__name__}",
        f"Error Message: {str(error)}",
        "",
        "Traceback:",
        traceback.format_exc(),
    ]
    
    if context:
        lines.extend([
            "",
            "Context:",
        ])
        for key, value in context.items():
            lines.append(f"  {key}: {value}")
    
    lines.append("=" * 60)
    return "\n".join(lines)


__all__ = [
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
]
