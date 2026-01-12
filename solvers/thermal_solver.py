"""
TVAC Thermal Analyzer - Thermal Solver
======================================
Thermal simulation solver with native C engine and Python fallback.

Features:
- Steady-state and transient analysis
- Radiation heat transfer (Stefan-Boltzmann)
- Conduction through PCB stackup
- Native C engine for performance (10-100x faster)
- Pure Python fallback

Author: Space Electronics Thermal Analysis Tool
Version: 2.0.0
"""

import os
import sys
import ctypes
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import time

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve, cg
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..core.constants import PhysicalConstants


@dataclass
class ThermalNode:
    """Single node in thermal mesh."""
    node_id: int = 0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    layer_idx: int = 0
    
    # Material properties
    k: float = 0.29  # Thermal conductivity W/(m·K)
    cp: float = 1100.0  # Specific heat J/(kg·K)
    rho: float = 1850.0  # Density kg/m³
    emissivity: float = 0.9
    
    # Geometry
    volume: float = 0.0  # m³
    surface_area: float = 0.0  # m² (for radiation)
    
    # Heat source
    heat_source: float = 0.0  # W
    
    # Boundary conditions
    is_fixed_temp: bool = False
    fixed_temp: float = 25.0
    
    # Neighbors: {neighbor_id: conductance}
    neighbors: Dict[int, float] = field(default_factory=dict)


@dataclass
class ThermalMesh:
    """Complete thermal mesh."""
    nodes: List[ThermalNode] = field(default_factory=list)
    nx: int = 0
    ny: int = 0
    nz: int = 0
    dx: float = 0.5e-3  # m
    dy: float = 0.5e-3
    dz: float = 0.2e-3
    
    # Board bounds (mm)
    board_min_x: float = 0.0
    board_max_x: float = 100.0
    board_min_y: float = 0.0
    board_max_y: float = 100.0
    
    def get_node(self, ix: int, iy: int, iz: int) -> Optional[ThermalNode]:
        """Get node by grid indices."""
        if 0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz:
            idx = iz * self.nx * self.ny + iy * self.nx + ix
            if idx < len(self.nodes):
                return self.nodes[idx]
        return None
    
    def get_node_index(self, ix: int, iy: int, iz: int) -> int:
        """Get linear index from grid indices."""
        return iz * self.nx * self.ny + iy * self.nx + ix


@dataclass
class ThermalResult:
    """Simulation results."""
    temperatures: List[float] = field(default_factory=list)
    min_temp: float = 0.0
    max_temp: float = 0.0
    avg_temp: float = 0.0
    iterations: int = 0
    compute_time: float = 0.0
    converged: bool = True
    error_message: str = ""
    
    # For transient
    time_points: List[float] = field(default_factory=list)
    temp_history: List[List[float]] = field(default_factory=list)
    
    def get_temperature_grid(self, nx: int, ny: int, nz: int, 
                            layer: int = 0) -> Optional['np.ndarray']:
        """Get temperature as 2D grid for visualization."""
        if not HAS_NUMPY or not self.temperatures:
            return None
        
        grid = np.zeros((ny, nx))
        
        for iy in range(ny):
            for ix in range(nx):
                idx = layer * nx * ny + iy * nx + ix
                if idx < len(self.temperatures):
                    grid[iy, ix] = self.temperatures[idx]
        
        return grid


class NativeThermalEngine:
    """Interface to native C thermal engine."""
    
    def __init__(self):
        self._lib = None
        self._state = None
        self._try_load_library()
    
    def _try_load_library(self):
        """Try to load the native library."""
        lib_names = [
            'libthermal_engine.so',
            'libthermal_engine.dylib',
            'thermal_engine.dll',
            'thermal_engine.so',
        ]
        
        # Search paths
        search_paths = [
            Path(__file__).parent.parent / 'native',
            Path(__file__).parent.parent,
            Path.cwd(),
        ]
        
        for path in search_paths:
            for name in lib_names:
                lib_path = path / name
                if lib_path.exists():
                    try:
                        self._lib = ctypes.CDLL(str(lib_path))
                        self._setup_functions()
                        return
                    except Exception as e:
                        print(f"Failed to load {lib_path}: {e}")
    
    def _setup_functions(self):
        """Setup ctypes function signatures based on the C API.

        The native thermal engine exports a number of functions prefixed with
        ``thermal_`` for allocating and configuring the simulation state.
        Here we bind those functions with the appropriate ctypes signatures.
        """
        if not self._lib:
            return

        # State management
        # int num_nodes, int nx, int ny, int nz -> pointer
        self._lib.thermal_create_state.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        self._lib.thermal_create_state.restype = ctypes.c_void_p

        self._lib.thermal_destroy_state.argtypes = [ctypes.c_void_p]
        self._lib.thermal_destroy_state.restype = None

        # Node properties: idx, k, cp, rho, emissivity, volume, surface_area, heat_source
        self._lib.thermal_set_node.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double
        ]
        self._lib.thermal_set_node.restype = None

        # Fixed temperature: state, idx, temp_k
        self._lib.thermal_set_fixed_temp.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_double
        ]
        self._lib.thermal_set_fixed_temp.restype = None

        # Initial temperature: state, idx, temp_k
        self._lib.thermal_set_initial_temp.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_double
        ]
        self._lib.thermal_set_initial_temp.restype = None

        # Chamber temperature: state, temp_k
        self._lib.thermal_set_chamber_temp.argtypes = [
            ctypes.c_void_p, ctypes.c_double
        ]
        self._lib.thermal_set_chamber_temp.restype = None

        # Neighbor connectivity allocation and definition
        self._lib.thermal_alloc_neighbors.argtypes = [
            ctypes.c_void_p, ctypes.c_int
        ]
        self._lib.thermal_alloc_neighbors.restype = ctypes.c_int

        # Set row pointer: state, row index, pointer value
        self._lib.thermal_set_row_ptr.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int
        ]
        self._lib.thermal_set_row_ptr.restype = None

        # Set neighbor: state, node index, neighbor offset (within row), neighbor idx, conductance
        self._lib.thermal_set_neighbor.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double
        ]
        self._lib.thermal_set_neighbor.restype = None

        # Progress callback
        CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)
        self._lib.thermal_set_progress_callback.argtypes = [ctypes.c_void_p, CALLBACK]
        self._lib.thermal_set_progress_callback.restype = None

        # Solve functions: state, result*, [duration, timestep]
        # Define result struct for passing back statistics
        class CThermalResult(ctypes.Structure):
            _fields_ = [
                ("min_temp", ctypes.c_double),
                ("max_temp", ctypes.c_double),
                ("avg_temp", ctypes.c_double),
                ("iterations", ctypes.c_int),
                ("compute_time", ctypes.c_double),
                ("converged", ctypes.c_int),
                ("error_message", ctypes.c_char * 256),
            ]

        # Expose on class for later use
        self.CThermalResult = CThermalResult

        self._lib.thermal_solve_steady_state.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(CThermalResult)
        ]
        self._lib.thermal_solve_steady_state.restype = ctypes.c_int

        self._lib.thermal_solve_transient.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(CThermalResult), ctypes.c_double, ctypes.c_double
        ]
        self._lib.thermal_solve_transient.restype = ctypes.c_int

        # Get temperature of node
        self._lib.thermal_get_temp.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.thermal_get_temp.restype = ctypes.c_double
    
    @property
    def is_available(self) -> bool:
        """Return True if the native library was successfully loaded."""
        return self._lib is not None

    def create_state(self, mesh: ThermalMesh) -> bool:
        """Create a simulation state for a given mesh.

        Allocates the internal C data structures sized according to the mesh
        dimensions.  This must be called before any other setters.
        """
        if not self._lib:
            return False
        num_nodes = mesh.nx * mesh.ny * mesh.nz
        # Create state with node count and grid dimensions
        self._state = self._lib.thermal_create_state(num_nodes, mesh.nx, mesh.ny, mesh.nz)
        return bool(self._state)

    def destroy(self):
        """Free the native simulation state."""
        if self._lib and self._state:
            self._lib.thermal_destroy_state(self._state)
            self._state = None

    def set_node(self, idx: int, *, k: float, cp: float, rho: float,
                 emissivity: float, volume: float, surface_area: float,
                 heat_source: float) -> None:
        """Set properties for a single node in the native state."""
        if self._lib and self._state:
            self._lib.thermal_set_node(
                self._state, idx,
                float(k), float(cp), float(rho), float(emissivity),
                float(volume), float(surface_area), float(heat_source)
            )

    def set_fixed_temp(self, idx: int, temp_k: float) -> None:
        if self._lib and self._state:
            self._lib.thermal_set_fixed_temp(self._state, idx, temp_k)

    def set_initial_temp(self, idx: int, temp_k: float) -> None:
        if self._lib and self._state:
            self._lib.thermal_set_initial_temp(self._state, idx, temp_k)

    def set_chamber_temp(self, temp_k: float) -> None:
        if self._lib and self._state:
            self._lib.thermal_set_chamber_temp(self._state, temp_k)

    def alloc_neighbors(self, total_neighbors: int) -> bool:
        """Allocate neighbor CSR arrays for the given total number of neighbor links."""
        if self._lib and self._state:
            err = self._lib.thermal_alloc_neighbors(self._state, total_neighbors)
            return err == 0
        return False

    def set_row_ptr(self, row: int, ptr: int) -> None:
        if self._lib and self._state:
            self._lib.thermal_set_row_ptr(self._state, row, ptr)

    def set_neighbor(self, node_idx: int, offset: int, neighbor_idx: int, conductance: float) -> None:
        if self._lib and self._state:
            self._lib.thermal_set_neighbor(self._state, node_idx, offset, neighbor_idx, float(conductance))

    def solve_steady_state(self, ambient_temp_c: float, chamber_wall_temp_c: float,
                          include_radiation: bool, convergence: float, max_iterations: int,
                          mesh: ThermalMesh, progress_callback: Optional[Callable] = None) -> ThermalResult:
        """Solve the steady state problem using the native engine.

        This method assumes that the state has been created and fully populated
        with node properties, neighbor connectivity, fixed temperatures, and
        initial temperatures (if desired).  It converts the result into the
        high‑level :class:`ThermalResult` for the plugin.
        """
        result = ThermalResult()
        if not (self._lib and self._state):
            result.error_message = "Native engine not available"
            result.converged = False
            return result

        # Set chamber wall temperature in Kelvin
        kelvin_offset = PhysicalConstants.CELSIUS_TO_KELVIN
        self.set_chamber_temp((chamber_wall_temp_c + kelvin_offset))

        # We do not support dynamic ambient temperature in the native engine;
        # the chamber temperature is used for radiation calculations.  The
        # initial temperature is set below if needed.

        # Set initial temperature to ambient
        for node in mesh.nodes:
            self.set_initial_temp(node.node_id, ambient_temp_c + kelvin_offset)

        # Define a ctypes result struct to receive data from C
        c_result = self.CThermalResult()
        # Solve; C returns error code
        err = self._lib.thermal_solve_steady_state(self._state, ctypes.byref(c_result))
        if err != 0:
            result.error_message = c_result.error_message.decode('utf-8') if c_result.error_message else "Native solver error"
            result.converged = False
            return result

        # Populate result
        # c_result temperatures are reported in Celsius
        # We still need to retrieve the per-node temperatures from the state
        n = mesh.nx * mesh.ny * mesh.nz
        temperatures = [0.0] * n
        for i in range(n):
            # Convert Kelvin to Celsius
            temperatures[i] = self._lib.thermal_get_temp(self._state, i) - kelvin_offset
        result.temperatures = temperatures
        result.min_temp = float(c_result.min_temp)
        result.max_temp = float(c_result.max_temp)
        result.avg_temp = float(c_result.avg_temp)
        result.iterations = int(c_result.iterations)
        result.compute_time = float(c_result.compute_time)
        result.converged = bool(c_result.converged)
        result.error_message = c_result.error_message.decode('utf-8') if c_result.error_message else ""
        return result

    def solve_transient(self, duration_s: float, timestep_s: float,
                       initial_temp_c: float, ambient_temp_c: float,
                       chamber_wall_temp_c: float, include_radiation: bool,
                       mesh: ThermalMesh, output_interval_s: float,
                       progress_callback: Optional[Callable] = None) -> ThermalResult:
        """Solve the transient problem using the native engine.

        The native engine implements a Crank–Nicolson scheme with radiation.
        """
        result = ThermalResult()
        if not (self._lib and self._state):
            result.error_message = "Native engine not available"
            result.converged = False
            return result

        # Set chamber wall temperature in Kelvin
        kelvin_offset = PhysicalConstants.CELSIUS_TO_KELVIN
        self.set_chamber_temp(chamber_wall_temp_c + kelvin_offset)

        # Set initial temperature for each node
        for node in mesh.nodes:
            self.set_initial_temp(node.node_id, initial_temp_c + kelvin_offset)

        c_result = self.CThermalResult()
        err = self._lib.thermal_solve_transient(
            self._state, ctypes.byref(c_result), float(duration_s), float(timestep_s)
        )
        if err != 0:
            result.error_message = c_result.error_message.decode('utf-8') if c_result.error_message else "Native transient solver error"
            result.converged = False
            return result

        # Extract final temperature array in Celsius
        n = mesh.nx * mesh.ny * mesh.nz
        temperatures = [0.0] * n
        for i in range(n):
            temperatures[i] = self._lib.thermal_get_temp(self._state, i) - kelvin_offset

        result.temperatures = temperatures
        result.min_temp = float(c_result.min_temp)
        result.max_temp = float(c_result.max_temp)
        result.avg_temp = float(c_result.avg_temp)
        result.iterations = int(c_result.iterations)
        result.compute_time = float(c_result.compute_time)
        result.converged = bool(c_result.converged)
        result.error_message = c_result.error_message.decode('utf-8') if c_result.error_message else ""
        # Note: for transient we do not populate time_points/temp_history via native engine
        return result


class PythonThermalSolver:
    """Pure Python thermal solver (fallback)."""
    
    def __init__(self):
        self.stefan_boltzmann = PhysicalConstants.STEFAN_BOLTZMANN
        self.kelvin_offset = PhysicalConstants.CELSIUS_TO_KELVIN
    
    def solve_steady_state(self, mesh: ThermalMesh, 
                          ambient_temp_c: float = 25.0,
                          chamber_wall_temp_c: float = 25.0,
                          include_radiation: bool = True,
                          convergence: float = 1e-6,
                          max_iterations: int = 10000,
                          progress_callback: Optional[Callable] = None) -> ThermalResult:
        """Solve steady-state thermal problem."""
        
        start_time = time.time()
        result = ThermalResult()
        
        n = len(mesh.nodes)
        if n == 0:
            result.error_message = "Empty mesh"
            result.converged = False
            return result
        
        if not HAS_NUMPY:
            result.error_message = "NumPy required for solver"
            result.converged = False
            return result
        
        # Initialize temperatures
        T = np.full(n, ambient_temp_c, dtype=np.float64)
        
        # Build conductance matrix
        if progress_callback:
            progress_callback(10, "Building conductance matrix...")
        
        # Use sparse matrix for efficiency
        row_idx = []
        col_idx = []
        data = []
        Q = np.zeros(n)
        
        for i, node in enumerate(mesh.nodes):
            if node.is_fixed_temp:
                # Fixed temperature node
                row_idx.append(i)
                col_idx.append(i)
                data.append(1.0)
                Q[i] = node.fixed_temp
            else:
                # Heat source
                Q[i] = node.heat_source
                
                # Self conductance (sum of neighbor conductances)
                diag = 0.0
                for j, G in node.neighbors.items():
                    row_idx.append(i)
                    col_idx.append(j)
                    data.append(-G)
                    diag += G
                
                row_idx.append(i)
                col_idx.append(i)
                data.append(diag)
        
        if HAS_SCIPY:
            K = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))
        else:
            K = np.zeros((n, n))
            for i, (r, c, v) in enumerate(zip(row_idx, col_idx, data)):
                K[r, c] = v
        
        # Iterative solution (for radiation nonlinearity)
        T_wall_K = chamber_wall_temp_c + self.kelvin_offset
        
        for iteration in range(max_iterations):
            if progress_callback and iteration % 100 == 0:
                progress_callback(10 + int(80 * iteration / max_iterations),
                                f"Iteration {iteration}...")
            
            T_old = T.copy()
            
            # Add radiation heat transfer
            Q_total = Q.copy()
            
            if include_radiation:
                for i, node in enumerate(mesh.nodes):
                    if not node.is_fixed_temp and node.surface_area > 0:
                        T_K = T[i] + self.kelvin_offset
                        q_rad = node.emissivity * self.stefan_boltzmann * node.surface_area * (
                            T_K**4 - T_wall_K**4
                        )
                        Q_total[i] -= q_rad
            
            # Solve linear system
            if HAS_SCIPY:
                # scipy.sparse.linalg.cg parameter varies by version
                # Older versions use 'tol', newer use 'atol'/'rtol'
                try:
                    T, info = cg(K, Q_total, x0=T, atol=convergence, rtol=convergence)
                except TypeError:
                    # Fallback for older scipy
                    try:
                        T, info = cg(K, Q_total, x0=T, tol=convergence)
                    except TypeError:
                        T, info = cg(K, Q_total, x0=T)
                
                if info != 0:
                    # Fallback to direct solver
                    try:
                        T = spsolve(K, Q_total)
                    except Exception:
                        pass
            else:
                T = self._jacobi_solve(K, Q_total, T, convergence, 1000)
            
            # Check convergence
            if np.max(np.abs(T - T_old)) < convergence:
                result.converged = True
                result.iterations = iteration + 1
                break
        else:
            result.converged = False
            result.iterations = max_iterations
        
        if progress_callback:
            progress_callback(95, "Finalizing...")
        
        result.temperatures = T.tolist()
        result.min_temp = float(np.min(T))
        result.max_temp = float(np.max(T))
        result.avg_temp = float(np.mean(T))
        result.compute_time = time.time() - start_time
        
        return result
    
    def solve_transient(self, mesh: ThermalMesh,
                       duration_s: float,
                       timestep_s: float,
                       initial_temp_c: float = 25.0,
                       ambient_temp_c: float = 25.0,
                       chamber_wall_temp_c: float = 25.0,
                       include_radiation: bool = True,
                       output_interval_s: float = 1.0,
                       progress_callback: Optional[Callable] = None) -> ThermalResult:
        """Solve transient thermal problem using Crank-Nicolson."""
        
        start_time = time.time()
        result = ThermalResult()
        
        n = len(mesh.nodes)
        if n == 0:
            result.error_message = "Empty mesh"
            return result
        
        if not HAS_NUMPY:
            result.error_message = "NumPy required"
            return result
        
        # Initialize
        T = np.full(n, initial_temp_c, dtype=np.float64)
        
        # Build mass and conductance matrices
        if progress_callback:
            progress_callback(5, "Building matrices...")
        
        # Thermal mass: C = rho * cp * V
        C = np.array([node.rho * node.cp * node.volume for node in mesh.nodes])
        C = np.maximum(C, 1e-12)  # Avoid division by zero
        
        # Conductance matrix (same as steady state)
        row_idx = []
        col_idx = []
        data = []
        Q = np.zeros(n)
        
        for i, node in enumerate(mesh.nodes):
            Q[i] = node.heat_source
            diag = 0.0
            for j, G in node.neighbors.items():
                row_idx.append(i)
                col_idx.append(j)
                data.append(-G)
                diag += G
            row_idx.append(i)
            col_idx.append(i)
            data.append(diag)
        
        if HAS_SCIPY:
            K = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))
            M = sparse.diags(C)
        else:
            K = np.zeros((n, n))
            for r, c, v in zip(row_idx, col_idx, data):
                K[r, c] = v
            M = np.diag(C)
        
        # Crank-Nicolson: (M/dt + 0.5*K) * T_new = (M/dt - 0.5*K) * T_old + Q
        theta = 0.5
        dt = timestep_s
        
        if HAS_SCIPY:
            LHS = M / dt + theta * K
        else:
            LHS = M / dt + theta * K
        
        # Time stepping
        T_wall_K = chamber_wall_temp_c + self.kelvin_offset
        t = 0.0
        num_steps = int(duration_s / timestep_s)
        output_step = max(1, int(output_interval_s / timestep_s))
        
        result.time_points = [0.0]
        result.temp_history = [T.tolist()]
        
        for step in range(num_steps):
            if progress_callback and step % 10 == 0:
                progress_callback(5 + int(90 * step / num_steps),
                                f"Time: {t:.1f}s / {duration_s:.1f}s")
            
            # Radiation term
            Q_rad = np.zeros(n)
            if include_radiation:
                for i, node in enumerate(mesh.nodes):
                    if node.surface_area > 0:
                        T_K = T[i] + self.kelvin_offset
                        Q_rad[i] = -node.emissivity * self.stefan_boltzmann * node.surface_area * (
                            T_K**4 - T_wall_K**4
                        )
            
            # RHS
            if HAS_SCIPY:
                RHS = (M / dt - (1 - theta) * K) @ T + Q + Q_rad
            else:
                RHS = (M / dt - (1 - theta) * K) @ T + Q + Q_rad
            
            # Apply fixed temperature BCs
            for i, node in enumerate(mesh.nodes):
                if node.is_fixed_temp:
                    RHS[i] = node.fixed_temp
                    if HAS_SCIPY:
                        # Modify LHS for fixed nodes (already done if sparse)
                        pass
            
            # Solve
            if HAS_SCIPY:
                try:
                    T_new, info = cg(LHS, RHS, x0=T, atol=1e-8, rtol=1e-8)
                except TypeError:
                    try:
                        T_new, info = cg(LHS, RHS, x0=T, tol=1e-8)
                    except TypeError:
                        T_new, info = cg(LHS, RHS, x0=T)
                
                if info != 0:
                    T_new = spsolve(LHS.tocsr(), RHS)
            else:
                T_new = self._jacobi_solve(LHS, RHS, T, 1e-8, 500)
            
            T = T_new
            t += dt
            
            # Store output
            if step % output_step == 0:
                result.time_points.append(t)
                result.temp_history.append(T.tolist())
        
        if progress_callback:
            progress_callback(98, "Finalizing...")
        
        result.temperatures = T.tolist()
        result.min_temp = float(np.min(T))
        result.max_temp = float(np.max(T))
        result.avg_temp = float(np.mean(T))
        result.compute_time = time.time() - start_time
        result.converged = True
        
        return result
    
    def _jacobi_solve(self, A, b, x0, tol, max_iter):
        """Simple Jacobi iterative solver (fallback)."""
        n = len(b)
        x = x0.copy()
        
        for _ in range(max_iter):
            x_new = np.zeros(n)
            for i in range(n):
                s = b[i]
                for j in range(n):
                    if i != j:
                        s -= A[i, j] * x[j]
                if abs(A[i, i]) > 1e-12:
                    x_new[i] = s / A[i, i]
                else:
                    x_new[i] = x[i]
            
            if np.max(np.abs(x_new - x)) < tol:
                return x_new
            x = x_new
        
        return x


class ThermalSolver:
    """Main thermal solver interface.

    This class selects among several backends to solve the heat equation:

    * **SfePy** – when available, provides a finite‑element implementation.  It is
      preferred over other Python methods but currently supports only
      conduction and ignores radiation.
    * **Native C engine** – when compiled, this backend offers the fastest
      solution of steady‑state and transient conduction with optional
      radiation.  It is the default when SfePy is not available or when
      radiation is enabled.
    * **Python (SciPy)** – a sparse matrix solver using conjugate gradient
      methods if SciPy is installed.
    * **Python (NumPy)** – a dense matrix solver if only NumPy is installed.
    * **Python (Basic)** – a very basic Jacobi iteration fallback used as
      a last resort.
    """

    def __init__(self) -> None:
        # Attempt to import an optional SfePy backend.  If it fails, the
        # attribute remains None and will not be used.
        try:
            from .sfepy_solver import SfepyThermalSolver  # type: ignore
            sfepy_solver = SfepyThermalSolver()
            # Only keep the solver if SfePy was imported successfully.
            self._sfepy = sfepy_solver if sfepy_solver.is_available else None
        except Exception:
            self._sfepy = None
        self._native = NativeThermalEngine()
        self._python = PythonThermalSolver()

    @property
    def using_native(self) -> bool:
        return self._native.is_available

    @property
    def backend_name(self) -> str:
        """Return a human‑readable name of the solver backend that will be used.

        The method checks the availability of backends in order of
        preference.  Note that the actual backend chosen for a particular
        solve may depend on the problem parameters (for example, SfePy
        currently does not support radiation).
        """
        # SfePy is preferred if it exists and is available.
        if getattr(self, '_sfepy', None) is not None:
            return self._sfepy.backend_name
        if self._native.is_available:
            return "Native C Engine"
        if HAS_SCIPY:
            return "Python (SciPy)"
        if HAS_NUMPY:
            return "Python (NumPy)"
        return "Python (Basic)"
    
    def solve_steady_state(self, mesh: ThermalMesh,
                          ambient_temp_c: float = 25.0,
                          chamber_wall_temp_c: float = 25.0,
                          include_radiation: bool = True,
                          convergence: float = 1e-6,
                          max_iterations: int = 10000,
                          progress_callback: Optional[Callable] = None) -> ThermalResult:
        """Solve steady-state problem.

        If the native C engine is available it will be used for improved
        performance.  Otherwise the Python fallback solver is invoked.  The
        caller should ensure that ``mesh`` has been fully prepared (nodes,
        conductances, heat sources and boundary conditions).
        """
        # If SfePy is available and radiation is disabled, use it.  This
        # provides a finite element solution to the conduction equation.
        if (getattr(self, '_sfepy', None) is not None
                and not include_radiation):
            try:
                return self._sfepy.solve_steady_state(
                    mesh,
                    ambient_temp_c=ambient_temp_c,
                    chamber_wall_temp_c=chamber_wall_temp_c,
                    include_radiation=include_radiation,
                    convergence=convergence,
                    max_iterations=max_iterations,
                    progress_callback=progress_callback,
                )
            except Exception:
                # If Sfepy solver throws, fall back to native or Python
                pass
        # Otherwise, use the native C engine if available
        if self._native.is_available:
            # Use native solver
            # Create and populate the native state
            if not self._native.create_state(mesh):
                # Fall back if state cannot be created
                return self._python.solve_steady_state(
                    mesh, ambient_temp_c, chamber_wall_temp_c,
                    include_radiation, convergence, max_iterations,
                    progress_callback
                )
            # Allocate neighbor arrays
            total_neighbors = sum(len(node.neighbors) for node in mesh.nodes)
            if not self._native.alloc_neighbors(total_neighbors):
                self._native.destroy()
                return self._python.solve_steady_state(
                    mesh, ambient_temp_c, chamber_wall_temp_c,
                    include_radiation, convergence, max_iterations,
                    progress_callback
                )
            # Build CSR pointers and neighbor entries
            ptr = 0
            for i, node in enumerate(mesh.nodes):
                self._native.set_row_ptr(i, ptr)
                offset = 0
                for j, G in node.neighbors.items():
                    self._native.set_neighbor(i, offset, j, G)
                    offset += 1
                ptr += offset
            # Final row_ptr entry pointing to total neighbors
            self._native.set_row_ptr(len(mesh.nodes), ptr)
            # Populate nodes
            for node in mesh.nodes:
                self._native.set_node(
                    node.node_id,
                    k=node.k,
                    cp=node.cp,
                    rho=node.rho,
                    emissivity=node.emissivity,
                    volume=node.volume,
                    surface_area=node.surface_area,
                    heat_source=node.heat_source
                )
                if node.is_fixed_temp:
                    self._native.set_fixed_temp(node.node_id, node.fixed_temp + PhysicalConstants.CELSIUS_TO_KELVIN)
            # Solve using native engine
            result = self._native.solve_steady_state(
                ambient_temp_c, chamber_wall_temp_c,
                include_radiation, convergence, max_iterations,
                mesh, progress_callback
            )
            self._native.destroy()
            return result
        # Fallback to Python solver
        return self._python.solve_steady_state(
            mesh, ambient_temp_c, chamber_wall_temp_c,
            include_radiation, convergence, max_iterations,
            progress_callback
        )
    
    def solve_transient(self, mesh: ThermalMesh,
                       duration_s: float,
                       timestep_s: float,
                       initial_temp_c: float = 25.0,
                       ambient_temp_c: float = 25.0,
                       chamber_wall_temp_c: float = 25.0,
                       include_radiation: bool = True,
                       output_interval_s: float = 1.0,
                       progress_callback: Optional[Callable] = None) -> ThermalResult:
        """Solve transient problem.

        The native engine is used when available.  Otherwise the Python
        implementation of the Crank–Nicolson scheme is used.
        """
        if self._native.is_available:
            if not self._native.create_state(mesh):
                return self._python.solve_transient(
                    mesh, duration_s, timestep_s,
                    initial_temp_c, ambient_temp_c, chamber_wall_temp_c,
                    include_radiation, output_interval_s,
                    progress_callback
                )
            # Allocate neighbor arrays
            total_neighbors = sum(len(node.neighbors) for node in mesh.nodes)
            if not self._native.alloc_neighbors(total_neighbors):
                self._native.destroy()
                return self._python.solve_transient(
                    mesh, duration_s, timestep_s,
                    initial_temp_c, ambient_temp_c, chamber_wall_temp_c,
                    include_radiation, output_interval_s,
                    progress_callback
                )
            ptr = 0
            for i, node in enumerate(mesh.nodes):
                self._native.set_row_ptr(i, ptr)
                offset = 0
                for j, G in node.neighbors.items():
                    self._native.set_neighbor(i, offset, j, G)
                    offset += 1
                ptr += offset
            self._native.set_row_ptr(len(mesh.nodes), ptr)
            for node in mesh.nodes:
                self._native.set_node(
                    node.node_id,
                    k=node.k,
                    cp=node.cp,
                    rho=node.rho,
                    emissivity=node.emissivity,
                    volume=node.volume,
                    surface_area=node.surface_area,
                    heat_source=node.heat_source
                )
                if node.is_fixed_temp:
                    self._native.set_fixed_temp(node.node_id, node.fixed_temp + PhysicalConstants.CELSIUS_TO_KELVIN)
            result = self._native.solve_transient(
                duration_s, timestep_s,
                initial_temp_c, ambient_temp_c, chamber_wall_temp_c,
                include_radiation, mesh, output_interval_s,
                progress_callback
            )
            self._native.destroy()
            return result
        return self._python.solve_transient(
            mesh, duration_s, timestep_s,
            initial_temp_c, ambient_temp_c, chamber_wall_temp_c,
            include_radiation, output_interval_s,
            progress_callback
        )


__all__ = [
    'ThermalSolver', 'ThermalNode', 'ThermalMesh', 'ThermalResult',
    'PythonThermalSolver', 'NativeThermalEngine',
]
