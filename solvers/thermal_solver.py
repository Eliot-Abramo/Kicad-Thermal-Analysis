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

    # Robin (conductance) boundary condition: q = bc_g*(T - bc_temp)
    # bc_g in W/K, bc_temp in °C (or K offset cancels for deltas)
    bc_g: float = 0.0
    bc_temp: float = 25.0
    
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
    warnings: List[str] = field(default_factory=list)
    
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
        """Setup ctypes function signatures."""
        if not self._lib:
            return
        
        # thermal_state_create
        self._lib.thermal_state_create.argtypes = [ctypes.c_int]
        self._lib.thermal_state_create.restype = ctypes.c_void_p
        
        # thermal_state_destroy
        self._lib.thermal_state_destroy.argtypes = [ctypes.c_void_p]
        self._lib.thermal_state_destroy.restype = None
        
        # thermal_set_node
        self._lib.thermal_set_node.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double
        ]
        self._lib.thermal_set_node.restype = None
        
        # thermal_solve_steady_state
        self._lib.thermal_solve_steady_state.argtypes = [
            ctypes.c_void_p, ctypes.c_double, ctypes.c_int
        ]
        self._lib.thermal_solve_steady_state.restype = ctypes.c_int
        
        # thermal_solve_transient
        self._lib.thermal_solve_transient.argtypes = [
            ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double
        ]
        self._lib.thermal_solve_transient.restype = ctypes.c_int
        
        # thermal_get_temperature
        self._lib.thermal_get_temperature.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.thermal_get_temperature.restype = ctypes.c_double
    
    @property
    def is_available(self) -> bool:
        return self._lib is not None
    
    def create_state(self, num_nodes: int) -> bool:
        if not self._lib:
            return False
        self._state = self._lib.thermal_state_create(num_nodes)
        return self._state is not None
    
    def destroy(self):
        if self._lib and self._state:
            self._lib.thermal_state_destroy(self._state)
            self._state = None
    
    def set_node(self, idx: int, k: float, cp: float, rho: float,
                 volume: float, heat_source: float, emissivity: float):
        if self._lib and self._state:
            self._lib.thermal_set_node(
                self._state, idx, k, cp, rho, volume, heat_source, emissivity
            )


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
        """Solve steady-state thermal problem.

        Numerical model (per node i):
            sum_j G_ij (T_i - T_j) + q_rad_i(T) + q_bc_i(T) = Q_i

        - Conduction is assembled into a sparse Laplacian-like matrix (SPD when anchored).
        - Radiation is *linearized* into an equivalent conductance h_rad(Tm):
              q_rad ≈ h_rad * (T - T_wall)
          where h_rad = 4 * eps * sigma * A * Tm^3, evaluated using a mean temperature.
        - Robin BC (mounting to box) is added as:
              q_bc = bc_g * (T - bc_temp)

        Fixed-temperature nodes are eliminated from the free-node system so we keep a
        symmetric positive definite (SPD) matrix for CG stability.
        """

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

        # Initial guess
        T = np.full(n, float(ambient_temp_c), dtype=np.float64)
        for i, node in enumerate(mesh.nodes):
            if getattr(node, 'is_fixed_temp', False):
                T[i] = float(getattr(node, 'fixed_temp', ambient_temp_c))

        fixed = np.array([bool(getattr(nd, 'is_fixed_temp', False)) for nd in mesh.nodes], dtype=bool)
        free_ids = np.where(~fixed)[0]

        # Detect if system is anchored (otherwise conduction-only is singular)
        has_dirichlet = bool(np.any(fixed))
        has_robin = any((float(getattr(nd, 'bc_g', 0.0) or 0.0) > 0.0) for nd in mesh.nodes)
        has_rad = False
        if include_radiation:
            for nd in mesh.nodes:
                if (not getattr(nd, 'is_fixed_temp', False)) and (float(getattr(nd, 'surface_area', 0.0) or 0.0) > 0.0) and (float(getattr(nd, 'emissivity', 0.0) or 0.0) > 0.0):
                    has_rad = True
                    break

        if not (has_dirichlet or has_robin or has_rad):
            result.error_message = "Unanchored thermal system (no fixed temp, no Robin BC, no radiation sink)."
            result.converged = False
            return result

        if len(free_ids) == 0:
            # All nodes fixed
            result.temperatures = T.tolist()
            result.min_temp = float(np.min(T))
            result.max_temp = float(np.max(T))
            result.avg_temp = float(np.mean(T))
            result.iterations = 0
            result.compute_time = time.time() - start_time
            result.converged = True
            return result

        # Map global index -> free index
        map_free = -np.ones(n, dtype=np.int64)
        map_free[free_ids] = np.arange(len(free_ids), dtype=np.int64)

        Twall_C = float(chamber_wall_temp_c)
        Twall_K = Twall_C + self.kelvin_offset

        # Iteration on radiation linearization (and optional damping)
        damp = 0.7

        for iteration in range(max_iterations):
            if progress_callback and iteration % 50 == 0:
                progress_callback(10 + int(80 * iteration / max_iterations), f"Iteration {iteration}...")

            T_old = T.copy()

            rows = []
            cols = []
            data = []
            b = np.zeros(len(free_ids), dtype=np.float64)

            for gi in free_ids:
                li = int(map_free[gi])
                node = mesh.nodes[gi]

                rhs = float(getattr(node, 'heat_source', 0.0) or 0.0)
                diag = 0.0

                # Conduction neighbors
                for gj, G in (getattr(node, 'neighbors', {}) or {}).items():
                    if gj < 0 or gj >= n:
                        continue
                    G = float(G)
                    if G <= 0:
                        continue
                    if fixed[gj]:
                        rhs += G * float(getattr(mesh.nodes[gj], 'fixed_temp', Twall_C))
                        diag += G
                    else:
                        lj = int(map_free[gj])
                        rows.append(li)
                        cols.append(lj)
                        data.append(-G)
                        diag += G

                # Robin BC (mounting / box)
                g_bc = float(getattr(node, 'bc_g', 0.0) or 0.0)
                if g_bc > 0:
                    diag += g_bc
                    rhs += g_bc * float(getattr(node, 'bc_temp', Twall_C))

                # Radiation linearization as equivalent conductance
                if include_radiation:
                    A = float(getattr(node, 'surface_area', 0.0) or 0.0)
                    eps = float(getattr(node, 'emissivity', 0.0) or 0.0)
                    if A > 0 and eps > 0:
                        Tm_K = 0.5 * ((float(T_old[gi]) + self.kelvin_offset) + Twall_K)
                        if Tm_K < 1.0:
                            Tm_K = 1.0
                        h = 4.0 * eps * self.stefan_boltzmann * A * (Tm_K ** 3)
                        if h > 0:
                            diag += h
                            rhs += h * Twall_C

                rows.append(li)
                cols.append(li)
                data.append(diag if diag > 0 else 1e-12)
                b[li] = rhs

            # Solve A x = b for free nodes
            if HAS_SCIPY:
                A_mat = sparse.csr_matrix((data, (rows, cols)), shape=(len(free_ids), len(free_ids)))
                x0 = T_old[free_ids]
                try:
                    x, info = cg(A_mat, b, x0=x0, atol=convergence, rtol=convergence, maxiter=2000)
                except TypeError:
                    try:
                        x, info = cg(A_mat, b, x0=x0, tol=convergence, maxiter=2000)
                    except TypeError:
                        x, info = cg(A_mat, b, x0=x0, maxiter=2000)

                if info != 0:
                    try:
                        x = spsolve(A_mat, b)
                    except Exception:
                        result.warnings.append(f"Steady-state solver nonconvergence (info={info}).")
                        # Keep last iterate
                        x = x0
            else:
                # Dense fallback
                A_dense = np.zeros((len(free_ids), len(free_ids)), dtype=np.float64)
                for r, c, v in zip(rows, cols, data):
                    A_dense[r, c] += v
                x = self._jacobi_solve(A_dense, b, T_old[free_ids], convergence, 2000)

            # Update temperatures
            T[free_ids] = damp * x + (1.0 - damp) * T_old[free_ids]
            for i, nd in enumerate(mesh.nodes):
                if fixed[i]:
                    T[i] = float(getattr(nd, 'fixed_temp', ambient_temp_c))

            err = float(np.max(np.abs(T - T_old)))
            if err < convergence:
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
        """Solve transient thermal problem.

        Robust implicit Euler with linearized radiation and Robin BCs.
        This is slower than the native engine but stable in TVAC-style cases.
        """

        start_time = time.time()
        result = ThermalResult()

        n = len(mesh.nodes)
        if n == 0:
            result.error_message = "Empty mesh"
            result.converged = False
            return result
        if not HAS_NUMPY:
            result.error_message = "NumPy required"
            result.converged = False
            return result

        dt = float(timestep_s)
        if dt <= 0:
            result.error_message = "Invalid timestep"
            result.converged = False
            return result

        # Initial temps
        T = np.full(n, float(initial_temp_c), dtype=np.float64)
        fixed = np.array([bool(getattr(nd, 'is_fixed_temp', False)) for nd in mesh.nodes], dtype=bool)
        for i, nd in enumerate(mesh.nodes):
            if fixed[i]:
                T[i] = float(getattr(nd, 'fixed_temp', initial_temp_c))

        free_ids = np.where(~fixed)[0]
        map_free = -np.ones(n, dtype=np.int64)
        map_free[free_ids] = np.arange(len(free_ids), dtype=np.int64)

        # Thermal mass
        C = np.array([float(getattr(nd, 'rho', 0.0)) * float(getattr(nd, 'cp', 0.0)) * float(getattr(nd, 'volume', 0.0)) for nd in mesh.nodes], dtype=np.float64)
        C = np.maximum(C, 1e-12)

        Twall_C = float(chamber_wall_temp_c)
        Twall_K = Twall_C + self.kelvin_offset

        num_steps = int(max(0, round(float(duration_s) / dt)))
        output_step = max(1, int(round(float(output_interval_s) / dt)))

        result.time_points = [0.0]
        result.temp_history = [T.tolist()]

        t = 0.0
        for step in range(num_steps):
            if progress_callback and step % 10 == 0:
                progress_callback(5 + int(90 * step / max(1, num_steps)), f"Time: {t:.1f}s / {duration_s:.1f}s")

            T_old = T.copy()

            if len(free_ids) == 0:
                # Fully fixed
                t += dt
                if step % output_step == 0:
                    result.time_points.append(t)
                    result.temp_history.append(T.tolist())
                continue

            rows = []
            cols = []
            data = []
            b = np.zeros(len(free_ids), dtype=np.float64)

            for gi in free_ids:
                li = int(map_free[gi])
                node = mesh.nodes[gi]

                rhs = float(getattr(node, 'heat_source', 0.0) or 0.0)
                rhs += (C[gi] / dt) * float(T_old[gi])

                diag = (C[gi] / dt)

                # Conduction
                for gj, G in (getattr(node, 'neighbors', {}) or {}).items():
                    if gj < 0 or gj >= n:
                        continue
                    G = float(G)
                    if G <= 0:
                        continue
                    if fixed[gj]:
                        rhs += G * float(getattr(mesh.nodes[gj], 'fixed_temp', Twall_C))
                        diag += G
                    else:
                        lj = int(map_free[gj])
                        rows.append(li)
                        cols.append(lj)
                        data.append(-G)
                        diag += G

                # Robin BC
                g_bc = float(getattr(node, 'bc_g', 0.0) or 0.0)
                if g_bc > 0:
                    diag += g_bc
                    rhs += g_bc * float(getattr(node, 'bc_temp', Twall_C))

                # Radiation linearized
                if include_radiation:
                    A = float(getattr(node, 'surface_area', 0.0) or 0.0)
                    eps = float(getattr(node, 'emissivity', 0.0) or 0.0)
                    if A > 0 and eps > 0:
                        Tm_K = 0.5 * ((float(T_old[gi]) + self.kelvin_offset) + Twall_K)
                        if Tm_K < 1.0:
                            Tm_K = 1.0
                        h = 4.0 * eps * self.stefan_boltzmann * A * (Tm_K ** 3)
                        if h > 0:
                            diag += h
                            rhs += h * Twall_C

                rows.append(li)
                cols.append(li)
                data.append(diag if diag > 0 else 1e-12)
                b[li] = rhs

            # Solve
            if HAS_SCIPY:
                A_mat = sparse.csr_matrix((data, (rows, cols)), shape=(len(free_ids), len(free_ids)))
                x0 = T_old[free_ids]
                try:
                    x, info = cg(A_mat, b, x0=x0, atol=1e-8, rtol=1e-8, maxiter=4000)
                except TypeError:
                    try:
                        x, info = cg(A_mat, b, x0=x0, tol=1e-8, maxiter=4000)
                    except TypeError:
                        x, info = cg(A_mat, b, x0=x0, maxiter=4000)
                if info != 0:
                    x = spsolve(A_mat, b)
            else:
                A_dense = np.zeros((len(free_ids), len(free_ids)), dtype=np.float64)
                for r, c, v in zip(rows, cols, data):
                    A_dense[r, c] += v
                x = self._jacobi_solve(A_dense, b, T_old[free_ids], 1e-8, 2000)

            T[free_ids] = x
            for i, nd in enumerate(mesh.nodes):
                if fixed[i]:
                    T[i] = float(getattr(nd, 'fixed_temp', initial_temp_c))

            t += dt
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
    """Main thermal solver interface."""
    
    def __init__(self):
        self._native = NativeThermalEngine()
        self._python = PythonThermalSolver()
    
    @property
    def using_native(self) -> bool:
        return self._native.is_available
    
    @property
    def backend_name(self) -> str:
        if self._native.is_available:
            return "Native C Engine"
        elif HAS_SCIPY:
            return "Python (SciPy)"
        elif HAS_NUMPY:
            return "Python (NumPy)"
        else:
            return "Python (Basic)"
    
    def solve_steady_state(self, mesh: ThermalMesh,
                          ambient_temp_c: float = 25.0,
                          chamber_wall_temp_c: float = 25.0,
                          include_radiation: bool = True,
                          convergence: float = 1e-6,
                          max_iterations: int = 10000,
                          progress_callback: Optional[Callable] = None) -> ThermalResult:
        """Solve steady-state problem."""
        
        # For now, always use Python solver
        # Native engine integration would go here
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
        """Solve transient problem."""
        
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
