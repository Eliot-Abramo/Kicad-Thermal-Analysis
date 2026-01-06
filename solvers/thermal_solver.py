"""
TVAC Thermal Analyzer - Thermal Solver
======================================
Finite difference thermal solver for PCB analysis in vacuum conditions.

Supports:
- 3D and 2D thermal analysis
- Transient and steady-state simulation
- Conduction and radiation heat transfer
- Multiple boundary condition types

Author: Space Electronics Thermal Analysis Tool
Version: 1.0.0
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import time
import threading
from enum import Enum

from .mesh_generator import ThermalMesh, MeshNode, NodeType
from ..core.config import SimulationParameters, MountingPoint, HeatsinkConfig
from ..core.constants import PhysicalConstants, MaterialsDatabase
from ..utils.logger import get_logger, timed_function, log_section


class SolverState(Enum):
    """Solver execution state."""
    IDLE = 0
    RUNNING = 1
    PAUSED = 2
    COMPLETED = 3
    FAILED = 4
    CANCELLED = 5


@dataclass
class ThermalFrame:
    """Single time frame of thermal results."""
    time_s: float
    temperature: np.ndarray  # Temperature at each node
    min_temp: float
    max_temp: float
    avg_temp: float
    heat_flux: Optional[np.ndarray] = None  # Optional heat flux data


@dataclass
class ThermalResults:
    """Complete thermal simulation results."""
    # Time series data
    frames: List[ThermalFrame] = field(default_factory=list)
    time_points: List[float] = field(default_factory=list)
    
    # Final state
    final_temperature: Optional[np.ndarray] = None
    steady_state_reached: bool = False
    steady_state_time: float = 0.0
    
    # Statistics over time
    min_temp_history: List[float] = field(default_factory=list)
    max_temp_history: List[float] = field(default_factory=list)
    avg_temp_history: List[float] = field(default_factory=list)
    
    # Per-component results
    component_temps: Dict[str, List[float]] = field(default_factory=dict)  # ref -> temp history
    component_max_temps: Dict[str, float] = field(default_factory=dict)
    
    # Simulation info
    total_simulation_time: float = 0.0
    total_compute_time: float = 0.0
    iterations: int = 0
    convergence_history: List[float] = field(default_factory=list)
    
    # Warnings and errors
    warnings: List[str] = field(default_factory=list)
    
    def get_frame_at_time(self, t: float) -> Optional[ThermalFrame]:
        """Get frame nearest to specified time."""
        if not self.frames:
            return None
        
        # Binary search for nearest time
        idx = np.searchsorted(self.time_points, t)
        if idx == 0:
            return self.frames[0]
        if idx >= len(self.frames):
            return self.frames[-1]
        
        # Return closer frame
        if abs(self.time_points[idx] - t) < abs(self.time_points[idx-1] - t):
            return self.frames[idx]
        return self.frames[idx-1]
    
    def get_temperature_at_node(self, node_id: int) -> List[Tuple[float, float]]:
        """Get temperature history for a specific node."""
        return [(f.time_s, f.temperature[node_id]) for f in self.frames 
                if node_id < len(f.temperature)]


class ThermalSolver:
    """
    Finite difference thermal solver for PCB thermal analysis.
    
    Implements:
    - Implicit Crank-Nicolson scheme for stability
    - Sparse matrix operations for efficiency
    - Radiation heat transfer (Stefan-Boltzmann)
    - Adaptive time stepping
    """
    
    def __init__(self, mesh: ThermalMesh, params: SimulationParameters):
        """Initialize thermal solver."""
        self.mesh = mesh
        self.params = params
        self.logger = get_logger()
        
        # Solver state
        self.state = SolverState.IDLE
        self.progress = 0.0
        self.current_time = 0.0
        
        # Results
        self.results = ThermalResults()
        
        # Matrices (built during solve)
        self.K_matrix: Optional[sparse.csr_matrix] = None  # Conductivity matrix
        self.C_matrix: Optional[sparse.csr_matrix] = None  # Capacitance matrix
        self.Q_vector: Optional[np.ndarray] = None  # Heat source vector
        
        # Temperature state
        self.T: Optional[np.ndarray] = None  # Current temperature
        self.T_prev: Optional[np.ndarray] = None  # Previous temperature
        
        # Callbacks
        self.progress_callback: Optional[Callable[[float, str], None]] = None
        self.frame_callback: Optional[Callable[[ThermalFrame], None]] = None
        
        # Thread control
        self._cancel_flag = threading.Event()
        self._pause_flag = threading.Event()
    
    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set callback for progress updates."""
        self.progress_callback = callback
    
    def set_frame_callback(self, callback: Callable[[ThermalFrame], None]):
        """Set callback for new frame availability."""
        self.frame_callback = callback
    
    def cancel(self):
        """Cancel running simulation."""
        self._cancel_flag.set()
        self.state = SolverState.CANCELLED
    
    def pause(self):
        """Pause running simulation."""
        self._pause_flag.set()
        self.state = SolverState.PAUSED
    
    def resume(self):
        """Resume paused simulation."""
        self._pause_flag.clear()
        self.state = SolverState.RUNNING
    
    def _update_progress(self, progress: float, message: str = ""):
        """Update progress and call callback."""
        self.progress = progress
        if self.progress_callback:
            self.progress_callback(progress, message)
    
    @timed_function("thermal_matrix_build")
    def _build_matrices(self):
        """Build system matrices for thermal analysis."""
        n = len(self.mesh.nodes)
        self.logger.info(f"Building thermal matrices for {n} nodes")
        
        # Initialize arrays for sparse matrix construction
        K_rows, K_cols, K_data = [], [], []
        C_diag = np.zeros(n, dtype=np.float64)
        
        for node in self.mesh.nodes:
            i = node.node_id
            
            # Diagonal capacitance: ρ * c * V
            C_diag[i] = node.density * node.specific_heat * node.volume_m3
            
            # Self-conductance (sum of conductances to neighbors)
            self_conductance = 0.0
            
            for j, (neighbor_idx, conductance) in enumerate(zip(node.neighbors, node.conductances)):
                if conductance > 0:
                    # Off-diagonal: -conductance
                    K_rows.append(i)
                    K_cols.append(neighbor_idx)
                    K_data.append(-conductance)
                    
                    self_conductance += conductance
            
            # Diagonal: sum of conductances
            K_rows.append(i)
            K_cols.append(i)
            K_data.append(self_conductance)
        
        # Build sparse matrices
        self.K_matrix = sparse.csr_matrix(
            (K_data, (K_rows, K_cols)),
            shape=(n, n),
            dtype=np.float64
        )
        
        self.C_matrix = sparse.diags(C_diag, format='csr', dtype=np.float64)
        
        # Heat source vector
        self.Q_vector = np.array([node.heat_generation_w for node in self.mesh.nodes], 
                                 dtype=np.float64)
        
        self.logger.debug(f"K matrix: {self.K_matrix.nnz} non-zeros")
        self.logger.debug(f"Total heat generation: {np.sum(self.Q_vector):.3f} W")
    
    def _apply_boundary_conditions(self, A: sparse.csr_matrix, b: np.ndarray) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """Apply boundary conditions to system matrix and RHS."""
        A_mod = A.tolil()
        
        for node in self.mesh.nodes:
            if node.is_fixed_temp:
                i = node.node_id
                # Zero out row except diagonal
                A_mod[i, :] = 0
                A_mod[i, i] = 1.0
                b[i] = node.fixed_temp_c
        
        return A_mod.tocsr(), b
    
    def _compute_radiation_heat_transfer(self, T: np.ndarray) -> np.ndarray:
        """
        Compute radiation heat transfer to/from chamber walls.
        
        Q_rad = ε * σ * A * (T⁴ - T_wall⁴)
        """
        if not self.params.include_radiation:
            return np.zeros_like(T)
        
        sigma = PhysicalConstants.STEFAN_BOLTZMANN
        T_wall = self.params.chamber_wall_temp_c + PhysicalConstants.CELSIUS_TO_KELVIN
        
        Q_rad = np.zeros_like(T)
        
        for node in self.mesh.nodes:
            if node.area_m2 > 0 and node.emissivity > 0:
                # Only surface nodes radiate (top and bottom layers)
                if node.layer_index == 0 or node.layer_index == self.mesh.nz - 1:
                    T_node_k = T[node.node_id] + PhysicalConstants.CELSIUS_TO_KELVIN
                    
                    # Radiation to chamber walls (negative = heat loss)
                    Q_rad[node.node_id] = -node.emissivity * sigma * node.area_m2 * (
                        T_node_k**4 - T_wall**4
                    )
        
        return Q_rad
    
    @timed_function("thermal_solve_transient")
    def solve_transient(self) -> ThermalResults:
        """
        Solve transient thermal problem.
        
        Uses Crank-Nicolson implicit scheme:
        [C/dt + θK] * T^{n+1} = [C/dt - (1-θ)K] * T^n + Q
        
        where θ = 0.5 for Crank-Nicolson (unconditionally stable)
        """
        self.state = SolverState.RUNNING
        self._cancel_flag.clear()
        self._pause_flag.clear()
        
        start_time = time.time()
        
        with log_section("Transient Thermal Simulation"):
            # Build matrices
            self._build_matrices()
            
            n = len(self.mesh.nodes)
            dt = self.params.timestep_s
            duration = self.params.duration_s
            output_interval = self.params.output_interval_s
            
            # Initialize temperature
            self.T = np.full(n, self.params.initial_board_temp_c, dtype=np.float64)
            
            # Apply initial fixed temperatures
            for node in self.mesh.nodes:
                if node.is_fixed_temp:
                    self.T[node.node_id] = node.fixed_temp_c
            
            # Crank-Nicolson parameter
            theta = 0.5
            
            # Build LHS matrix: C/dt + θK
            C_dt = self.C_matrix / dt
            LHS = C_dt + theta * self.K_matrix
            
            # Pre-factorize for efficiency (if using direct solver)
            try:
                LHS_factored = sparse_linalg.splu(LHS.tocsc())
                use_direct = True
            except:
                use_direct = False
                self.logger.warning("Direct factorization failed, using iterative solver")
            
            # Time stepping
            t = 0.0
            step = 0
            last_output_time = -output_interval
            
            self.results = ThermalResults()
            
            # Store initial state
            self._store_frame(t)
            
            total_steps = int(duration / dt)
            
            self.logger.info(f"Starting transient simulation: {duration}s, dt={dt}s, {total_steps} steps")
            
            while t < duration:
                # Check for cancellation
                if self._cancel_flag.is_set():
                    self.logger.info("Simulation cancelled")
                    break
                
                # Check for pause
                while self._pause_flag.is_set():
                    time.sleep(0.1)
                    if self._cancel_flag.is_set():
                        break
                
                # Compute radiation contribution
                Q_rad = self._compute_radiation_heat_transfer(self.T)
                
                # Build RHS: [C/dt - (1-θ)K] * T^n + Q + Q_rad
                RHS = C_dt.dot(self.T) - (1 - theta) * self.K_matrix.dot(self.T) + self.Q_vector + Q_rad
                
                # Apply boundary conditions
                LHS_bc, RHS_bc = self._apply_boundary_conditions(LHS, RHS)
                
                # Solve
                if use_direct:
                    # Need to re-apply BCs to factored matrix for fixed temp nodes
                    # For simplicity, use iterative for now
                    self.T, info = sparse_linalg.cg(LHS_bc, RHS_bc, x0=self.T, 
                                                    rtol=self.params.convergence_criterion,
                                                    maxiter=100)
                else:
                    self.T, info = sparse_linalg.cg(LHS_bc, RHS_bc, x0=self.T,
                                                    rtol=self.params.convergence_criterion,
                                                    maxiter=100)
                
                t += dt
                step += 1
                
                # Store frame at output intervals
                if t - last_output_time >= output_interval:
                    self._store_frame(t)
                    last_output_time = t
                    
                    # Check for steady state
                    if self._check_steady_state():
                        self.results.steady_state_reached = True
                        self.results.steady_state_time = t
                        self.logger.info(f"Steady state reached at t={t:.1f}s")
                
                # Update progress
                progress = min(1.0, t / duration)
                self._update_progress(progress, f"t={t:.1f}s, Tmax={np.max(self.T):.1f}°C")
                
                # Log periodically
                if step % 100 == 0:
                    self.logger.debug(f"Step {step}: t={t:.2f}s, Tmin={np.min(self.T):.2f}°C, "
                                     f"Tmax={np.max(self.T):.2f}°C")
            
            # Store final state
            self.results.final_temperature = self.T.copy()
            self.results.iterations = step
            self.results.total_simulation_time = duration
            self.results.total_compute_time = time.time() - start_time
            
            # Extract component temperatures
            self._extract_component_temperatures()
            
            self.state = SolverState.COMPLETED
            self.logger.info(f"Transient simulation complete: {step} steps in {self.results.total_compute_time:.1f}s")
        
        return self.results
    
    @timed_function("thermal_solve_steady")
    def solve_steady_state(self) -> ThermalResults:
        """
        Solve steady-state thermal problem.
        
        Solves K * T = Q for equilibrium temperature distribution.
        """
        self.state = SolverState.RUNNING
        start_time = time.time()
        
        with log_section("Steady-State Thermal Simulation"):
            # Build matrices
            self._build_matrices()
            
            n = len(self.mesh.nodes)
            
            # Initialize temperature
            self.T = np.full(n, self.params.initial_board_temp_c, dtype=np.float64)
            
            # For steady state with radiation, we need iterative approach
            if self.params.include_radiation:
                self.T = self._solve_steady_with_radiation()
            else:
                # Direct solve: K * T = Q
                K_bc, Q_bc = self._apply_boundary_conditions(self.K_matrix, self.Q_vector.copy())
                
                try:
                    self.T = sparse_linalg.spsolve(K_bc.tocsc(), Q_bc)
                except Exception as e:
                    self.logger.error(f"Steady-state solve failed: {e}")
                    # Try iterative
                    self.T, _ = sparse_linalg.cg(K_bc, Q_bc, x0=self.T, maxiter=1000)
            
            # Store result
            self.results = ThermalResults()
            self._store_frame(0.0)
            self.results.final_temperature = self.T.copy()
            self.results.steady_state_reached = True
            self.results.total_compute_time = time.time() - start_time
            
            self._extract_component_temperatures()
            
            self.state = SolverState.COMPLETED
            self.logger.info(f"Steady-state simulation complete in {self.results.total_compute_time:.1f}s")
        
        return self.results
    
    def _solve_steady_with_radiation(self) -> np.ndarray:
        """Iterative solution for steady-state with radiation."""
        max_iter = self.params.max_iterations
        tol = self.params.steady_state_tolerance
        
        T = self.T.copy()
        
        for iteration in range(max_iter):
            T_old = T.copy()
            
            # Compute radiation
            Q_rad = self._compute_radiation_heat_transfer(T)
            Q_total = self.Q_vector + Q_rad
            
            # Solve linear system
            K_bc, Q_bc = self._apply_boundary_conditions(self.K_matrix, Q_total.copy())
            T, _ = sparse_linalg.cg(K_bc, Q_bc, x0=T, rtol=1e-6, maxiter=100)
            
            # Check convergence
            residual = np.max(np.abs(T - T_old))
            self.results.convergence_history.append(residual)
            
            if residual < tol:
                self.logger.info(f"Steady-state converged in {iteration+1} iterations")
                break
            
            if iteration % 10 == 0:
                self.logger.debug(f"Iteration {iteration}: residual = {residual:.2e}")
        
        return T
    
    def _store_frame(self, t: float):
        """Store current temperature state as a frame."""
        frame = ThermalFrame(
            time_s=t,
            temperature=self.T.copy(),
            min_temp=float(np.min(self.T)),
            max_temp=float(np.max(self.T)),
            avg_temp=float(np.mean(self.T))
        )
        
        self.results.frames.append(frame)
        self.results.time_points.append(t)
        self.results.min_temp_history.append(frame.min_temp)
        self.results.max_temp_history.append(frame.max_temp)
        self.results.avg_temp_history.append(frame.avg_temp)
        
        # Notify callback
        if self.frame_callback:
            self.frame_callback(frame)
    
    def _check_steady_state(self) -> bool:
        """Check if system has reached steady state."""
        if len(self.results.max_temp_history) < 10:
            return False
        
        # Check rate of change of max temperature
        recent_temps = self.results.max_temp_history[-10:]
        rate = abs(recent_temps[-1] - recent_temps[0]) / (10 * self.params.output_interval_s)
        
        return rate < self.params.steady_state_tolerance
    
    def _extract_component_temperatures(self):
        """Extract temperature at each component location."""
        for node in self.mesh.nodes:
            if node.component_ref and node.node_type == NodeType.COMPONENT:
                ref = node.component_ref
                
                if ref not in self.results.component_temps:
                    self.results.component_temps[ref] = []
                
                # Get temperature from all frames
                for frame in self.results.frames:
                    if node.node_id < len(frame.temperature):
                        if len(self.results.component_temps[ref]) < len(self.results.frames):
                            self.results.component_temps[ref].append(frame.temperature[node.node_id])
                
                # Store max temp
                if self.results.final_temperature is not None:
                    self.results.component_max_temps[ref] = float(
                        self.results.final_temperature[node.node_id]
                    )
    
    def get_temperature_at_point(self, x: float, y: float, z: float = None, 
                                 time_s: float = None) -> Optional[float]:
        """Get temperature at a specific point and time."""
        node = self.mesh.get_node_at_position(x, y, z if z else 0)
        if node is None:
            return None
        
        if time_s is None:
            # Return final temperature
            if self.results.final_temperature is not None:
                return float(self.results.final_temperature[node.node_id])
            return None
        
        frame = self.results.get_frame_at_time(time_s)
        if frame and node.node_id < len(frame.temperature):
            return float(frame.temperature[node.node_id])
        
        return None
    
    def get_temperature_grid(self, layer: int = 0, time_s: float = None) -> Optional[np.ndarray]:
        """Get 2D temperature grid for a specific layer."""
        if time_s is None:
            T = self.results.final_temperature
        else:
            frame = self.results.get_frame_at_time(time_s)
            T = frame.temperature if frame else None
        
        if T is None:
            return None
        
        # Reshape to 2D grid
        grid = np.zeros((self.mesh.ny, self.mesh.nx), dtype=np.float64)
        
        for ix in range(self.mesh.nx):
            for iy in range(self.mesh.ny):
                idx = self.mesh.node_index_map.get((ix, iy, layer))
                if idx is not None and idx < len(T):
                    grid[iy, ix] = T[idx]
        
        return grid


def run_simulation(mesh: ThermalMesh, params: SimulationParameters,
                  progress_callback: Callable = None) -> ThermalResults:
    """
    Convenience function to run thermal simulation.
    
    Args:
        mesh: Generated thermal mesh
        params: Simulation parameters
        progress_callback: Optional progress callback
    
    Returns:
        ThermalResults with complete simulation data
    """
    solver = ThermalSolver(mesh, params)
    
    if progress_callback:
        solver.set_progress_callback(progress_callback)
    
    if params.simulation_mode == "steady_state":
        return solver.solve_steady_state()
    else:
        return solver.solve_transient()


class ThermalAnalysisEngine:
    """
    High-level thermal analysis engine that orchestrates the complete analysis workflow.
    
    This class coordinates:
    1. PCB geometry extraction
    2. Mesh generation
    3. Current distribution solving
    4. Thermal simulation
    5. Results aggregation
    """
    
    def __init__(self):
        """Initialize the analysis engine."""
        self.logger = get_logger()
        self.mesh: Optional[ThermalMesh] = None
        self.thermal_result: Optional[ThermalResults] = None
        self.current_result = None
        self.progress_callback: Optional[Callable[[float, str], None]] = None
    
    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set progress callback function."""
        self.progress_callback = callback
    
    def _report_progress(self, progress: float, message: str):
        """Report progress via callback."""
        if self.progress_callback:
            self.progress_callback(progress, message)
    
    def run_analysis(self, pcb_data, config) -> ThermalResults:
        """
        Run complete thermal analysis.
        
        Args:
            pcb_data: Extracted PCB geometry data
            config: ThermalAnalysisConfig with all settings
        
        Returns:
            ThermalResults with simulation data
        """
        from .mesh_generator import MeshGenerator
        from .current_solver import CurrentDistributionSolver
        
        self.logger.info("Starting thermal analysis engine")
        start_time = time.time()
        
        try:
            # Step 1: Generate mesh
            self._report_progress(0.05, "Generating thermal mesh...")
            mesh_gen = MeshGenerator(
                pcb_data,
                resolution_mm=config.simulation.resolution_mm,
                use_adaptive=config.simulation.use_adaptive_mesh,
                adaptive_factor=config.simulation.adaptive_refinement_factor
            )
            self.mesh = mesh_gen.generate(config.stackup, config.simulation.simulation_3d)
            self.logger.info(f"Mesh generated: {len(self.mesh.nodes)} nodes")
            
            # Step 2: Current distribution (if injection points defined)
            if config.current_injection_points:
                self._report_progress(0.15, "Solving current distribution...")
                
                copper_thickness_m = {
                    layer: um / 1e6 for layer, um in config.stackup.copper_thickness_um.items()
                }
                
                current_solver = CurrentDistributionSolver(pcb_data)
                current_solver.build_network(
                    copper_thickness_m,
                    include_ac_effects=config.simulation.include_ac_effects,
                    frequency_hz=config.simulation.ac_frequency_hz
                )
                
                self.current_result = current_solver.solve(
                    config.current_injection_points,
                    config.trace_current_overrides
                )
                
                # Apply Joule heating to mesh
                self._report_progress(0.25, "Applying heat sources...")
                mesh_gen.apply_heat_sources(
                    self.current_result.trace_power,
                    {c.reference: c.power_w for c in config.component_power}
                )
            else:
                # Just component power
                mesh_gen.apply_heat_sources(
                    {},
                    {c.reference: c.power_w for c in config.component_power}
                )
            
            # Step 3: Apply boundary conditions
            self._report_progress(0.30, "Applying boundary conditions...")
            mesh_gen.apply_boundary_conditions(
                config.mounting_points,
                {'ambient': config.simulation.ambient_temp_c}
            )
            
            # Step 4: Run thermal solver
            self._report_progress(0.35, "Starting thermal solver...")
            
            solver = ThermalSolver(self.mesh, config.simulation)
            solver.set_progress_callback(
                lambda p, m: self._report_progress(0.35 + 0.6 * p, m)
            )
            
            if config.simulation.simulation_mode == "steady_state":
                self.thermal_result = solver.solve_steady_state()
            else:
                self.thermal_result = solver.solve_transient()
            
            # Add any current solver warnings
            if self.current_result and hasattr(self.current_result, 'warnings'):
                self.thermal_result.warnings.extend(self.current_result.warnings)
            
            elapsed = time.time() - start_time
            self._report_progress(1.0, f"Analysis complete ({elapsed:.1f}s)")
            self.logger.info(f"Thermal analysis completed in {elapsed:.1f}s")
            
            return self.thermal_result
            
        except Exception as e:
            self.logger.exception(f"Analysis failed: {e}")
            raise


# Alias for backwards compatibility
ThermalSimulationResult = ThermalResults

__all__ = [
    'SolverState',
    'ThermalFrame',
    'ThermalResults',
    'ThermalSimulationResult',
    'ThermalSolver',
    'ThermalAnalysisEngine',
    'run_simulation',
]
