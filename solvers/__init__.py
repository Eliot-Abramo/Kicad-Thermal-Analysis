"""
TVAC Thermal Analyzer - Solvers Module
======================================
Thermal and electrical solvers for PCB analysis.
"""

from .mesh_generator import (
    NodeType,
    MeshNode,
    MeshElement,
    ThermalMesh,
    MeshGenerator,
)

from .current_solver import (
    ElectricalNode,
    ElectricalSegment,
    CurrentDistributionResult,
    CurrentDistributionSolver,
)

from .thermal_solver import (
    SolverState,
    ThermalFrame,
    ThermalResults,
    ThermalSimulationResult,
    ThermalSolver,
    ThermalAnalysisEngine,
    run_simulation,
)

__all__ = [
    # Mesh
    'NodeType',
    'MeshNode',
    'MeshElement',
    'ThermalMesh',
    'MeshGenerator',
    # Current
    'ElectricalNode',
    'ElectricalSegment',
    'CurrentDistributionResult',
    'CurrentDistributionSolver',
    # Thermal
    'SolverState',
    'ThermalFrame',
    'ThermalResults',
    'ThermalSimulationResult',
    'ThermalSolver',
    'ThermalAnalysisEngine',
    'run_simulation',
]
