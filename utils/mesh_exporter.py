"""
Utility functions for exporting thermal meshes.

This module provides a simple helper to serialize the internal
``ThermalMesh`` representation into an external file.  The default
export format is JSON and captures all essential fields of each
``ThermalNode`` (coordinates, material properties, geometry and heat
source) along with the mesh dimensions and grid spacing.  Conductive
connections between nodes (the neighbour conductances) are recorded
as a list of pairs ``[neighbor_id, conductance]``.  This format is
designed to be both human‐readable and easily ingested by other
analysis tools or custom scripts.

The primary entry point is :func:`export_mesh` which writes a
``ThermalMesh`` to disk.  Callers should ensure that the mesh has
already been fully generated (e.g. via ``MeshGenerator.generate``)
prior to invoking this function.

Example usage within the KiCad plugin:

.. code-block:: python

    from tvac_thermal_analyzer.utils.mesh_exporter import export_mesh
    mesh = MeshGenerator(pcb_data, config).generate()
    export_path = os.path.splitext(board.GetFileName())[0] + "_tvac_mesh.json"
    export_mesh(mesh, export_path)

"""

from __future__ import annotations

import json
import os
from typing import Dict, Any, List, Tuple

from ..solvers.thermal_solver import ThermalMesh, ThermalNode

def _serialize_node(node: ThermalNode) -> Dict[str, Any]:
    """Serialize a single :class:`ThermalNode` to a JSON‐serialisable dict.

    Parameters
    ----------
    node : ThermalNode
        The node to serialize.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing all relevant fields of the node.  The
        ``neighbors`` field is converted to a list of ``[id, conductance]``
        pairs to ensure the keys remain numeric when encoded to JSON.
    """
    return {
        "id": node.node_id,
        "x_mm": node.x,
        "y_mm": node.y,
        "z_mm": node.z,
        "layer_idx": node.layer_idx,
        "k": node.k,
        "cp": node.cp,
        "rho": node.rho,
        "emissivity": node.emissivity,
        "volume_m3": node.volume,
        "surface_area_m2": node.surface_area,
        "heat_source_w": node.heat_source,
        "is_fixed_temp": node.is_fixed_temp,
        "fixed_temp_c": node.fixed_temp,
        "neighbors": [[int(n_id), float(g)] for n_id, g in node.neighbors.items()],
    }


def export_mesh(mesh: ThermalMesh, file_path: str, *, include_bounds: bool = True) -> None:
    """Export a complete :class:`ThermalMesh` to a JSON file.

    The resulting file contains the grid dimensions (``nx``, ``ny``, ``nz``),
    the grid spacing (``dx``, ``dy``, ``dz`` in metres) and a list of
    serialized nodes.  Optionally the board outline (in millimetres) is
    included when ``include_bounds`` is ``True``.

    Parameters
    ----------
    mesh : ThermalMesh
        The thermal mesh to export.  Must have its nodes populated by
        ``MeshGenerator.generate``.
    file_path : str
        Path to the output file.  Any necessary parent directories will
        be created automatically.  The file extension should typically
        be ``.json``.
    include_bounds : bool, optional
        When ``True`` the board bounding box is added to the JSON.  Set
        this to ``False`` to omit these fields.  Defaults to ``True``.
    """
    if mesh is None:
        raise ValueError("Mesh must not be None")
    # Ensure the directory exists
    out_dir = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(out_dir, exist_ok=True)

    # Build top-level structure
    mesh_dict: Dict[str, Any] = {
        "nx": mesh.nx,
        "ny": mesh.ny,
        "nz": mesh.nz,
        "dx_m": mesh.dx,
        "dy_m": mesh.dy,
        "dz_m": mesh.dz,
        "nodes": [_serialize_node(node) for node in mesh.nodes],
    }
    if include_bounds:
        mesh_dict.update({
            "board_min_x_mm": mesh.board_min_x,
            "board_max_x_mm": mesh.board_max_x,
            "board_min_y_mm": mesh.board_min_y,
            "board_max_y_mm": mesh.board_max_y,
        })

    # Write to disk atomically: write to temp then rename
    tmp_path = file_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(mesh_dict, fh, indent=2)
    os.replace(tmp_path, file_path)
