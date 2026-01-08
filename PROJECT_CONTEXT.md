# TVAC Thermal Analyzer - KiCad Plugin Project Context

## Overview

This is a professional-grade KiCad plugin for thermal analysis of PCBs, specifically designed for space electronics and TVAC (Thermal Vacuum) testing conditions. The plugin performs finite-element thermal simulation directly within KiCad.

**Developer:** Eliot Abramo  
**KiCad Version:** 9.0+  
**Python:** 3.x with wxPython (KiCad's embedded Python)

---

## Project Goals

1. **Industrial-grade thermal analysis** integrated directly into KiCad
2. **Two heat source modes:**
   - Component Power Dissipation: Manual power values per component
   - Current Injection (I²R Joule Heating): Heat calculated from current flow through traces
3. **TVAC simulation conditions:** Radiation heat transfer, chamber wall temperature
4. **Professional UI:** Interactive BOM-style PCB visualization with thermal overlay
5. **Automatic PCB data extraction:** Components, traces, pads, board outline from KiCad

---

## Architecture

```
tvac_rewrite/
├── __init__.py              # Plugin registration for KiCad
├── core/
│   ├── __init__.py
│   ├── config.py            # Configuration dataclasses, JSON serialization
│   ├── constants.py         # Materials database, physical constants
│   └── pcb_extractor.py     # Extract PCB data from KiCad board
├── solvers/
│   ├── __init__.py
│   ├── mesh_gen.py          # Thermal mesh generation
│   └── thermal_solver.py    # Steady-state and transient solvers
├── ui/
│   ├── __init__.py
│   ├── design_system.py     # wxPython widgets, colors, fonts
│   ├── main_dialog.py       # Main UI with tabs (Components, Heatsinks, Mounting, Current Path, Simulation)
│   ├── pcb_visualization.py # Interactive PCB canvas with thermal overlay
│   └── results_viewer.py    # Results display
├── utils/
│   ├── __init__.py
│   └── report_generator.py  # PDF report generation
├── native/
│   ├── thermal_engine.c     # Optional C solver (not required)
│   ├── build.sh
│   └── build.bat
└── resources/
    └── icon.svg
```

---

## Key Components

### 1. PCB Extractor (`core/pcb_extractor.py`)

Extracts from KiCad board:
- **Components:** Reference, value, footprint, position, rotation, bounding box, pads
- **Traces:** Start/end points, width, layer, net
- **Pads:** Position, size, net, drill info
- **Board outline:** From Edge.Cuts layer
- **Zones/Pours:** Copper fills
- **User shapes:** From User.1-User.9 layers (used for heatsinks)
- **Power field extraction:** Reads POWER_DISSIPATION field from schematic

**Important methods:**
- `extract_all()` - Returns `PCBData` dataclass with all extracted info
- `_extract_power_field(fp)` - Reads power from footprint fields
- `_extract_user_shapes()` - Gets heatsink outlines from User layers

### 2. Configuration (`core/config.py`)

Key dataclasses:
- `CurrentPath` - Source net, sink net, current (simplified I²R model)
- `ComponentPowerConfig` - Reference, power_w, source
- `MountingPointConfig` - Thermal mounting holes with optional fixed temperature
- `HeatsinkConfig` - Polygon outline, material, thickness
- `SimulationConfig` - Resolution, mode (steady/transient), heat_source_mode
- `ThermalAnalysisConfig` - Top-level config containing all above

### 3. Mesh Generator (`solvers/mesh_gen.py`)

Creates 3D thermal mesh from PCB data:
- Adaptive refinement near heat sources
- Material property assignment (FR4, copper, heatsink materials)
- Conductance calculation between nodes
- Boundary conditions from mounting points

**Heat source modes:**
- `_add_component_heat_sources()` - Distributes component power to mesh nodes
- `_add_current_heat_sources()` - Calculates I²R from current paths through traces

### 4. Thermal Solver (`solvers/thermal_solver.py`)

Two solver modes:
- **Steady-state:** Iterative solution with optional radiation
- **Transient:** Time-stepping with thermal mass

Uses scipy.sparse for efficiency. Falls back to pure NumPy/Python if scipy unavailable.

**Important:** scipy.sparse.linalg.cg() parameter compatibility:
- Newer scipy uses `atol`/`rtol`
- Older scipy uses `tol`
- Code has try/except to handle both

### 5. Main Dialog (`ui/main_dialog.py`)

Tabs:
1. **Components** - Set power dissipation per component
2. **Heatsinks** - Configure heatsink regions from User layers
3. **Mounting** - Define thermal mounting points (chassis contact)
4. **Current Path** - Define current flow paths (source net → sink net)
5. **Simulation** - Settings: mode, temperatures, mesh resolution

### 6. PCB Visualization (`ui/pcb_visualization.py`)

Interactive canvas showing:
- Board outline
- Components (top/bottom)
- Heatsinks (from User layers)
- Mounting points
- Current points
- Thermal overlay (color-coded temperature)

Features:
- Pan/zoom
- Component selection highlighting
- Layer visibility toggles

---

## Bugs Fixed (This Session)

1. **`chamber_temp_c` parameter error** - Wrong parameter name in solve_steady_state call
   - Fixed: Changed to `chamber_wall_temp_c`

2. **Gauge.SetValue float error** - Progress callback passed float instead of int
   - Fixed: `int(percent)` in ProgressDialog.update()

3. **scipy cg() `tol` parameter** - Newer scipy versions use `atol`/`rtol`
   - Fixed: Try/except wrapper for compatibility

4. **Heatsink detection** - User layers not being read correctly
   - Fixed: Enhanced `_extract_user_shapes()` with proper layer ID mapping

5. **Selection highlighting** - List selection not highlighting on PCB
   - Fixed: Added `_on_select()` handlers to all panels

6. **POWER_DISSIPATION field** - Not importing from schematic
   - Fixed: Added `_extract_power_field()` with multiple field name variants

7. **Simulation crash during material assignment** - Per-node copper detection too slow
   - Fixed: Simplified material assignment, removed expensive per-node checks

---

## Current Path Feature (I²R Joule Heating)

**Simplified model for user:**
1. Select source net (where current enters, e.g., "+12V")
2. Select sink net (where current exits, e.g., "GND")
3. Enter current magnitude in Amps
4. Solver calculates heat distribution

**How it works internally:**
1. Find all traces on the board
2. Calculate each trace's resistance: R = ρL/A
   - ρ = copper resistivity (1.68e-8 Ω·m)
   - L = trace length
   - A = trace width × copper thickness (35µm default)
3. Total power P = I²R summed over traces
4. Distribute heat to mesh nodes along traces

**Limitations:**
- Currently simplified model - distributes heat proportionally across all traces
- Does not do full Kirchhoff network analysis for current distribution
- For accurate results with complex routing, would need net connectivity tracing

---

## Installation

1. Extract `tvac_rewrite` folder to KiCad plugins directory:
   - Windows: `C:\Users\<user>\Documents\KiCad\9.0\scripting\plugins\`
   - Linux: `~/.local/share/kicad/9.0/scripting/plugins/`
   - macOS: `~/Documents/KiCad/9.0/scripting/plugins/`

2. Rename folder to `tvac_thermal_analyzer` (optional)

3. Restart KiCad or refresh plugins

4. Access via Tools → External Plugins → TVAC Thermal Analyzer

---

## Dependencies

**Required:**
- KiCad 9.0+ (uses pcbnew Python API)
- wxPython (included with KiCad)

**Optional (recommended):**
- NumPy - For efficient array operations
- SciPy - For sparse matrix solvers (much faster)

---

## Known Issues / TODO

1. **Current path model is simplified** - Heat distributed to all traces, not just connected ones
   - Enhancement: Build net connectivity graph, trace current paths properly

2. **Heatsink properties only on top layer** - Bottom layer heatsinks not fully supported

3. **No copper layer detection in mesh** - All nodes use substrate properties
   - Enhancement: Detect copper coverage per node for accurate thermal conductivity

4. **Transient solver convergence** - May need tuning for large meshes

5. **No GPU acceleration** - CPU only currently

---

## Testing Checklist

- [ ] Plugin loads without errors in KiCad
- [ ] PCB visualization shows board outline and components
- [ ] Can add component power values
- [ ] Heatsinks detected from User.1 layer
- [ ] Mounting points can be added
- [ ] Current paths can be defined
- [ ] Steady-state simulation completes
- [ ] Transient simulation completes
- [ ] Thermal overlay displays on PCB
- [ ] No crashes on empty board
- [ ] No crashes on complex board

---

## Code Style Notes

- Defensive programming throughout (try/except, None checks)
- Deferred imports inside methods for KiCad plugin compatibility
- Type hints used where practical
- Dataclasses for configuration structures
- wxPython for all UI (KiCad requirement)

---

## Contact

Developer: Eliot Abramo
