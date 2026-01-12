# AI Agent Handoff Prompt

Use this prompt when starting a new conversation about this project:

---

## PROMPT TO COPY:

```
I'm working on a KiCad plugin called "TVAC Thermal Analyzer" for thermal analysis of PCBs in space electronics applications. This is a complex Python/wxPython project that runs inside KiCad's plugin system.

**Project Summary:**
- Finite-element thermal simulation integrated into KiCad
- Two heat source modes: component power dissipation OR current injection (I²R Joule heating)
- Supports TVAC conditions (radiation, chamber wall temperature)
- Interactive PCB visualization with thermal overlay
- Extracts PCB data automatically from KiCad (components, traces, pads, outline)

**Key Files:**
- `core/config.py` - Configuration dataclasses (CurrentPath, ComponentPowerConfig, SimulationConfig, etc.)
- `core/pcb_extractor.py` - Extracts data from KiCad board via pcbnew API
- `solvers/mesh_gen.py` - Generates thermal mesh, assigns heat sources
- `solvers/thermal_solver.py` - Steady-state and transient solvers (uses scipy.sparse)
- `ui/main_dialog.py` - Main wxPython dialog with tabs (Components, Heatsinks, Mounting, Current Path, Simulation)
- `ui/pcb_visualization.py` - Interactive PCB canvas with thermal overlay

**Heat Source Modes:**
1. Component Power: Manual power values per component
2. Current Injection: Define source/sink nets + current, calculates I²R heat in traces

**Recent Bug Fixes:**
- scipy.sparse.linalg.cg() parameter compatibility (tol vs atol/rtol)
- Progress gauge float→int conversion
- Parameter name mismatches
- Heatsink detection from User layers
- POWER_DISSIPATION field import from schematic

**Important Constraints:**
- Must use wxPython (KiCad requirement)
- Deferred imports inside methods (not at module level)
- Defensive programming (try/except, None checks)
- Must handle missing scipy/numpy gracefully

Please read the attached PROJECT_CONTEXT.md and PROJECT_CONTEXT.json files for full details before making any changes.

[Then describe your specific issue or request]
```

---

## WHAT TO INCLUDE WITH THIS PROMPT:

1. **The full plugin source code** - ZIP file or paste relevant files
2. **PROJECT_CONTEXT.md** - Markdown documentation
3. **PROJECT_CONTEXT.json** - Structured metadata
4. **Error screenshots** - If debugging
5. **KiCad version** - Currently developed for KiCad 9.0

---

## COMMON ISSUES AND SOLUTIONS:

### "Unexpected keyword argument" errors
- Usually scipy version differences (tol vs atol/rtol)
- Or parameter name mismatches between method signature and call

### "Gauge.SetValue" type errors
- Progress values must be int, not float
- Wrap in int() before passing

### Plugin doesn't appear in KiCad
- Check folder is in correct plugins directory
- Check __init__.py has proper ActionPlugin subclass
- Restart KiCad completely

### Simulation crashes during material assignment
- Usually performance issue with per-node checks
- Simplify by assigning properties to all nodes at once

### Heatsinks not detected
- Check shapes are on User.1-User.9 layers
- Check _extract_user_shapes() handles the shape type

---

## DEVELOPMENT ENVIRONMENT:

- KiCad 9.0 on Windows
- Python 3.x (KiCad's embedded Python)
- wxPython (included with KiCad)
- NumPy and SciPy (optional but recommended)

---

## DEVELOPER PREFERENCES (Eliot):

- Prefers simple, intuitive workflows over complex architectures
- Values stability and defensive programming
- Wants Altium-quality professional UI
- Testing on real PCB designs
- Iterative debugging with detailed error feedback
