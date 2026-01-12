# TVAC Thermal Analyzer — Industrial Refactor (WIP Snapshot)

This ZIP is a **work-in-progress snapshot** of the ongoing refactor toward an industrial-grade, routing-aware electro‑thermal tool for KiCad 9.0+.

It is meant to demonstrate active progress and to provide a **more stable** baseline than the original ZIP.
Some key industrial features (terminal-based current injection UI, zone sheet-resistance modeling, radiation Newton linearization) are still in progress.

---

## What changed in this snapshot

### Dependency handling (NumPy/SciPy required)
- The plugin now **requires** NumPy + SciPy at runtime.
- At launch, the plugin checks imports and shows a **wx error dialog** if missing.

### Config reliability
- `ThermalAnalysisConfig.from_dict()` now loads `current_paths` (it was previously ignored).
- JSON loading is now **defensive**: unknown keys are ignored (forward/backward compatibility).
- `CurrentPath` supports both:
  - legacy `source_net/sink_net`
  - **new fields (preferred, WIP)**: `source_ref/source_pad`, `sink_ref/sink_pad`

### Current solver import crash fixed
- `solvers/current_solver.py` no longer hard-crashes if `TraceCurrentOverride` is missing.
  A fallback dataclass is provided for backward compatibility.

### Joule heating is now routing-aware (DC resistive-network solve)
- Current Injection mode no longer uses the old “spread over all traces” heuristic.
- For each `CurrentPath` with terminals (`source_ref/source_pad` → `sink_ref/sink_pad`):
  - Builds a resistive network from **routed traces + vias** on the selected net.
  - Solves for node voltages using sparse nodal analysis (SciPy).
  - Computes per-segment Joule losses (**I²R**) and **deposits them onto the thermal mesh** near each copper segment.

**Still not modeled (yet):**
- Copper zones/pours as a 2D sheet-resistance grid (important for planes/returns)
- Temperature-dependent resistivity iteration (electro-thermal coupling loop)

### Simulation launch gating fixed
- `Run Simulation` no longer blocks when component power is 0 **if** the mode is `current_injection`
  and current paths exist.

### Mounting points applied on **B.Cu side**
- Mesh boundary selection for mounting points now uses the **bottom layer** (B.Cu side).

---

## Work still in progress (next steps)

1. **Terminal-based current injection (pads REF:PAD)**
   - UI: searchable selection of pads as current terminals.
   - Enforce that a *single* current path lives on a single electrical net.
   - Support explicit **supply** and **return** paths.

2. **Routing-aware DC solve**
   - Build a conductance (Laplacian) matrix for copper network and solve:
     \[
       G V = I
     \]
   - Extract edge currents and Joule heating:
     \[
       I_e = \frac{V_i - V_j}{R_e}, \quad P_e = I_e^2 R_e = \frac{(V_i - V_j)^2}{R_e}
     \]
   - Include:
     - traces: \(R=\rho L/(t w)\)
     - vias: plated barrel (annular cylinder)
     - copper planes/zones: sheet resistance mesh (grid, parameter `plane_grid_mm`)

3. **Joule-to-thermal mesh deposition**
   - Deposit per-edge Joule power into the 3D thermal mesh on the correct copper layer.
   - Use geometry-aware mapping (nearest-node / cell intersection) to avoid bias.

4. **TVAC radiation: stable Newton linearization**
   - Radiation boundary condition:
     \[
       q = \varepsilon \sigma A (T^4 - T_w^4)
     \]
   - Newton linearization around \(T_k\):
     \[
       q(T) \approx q(T_k) + 4\varepsilon\sigma A T_k^3 (T - T_k)
     \]
   - Which becomes a **diagonal conductance** term \(h_{rad}=4\varepsilon\sigma A T_k^3\) in the matrix (improves solvability).

5. **Mounting to a temperature box**
   - Replace pure Dirichlet by (optional) Robin/contact model:
     \[
       q = \frac{T - T_{box}}{R_\theta}
     \]
   - Implemented as diagonal conductance + RHS contribution.

---

## Accuracy & approximations (current intended model)

### Thermal
- 3D conduction on a structured mesh (finite volume / finite difference style).
- Copper features increase local conductivity on copper layers.
- FR-4 uses nominal constants (as requested).

### Electrical (DC)
- DC only (no inductive/skin/AC return current effects).
- Vias modeled via plated barrel thickness (default 25 µm).
- Planes modeled via a resistive grid (approximation controlled by `plane_grid_mm`).

### TVAC radiation
- Uses an isothermal chamber wall temperature `T_w` as an effective sink.
- No view-factor geometry model in this version (documented limitation).

---

## Installation
Copy the `tvac_thermal_analyzer/` folder into KiCad's `scripting/plugins/` directory and restart KiCad.

---

## Notes
This snapshot is intentionally marked WIP. The next ZIP will include:
- pad-terminal current injection UI,
- DC solve for currents + Joule mapping,
- full README with numerical analysis discussion and a clearer “what’s modeled vs not modeled”.
