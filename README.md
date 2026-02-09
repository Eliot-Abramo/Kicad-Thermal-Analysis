# TVAC Thermal Analyzer v3.0

**Standalone thermal analysis tool for PCBs operating in Thermal Vacuum (TVAC) conditions.**

Reads KiCad `.kicad_pcb` files directly — no KiCad or pcbnew dependency required. Generates thermal meshes, solves steady-state and transient heat conduction with radiation, and displays interactive results.

## Quick Start

```bash
chmod +x build.sh
./build.sh                              # install deps + compile C engine
python3 tvac_analyzer.py                # opens file dialog
python3 tvac_analyzer.py board.kicad_pcb  # open specific file
```

## Requirements

| Dependency | Purpose |
|-----------|---------|
| Python 3.8+ | Runtime |
| PyQt5 | GUI |
| NumPy | Mesh & arrays |
| SciPy | Sparse solver (fallback) |
| GCC (optional) | C thermal engine (3-5× faster) |

Install: `pip install PyQt5 numpy scipy`

## Features

### 5-Tab Interface

1. **Components** — Click any component on the PCB or in the table to set its power dissipation (Watts). Filter by prefix (U*, Q*, R*). Auto-imports `POWER_DISSIPATION` fields from KiCad.

2. **Heatsinks** — Define heatsink polygons with material selection (Aluminum 6061/6063, Anodized Al, Copper). Set thickness and emissivity. Polygons rendered on PCB preview.

3. **Mounting Points** — Auto-detect mounting holes from PCB. Set fixed-temperature boundary conditions. Add manual mounting points.

4. **Current Path** — Define source/sink net pairs for I²R Joule heating analysis. Select nets from dropdowns populated from the PCB netlist. Set current in Amps.

5. **Simulation** — Choose heat source mode (Component Power or Current Injection), analysis type (Steady-State or Transient), environment temps, mesh resolution, radiation toggle.

### Interactive PCB Visualization

- Pan (drag) and zoom (scroll wheel) the board
- Click components to select and edit power
- Layer toggles: Grid, Outline, Top/Bottom Cu, Traces, Heatsinks, Mounting, Thermal overlay
- Thermal heatmap overlay after simulation

### Dual Solver Backend

| Backend | Speed | Accuracy | Availability |
|---------|-------|----------|-------------|
| **C Engine** | ~0.03s typical | Full double precision, PCG + Picard | Requires GCC |
| **SciPy** | ~0.15s typical | Full double precision, direct + Picard | Always available |

Both backends produce **identical results** — validated to < 0.01°C difference.

### Physics

- **Conduction**: Anisotropic thermal conductivity through FR4/copper/heatsink materials
- **Radiation**: Stefan-Boltzmann law with surface emissivity, both surfaces (top + bottom)
- **Picard iteration**: Under-relaxed nonlinear iteration for radiation coupling
- **Copper blending**: Zone and trace copper mapped to mesh with effective thermal conductivity
- **I²R heating**: Current paths compute trace resistance from copper geometry and distribute Joule heat

## File Structure

```
tvac_analyzer/
├── tvac_analyzer.py          # Complete standalone application (1700+ lines)
├── thermal_engine.c          # Optimized C solver (~680 lines)
├── libthermal_engine.so      # Compiled C library (auto-generated)
├── build.sh                  # Build & dependency installer
└── README.md                 # This file
```

## Configuration

Settings auto-save to `<board>.tvac_config.json` alongside the PCB file. Config includes:
- Component power assignments
- Heatsink definitions with polygons
- Mounting point locations and fixed temperatures
- Current injection paths
- All simulation parameters

## Architecture

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  KiCad PCB   │──▶│  S-Expr      │──▶│   PCBData    │
│  .kicad_pcb  │   │  Parser      │   │  Components  │
└──────────────┘   └──────────────┘   │  Traces/Vias │
                                      │  Zones/Nets  │
                                      └──────┬───────┘
                                             │
┌──────────────┐   ┌──────────────┐          │
│  Config JSON │──▶│  Config      │──────────┤
│  .json       │   │  (powers,    │          │
└──────────────┘   │   heatsinks, │   ┌──────▼───────┐
                   │   mounting)  │──▶│  Mesh Gen    │
                   └──────────────┘   │  Adaptive    │
                                      └──────┬───────┘
                                             │
                   ┌──────────────┐   ┌──────▼───────┐
                   │  C Engine    │◀──│  Thermal     │
                   │  PCG+Picard  │──▶│  Solver      │
                   └──────────────┘   └──────┬───────┘
                                             │
                                      ┌──────▼───────┐
                                      │  Results     │
                                      │  Heatmap     │
                                      └──────────────┘
```

## Materials Database

### PCB Substrates
- FR4 (k=0.29 W/mK, ε=0.90)
- FR4 High-Tg (k=0.35 W/mK, ε=0.90)
- Copper (k=385 W/mK, ε=0.03/0.65 oxidized)

### Heatsink Materials
- Aluminum 6061-T6 (k=167 W/mK, ε=0.09)
- Aluminum 6063-T5 (k=200 W/mK, ε=0.09)
- Anodized Aluminum (k=167 W/mK, ε=0.85)
- Copper Heatsink (k=385 W/mK, ε=0.65)

## License

Internal tool for space electronics thermal analysis.
