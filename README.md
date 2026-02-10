# TVAC Thermal Analyzer v4.0

**Physically rigorous thermal analysis for PCBs operating in Thermal Vacuum (TVAC) conditions.**

Reads KiCad 9 `.kicad_pcb` files directly — no KiCad or pcbnew dependency required. Generates adaptive thermal meshes with via modeling, solves steady-state and transient heat conduction with radiation, and displays interactive results with per-layer visualization.

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

## What's New in v4

### Physics Improvements
- **Adaptive mesh refinement** — quadtree-based, fine cells (0.15mm) near vias/pads/heat sources, coarse cells (3mm) in bulk FR4
- **Explicit via thermal modeling** — barrel conductance G = k_cu · A_barrel / L bridges copper layers
- **Proper copper fraction** — per-cell computation from filled zones, traces, and via pads
- **Linearized radiation Picard** — adds 4·ε·σ·T³·A to diagonal for dramatically faster convergence
- **Separate top/bottom emissivity** — solder mask (ε=0.90) on exposed surfaces, none on internal layers
- **Energy balance verification** — tracks input power vs radiation + conduction losses
- **IC(0) preconditioner** — Incomplete Cholesky with automatic Jacobi fallback

### KiCad 9 Support
- `filled_polygon` zones (actual copper fill geometry)
- `property` nodes for Reference/Value
- Arc trace support (`arc` with start/mid/end)
- Pad mirroring for back-side components
- Stackup parsing for per-layer copper thickness

### Performance
- OpenMP parallelization in C engine
- Adaptive mesh reduces node count ~75% vs uniform at same accuracy
- Anderson acceleration for Picard iteration
- Adaptive under-relaxation (ω: 0.15 → 0.95)

## Features

### 5-Tab Interface

1. **Components** — Click any component to set power dissipation. Filter by prefix (U*, Q*, R*). Auto-imports `POWER_DISSIPATION` fields from KiCad.

2. **Heatsinks** — Define heatsink polygons with material selection (Aluminum 6061/6063, Anodized Al, Copper). Set thickness and emissivity.

3. **Mounting Points** — Auto-detect mounting holes. Set fixed-temperature boundary conditions.

4. **Current Path** — Define source/sink net pairs for I²R Joule heating analysis. Temperature-dependent copper resistivity.

5. **Simulation** — Heat source mode, analysis type, environment temps, mesh resolution, adaptive settings, radiation toggle.

### Interactive PCB Visualization

- Pan (drag) and zoom (scroll wheel) the board
- Click components to select and edit power
- Per-layer toggles: Grid, Outline, Top Cu, Bottom Cu, Inner Cu, Traces, Vias, Heatsinks, Mounting, Thermal overlay
- Layer-specific colors: F.Cu=red, B.Cu=blue, inner layers=yellow/green
- Thermal heatmap overlay with Blue→Cyan→Green→Yellow→Red gradient

### Dual Solver Backend

| Backend | Speed | Features | Availability |
|---------|-------|----------|-------------|
| **C Engine** | ~0.03s typical | IC(0) preconditioner, OpenMP, Anderson accel | Requires GCC |
| **SciPy** | ~0.15s typical | Direct solver + Picard | Always available |

Both backends produce identical physics — validated to < 0.01°C difference.

### Physics Model

- **Conduction**: Harmonic mean k_eff for series resistance, copper fraction blending
- **Radiation**: Stefan-Boltzmann with separate top/bottom surface emissivity
- **Vias**: Barrel conductance bridging copper layers (G = k_cu · A / L)
- **Picard iteration**: Linearized radiation with adaptive under-relaxation
- **Copper blending**: k_eff = f·k_cu + (1-f)·k_FR4 with fraction from zones/traces/pads
- **I²R heating**: Temperature-dependent resistivity ρ(T) = ρ₀·(1 + α·ΔT)
- **Transient**: Crank-Nicolson (θ=0.5), 2nd-order time accuracy

## File Structure

```
tvac_analyzer/
├── tvac_analyzer.py          # Complete standalone application (~3700 lines)
├── thermal_engine.c          # Optimized C solver with IC(0) + OpenMP
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
- All simulation parameters (mesh resolution, adaptive settings, etc.)

## Architecture

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  KiCad 9 PCB │──▶│  S-Expr      │──▶│   PCBData    │
│  .kicad_pcb  │   │  Parser      │   │  Components  │
└──────────────┘   └──────────────┘   │  Traces/Arcs │
                                      │  Zones/Nets  │
                                      │  Vias/Pads   │
                                      └──────┬───────┘
                                             │
┌──────────────┐   ┌──────────────┐          │
│  Config JSON │──▶│  Config      │──────────┤
│  .json       │   │  (powers,    │          │
└──────────────┘   │   heatsinks, │   ┌──────▼───────┐
                   │   mounting)  │──▶│  Adaptive    │
                   └──────────────┘   │  Mesh Gen    │
                                      │  + Via Model │
                                      └──────┬───────┘
                                             │
                   ┌──────────────┐   ┌──────▼───────┐
                   │  C Engine    │◀──│  Thermal     │
                   │  IC(0)+PCG   │──▶│  Solver      │
                   │  OpenMP      │   │  Linearized  │
                   └──────────────┘   │  Radiation   │
                                      └──────┬───────┘
                                             │
                                      ┌──────▼───────┐
                                      │  Results     │
                                      │  Per-layer   │
                                      │  Heatmap     │
                                      │  Energy Bal  │
                                      └──────────────┘
```

## Materials Database

### PCB Substrates
| Material | k (W/m·K) | ε | ρ (kg/m³) |
|----------|-----------|---|-----------|
| FR4 | 0.29 | 0.90 | 1850 |
| FR4 High-Tg | 0.35 | 0.90 | 1900 |
| Copper | 385 | 0.03 | 8960 |
| Cu Oxidized | 385 | 0.65 | 8960 |
| Solder Mask | 0.25 | 0.90 | 1200 |

### Heatsink Materials
| Material | k (W/m·K) | ε | ρ (kg/m³) |
|----------|-----------|---|-----------|
| Al 6061-T6 | 167 | 0.09 | 2700 |
| Al 6063-T5 | 200 | 0.09 | 2690 |
| Anodized Al | 167 | 0.85 | 2700 |
| Copper | 385 | 0.65 | 8960 |

## Validation

### Energy Balance
At steady state, the solver verifies: `Σ Q_input ≈ Σ Q_radiation + Σ Q_conduction`
- Typical imbalance: <1% for well-conditioned problems
- Displayed in results dialog after simulation

### Via Thermal Impact
Example: 0.6mm via, 0.3mm drill, 1.6mm board
- Via barrel conductance: ~51 W/K
- FR4 conductance (same area): ~0.04 W/K
- **Vias carry ~1300× more heat than bare FR4**

### Adaptive Mesh
Typical 100×80mm board, 4 layers:
- Uniform 1mm mesh: ~32,000 nodes
- Adaptive mesh: ~8,000 nodes (75% reduction) with better accuracy near features

## License

Internal tool for space electronics thermal analysis.
