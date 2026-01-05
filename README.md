# TVAC Thermal Analyzer for KiCad

A professional-grade thermal simulation plugin for KiCad PCB Editor, designed for analyzing PCB thermal behavior under thermal-vacuum (TVAC) conditions. Ideal for space electronics, aerospace applications, and high-reliability designs.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![KiCad](https://img.shields.io/badge/KiCad-9.0+-green)
![License](https://img.shields.io/badge/license-MIT-brightgreen)

## Features

### Thermal Simulation
- **3D Finite Element Analysis**: Full 3D thermal modeling through PCB stackup
- **Transient & Steady-State**: Both simulation modes with configurable duration
- **Adaptive Mesh**: Automatic mesh refinement near traces and components
- **Radiation Modeling**: Stefan-Boltzmann radiation for vacuum conditions
- **Conduction**: Full anisotropic thermal conductivity modeling

### Electrical Analysis
- **Current Distribution Solver**: Kirchhoff nodal analysis for current paths
- **Joule Heating**: Automatic I²R power dissipation calculation
- **Via Thermal Conductance**: Through-via heat transfer modeling
- **AC/Skin Effect**: Optional high-frequency resistance correction

### User Interface
- **Dark Theme**: Professional Altium-inspired interface
- **Point-to-Point Current Definition**: Easy current injection setup
- **Component Power Entry**: Manual or schematic-imported power values
- **Real-Time Progress**: Live simulation progress tracking
- **Heat Map Visualization**: Color-coded temperature overlay

### Reporting
- **Professional PDF Reports**: Comprehensive analysis documentation
- **Executive Summary**: Pass/fail assessment with key metrics
- **Detailed Tables**: Component temperatures, hot spots, warnings
- **Temperature History**: Time-evolution data for transient analysis

## Installation

### Method 1: Copy to KiCad Plugins Directory

1. Download or clone this repository
2. Copy the entire `tvac_thermal_analyzer` folder to your KiCad plugins directory:
   - **Windows**: `%APPDATA%/kicad/9.0/scripting/plugins/`
   - **Linux**: `~/.local/share/kicad/9.0/scripting/plugins/`
   - **macOS**: `~/Library/Preferences/kicad/9.0/scripting/plugins/`
3. Copy `tvac_thermal_analyzer_plugin.py` to the same directory
4. Restart KiCad

### Method 2: Using Package Manager (Coming Soon)

The plugin will be available through the KiCad Plugin and Content Manager.

### Dependencies

The plugin requires the following Python packages (install via pip if not present):

```bash
pip install numpy scipy reportlab
```

## Quick Start

1. Open your PCB in KiCad's PCB Editor
2. Go to **Tools → External Plugins → TVAC Thermal Analyzer**
3. Configure simulation parameters in the **Simulation** tab
4. Define current injection points in the **Current** tab
5. Set component power dissipation in the **Components** tab
6. Click **Run Simulation**
7. View results or export a PDF report

## Usage Guide

### Simulation Settings

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| Grid Resolution | Mesh spacing (smaller = more accurate, slower) | 0.5mm for 80x80mm boards |
| 3D Simulation | Enable full Z-axis thermal modeling | Yes for multi-layer boards |
| Include Radiation | Model vacuum radiation heat transfer | Yes for TVAC analysis |
| Duration | Simulation time for transient analysis | 10-30 minutes typical |

### Current Injection

Define current paths through your PCB:
- **Positive current**: Where current enters the board
- **Negative current**: Where current exits the board
- **Total must sum to zero** for a balanced analysis

Example for a 5A power stage:
- Input connector: +5.0 A
- Output connector: -5.0 A

### Component Power

Specify power dissipation for active components:
- Import from schematic `POWER_DISSIPATION` field
- Manual entry for each component
- Database lookup for common packages

### Interpretation

| Temperature | Status | Action |
|-------------|--------|--------|
| < 70°C | Safe | No action needed |
| 70-85°C | Marginal | Review thermal design |
| 85-100°C | Warning | Add thermal management |
| > 100°C | Critical | Redesign required |

## Technical Details

### Thermal Model

The solver implements the heat equation:

```
ρ·c·∂T/∂t = ∇·(k·∇T) + Q - σ·ε·(T⁴ - T_amb⁴)
```

Where:
- ρ = density (kg/m³)
- c = specific heat (J/kg·K)
- k = thermal conductivity (W/m·K)
- Q = heat generation (W/m³)
- σ = Stefan-Boltzmann constant
- ε = emissivity

### Solver Method

- **Spatial**: Finite difference on structured grid
- **Temporal**: Implicit Euler (backward difference)
- **Matrix**: Sparse LU factorization for efficiency
- **Radiation**: Newton-Raphson iteration

### Material Properties

Default material properties are based on industry standards:
- **FR4**: k = 0.3 W/(m·K), ε = 0.90
- **Copper**: k = 401 W/(m·K), ρ = 1.68×10⁻⁸ Ω·m
- **Solder Mask**: ε = 0.92 (green)

All values are configurable through the UI.

## File Structure

```
tvac_thermal_analyzer/
├── __init__.py              # Package initialization
├── tvac_thermal_analyzer_plugin.py  # KiCad plugin entry point
├── core/
│   ├── __init__.py
│   ├── constants.py         # Physical constants & materials
│   ├── config.py            # Configuration management
│   └── pcb_extractor.py     # PCB geometry extraction
├── solvers/
│   ├── __init__.py
│   ├── mesh_generator.py    # Thermal mesh generation
│   ├── current_solver.py    # Electrical network solver
│   └── thermal_solver.py    # FEM thermal solver
├── ui/
│   ├── __init__.py
│   ├── main_dialog.py       # Main plugin dialog
│   └── heat_map.py          # Visualization components
├── utils/
│   ├── __init__.py
│   ├── logger.py            # Logging system
│   └── report_generator.py  # PDF report generation
├── resources/
│   └── icon.png             # Plugin icon
└── data/
    └── (user configurations)
```

## Configuration Persistence

Simulation configurations are automatically saved alongside your PCB project:
- `<pcb_name>_tvac_thermal_config.json`

This allows you to:
- Re-run simulations with the same settings
- Share configurations with team members
- Version control thermal analysis setup

## Troubleshooting

### "No PCB is currently open"
- Ensure you have a PCB file open in the PCB Editor
- The plugin cannot run from the schematic editor

### Simulation is slow
- Increase grid resolution (larger value = fewer nodes)
- Disable 3D simulation for quick estimates
- Reduce simulation duration for initial testing

### Temperature results seem wrong
- Verify current injection points sum to zero
- Check component power values are in Watts
- Ensure material properties match your stackup

### Report generation fails
- Install reportlab: `pip install reportlab`
- Check write permissions to output directory

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - See LICENSE file for details.

## Support

For issues and feature requests, please use the GitHub issue tracker.

## Acknowledgments

- KiCad development team
- ECSS/IEC TR 62380 standards for reliability data
- Open source thermal analysis community

---

**Note**: This tool is for engineering analysis purposes. Always validate results with physical testing for critical applications.
