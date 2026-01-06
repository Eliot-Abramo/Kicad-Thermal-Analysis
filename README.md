# TVAC Thermal Analyzer

Industrial-grade thermal analysis plugin for KiCad, designed for space electronics PCB design under Thermal Vacuum (TVAC) chamber conditions.

## Features

- **TVAC Simulation**: Radiation-dominated heat transfer modeling for vacuum conditions
- **Interactive BOM-style Visualization**: Pan, zoom, and inspect PCB with thermal overlay
- **Component Power Management**: Configure power dissipation for ICs, regulators, and passives
- **Heatsink Configuration**: Define heatsinks via User layer polygons
- **Mounting Point Thermal Boundaries**: Set fixed temperatures at mounting locations
- **Transient & Steady-State Analysis**: Full time-domain or equilibrium solutions
- **3D Stackup Modeling**: Multi-layer thermal conductance with realistic materials
- **Professional Reporting**: PDF reports with thermal maps and component tables

## Requirements

### Required
- KiCad 9.0 or later
- Python 3.9+
- wxPython (bundled with KiCad)

### Optional (for enhanced performance)
- NumPy (for Python solver)
- SciPy (for sparse matrix solvers)
- reportlab (for PDF report generation)
- Pillow/PIL (for thermal map images)

### Native Engine (optional, 10-100x faster)
- GCC with OpenMP support
- See build instructions below

## Installation

### Method 1: KiCad Plugin Manager (Recommended)
1. Open KiCad
2. Go to Plugin and Content Manager
3. Search for "TVAC Thermal Analyzer"
4. Click Install

### Method 2: Manual Installation
1. Locate your KiCad plugins directory:
   - Windows: `%APPDATA%\kicad\9.0\scripting\plugins`
   - macOS: `~/Library/Preferences/kicad/9.0/scripting/plugins`
   - Linux: `~/.local/share/kicad/9.0/scripting/plugins`

2. Copy the `tvac_thermal_analyzer` folder to the plugins directory

3. Restart KiCad or reload plugins (Tools → Scripting → Reload Plugins)

## Usage

1. Open a PCB in KiCad's PCB Editor
2. Launch via Tools → External Plugins → TVAC Thermal Analyzer
3. Configure:
   - **Components Tab**: Set power dissipation for heat-generating components
   - **Heatsinks Tab**: Select layer and import heatsink polygons
   - **Mounting Tab**: Configure thermal mounting points
   - **Simulation Tab**: Set temperatures and solver parameters
4. Click "Run Simulation"
5. View results with thermal overlay on PCB
6. Export PDF report if needed

## Heatsink Definition

To define heatsinks in your PCB:

1. In KiCad PCB Editor, switch to a User layer (default: User.1)
2. Draw a polygon outline around the heatsink area
3. In the TVAC Analyzer, go to Heatsinks tab
4. Select the layer and click "Detect Shapes"
5. Configure material and thickness for each heatsink

## Mounting Points

Thermal mounting points provide heat paths to a chassis:

1. Use standard mounting hole footprints in your design
2. In TVAC Analyzer, go to Mounting tab
3. Click "Auto-Detect Holes" to import mounting locations
4. Set fixed temperatures for thermal boundary conditions

## Building the Native Engine (Optional)

For best performance, compile the C thermal engine:

### Linux/macOS
```bash
cd tvac_thermal_analyzer/native
./build.sh
```

### Windows (MinGW)
```bash
cd tvac_thermal_analyzer\native
build.bat
```

### Manual Build
```bash
gcc -O3 -fopenmp -shared -fPIC -o libthermal_engine.so thermal_engine.c -lm
```

The plugin automatically uses the native engine if available, falling back to Python.

## Configuration Files

The plugin saves configuration alongside your PCB:
- `<pcb_name>_tvac_thermal_config.json` - All settings and component power

## Physical Models

### Heat Transfer
- **Conduction**: 3D finite difference through PCB stackup
- **Radiation**: Stefan-Boltzmann to chamber walls (emissivity-based)
- **Convection**: Disabled (vacuum conditions)

### Materials Database
- PCB substrates: FR4, Polyimide, Rogers, Ceramic, PTFE
- Conductors: Copper, Aluminum, Gold
- Heatsinks: Aluminum alloys, Copper
- Thermal interfaces: Pastes, pads, gap fillers

### Component Thermal Models
- Package theta values (θja, θjc, θjb)
- Thermal mass for transient analysis
- Auto-estimation from footprint

## Troubleshooting

### Plugin doesn't appear
- Ensure Python dependencies are installed
- Check KiCad console for import errors
- Verify plugin folder structure

### Simulation is slow
- Reduce mesh resolution (increase mm value)
- Disable adaptive refinement
- Build native engine for 10-100x speedup

### High temperatures
- Check component power values
- Verify heatsink thermal contact
- Review mounting point boundary conditions

## License

MIT License - See LICENSE file

## Author

Space Electronics Thermal Analysis Tool v2.0.0

## Acknowledgments

- KiCad development team
- Interactive BOM for visualization inspiration
- Thermal modeling community
