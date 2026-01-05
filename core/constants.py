"""
TVAC Thermal Analyzer - Physical Constants and Materials Database
=================================================================
Contains all physical constants, material properties, and component thermal data
for accurate thermal simulation in vacuum conditions.

Author: Space Electronics Thermal Analysis Tool
Version: 1.0.0
License: MIT
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from enum import Enum
import json
import os


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

class PhysicalConstants:
    """Fundamental physical constants for thermal calculations."""
    
    # Stefan-Boltzmann constant [W/(m²·K⁴)]
    STEFAN_BOLTZMANN = 5.670374419e-8
    
    # Boltzmann constant [J/K]
    BOLTZMANN = 1.380649e-23
    
    # Absolute zero offset [K]
    CELSIUS_TO_KELVIN = 273.15
    
    # Reference temperature for material properties [°C]
    REFERENCE_TEMP = 25.0
    
    # Gravity (for natural convection reference, not used in vacuum) [m/s²]
    GRAVITY = 9.81


# =============================================================================
# MATERIAL PROPERTIES
# =============================================================================

@dataclass
class ThermalMaterial:
    """Material thermal properties dataclass."""
    name: str
    thermal_conductivity: float  # W/(m·K)
    specific_heat: float  # J/(kg·K)
    density: float  # kg/m³
    emissivity: float  # dimensionless (0-1)
    electrical_resistivity: Optional[float] = None  # Ω·m (for conductors)
    temp_coeff_resistivity: Optional[float] = None  # 1/K (temperature coefficient)
    
    def get_resistivity_at_temp(self, temp_c: float) -> Optional[float]:
        """Calculate electrical resistivity at given temperature."""
        if self.electrical_resistivity is None:
            return None
        if self.temp_coeff_resistivity is None:
            return self.electrical_resistivity
        delta_t = temp_c - PhysicalConstants.REFERENCE_TEMP
        return self.electrical_resistivity * (1 + self.temp_coeff_resistivity * delta_t)
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'thermal_conductivity': self.thermal_conductivity,
            'specific_heat': self.specific_heat,
            'density': self.density,
            'emissivity': self.emissivity,
            'electrical_resistivity': self.electrical_resistivity,
            'temp_coeff_resistivity': self.temp_coeff_resistivity
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ThermalMaterial':
        """Deserialize from dictionary."""
        return cls(**data)


class MaterialsDatabase:
    """Database of thermal materials with default values and user customization."""
    
    # Default PCB substrate materials
    PCB_SUBSTRATES = {
        'FR4': ThermalMaterial(
            name='FR4 (Standard)',
            thermal_conductivity=0.3,  # W/(m·K) - through thickness
            specific_heat=1100,
            density=1850,
            emissivity=0.90
        ),
        'FR4_HIGH_TG': ThermalMaterial(
            name='FR4 High-Tg',
            thermal_conductivity=0.35,
            specific_heat=1050,
            density=1900,
            emissivity=0.90
        ),
        'POLYIMIDE': ThermalMaterial(
            name='Polyimide (Kapton)',
            thermal_conductivity=0.12,
            specific_heat=1090,
            density=1420,
            emissivity=0.85
        ),
        'ROGERS_4350B': ThermalMaterial(
            name='Rogers RO4350B',
            thermal_conductivity=0.69,
            specific_heat=850,
            density=1860,
            emissivity=0.85
        ),
        'ROGERS_4003C': ThermalMaterial(
            name='Rogers RO4003C',
            thermal_conductivity=0.71,
            specific_heat=840,
            density=1790,
            emissivity=0.85
        ),
        'CERAMIC_ALUMINA': ThermalMaterial(
            name='Ceramic (Al₂O₃)',
            thermal_conductivity=24.0,
            specific_heat=880,
            density=3900,
            emissivity=0.80
        ),
        'CERAMIC_ALN': ThermalMaterial(
            name='Ceramic (AlN)',
            thermal_conductivity=170.0,
            specific_heat=740,
            density=3260,
            emissivity=0.75
        ),
    }
    
    # Conductor materials
    CONDUCTORS = {
        'COPPER': ThermalMaterial(
            name='Copper (Annealed)',
            thermal_conductivity=401.0,
            specific_heat=385,
            density=8960,
            emissivity=0.03,  # Polished, increases with oxidation
            electrical_resistivity=1.68e-8,
            temp_coeff_resistivity=0.00393
        ),
        'COPPER_ED': ThermalMaterial(
            name='Copper (Electrodeposited)',
            thermal_conductivity=385.0,
            specific_heat=385,
            density=8900,
            emissivity=0.05,
            electrical_resistivity=1.72e-8,
            temp_coeff_resistivity=0.00393
        ),
        'ALUMINUM': ThermalMaterial(
            name='Aluminum',
            thermal_conductivity=237.0,
            specific_heat=897,
            density=2700,
            emissivity=0.05,
            electrical_resistivity=2.65e-8,
            temp_coeff_resistivity=0.00429
        ),
        'GOLD': ThermalMaterial(
            name='Gold',
            thermal_conductivity=318.0,
            specific_heat=129,
            density=19300,
            emissivity=0.02,
            electrical_resistivity=2.44e-8,
            temp_coeff_resistivity=0.0034
        ),
        'SILVER': ThermalMaterial(
            name='Silver',
            thermal_conductivity=429.0,
            specific_heat=235,
            density=10490,
            emissivity=0.02,
            electrical_resistivity=1.59e-8,
            temp_coeff_resistivity=0.0038
        ),
        'TIN_LEAD': ThermalMaterial(
            name='Solder (Sn63Pb37)',
            thermal_conductivity=50.0,
            specific_heat=180,
            density=8400,
            emissivity=0.05,
            electrical_resistivity=1.5e-7,
            temp_coeff_resistivity=0.004
        ),
        'SAC305': ThermalMaterial(
            name='Solder (SAC305)',
            thermal_conductivity=58.0,
            specific_heat=220,
            density=7400,
            emissivity=0.05,
            electrical_resistivity=1.3e-7,
            temp_coeff_resistivity=0.004
        ),
    }
    
    # Heatsink materials
    HEATSINK_MATERIALS = {
        'ALUMINUM_6061': ThermalMaterial(
            name='Aluminum 6061-T6',
            thermal_conductivity=167.0,
            specific_heat=896,
            density=2700,
            emissivity=0.09  # Bare, 0.77-0.85 if anodized
        ),
        'ALUMINUM_ANODIZED': ThermalMaterial(
            name='Aluminum (Anodized)',
            thermal_conductivity=167.0,
            specific_heat=896,
            density=2700,
            emissivity=0.82
        ),
        'COPPER_C101': ThermalMaterial(
            name='Copper C101',
            thermal_conductivity=388.0,
            specific_heat=385,
            density=8940,
            emissivity=0.07
        ),
        'COPPER_NICKEL_PLATED': ThermalMaterial(
            name='Copper (Nickel Plated)',
            thermal_conductivity=380.0,
            specific_heat=385,
            density=8940,
            emissivity=0.35
        ),
    }
    
    # Surface finishes (affects emissivity)
    SURFACE_FINISHES = {
        'BARE_COPPER': {'emissivity': 0.03, 'name': 'Bare Copper'},
        'HASL': {'emissivity': 0.05, 'name': 'HASL (Tin-Lead)'},
        'ENIG': {'emissivity': 0.03, 'name': 'ENIG (Gold)'},
        'OSP': {'emissivity': 0.80, 'name': 'OSP (Organic)'},
        'IMMERSION_SILVER': {'emissivity': 0.02, 'name': 'Immersion Silver'},
        'IMMERSION_TIN': {'emissivity': 0.05, 'name': 'Immersion Tin'},
    }
    
    # Solder mask colors (affects emissivity)
    SOLDER_MASK = {
        'GREEN': {'emissivity': 0.92, 'name': 'Green'},
        'BLACK': {'emissivity': 0.95, 'name': 'Black'},
        'WHITE': {'emissivity': 0.88, 'name': 'White'},
        'RED': {'emissivity': 0.90, 'name': 'Red'},
        'BLUE': {'emissivity': 0.91, 'name': 'Blue'},
        'YELLOW': {'emissivity': 0.89, 'name': 'Yellow'},
        'MATTE_BLACK': {'emissivity': 0.97, 'name': 'Matte Black'},
    }
    
    # Thermal interface materials
    THERMAL_INTERFACE = {
        'THERMAL_PASTE_STANDARD': {
            'name': 'Thermal Paste (Standard)',
            'thermal_conductivity': 4.0,  # W/(m·K)
            'typical_thickness': 0.05e-3  # 50 µm
        },
        'THERMAL_PASTE_HIGH': {
            'name': 'Thermal Paste (High Performance)',
            'thermal_conductivity': 8.5,
            'typical_thickness': 0.03e-3
        },
        'THERMAL_PAD': {
            'name': 'Thermal Pad',
            'thermal_conductivity': 3.0,
            'typical_thickness': 0.5e-3
        },
        'THERMAL_ADHESIVE': {
            'name': 'Thermal Adhesive',
            'thermal_conductivity': 1.5,
            'typical_thickness': 0.1e-3
        },
        'INDIUM_FOIL': {
            'name': 'Indium Foil',
            'thermal_conductivity': 86.0,
            'typical_thickness': 0.1e-3
        },
    }


# =============================================================================
# COMPONENT THERMAL DATABASE
# =============================================================================

@dataclass
class ComponentThermalData:
    """Thermal properties for electronic component packages."""
    package_type: str
    theta_ja: float  # Junction-to-ambient [°C/W]
    theta_jc: float  # Junction-to-case [°C/W]
    theta_jb: Optional[float] = None  # Junction-to-board [°C/W]
    thermal_mass: float = 0.0  # Thermal capacitance [J/K]
    package_area: Tuple[float, float] = (0.0, 0.0)  # (width, height) in mm
    typical_power: float = 0.0  # Typical power dissipation [W]
    max_junction_temp: float = 125.0  # Maximum junction temperature [°C]
    
    def to_dict(self) -> Dict:
        return {
            'package_type': self.package_type,
            'theta_ja': self.theta_ja,
            'theta_jc': self.theta_jc,
            'theta_jb': self.theta_jb,
            'thermal_mass': self.thermal_mass,
            'package_area': self.package_area,
            'typical_power': self.typical_power,
            'max_junction_temp': self.max_junction_temp
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ComponentThermalData':
        data['package_area'] = tuple(data.get('package_area', (0.0, 0.0)))
        return cls(**data)


class ComponentThermalDatabase:
    """Database of component package thermal properties."""
    
    # Resistors (SMD)
    RESISTORS = {
        '0201': ComponentThermalData('0201', 800, 400, thermal_mass=0.0001, package_area=(0.6, 0.3)),
        '0402': ComponentThermalData('0402', 500, 250, thermal_mass=0.0003, package_area=(1.0, 0.5)),
        '0603': ComponentThermalData('0603', 300, 150, thermal_mass=0.001, package_area=(1.6, 0.8)),
        '0805': ComponentThermalData('0805', 200, 100, thermal_mass=0.003, package_area=(2.0, 1.25)),
        '1206': ComponentThermalData('1206', 100, 50, thermal_mass=0.008, package_area=(3.2, 1.6)),
        '1210': ComponentThermalData('1210', 80, 40, thermal_mass=0.012, package_area=(3.2, 2.5)),
        '2010': ComponentThermalData('2010', 50, 25, thermal_mass=0.02, package_area=(5.0, 2.5)),
        '2512': ComponentThermalData('2512', 35, 18, thermal_mass=0.035, package_area=(6.3, 3.2)),
    }
    
    # Capacitors (SMD) - similar to resistors but with different thermal mass
    CAPACITORS = {
        '0201': ComponentThermalData('0201', 900, 450, thermal_mass=0.00008, package_area=(0.6, 0.3)),
        '0402': ComponentThermalData('0402', 600, 300, thermal_mass=0.0002, package_area=(1.0, 0.5)),
        '0603': ComponentThermalData('0603', 350, 175, thermal_mass=0.0008, package_area=(1.6, 0.8)),
        '0805': ComponentThermalData('0805', 220, 110, thermal_mass=0.002, package_area=(2.0, 1.25)),
        '1206': ComponentThermalData('1206', 120, 60, thermal_mass=0.006, package_area=(3.2, 1.6)),
        '1210': ComponentThermalData('1210', 90, 45, thermal_mass=0.01, package_area=(3.2, 2.5)),
    }
    
    # ICs - SOT packages
    SOT = {
        'SOT-23': ComponentThermalData('SOT-23', 250, 100, 150, thermal_mass=0.005, package_area=(2.9, 1.3)),
        'SOT-23-5': ComponentThermalData('SOT-23-5', 220, 90, 130, thermal_mass=0.006, package_area=(2.9, 1.6)),
        'SOT-23-6': ComponentThermalData('SOT-23-6', 200, 85, 120, thermal_mass=0.007, package_area=(2.9, 1.6)),
        'SOT-223': ComponentThermalData('SOT-223', 60, 15, 40, thermal_mass=0.03, package_area=(6.5, 3.5)),
        'SOT-89': ComponentThermalData('SOT-89', 120, 40, 80, thermal_mass=0.015, package_area=(4.5, 2.5)),
        'SC-70': ComponentThermalData('SC-70', 350, 140, 200, thermal_mass=0.003, package_area=(2.0, 1.25)),
    }
    
    # ICs - SOIC packages
    SOIC = {
        'SOIC-8': ComponentThermalData('SOIC-8', 120, 40, 60, thermal_mass=0.04, package_area=(5.0, 4.0)),
        'SOIC-14': ComponentThermalData('SOIC-14', 100, 35, 50, thermal_mass=0.06, package_area=(8.7, 4.0)),
        'SOIC-16': ComponentThermalData('SOIC-16', 90, 30, 45, thermal_mass=0.07, package_area=(10.0, 4.0)),
        'SOIC-20': ComponentThermalData('SOIC-20', 80, 28, 40, thermal_mass=0.09, package_area=(12.8, 7.5)),
        'SOIC-28': ComponentThermalData('SOIC-28', 60, 22, 35, thermal_mass=0.12, package_area=(17.9, 7.5)),
    }
    
    # ICs - QFP packages
    QFP = {
        'TQFP-32': ComponentThermalData('TQFP-32', 70, 20, 35, thermal_mass=0.08, package_area=(7.0, 7.0)),
        'TQFP-44': ComponentThermalData('TQFP-44', 60, 18, 30, thermal_mass=0.12, package_area=(10.0, 10.0)),
        'TQFP-48': ComponentThermalData('TQFP-48', 55, 15, 28, thermal_mass=0.14, package_area=(9.0, 9.0)),
        'TQFP-64': ComponentThermalData('TQFP-64', 45, 12, 24, thermal_mass=0.18, package_area=(12.0, 12.0)),
        'TQFP-100': ComponentThermalData('TQFP-100', 35, 8, 18, thermal_mass=0.28, package_area=(14.0, 14.0)),
        'TQFP-144': ComponentThermalData('TQFP-144', 30, 6, 15, thermal_mass=0.4, package_area=(20.0, 20.0)),
        'LQFP-48': ComponentThermalData('LQFP-48', 50, 14, 26, thermal_mass=0.15, package_area=(9.0, 9.0)),
        'LQFP-64': ComponentThermalData('LQFP-64', 40, 10, 22, thermal_mass=0.2, package_area=(12.0, 12.0)),
        'LQFP-100': ComponentThermalData('LQFP-100', 32, 7, 16, thermal_mass=0.3, package_area=(14.0, 14.0)),
        'LQFP-144': ComponentThermalData('LQFP-144', 28, 5, 14, thermal_mass=0.42, package_area=(20.0, 20.0)),
    }
    
    # ICs - QFN/DFN packages (excellent thermal)
    QFN = {
        'QFN-8': ComponentThermalData('QFN-8', 80, 8, 30, thermal_mass=0.02, package_area=(3.0, 3.0)),
        'QFN-16': ComponentThermalData('QFN-16', 50, 5, 20, thermal_mass=0.04, package_area=(4.0, 4.0)),
        'QFN-20': ComponentThermalData('QFN-20', 45, 4, 18, thermal_mass=0.05, package_area=(4.0, 4.0)),
        'QFN-24': ComponentThermalData('QFN-24', 40, 3.5, 16, thermal_mass=0.06, package_area=(5.0, 5.0)),
        'QFN-32': ComponentThermalData('QFN-32', 35, 3, 14, thermal_mass=0.08, package_area=(5.0, 5.0)),
        'QFN-40': ComponentThermalData('QFN-40', 30, 2.5, 12, thermal_mass=0.1, package_area=(6.0, 6.0)),
        'QFN-48': ComponentThermalData('QFN-48', 28, 2, 10, thermal_mass=0.12, package_area=(7.0, 7.0)),
        'QFN-64': ComponentThermalData('QFN-64', 25, 1.8, 8, thermal_mass=0.16, package_area=(9.0, 9.0)),
        'DFN-6': ComponentThermalData('DFN-6', 100, 12, 40, thermal_mass=0.01, package_area=(2.0, 2.0)),
        'DFN-8': ComponentThermalData('DFN-8', 85, 10, 35, thermal_mass=0.015, package_area=(3.0, 2.0)),
    }
    
    # ICs - BGA packages
    BGA = {
        'BGA-49': ComponentThermalData('BGA-49', 40, 5, 15, thermal_mass=0.08, package_area=(5.0, 5.0)),
        'BGA-64': ComponentThermalData('BGA-64', 35, 4, 12, thermal_mass=0.1, package_area=(6.0, 6.0)),
        'BGA-100': ComponentThermalData('BGA-100', 30, 3, 10, thermal_mass=0.15, package_area=(8.0, 8.0)),
        'BGA-144': ComponentThermalData('BGA-144', 25, 2.5, 8, thermal_mass=0.22, package_area=(10.0, 10.0)),
        'BGA-256': ComponentThermalData('BGA-256', 20, 2, 6, thermal_mass=0.35, package_area=(13.0, 13.0)),
        'BGA-324': ComponentThermalData('BGA-324', 18, 1.8, 5, thermal_mass=0.45, package_area=(15.0, 15.0)),
        'BGA-484': ComponentThermalData('BGA-484', 15, 1.5, 4, thermal_mass=0.6, package_area=(19.0, 19.0)),
        'BGA-676': ComponentThermalData('BGA-676', 12, 1.2, 3, thermal_mass=0.8, package_area=(23.0, 23.0)),
        'BGA-900': ComponentThermalData('BGA-900', 10, 1.0, 2.5, thermal_mass=1.0, package_area=(27.0, 27.0)),
        'FBGA-96': ComponentThermalData('FBGA-96', 35, 3.5, 11, thermal_mass=0.12, package_area=(8.0, 8.0)),
    }
    
    # Power devices
    POWER = {
        'TO-220': ComponentThermalData('TO-220', 62, 1.5, 25, thermal_mass=0.8, package_area=(10.0, 15.0), max_junction_temp=150),
        'TO-220F': ComponentThermalData('TO-220F', 65, 3.5, 28, thermal_mass=0.6, package_area=(10.0, 15.0), max_junction_temp=150),
        'TO-247': ComponentThermalData('TO-247', 40, 0.7, 18, thermal_mass=1.5, package_area=(16.0, 21.0), max_junction_temp=175),
        'TO-263': ComponentThermalData('TO-263', 50, 2.5, 20, thermal_mass=0.5, package_area=(10.0, 15.0), max_junction_temp=150),
        'TO-252': ComponentThermalData('TO-252', 80, 4, 30, thermal_mass=0.3, package_area=(6.5, 6.0), max_junction_temp=150),
        'DPAK': ComponentThermalData('DPAK', 80, 4, 30, thermal_mass=0.3, package_area=(6.5, 6.0), max_junction_temp=150),
        'D2PAK': ComponentThermalData('D2PAK', 50, 2.5, 20, thermal_mass=0.5, package_area=(10.0, 15.0), max_junction_temp=150),
        'LFPAK': ComponentThermalData('LFPAK', 60, 3, 25, thermal_mass=0.2, package_area=(5.0, 6.0), max_junction_temp=175),
        'PowerQFN': ComponentThermalData('PowerQFN', 40, 1.5, 15, thermal_mass=0.15, package_area=(5.0, 6.0), max_junction_temp=150),
    }
    
    # Connectors (approximations)
    CONNECTORS = {
        'USB-A': ComponentThermalData('USB-A', 100, 50, 60, thermal_mass=0.5, package_area=(12.0, 14.0)),
        'USB-C': ComponentThermalData('USB-C', 80, 40, 50, thermal_mass=0.3, package_area=(9.0, 7.5)),
        'RJ45': ComponentThermalData('RJ45', 60, 30, 40, thermal_mass=0.8, package_area=(16.0, 21.5)),
        'SMA': ComponentThermalData('SMA', 70, 35, 45, thermal_mass=0.2, package_area=(6.3, 6.3)),
        'HEADER_2.54': ComponentThermalData('Header 2.54mm', 150, 80, 100, thermal_mass=0.1, package_area=(2.54, 2.54)),
    }
    
    @classmethod
    def get_all_packages(cls) -> Dict[str, ComponentThermalData]:
        """Get all packages as a flat dictionary."""
        all_packages = {}
        for category_name in ['RESISTORS', 'CAPACITORS', 'SOT', 'SOIC', 'QFP', 'QFN', 'BGA', 'POWER', 'CONNECTORS']:
            category = getattr(cls, category_name, {})
            all_packages.update(category)
        return all_packages
    
    @classmethod
    def get_by_footprint(cls, footprint: str) -> Optional[ComponentThermalData]:
        """Attempt to match a KiCad footprint to thermal data."""
        footprint_upper = footprint.upper()
        all_packages = cls.get_all_packages()
        
        # Direct match
        for key, data in all_packages.items():
            if key.upper() in footprint_upper:
                return data
        
        # Partial matches
        if 'SOT23' in footprint_upper or 'SOT-23' in footprint_upper:
            if '6' in footprint_upper:
                return all_packages.get('SOT-23-6')
            elif '5' in footprint_upper:
                return all_packages.get('SOT-23-5')
            return all_packages.get('SOT-23')
        
        if 'QFN' in footprint_upper:
            # Try to extract pin count
            import re
            match = re.search(r'(\d+)', footprint_upper)
            if match:
                pins = int(match.group(1))
                return all_packages.get(f'QFN-{pins}')
        
        if 'BGA' in footprint_upper:
            import re
            match = re.search(r'(\d+)', footprint_upper)
            if match:
                pins = int(match.group(1))
                return all_packages.get(f'BGA-{pins}')
        
        # Size-based matching for passives
        for size in ['0201', '0402', '0603', '0805', '1206', '1210', '2010', '2512']:
            if size in footprint_upper:
                if 'C_' in footprint_upper or 'CAP' in footprint_upper:
                    return cls.CAPACITORS.get(size)
                return cls.RESISTORS.get(size)
        
        return None


# =============================================================================
# PCB STACKUP DEFAULTS
# =============================================================================

@dataclass
class PCBLayerDefaults:
    """Default PCB layer parameters."""
    copper_thickness_oz: Dict[str, float] = field(default_factory=lambda: {
        '0.5oz': 17.5e-6,  # 17.5 µm
        '1oz': 35e-6,      # 35 µm
        '2oz': 70e-6,      # 70 µm
        '3oz': 105e-6,     # 105 µm
    })
    
    prepreg_thickness: Dict[str, float] = field(default_factory=lambda: {
        '1080': 65e-6,   # µm
        '2116': 120e-6,
        '7628': 180e-6,
    })
    
    core_thickness: Dict[str, float] = field(default_factory=lambda: {
        '0.2mm': 200e-6,
        '0.4mm': 400e-6,
        '0.5mm': 500e-6,
        '0.8mm': 800e-6,
        '1.0mm': 1000e-6,
        '1.2mm': 1200e-6,
    })


# =============================================================================
# SIMULATION DEFAULTS
# =============================================================================

class SimulationDefaults:
    """Default simulation parameters."""
    
    # Grid resolution
    DEFAULT_RESOLUTION_MM = 0.5  # mm
    MIN_RESOLUTION_MM = 0.1
    MAX_RESOLUTION_MM = 2.0
    
    # Time stepping
    DEFAULT_TIMESTEP_S = 0.1  # seconds
    MIN_TIMESTEP_S = 0.01
    MAX_TIMESTEP_S = 10.0
    
    # Simulation durations
    DURATION_PRESETS = {
        'quick': 60,      # 1 minute
        'short': 300,     # 5 minutes
        'medium': 600,    # 10 minutes
        'long': 1800,     # 30 minutes
        'extended': 3600, # 1 hour
        'full': 18000,    # 5 hours
    }
    
    # Temperature bounds
    DEFAULT_AMBIENT_TEMP_C = 25.0
    DEFAULT_CHAMBER_TEMP_C = 25.0
    MIN_TEMP_C = -200.0
    MAX_TEMP_C = 500.0
    
    # Convergence
    STEADY_STATE_TOLERANCE = 0.01  # °C/s rate of change
    MAX_ITERATIONS = 10000
    
    # Memory management
    MAX_RESULT_FRAMES = 1000
    FRAME_COMPRESSION = True


# =============================================================================
# EXPORT ALL
# =============================================================================

__all__ = [
    'PhysicalConstants',
    'ThermalMaterial',
    'MaterialsDatabase',
    'ComponentThermalData',
    'ComponentThermalDatabase',
    'PCBLayerDefaults',
    'SimulationDefaults',
]
