"""
TVAC Thermal Analyzer - Constants and Materials Database
========================================================
Physical constants, material properties, and component thermal data.

Author: Space Electronics Thermal Analysis Tool
Version: 2.0.0
"""

from dataclasses import dataclass
from typing import Dict, Optional


class PhysicalConstants:
    """Fundamental physical constants."""
    STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)
    CELSIUS_TO_KELVIN = 273.15
    COPPER_RESISTIVITY = 1.68e-8  # Ω·m at 20°C
    COPPER_TCR = 0.00393  # Temperature coefficient of resistance


@dataclass
class MaterialProperties:
    """Material thermal and electrical properties."""
    name: str
    thermal_conductivity: float  # W/(m·K)
    specific_heat: float  # J/(kg·K)
    density: float  # kg/m³
    emissivity: float = 0.9
    electrical_resistivity: Optional[float] = None  # Ω·m
    tcr: Optional[float] = None  # Temperature coefficient
    dielectric_constant: Optional[float] = None


class MaterialsDatabase:
    """Database of thermal material properties."""
    
    # PCB Substrates
    PCB_SUBSTRATES = {
        'FR4': MaterialProperties(
            name='FR4 (Standard)',
            thermal_conductivity=0.29,
            specific_heat=1100,
            density=1850,
            emissivity=0.9,
            dielectric_constant=4.5
        ),
        'FR4_HIGH_TG': MaterialProperties(
            name='FR4 (High-Tg)',
            thermal_conductivity=0.35,
            specific_heat=1100,
            density=1900,
            emissivity=0.9,
            dielectric_constant=4.3
        ),
        'POLYIMIDE': MaterialProperties(
            name='Polyimide',
            thermal_conductivity=0.35,
            specific_heat=1100,
            density=1420,
            emissivity=0.8,
            dielectric_constant=3.4
        ),
        'ROGERS_4350B': MaterialProperties(
            name='Rogers 4350B',
            thermal_conductivity=0.69,
            specific_heat=900,
            density=1860,
            emissivity=0.85,
            dielectric_constant=3.48
        ),
        'CERAMIC_ALUMINA': MaterialProperties(
            name='Alumina (Al₂O₃)',
            thermal_conductivity=25.0,
            specific_heat=880,
            density=3900,
            emissivity=0.9
        ),
    }
    
    # Conductors
    CONDUCTORS = {
        'COPPER': MaterialProperties(
            name='Copper',
            thermal_conductivity=385.0,
            specific_heat=385,
            density=8960,
            emissivity=0.03,
            electrical_resistivity=1.68e-8,
            tcr=0.00393
        ),
        'COPPER_OXIDIZED': MaterialProperties(
            name='Copper (Oxidized)',
            thermal_conductivity=385.0,
            specific_heat=385,
            density=8960,
            emissivity=0.65,
            electrical_resistivity=1.68e-8
        ),
        'ALUMINUM': MaterialProperties(
            name='Aluminum',
            thermal_conductivity=205.0,
            specific_heat=900,
            density=2700,
            emissivity=0.05,
            electrical_resistivity=2.65e-8,
            tcr=0.00429
        ),
        'GOLD': MaterialProperties(
            name='Gold',
            thermal_conductivity=317.0,
            specific_heat=129,
            density=19300,
            emissivity=0.02,
            electrical_resistivity=2.44e-8
        ),
    }
    
    # Heatsink Materials
    HEATSINK_MATERIALS = {
        'ALUMINUM_6061': MaterialProperties(
            name='Aluminum 6061-T6',
            thermal_conductivity=167.0,
            specific_heat=896,
            density=2700,
            emissivity=0.09
        ),
        'ALUMINUM_6063': MaterialProperties(
            name='Aluminum 6063-T5',
            thermal_conductivity=200.0,
            specific_heat=900,
            density=2690,
            emissivity=0.09
        ),
        'ALUMINUM_ANODIZED': MaterialProperties(
            name='Aluminum (Anodized)',
            thermal_conductivity=167.0,
            specific_heat=896,
            density=2700,
            emissivity=0.85
        ),
        'COPPER_HEATSINK': MaterialProperties(
            name='Copper (Heatsink)',
            thermal_conductivity=385.0,
            specific_heat=385,
            density=8960,
            emissivity=0.65
        ),
    }
    
    # Thermal Interface Materials
    THERMAL_INTERFACE = {
        'PASTE_GENERIC': MaterialProperties(
            name='Thermal Paste (Generic)',
            thermal_conductivity=1.0,
            specific_heat=1000,
            density=2500,
            emissivity=0.9
        ),
        'PASTE_SILVER': MaterialProperties(
            name='Thermal Paste (Silver)',
            thermal_conductivity=8.0,
            specific_heat=1000,
            density=3500,
            emissivity=0.9
        ),
        'PAD_SOFT': MaterialProperties(
            name='Thermal Pad (Soft)',
            thermal_conductivity=3.0,
            specific_heat=1200,
            density=1800,
            emissivity=0.9
        ),
        'PAD_FIRM': MaterialProperties(
            name='Thermal Pad (Firm)',
            thermal_conductivity=6.0,
            specific_heat=1200,
            density=2000,
            emissivity=0.9
        ),
    }
    
    # Solder Mask Emissivities
    SOLDER_MASK = {
        'green': 0.90,
        'black': 0.95,
        'white': 0.80,
        'blue': 0.88,
        'red': 0.87,
        'matte_black': 0.97,
    }
    
    @classmethod
    def get_material(cls, category: str, name: str) -> Optional[MaterialProperties]:
        """Get material by category and name."""
        categories = {
            'substrate': cls.PCB_SUBSTRATES,
            'conductor': cls.CONDUCTORS,
            'heatsink': cls.HEATSINK_MATERIALS,
            'interface': cls.THERMAL_INTERFACE,
        }
        
        cat = categories.get(category.lower())
        if cat:
            return cat.get(name)
        return None


@dataclass
class PackageThermalData:
    """Thermal characteristics for IC packages."""
    theta_ja: float  # Junction-to-ambient (°C/W)
    theta_jc: float  # Junction-to-case (°C/W)
    theta_jb: float = 0.0  # Junction-to-board (°C/W)
    thermal_mass: float = 0.0  # J/K
    typical_area_mm2: float = 0.0


class ComponentThermalDatabase:
    """Thermal data for common component packages."""
    
    PACKAGES = {
        # QFP Family
        'LQFP-32': PackageThermalData(theta_ja=55.0, theta_jc=10.0, theta_jb=25.0, thermal_mass=0.08, typical_area_mm2=49),
        'LQFP-48': PackageThermalData(theta_ja=45.0, theta_jc=8.0, theta_jb=20.0, thermal_mass=0.12, typical_area_mm2=81),
        'LQFP-64': PackageThermalData(theta_ja=40.0, theta_jc=6.0, theta_jb=18.0, thermal_mass=0.15, typical_area_mm2=100),
        'LQFP-100': PackageThermalData(theta_ja=35.0, theta_jc=5.0, theta_jb=15.0, thermal_mass=0.20, typical_area_mm2=196),
        'LQFP-144': PackageThermalData(theta_ja=30.0, theta_jc=4.0, theta_jb=12.0, thermal_mass=0.25, typical_area_mm2=400),
        
        # QFN Family
        'QFN-16': PackageThermalData(theta_ja=45.0, theta_jc=4.0, theta_jb=15.0, thermal_mass=0.03, typical_area_mm2=9),
        'QFN-20': PackageThermalData(theta_ja=40.0, theta_jc=3.5, theta_jb=12.0, thermal_mass=0.04, typical_area_mm2=16),
        'QFN-24': PackageThermalData(theta_ja=38.0, theta_jc=3.0, theta_jb=10.0, thermal_mass=0.05, typical_area_mm2=16),
        'QFN-32': PackageThermalData(theta_ja=35.0, theta_jc=2.5, theta_jb=8.0, thermal_mass=0.06, typical_area_mm2=25),
        'QFN-48': PackageThermalData(theta_ja=30.0, theta_jc=2.0, theta_jb=6.0, thermal_mass=0.08, typical_area_mm2=49),
        'QFN-64': PackageThermalData(theta_ja=28.0, theta_jc=1.5, theta_jb=5.0, thermal_mass=0.10, typical_area_mm2=81),
        
        # BGA Family
        'BGA-64': PackageThermalData(theta_ja=40.0, theta_jc=8.0, theta_jb=15.0, thermal_mass=0.10, typical_area_mm2=49),
        'BGA-100': PackageThermalData(theta_ja=35.0, theta_jc=6.0, theta_jb=12.0, thermal_mass=0.15, typical_area_mm2=100),
        'BGA-144': PackageThermalData(theta_ja=30.0, theta_jc=5.0, theta_jb=10.0, thermal_mass=0.20, typical_area_mm2=169),
        'BGA-256': PackageThermalData(theta_ja=25.0, theta_jc=4.0, theta_jb=8.0, thermal_mass=0.30, typical_area_mm2=289),
        'BGA-400': PackageThermalData(theta_ja=20.0, theta_jc=3.0, theta_jb=6.0, thermal_mass=0.40, typical_area_mm2=529),
        
        # SOIC Family
        'SOIC-8': PackageThermalData(theta_ja=120.0, theta_jc=30.0, theta_jb=50.0, thermal_mass=0.02, typical_area_mm2=25),
        'SOIC-14': PackageThermalData(theta_ja=100.0, theta_jc=25.0, theta_jb=45.0, thermal_mass=0.03, typical_area_mm2=35),
        'SOIC-16': PackageThermalData(theta_ja=90.0, theta_jc=22.0, theta_jb=40.0, thermal_mass=0.035, typical_area_mm2=40),
        
        # SOT Family
        'SOT-23': PackageThermalData(theta_ja=230.0, theta_jc=100.0, theta_jb=150.0, thermal_mass=0.005, typical_area_mm2=3),
        'SOT-223': PackageThermalData(theta_ja=60.0, theta_jc=15.0, theta_jb=30.0, thermal_mass=0.02, typical_area_mm2=25),
        'SOT-89': PackageThermalData(theta_ja=100.0, theta_jc=25.0, theta_jb=50.0, thermal_mass=0.01, typical_area_mm2=12),
        
        # Power Packages
        'TO-220': PackageThermalData(theta_ja=50.0, theta_jc=1.5, theta_jb=5.0, thermal_mass=0.3, typical_area_mm2=150),
        'TO-252': PackageThermalData(theta_ja=80.0, theta_jc=3.0, theta_jb=8.0, thermal_mass=0.15, typical_area_mm2=40),
        'TO-263': PackageThermalData(theta_ja=45.0, theta_jc=2.0, theta_jb=6.0, thermal_mass=0.2, typical_area_mm2=90),
        
        # Passives
        '0201': PackageThermalData(theta_ja=500.0, theta_jc=200.0, theta_jb=300.0, thermal_mass=0.0001, typical_area_mm2=0.3),
        '0402': PackageThermalData(theta_ja=400.0, theta_jc=150.0, theta_jb=250.0, thermal_mass=0.0003, typical_area_mm2=0.5),
        '0603': PackageThermalData(theta_ja=300.0, theta_jc=100.0, theta_jb=200.0, thermal_mass=0.0005, typical_area_mm2=1.0),
        '0805': PackageThermalData(theta_ja=200.0, theta_jc=70.0, theta_jb=150.0, thermal_mass=0.001, typical_area_mm2=2.0),
        '1206': PackageThermalData(theta_ja=150.0, theta_jc=50.0, theta_jb=100.0, thermal_mass=0.002, typical_area_mm2=3.5),
        '2512': PackageThermalData(theta_ja=80.0, theta_jc=25.0, theta_jb=50.0, thermal_mass=0.005, typical_area_mm2=15),
    }
    
    @classmethod
    def get_package(cls, name: str) -> Optional[PackageThermalData]:
        """Get package thermal data by name."""
        return cls.PACKAGES.get(name)
    
    @classmethod
    def estimate_from_footprint(cls, footprint_name: str, area_mm2: float = 0) -> Optional[PackageThermalData]:
        """Estimate thermal data from footprint name or area."""
        name = footprint_name.upper()
        
        # Try direct match
        for pkg_name in cls.PACKAGES:
            if pkg_name in name:
                return cls.PACKAGES[pkg_name]
        
        # Estimate from area
        if area_mm2 > 0:
            if area_mm2 < 2:
                return cls.PACKAGES.get('0603')
            elif area_mm2 < 10:
                return cls.PACKAGES.get('0805')
            elif area_mm2 < 30:
                return cls.PACKAGES.get('SOIC-8')
            elif area_mm2 < 60:
                return cls.PACKAGES.get('QFN-32')
            elif area_mm2 < 120:
                return cls.PACKAGES.get('LQFP-64')
            else:
                return cls.PACKAGES.get('BGA-256')
        
        return None


class SimulationDefaults:
    """Default simulation parameters."""
    RESOLUTION_MM = 0.5
    USE_ADAPTIVE_MESH = True
    ADAPTIVE_FACTOR = 2.0
    TIMESTEP_S = 0.5
    DURATION_S = 600.0
    CONVERGENCE = 1e-6
    MAX_ITERATIONS = 10000
    AMBIENT_TEMP_C = 25.0
    CHAMBER_WALL_TEMP_C = 25.0
    COPPER_THICKNESS_UM = 35
    COPPER_INNER_UM = 35
    DIELECTRIC_THICKNESS_UM = 200


__all__ = [
    'PhysicalConstants', 'MaterialProperties', 'MaterialsDatabase',
    'PackageThermalData', 'ComponentThermalDatabase', 'SimulationDefaults',
]
