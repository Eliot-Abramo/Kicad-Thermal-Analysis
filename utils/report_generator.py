"""
TVAC Thermal Analyzer - Report Generator
========================================
Generate professional PDF thermal analysis reports.

Features:
- Executive summary with key findings
- Thermal map images
- Component temperature tables
- Simulation parameters
- Material properties summary

Author: Space Electronics Thermal Analysis Tool
Version: 2.0.0
"""

import os
import io
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, PageBreak, KeepTogether
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from .core.config import ThermalAnalysisConfig
from .core.pcb_extractor import PCBData
from .solvers.thermal_solver import ThermalResult, ThermalMesh


@dataclass
class ReportSettings:
    """Settings for report generation."""
    title: str = "TVAC Thermal Analysis Report"
    project_name: str = ""
    author: str = ""
    company: str = ""
    page_size: str = "letter"  # 'letter' or 'A4'
    include_thermal_map: bool = True
    include_component_table: bool = True
    include_material_summary: bool = True
    include_simulation_params: bool = True
    max_components_per_page: int = 40
    thermal_map_dpi: int = 150


class ThermalMapGenerator:
    """Generate thermal map images from simulation results."""
    
    # Color map: blue -> cyan -> green -> yellow -> red
    COLORMAP = [
        (0.0, (0, 0, 255)),      # Blue
        (0.25, (0, 255, 255)),   # Cyan
        (0.5, (0, 255, 0)),      # Green
        (0.75, (255, 255, 0)),   # Yellow
        (1.0, (255, 0, 0)),      # Red
    ]
    
    @classmethod
    def generate_thermal_image(cls, result: ThermalResult, mesh: ThermalMesh,
                               layer: int = 0, dpi: int = 150) -> Optional[bytes]:
        """Generate a thermal map image as PNG bytes."""
        if not HAS_NUMPY or not HAS_PIL:
            return None
        
        # Get temperature grid
        temp_grid = result.get_temperature_grid(mesh.nx, mesh.ny, mesh.nz, layer)
        if temp_grid is None:
            return None
        
        # Normalize to 0-1
        t_min, t_max = result.min_temp, result.max_temp
        if t_max - t_min < 0.1:
            t_max = t_min + 1.0
        
        normalized = (temp_grid - t_min) / (t_max - t_min)
        normalized = np.clip(normalized, 0, 1)
        
        # Map to colors
        h, w = normalized.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        for y in range(h):
            for x in range(w):
                v = normalized[y, x]
                rgb[y, x] = cls._value_to_color(v)
        
        # Create PIL image
        img = PILImage.fromarray(rgb, 'RGB')
        
        # Scale to reasonable size
        scale = max(1, 400 // max(w, h))
        if scale > 1:
            img = img.resize((w * scale, h * scale), PILImage.NEAREST)
        
        # Add color bar
        img = cls._add_colorbar(img, t_min, t_max)
        
        # Convert to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='PNG', dpi=(dpi, dpi))
        return buffer.getvalue()
    
    @classmethod
    def _value_to_color(cls, v: float) -> Tuple[int, int, int]:
        """Map normalized value to RGB color."""
        for i in range(len(cls.COLORMAP) - 1):
            v1, c1 = cls.COLORMAP[i]
            v2, c2 = cls.COLORMAP[i + 1]
            
            if v1 <= v <= v2:
                t = (v - v1) / (v2 - v1)
                r = int(c1[0] + t * (c2[0] - c1[0]))
                g = int(c1[1] + t * (c2[1] - c1[1]))
                b = int(c1[2] + t * (c2[2] - c1[2]))
                return (r, g, b)
        
        return cls.COLORMAP[-1][1]
    
    @classmethod
    def _add_colorbar(cls, img: 'PILImage.Image', t_min: float, t_max: float) -> 'PILImage.Image':
        """Add a color bar to the thermal image."""
        from PIL import ImageDraw, ImageFont
        
        # Create new image with space for colorbar
        w, h = img.size
        bar_width = 30
        margin = 50
        new_w = w + bar_width + margin
        
        new_img = PILImage.new('RGB', (new_w, h), (255, 255, 255))
        new_img.paste(img, (0, 0))
        
        draw = ImageDraw.Draw(new_img)
        
        # Draw colorbar
        bar_x = w + 10
        bar_top = 20
        bar_height = h - 40
        
        for y in range(bar_height):
            v = 1.0 - (y / bar_height)
            color = cls._value_to_color(v)
            draw.line([(bar_x, bar_top + y), (bar_x + 20, bar_top + y)], fill=color)
        
        # Draw border
        draw.rectangle([bar_x, bar_top, bar_x + 20, bar_top + bar_height], outline=(0, 0, 0))
        
        # Add labels
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        
        draw.text((bar_x + 25, bar_top - 5), f"{t_max:.1f}°C", fill=(0, 0, 0), font=font)
        draw.text((bar_x + 25, bar_top + bar_height - 10), f"{t_min:.1f}°C", fill=(0, 0, 0), font=font)
        
        mid_temp = (t_min + t_max) / 2
        draw.text((bar_x + 25, bar_top + bar_height // 2 - 5), f"{mid_temp:.1f}°C", fill=(0, 0, 0), font=font)
        
        return new_img


class ReportGenerator:
    """Generate PDF thermal analysis reports."""
    
    def __init__(self, settings: Optional[ReportSettings] = None):
        self.settings = settings or ReportSettings()
        
        if not HAS_REPORTLAB:
            raise RuntimeError("reportlab is required for PDF generation. "
                             "Install with: pip install reportlab")
    
    def generate(self, output_path: str,
                 config: ThermalAnalysisConfig,
                 pcb_data: PCBData,
                 result: ThermalResult,
                 mesh: ThermalMesh) -> bool:
        """
        Generate a complete thermal analysis report.
        
        Returns True on success.
        """
        try:
            page_size = A4 if self.settings.page_size.lower() == 'a4' else letter
            
            doc = SimpleDocTemplate(
                output_path,
                pagesize=page_size,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch
            )
            
            # Build content
            story = []
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                spaceBefore=20,
                spaceAfter=10
            )
            
            # Title page
            story.append(Spacer(1, 2*inch))
            story.append(Paragraph(self.settings.title, title_style))
            story.append(Spacer(1, 0.5*inch))
            
            if self.settings.project_name:
                story.append(Paragraph(f"Project: {self.settings.project_name}", styles['Heading2']))
            
            story.append(Spacer(1, 0.5*inch))
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
            
            if self.settings.author:
                story.append(Paragraph(f"Author: {self.settings.author}", styles['Normal']))
            if self.settings.company:
                story.append(Paragraph(f"Company: {self.settings.company}", styles['Normal']))
            
            story.append(PageBreak())
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", heading_style))
            
            summary_data = [
                ["Parameter", "Value"],
                ["Simulation Mode", config.simulation.mode.replace('_', ' ').title()],
                ["Minimum Temperature", f"{result.min_temp:.2f} °C"],
                ["Maximum Temperature", f"{result.max_temp:.2f} °C"],
                ["Average Temperature", f"{result.avg_temp:.2f} °C"],
                ["Temperature Range", f"{result.max_temp - result.min_temp:.2f} °C"],
                ["Ambient Temperature", f"{config.simulation.ambient_temp_c:.1f} °C"],
                ["Chamber Wall Temperature", f"{config.simulation.chamber_wall_temp_c:.1f} °C"],
                ["Total Components", str(len(pcb_data.components))],
                ["Components with Power", str(len(config.component_power))],
                ["Total Power Dissipation", f"{sum(cp.power_w for cp in config.component_power):.3f} W"],
                ["Compute Time", f"{result.compute_time:.2f} s"],
            ]
            
            summary_table = Table(summary_data, colWidths=[3*inch, 2.5*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976D2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F5F5')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('TOPPADDING', (0, 1), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Thermal findings
            findings = []
            if result.max_temp > 85:
                findings.append("⚠ Maximum temperature exceeds 85°C - may require thermal mitigation")
            if result.max_temp > 125:
                findings.append("🔴 CRITICAL: Maximum temperature exceeds 125°C - design revision recommended")
            if result.max_temp - result.min_temp > 50:
                findings.append("⚠ Large temperature gradient (>50°C) detected - check for thermal stress")
            if not findings:
                findings.append("✓ All temperatures within acceptable range")
            
            story.append(Paragraph("Key Findings:", styles['Heading3']))
            for finding in findings:
                story.append(Paragraph(f"• {finding}", styles['Normal']))
            
            story.append(PageBreak())
            
            # Thermal Map
            if self.settings.include_thermal_map and HAS_PIL:
                story.append(Paragraph("Thermal Distribution Map", heading_style))
                
                img_data = ThermalMapGenerator.generate_thermal_image(
                    result, mesh, layer=0, dpi=self.settings.thermal_map_dpi
                )
                
                if img_data:
                    img_buffer = io.BytesIO(img_data)
                    img = Image(img_buffer, width=5*inch, height=4*inch)
                    story.append(img)
                    story.append(Paragraph("Figure 1: Top layer temperature distribution", 
                                          styles['Italic']))
                else:
                    story.append(Paragraph("(Thermal map generation requires NumPy and PIL)", 
                                          styles['Italic']))
                
                story.append(PageBreak())
            
            # Component Temperature Table
            if self.settings.include_component_table and config.component_power:
                story.append(Paragraph("Component Power Dissipation", heading_style))
                
                comp_data = [["Reference", "Value", "Power (W)", "Source"]]
                
                # Sort by power
                sorted_power = sorted(config.component_power, 
                                     key=lambda x: x.power_w, reverse=True)
                
                for cp in sorted_power[:self.settings.max_components_per_page]:
                    # Find component value
                    value = ""
                    for comp in pcb_data.components:
                        if comp.reference == cp.reference:
                            value = comp.value
                            break
                    
                    comp_data.append([
                        cp.reference,
                        value[:20],
                        f"{cp.power_w:.4f}",
                        cp.source
                    ])
                
                if len(sorted_power) > self.settings.max_components_per_page:
                    comp_data.append([
                        f"... and {len(sorted_power) - self.settings.max_components_per_page} more",
                        "", "", ""
                    ])
                
                comp_table = Table(comp_data, colWidths=[1*inch, 2*inch, 1*inch, 1*inch])
                comp_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976D2')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')]),
                ]))
                story.append(comp_table)
                story.append(PageBreak())
            
            # Simulation Parameters
            if self.settings.include_simulation_params:
                story.append(Paragraph("Simulation Parameters", heading_style))
                
                sim = config.simulation
                params_data = [
                    ["Parameter", "Value"],
                    ["Mode", sim.mode.replace('_', ' ').title()],
                    ["Mesh Resolution", f"{sim.resolution_mm} mm"],
                    ["Adaptive Mesh", "Yes" if sim.use_adaptive_mesh else "No"],
                    ["Include Radiation", "Yes" if sim.include_radiation else "No"],
                    ["Ambient Temperature", f"{sim.ambient_temp_c} °C"],
                    ["Chamber Wall Temperature", f"{sim.chamber_wall_temp_c} °C"],
                    ["Initial Board Temperature", f"{sim.initial_board_temp_c} °C"],
                ]
                
                if sim.mode == "transient":
                    params_data.extend([
                        ["Duration", f"{sim.duration_s} s"],
                        ["Timestep", f"{sim.timestep_s} s"],
                        ["Output Interval", f"{sim.output_interval_s} s"],
                    ])
                
                params_table = Table(params_data, colWidths=[3*inch, 2*inch])
                params_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976D2')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')]),
                ]))
                story.append(params_table)
                story.append(Spacer(1, 0.3*inch))
                
                # Stackup info
                story.append(Paragraph("Board Stackup", styles['Heading3']))
                
                stackup_data = [["Layer", "Type", "Thickness (µm)", "Material"]]
                for layer in config.stackup.layers:
                    stackup_data.append([
                        layer.name,
                        layer.layer_type.title(),
                        str(layer.thickness_um),
                        layer.material
                    ])
                
                stackup_table = Table(stackup_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1.5*inch])
                stackup_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#607D8B')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                ]))
                story.append(stackup_table)
            
            # Build document
            doc.build(story)
            return True
            
        except Exception as e:
            print(f"Report generation error: {e}")
            import traceback
            traceback.print_exc()
            return False


def generate_report(output_path: str,
                    config: ThermalAnalysisConfig,
                    pcb_data: PCBData,
                    result: ThermalResult,
                    mesh: ThermalMesh,
                    settings: Optional[ReportSettings] = None) -> bool:
    """
    Convenience function to generate a thermal report.
    
    Args:
        output_path: Path for the output PDF file
        config: Thermal analysis configuration
        pcb_data: Extracted PCB data
        result: Simulation results
        mesh: Thermal mesh used in simulation
        settings: Optional report settings
    
    Returns:
        True if report was generated successfully
    """
    generator = ReportGenerator(settings)
    return generator.generate(output_path, config, pcb_data, result, mesh)


__all__ = ['ReportGenerator', 'ReportSettings', 'ThermalMapGenerator', 'generate_report']
