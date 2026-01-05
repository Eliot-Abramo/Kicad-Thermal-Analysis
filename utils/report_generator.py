"""
TVAC Thermal Analyzer - Report Generator
========================================
Generates professional PDF reports for thermal analysis results.

Author: Space Electronics Thermal Analysis Tool
Version: 1.0.0
"""

from __future__ import annotations

import os
import io
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass

import numpy as np

# Report generation imports
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, ListFlowable, ListItem, KeepTogether
    )
    from reportlab.graphics.shapes import Drawing, Rect, String, Line
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.charts.legends import Legend
    from reportlab.graphics.widgets.markers import makeMarker
    from reportlab.pdfgen import canvas
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Use TYPE_CHECKING to avoid circular imports

if TYPE_CHECKING:
    from ..solvers.thermal_solver import ThermalSimulationResult, ThermalFrame
    from ..solvers.current_solver import CurrentDistributionResult
    from ..solvers.mesh_generator import ThermalMesh
    from ..core.config import ThermalAnalysisConfig, SimulationParameters
    from ..core.pcb_extractor import PCBData

# Logger can be imported directly without circular issues
from .logger import get_logger


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str = "TVAC Thermal Analysis Report"
    author: str = ""
    project_name: str = ""
    document_number: str = ""
    revision: str = "A"
    
    # Content options
    include_executive_summary: bool = True
    include_configuration: bool = True
    include_current_analysis: bool = True
    include_thermal_results: bool = True
    include_component_analysis: bool = True
    include_time_history: bool = True
    include_thermal_maps: bool = True
    include_appendix: bool = True
    
    # Thermal map options
    thermal_map_times: List[float] = None  # Times for thermal maps (None = auto)
    thermal_map_layers: List[int] = None  # Layers to include (None = all)
    colormap: str = "jet"  # Color map for thermal images
    
    # Page settings
    page_size: str = "A4"  # "A4" or "letter"
    
    def __post_init__(self):
        if self.thermal_map_times is None:
            self.thermal_map_times = []
        if self.thermal_map_layers is None:
            self.thermal_map_layers = []


class ThermalReportGenerator:
    """Generates professional PDF reports for thermal analysis."""
    
    def __init__(self, config: ReportConfig = None):
        """Initialize report generator."""
        self.config = config or ReportConfig()
        self.logger = get_logger()
        
        if not HAS_REPORTLAB:
            self.logger.warning("reportlab not available - PDF generation disabled")
        
        # Styles
        self.styles = None
        self._setup_styles()
        
        # Color scheme (professional blue/gray)
        self.colors = {
            'primary': colors.HexColor('#1a365d'),      # Dark blue
            'secondary': colors.HexColor('#2c5282'),    # Medium blue
            'accent': colors.HexColor('#3182ce'),       # Light blue
            'success': colors.HexColor('#276749'),      # Green
            'warning': colors.HexColor('#c05621'),      # Orange
            'danger': colors.HexColor('#c53030'),       # Red
            'gray_dark': colors.HexColor('#2d3748'),
            'gray_medium': colors.HexColor('#718096'),
            'gray_light': colors.HexColor('#e2e8f0'),
            'white': colors.white,
        }
    
    def _setup_styles(self):
        """Setup document styles."""
        if not HAS_REPORTLAB:
            return
        
        self.styles = getSampleStyleSheet()
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1a365d'),
            spaceAfter=20,
            alignment=TA_CENTER,
        ))
        
        # Subtitle
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#718096'),
            spaceAfter=30,
            alignment=TA_CENTER,
        ))
        
        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#1a365d'),
            spaceBefore=20,
            spaceAfter=10,
            borderWidth=0,
            borderColor=colors.HexColor('#3182ce'),
            borderPadding=5,
        ))
        
        # Subsection header
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#2c5282'),
            spaceBefore=15,
            spaceAfter=8,
        ))
        
        # Body text
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#2d3748'),
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            leading=14,
        ))
        
        # Table header
        self.styles.add(ParagraphStyle(
            name='TableHeader',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.white,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
        ))
        
        # Table cell
        self.styles.add(ParagraphStyle(
            name='TableCell',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#2d3748'),
            alignment=TA_CENTER,
        ))
        
        # Caption
        self.styles.add(ParagraphStyle(
            name='Caption',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#718096'),
            alignment=TA_CENTER,
            spaceBefore=5,
            spaceAfter=15,
            fontName='Helvetica-Oblique',
        ))
    
    def generate_report(self, 
                       output_path: str,
                       thermal_results: ThermalResults,
                       current_results: Optional[CurrentDistributionResult],
                       mesh: ThermalMesh,
                       analysis_config: ThermalAnalysisConfig,
                       pcb_data: PCBData) -> bool:
        """
        Generate complete thermal analysis report.
        
        Args:
            output_path: Path for output PDF file
            thermal_results: Results from thermal simulation
            current_results: Results from current distribution analysis
            mesh: Thermal mesh used in simulation
            analysis_config: Configuration used for analysis
            pcb_data: PCB geometry data
        
        Returns:
            True if report generated successfully
        """
        if not HAS_REPORTLAB:
            self.logger.error("Cannot generate PDF - reportlab not installed")
            return False
        
        self.logger.info(f"Generating thermal analysis report: {output_path}")
        
        try:
            # Select page size
            page_size = A4 if self.config.page_size == "A4" else letter
            
            # Create document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=page_size,
                rightMargin=20*mm,
                leftMargin=20*mm,
                topMargin=25*mm,
                bottomMargin=20*mm,
            )
            
            # Build story (content)
            story = []
            
            # Title page
            story.extend(self._create_title_page(analysis_config, pcb_data))
            story.append(PageBreak())
            
            # Table of contents placeholder
            story.extend(self._create_toc())
            story.append(PageBreak())
            
            # Executive summary
            if self.config.include_executive_summary:
                story.extend(self._create_executive_summary(
                    thermal_results, current_results, analysis_config
                ))
                story.append(PageBreak())
            
            # Configuration section
            if self.config.include_configuration:
                story.extend(self._create_configuration_section(
                    analysis_config, pcb_data
                ))
                story.append(PageBreak())
            
            # Current analysis section
            if self.config.include_current_analysis and current_results:
                story.extend(self._create_current_analysis_section(
                    current_results, pcb_data
                ))
                story.append(PageBreak())
            
            # Thermal results section
            if self.config.include_thermal_results:
                story.extend(self._create_thermal_results_section(
                    thermal_results, mesh
                ))
                story.append(PageBreak())
            
            # Component analysis
            if self.config.include_component_analysis:
                story.extend(self._create_component_analysis_section(
                    thermal_results, pcb_data, analysis_config
                ))
                story.append(PageBreak())
            
            # Time history
            if self.config.include_time_history and thermal_results.frames:
                story.extend(self._create_time_history_section(thermal_results))
                story.append(PageBreak())
            
            # Thermal maps
            if self.config.include_thermal_maps and HAS_MATPLOTLIB:
                story.extend(self._create_thermal_maps_section(
                    thermal_results, mesh
                ))
                story.append(PageBreak())
            
            # Appendix
            if self.config.include_appendix:
                story.extend(self._create_appendix(
                    analysis_config, thermal_results, current_results
                ))
            
            # Build PDF
            doc.build(story, onFirstPage=self._add_header_footer,
                     onLaterPages=self._add_header_footer)
            
            self.logger.info(f"Report generated successfully: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return False
    
    def _create_title_page(self, config: ThermalAnalysisConfig, 
                          pcb_data: PCBData) -> List:
        """Create title page content."""
        story = []
        
        # Add spacing at top
        story.append(Spacer(1, 50*mm))
        
        # Title
        story.append(Paragraph(self.config.title, self.styles['ReportTitle']))
        
        # Subtitle with project info
        subtitle = f"Project: {self.config.project_name}" if self.config.project_name else ""
        if self.config.document_number:
            subtitle += f"<br/>Document: {self.config.document_number} Rev. {self.config.revision}"
        story.append(Paragraph(subtitle, self.styles['ReportSubtitle']))
        
        story.append(Spacer(1, 30*mm))
        
        # PCB info box
        pcb_info = [
            ['PCB Information', ''],
            ['Board Dimensions', f'{pcb_data.board_outline.width_mm:.1f} × {pcb_data.board_outline.height_mm:.1f} mm'],
            ['Layer Count', str(len(pcb_data.copper_layers))],
            ['Component Count', str(pcb_data.component_count)],
            ['Via Count', str(pcb_data.via_count)],
            ['Total Trace Length', f'{pcb_data.total_trace_length_mm:.1f} mm'],
        ]
        
        table = Table(pcb_info, colWidths=[60*mm, 60*mm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, self.colors['gray_medium']),
            ('BACKGROUND', (0, 1), (-1, -1), self.colors['gray_light']),
            ('ROWHEIGHT', (0, 0), (-1, -1), 8*mm),
            ('SPAN', (0, 0), (1, 0)),
        ]))
        story.append(table)
        
        story.append(Spacer(1, 40*mm))
        
        # Generation info
        gen_info = f"""
        <b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>Author:</b> {self.config.author or 'TVAC Thermal Analyzer'}<br/>
        <b>Analysis Date:</b> {config.modified_date[:10] if config.modified_date else 'N/A'}
        """
        story.append(Paragraph(gen_info, self.styles['BodyText']))
        
        return story
    
    def _create_toc(self) -> List:
        """Create table of contents."""
        story = []
        
        story.append(Paragraph("Table of Contents", self.styles['SectionHeader']))
        story.append(Spacer(1, 10*mm))
        
        # TOC entries (simplified - reportlab doesn't auto-generate TOC)
        toc_items = []
        section = 1
        
        if self.config.include_executive_summary:
            toc_items.append(f"{section}. Executive Summary")
            section += 1
        if self.config.include_configuration:
            toc_items.append(f"{section}. Analysis Configuration")
            section += 1
        if self.config.include_current_analysis:
            toc_items.append(f"{section}. Current Distribution Analysis")
            section += 1
        if self.config.include_thermal_results:
            toc_items.append(f"{section}. Thermal Simulation Results")
            section += 1
        if self.config.include_component_analysis:
            toc_items.append(f"{section}. Component Temperature Analysis")
            section += 1
        if self.config.include_time_history:
            toc_items.append(f"{section}. Temperature Time History")
            section += 1
        if self.config.include_thermal_maps:
            toc_items.append(f"{section}. Thermal Distribution Maps")
            section += 1
        if self.config.include_appendix:
            toc_items.append(f"{section}. Appendix")
        
        for item in toc_items:
            story.append(Paragraph(item, self.styles['BodyText']))
        
        return story
    
    def _create_executive_summary(self, thermal_results: ThermalResults,
                                  current_results: Optional[CurrentDistributionResult],
                                  config: ThermalAnalysisConfig) -> List:
        """Create executive summary section."""
        story = []
        
        story.append(Paragraph("1. Executive Summary", self.styles['SectionHeader']))
        
        # Key findings
        story.append(Paragraph("1.1 Key Findings", self.styles['SubsectionHeader']))
        
        findings = []
        
        # Temperature summary
        if thermal_results.final_temperature is not None:
            t_min = float(np.min(thermal_results.final_temperature))
            t_max = float(np.max(thermal_results.final_temperature))
            t_avg = float(np.mean(thermal_results.final_temperature))
            
            findings.append(f"• Maximum board temperature: <b>{t_max:.1f}°C</b>")
            findings.append(f"• Minimum board temperature: <b>{t_min:.1f}°C</b>")
            findings.append(f"• Average board temperature: <b>{t_avg:.1f}°C</b>")
            findings.append(f"• Temperature range: <b>{t_max - t_min:.1f}°C</b>")
        
        # Current summary
        if current_results:
            findings.append(f"• Maximum trace current: <b>{current_results.max_current_a:.3f}A</b>")
            findings.append(f"• Total Joule heating: <b>{current_results.total_power_w:.3f}W</b>")
        
        # Steady state
        if thermal_results.steady_state_reached:
            findings.append(f"• Steady state reached at: <b>{thermal_results.steady_state_time:.1f}s</b>")
        
        for finding in findings:
            story.append(Paragraph(finding, self.styles['BodyText']))
        
        story.append(Spacer(1, 5*mm))
        
        # Summary table
        story.append(Paragraph("1.2 Results Summary", self.styles['SubsectionHeader']))
        
        summary_data = [
            ['Parameter', 'Value', 'Unit'],
            ['Simulation Duration', f'{config.simulation.duration_s:.0f}', 's'],
            ['Time Step', f'{config.simulation.timestep_s:.3f}', 's'],
            ['Ambient Temperature', f'{config.simulation.ambient_temp_c:.1f}', '°C'],
            ['Chamber Wall Temperature', f'{config.simulation.chamber_wall_temp_c:.1f}', '°C'],
            ['Computation Time', f'{thermal_results.total_compute_time:.1f}', 's'],
        ]
        
        if thermal_results.final_temperature is not None:
            summary_data.extend([
                ['Final Max Temperature', f'{np.max(thermal_results.final_temperature):.1f}', '°C'],
                ['Final Avg Temperature', f'{np.mean(thermal_results.final_temperature):.1f}', '°C'],
            ])
        
        table = self._create_styled_table(summary_data)
        story.append(table)
        
        # Warnings
        if thermal_results.warnings or (current_results and current_results.warnings):
            story.append(Spacer(1, 5*mm))
            story.append(Paragraph("1.3 Warnings", self.styles['SubsectionHeader']))
            
            all_warnings = thermal_results.warnings.copy()
            if current_results:
                all_warnings.extend(current_results.warnings)
            
            if all_warnings:
                for warning in all_warnings:
                    story.append(Paragraph(f"⚠ {warning}", self.styles['BodyText']))
            else:
                story.append(Paragraph("No warnings generated during analysis.", 
                                      self.styles['BodyText']))
        
        return story
    
    def _create_configuration_section(self, config: ThermalAnalysisConfig,
                                      pcb_data: PCBData) -> List:
        """Create configuration section."""
        story = []
        
        story.append(Paragraph("2. Analysis Configuration", self.styles['SectionHeader']))
        
        # PCB Stackup
        story.append(Paragraph("2.1 PCB Stackup", self.styles['SubsectionHeader']))
        
        stackup_data = [
            ['Parameter', 'Value'],
            ['Total Thickness', f'{config.stackup.total_thickness_mm:.2f} mm'],
            ['Layer Count', str(config.stackup.layer_count)],
            ['Substrate Material', config.stackup.substrate_material],
            ['Surface Finish', config.stackup.surface_finish],
            ['Solder Mask', config.stackup.solder_mask_color],
        ]
        
        story.append(self._create_styled_table(stackup_data))
        story.append(Spacer(1, 5*mm))
        
        # Copper thickness per layer
        story.append(Paragraph("2.2 Copper Thickness", self.styles['SubsectionHeader']))
        
        cu_data = [['Layer', 'Thickness (µm)']]
        for layer, thickness in config.stackup.copper_thickness_um.items():
            cu_data.append([layer, f'{thickness:.1f}'])
        
        story.append(self._create_styled_table(cu_data))
        story.append(Spacer(1, 5*mm))
        
        # Simulation parameters
        story.append(Paragraph("2.3 Simulation Parameters", self.styles['SubsectionHeader']))
        
        sim = config.simulation
        sim_data = [
            ['Parameter', 'Value'],
            ['Grid Resolution', f'{sim.resolution_mm:.2f} mm'],
            ['3D Simulation', 'Yes' if sim.simulation_3d else 'No'],
            ['Include Radiation', 'Yes' if sim.include_radiation else 'No'],
            ['AC Effects', 'Yes' if sim.include_ac_effects else 'No'],
            ['Simulation Mode', sim.simulation_mode.replace('_', ' ').title()],
            ['Duration', f'{sim.duration_s:.0f} s'],
            ['Time Step', f'{sim.timestep_s:.3f} s'],
            ['Convergence Criterion', f'{sim.convergence_criterion:.0e}'],
        ]
        
        story.append(self._create_styled_table(sim_data))
        story.append(Spacer(1, 5*mm))
        
        # Current injection points
        if config.current_injection_points:
            story.append(Paragraph("2.4 Current Injection Points", self.styles['SubsectionHeader']))
            
            current_data = [['ID', 'Net', 'Position (mm)', 'Layer', 'Current (A)', 'Description']]
            for pt in config.current_injection_points:
                current_data.append([
                    pt.point_id,
                    pt.net_name or 'Any',
                    f'({pt.x_mm:.1f}, {pt.y_mm:.1f})',
                    pt.layer or 'Any',
                    f'{pt.current_a:+.3f}',
                    pt.description[:20] + '...' if len(pt.description) > 20 else pt.description
                ])
            
            story.append(self._create_styled_table(current_data, col_widths=[15*mm, 25*mm, 30*mm, 20*mm, 20*mm, 40*mm]))
        
        return story
    
    def _create_current_analysis_section(self, results: CurrentDistributionResult,
                                         pcb_data: PCBData) -> List:
        """Create current analysis section."""
        story = []
        
        story.append(Paragraph("3. Current Distribution Analysis", self.styles['SectionHeader']))
        
        # Summary
        story.append(Paragraph("3.1 Summary", self.styles['SubsectionHeader']))
        
        summary_text = f"""
        The current distribution analysis solved for current flow through the PCB copper network
        using nodal analysis (Kirchhoff's current law). The analysis identified 
        <b>{len(results.segment_currents)}</b> trace segments with the following key results:
        """
        story.append(Paragraph(summary_text, self.styles['BodyText']))
        
        summary_data = [
            ['Metric', 'Value', 'Unit'],
            ['Maximum Current', f'{results.max_current_a:.4f}', 'A'],
            ['Total Joule Heating', f'{results.total_power_w:.4f}', 'W'],
            ['Max Current Segment', results.max_current_segment, '-'],
        ]
        
        story.append(self._create_styled_table(summary_data))
        story.append(Spacer(1, 5*mm))
        
        # High current traces
        story.append(Paragraph("3.2 Highest Current Traces", self.styles['SubsectionHeader']))
        
        # Sort by current and get top 20
        sorted_segments = sorted(
            results.segment_currents.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:20]
        
        high_current_data = [['Segment', 'Current (A)', 'Power (mW)', 'Voltage Drop (mV)']]
        for seg_id, current in sorted_segments:
            power = results.segment_power.get(seg_id, 0) * 1000  # Convert to mW
            v_drop = results.segment_voltage_drop.get(seg_id, 0) * 1000  # Convert to mV
            high_current_data.append([
                seg_id,
                f'{current:.4f}',
                f'{power:.3f}',
                f'{v_drop:.3f}'
            ])
        
        story.append(self._create_styled_table(high_current_data))
        
        return story
    
    def _create_thermal_results_section(self, results: ThermalResults,
                                        mesh: ThermalMesh) -> List:
        """Create thermal results section."""
        story = []
        
        story.append(Paragraph("4. Thermal Simulation Results", self.styles['SectionHeader']))
        
        # Overview
        story.append(Paragraph("4.1 Overview", self.styles['SubsectionHeader']))
        
        if results.final_temperature is not None:
            t_min = float(np.min(results.final_temperature))
            t_max = float(np.max(results.final_temperature))
            t_avg = float(np.mean(results.final_temperature))
            t_std = float(np.std(results.final_temperature))
        else:
            t_min = t_max = t_avg = t_std = 0
        
        overview_data = [
            ['Statistic', 'Value', 'Unit'],
            ['Minimum Temperature', f'{t_min:.2f}', '°C'],
            ['Maximum Temperature', f'{t_max:.2f}', '°C'],
            ['Average Temperature', f'{t_avg:.2f}', '°C'],
            ['Standard Deviation', f'{t_std:.2f}', '°C'],
            ['Temperature Range', f'{t_max - t_min:.2f}', '°C'],
            ['Mesh Nodes', str(len(mesh.nodes)), '-'],
            ['Grid Resolution', f'{mesh.dx_mm:.2f} × {mesh.dy_mm:.2f}', 'mm'],
        ]
        
        if results.steady_state_reached:
            overview_data.append(['Steady State Time', f'{results.steady_state_time:.1f}', 's'])
        
        story.append(self._create_styled_table(overview_data))
        story.append(Spacer(1, 5*mm))
        
        # Temperature distribution statistics
        story.append(Paragraph("4.2 Temperature Distribution", self.styles['SubsectionHeader']))
        
        if results.final_temperature is not None:
            # Create histogram data
            hist, bin_edges = np.histogram(results.final_temperature, bins=10)
            
            hist_data = [['Temperature Range (°C)', 'Node Count', 'Percentage']]
            total_nodes = len(results.final_temperature)
            for i in range(len(hist)):
                range_str = f'{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}'
                pct = hist[i] / total_nodes * 100
                hist_data.append([range_str, str(hist[i]), f'{pct:.1f}%'])
            
            story.append(self._create_styled_table(hist_data))
        
        return story
    
    def _create_component_analysis_section(self, results: ThermalResults,
                                           pcb_data: PCBData,
                                           config: ThermalAnalysisConfig) -> List:
        """Create component analysis section."""
        story = []
        
        story.append(Paragraph("5. Component Temperature Analysis", self.styles['SectionHeader']))
        
        # Component temperatures
        story.append(Paragraph("5.1 Component Temperatures", self.styles['SubsectionHeader']))
        
        comp_data = [['Reference', 'Power (W)', 'Max Temp (°C)', 'Status']]
        
        # Get power config
        power_map = {c.reference: c.power_w for c in config.component_power}
        
        # Sort by temperature (highest first)
        sorted_comps = sorted(
            results.component_max_temps.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for ref, max_temp in sorted_comps[:30]:  # Limit to top 30
            power = power_map.get(ref, 0)
            
            # Status based on temperature thresholds
            if max_temp > 100:
                status = '⚠ HIGH'
            elif max_temp > 85:
                status = 'Warm'
            else:
                status = 'OK'
            
            comp_data.append([
                ref,
                f'{power:.3f}' if power > 0 else '-',
                f'{max_temp:.1f}',
                status
            ])
        
        if len(comp_data) > 1:
            story.append(self._create_styled_table(comp_data))
        else:
            story.append(Paragraph("No component temperature data available.", 
                                  self.styles['BodyText']))
        
        return story
    
    def _create_time_history_section(self, results: ThermalResults) -> List:
        """Create time history section."""
        story = []
        
        story.append(Paragraph("6. Temperature Time History", self.styles['SectionHeader']))
        
        if not results.frames:
            story.append(Paragraph("No time history data available.", self.styles['BodyText']))
            return story
        
        # Create temperature history plot
        if HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=(7, 4))
            
            times = results.time_points
            ax.plot(times, results.max_temp_history, 'r-', label='Max', linewidth=1.5)
            ax.plot(times, results.avg_temp_history, 'b-', label='Average', linewidth=1.5)
            ax.plot(times, results.min_temp_history, 'g-', label='Min', linewidth=1.5)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Temperature (°C)')
            ax.set_title('Board Temperature vs Time')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Save to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            
            # Add to story
            img = Image(buf, width=160*mm, height=90*mm)
            story.append(img)
            story.append(Paragraph("Figure: Temperature evolution during simulation", 
                                  self.styles['Caption']))
        
        # Time history table
        story.append(Paragraph("6.1 Key Time Points", self.styles['SubsectionHeader']))
        
        # Select key time points
        n_frames = len(results.frames)
        indices = [0]
        if n_frames > 4:
            indices.extend([n_frames // 4, n_frames // 2, 3 * n_frames // 4])
        indices.append(n_frames - 1)
        indices = sorted(set(indices))
        
        time_data = [['Time (s)', 'Min (°C)', 'Max (°C)', 'Avg (°C)', 'Range (°C)']]
        for i in indices:
            frame = results.frames[i]
            time_data.append([
                f'{frame.time_s:.1f}',
                f'{frame.min_temp:.2f}',
                f'{frame.max_temp:.2f}',
                f'{frame.avg_temp:.2f}',
                f'{frame.max_temp - frame.min_temp:.2f}'
            ])
        
        story.append(self._create_styled_table(time_data))
        
        return story
    
    def _create_thermal_maps_section(self, results: ThermalResults,
                                     mesh: ThermalMesh) -> List:
        """Create thermal maps section with heat map images."""
        story = []
        
        story.append(Paragraph("7. Thermal Distribution Maps", self.styles['SectionHeader']))
        
        if not HAS_MATPLOTLIB:
            story.append(Paragraph("Matplotlib not available - cannot generate thermal maps.",
                                  self.styles['BodyText']))
            return story
        
        # Determine time points for maps
        if self.config.thermal_map_times:
            time_points = self.config.thermal_map_times
        else:
            # Auto-select: initial, 25%, 50%, 75%, final
            if results.frames:
                n = len(results.frames)
                indices = [0, n//4, n//2, 3*n//4, n-1]
                time_points = [results.frames[i].time_s for i in indices if i < n]
            else:
                time_points = [0]
        
        # Determine layers
        if self.config.thermal_map_layers:
            layers = self.config.thermal_map_layers
        else:
            layers = [0]  # Top layer only by default
            if mesh.nz > 1:
                layers.append(mesh.nz - 1)  # Also bottom layer
        
        # Generate maps
        for layer in layers:
            layer_name = "Top" if layer == 0 else ("Bottom" if layer == mesh.nz - 1 else f"Layer {layer}")
            story.append(Paragraph(f"7.{layer+1} {layer_name} Layer", self.styles['SubsectionHeader']))
            
            for t in time_points:
                frame = results.get_frame_at_time(t)
                if frame is None:
                    continue
                
                # Get temperature grid
                grid = np.zeros((mesh.ny, mesh.nx))
                for ix in range(mesh.nx):
                    for iy in range(mesh.ny):
                        idx = mesh.node_index_map.get((ix, iy, layer))
                        if idx is not None and idx < len(frame.temperature):
                            grid[iy, ix] = frame.temperature[idx]
                
                # Create figure
                fig, ax = plt.subplots(figsize=(6, 5))
                
                # Custom colormap (blue to red through yellow)
                cmap = plt.cm.get_cmap(self.config.colormap)
                
                im = ax.imshow(grid, cmap=cmap, origin='lower',
                              extent=[mesh.x_min, mesh.x_max, mesh.y_min, mesh.y_max])
                
                cbar = plt.colorbar(im, ax=ax, label='Temperature (°C)')
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                ax.set_title(f'{layer_name} at t = {t:.1f}s')
                ax.set_aspect('equal')
                
                # Save to buffer
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                plt.close(fig)
                
                # Add to story
                img = Image(buf, width=140*mm, height=115*mm)
                story.append(img)
                story.append(Paragraph(
                    f"Figure: Thermal distribution - {layer_name} layer at t = {t:.1f}s "
                    f"(Tmin={frame.min_temp:.1f}°C, Tmax={frame.max_temp:.1f}°C)",
                    self.styles['Caption']
                ))
        
        return story
    
    def _create_appendix(self, config: ThermalAnalysisConfig,
                        thermal_results: ThermalResults,
                        current_results: Optional[CurrentDistributionResult]) -> List:
        """Create appendix with detailed data."""
        story = []
        
        story.append(Paragraph("Appendix", self.styles['SectionHeader']))
        
        # A. Material properties used
        story.append(Paragraph("A. Material Properties", self.styles['SubsectionHeader']))
        
        from ..core.constants import MaterialsDatabase
        
        substrate = MaterialsDatabase.PCB_SUBSTRATES.get(
            config.stackup.substrate_material,
            MaterialsDatabase.PCB_SUBSTRATES['FR4']
        )
        copper = MaterialsDatabase.CONDUCTORS['COPPER']
        
        mat_data = [
            ['Material', 'k (W/m·K)', 'ρ (kg/m³)', 'c (J/kg·K)', 'ε'],
            [substrate.name, f'{substrate.thermal_conductivity:.2f}', 
             f'{substrate.density:.0f}', f'{substrate.specific_heat:.0f}',
             f'{substrate.emissivity:.2f}'],
            [copper.name, f'{copper.thermal_conductivity:.1f}',
             f'{copper.density:.0f}', f'{copper.specific_heat:.0f}',
             f'{copper.emissivity:.2f}'],
        ]
        
        story.append(self._create_styled_table(mat_data))
        story.append(Spacer(1, 5*mm))
        
        # B. Complete configuration dump
        story.append(Paragraph("B. Configuration Parameters", self.styles['SubsectionHeader']))
        
        config_text = config.export_config_summary() if hasattr(config, 'export_config_summary') else str(config.to_dict())
        
        # Truncate if too long
        if len(config_text) > 3000:
            config_text = config_text[:3000] + "\n... (truncated)"
        
        story.append(Paragraph(f"<pre>{config_text}</pre>", self.styles['BodyText']))
        
        return story
    
    def _create_styled_table(self, data: List[List], col_widths=None) -> Table:
        """Create a consistently styled table."""
        if col_widths is None:
            col_widths = None  # Auto-size
        
        table = Table(data, colWidths=col_widths)
        
        style = TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            
            # Data rows
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            
            # Alternating row colors
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.colors['gray_light']]),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, self.colors['gray_medium']),
            
            # Padding
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ])
        
        table.setStyle(style)
        return table
    
    def _add_header_footer(self, canvas_obj, doc):
        """Add header and footer to each page."""
        canvas_obj.saveState()
        
        # Header
        canvas_obj.setFillColor(self.colors['primary'])
        canvas_obj.setFont('Helvetica-Bold', 9)
        canvas_obj.drawString(20*mm, doc.pagesize[1] - 15*mm, self.config.title)
        
        if self.config.document_number:
            canvas_obj.setFont('Helvetica', 8)
            canvas_obj.drawRightString(
                doc.pagesize[0] - 20*mm,
                doc.pagesize[1] - 15*mm,
                f"{self.config.document_number} Rev. {self.config.revision}"
            )
        
        # Header line
        canvas_obj.setStrokeColor(self.colors['accent'])
        canvas_obj.setLineWidth(1)
        canvas_obj.line(20*mm, doc.pagesize[1] - 18*mm,
                       doc.pagesize[0] - 20*mm, doc.pagesize[1] - 18*mm)
        
        # Footer
        canvas_obj.setFillColor(self.colors['gray_medium'])
        canvas_obj.setFont('Helvetica', 8)
        
        # Page number
        page_num = canvas_obj.getPageNumber()
        canvas_obj.drawCentredString(doc.pagesize[0] / 2, 10*mm, f"Page {page_num}")
        
        # Footer line
        canvas_obj.line(20*mm, 15*mm, doc.pagesize[0] - 20*mm, 15*mm)
        
        # Confidentiality notice
        canvas_obj.setFont('Helvetica-Oblique', 7)
        canvas_obj.drawString(20*mm, 10*mm, "Generated by TVAC Thermal Analyzer")
        
        canvas_obj.restoreState()


__all__ = [
    'ReportConfig',
    'ThermalReportGenerator',
]
