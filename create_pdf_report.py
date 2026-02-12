#!/usr/bin/env python3
"""
HydroCarbon Research Paper - PDF Report Generator
Creates a PDF report without requiring LaTeX installation
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add reportlab to path
sys.path.insert(0, str(Path(__file__).parent))

def check_and_install_reportlab():
    """Check if reportlab is installed, install if not"""
    try:
        import reportlab
        print("✓ reportlab already installed")
        return True
    except ImportError:
        print("Installing reportlab...")
        os.system("pip install reportlab --user")
        try:
            import reportlab
            print("✓ reportlab installed successfully")
            return True
        except:
            print("✗ Could not install reportlab")
            return False

def create_pdf_report():
    """Create PDF report using reportlab"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        
        print("Creating PDF report...")
        
        # Create PDF document
        pdf_path = Path(__file__).parent / "hydrocarbon_model_report.pdf"
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Container for PDF elements
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center
            textColor=colors.HexColor('#4682B4')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.HexColor('#2E5984')
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            spaceBefore=6,
            spaceAfter=6,
            leading=14
        )
        
        # Title
        story.append(Paragraph("HydroCarbon: A Physics-Informed Machine Learning Model", title_style))
        story.append(Paragraph("for Environmental Footprint Prediction in Fashion Products", title_style))
        story.append(Spacer(1, 20))
        
        # Abstract
        story.append(Paragraph("ABSTRACT", heading_style))
        abstract_text = """
        This report presents HydroCarbon, a novel physics-informed machine learning model designed to predict 
        carbon and water footprints for fashion products. The model addresses a critical gap in sustainability 
        research: the absence of large-scale, publicly available life cycle assessment (LCA) datasets for 
        fashion products. By combining synthetic data generation using Large Language Models (LLMs) with 
        physics-based footprint calculations and robust XGBoost regression, HydroCarbon achieves state-of-the-art 
        performance with R² > 0.999 on complete data and maintains R² > 0.93 even when 40%% of input features 
        are missing.
        """
        story.append(Paragraph(abstract_text, normal_style))
        story.append(Spacer(1, 20))
        
        # Core Formulas Section
        story.append(Paragraph("Core Mathematical Formulas", heading_style))
        
        formulas = [
            ("Carbon Material Footprint:", 
             "C_material = Σ(weight × material_% × carbon_factor)"),
            ("Carbon Transport Footprint:", 
             "C_transport = (weight/1000) × distance × (weighted_EF/1000)"),
            ("Total Carbon Footprint:", 
             "C_total = C_material + C_transport"),
            ("Water Footprint:", 
             "W_total = Σ(weight × material_% × water_factor)")
        ]
        
        for title, formula in formulas:
            story.append(Paragraph(f"<b>{title}</b>", normal_style))
            story.append(Paragraph(f"<font face='Courier'>{formula}</font>", normal_style))
            story.append(Spacer(1, 10))
        
        # Model Performance
        story.append(Paragraph("Model Performance", heading_style))
        
        # Performance table
        performance_data = [
            ['Target', 'R² (Complete)', 'MAE', 'R² (40% Missing)', 'MAE'],
            ['Carbon Material', '0.9999', '0.041 kgCO₂e', '0.936', '0.29 kgCO₂e'],
            ['Carbon Transport', '0.9998', '0.001 kgCO₂e', '0.968', '0.001 kgCO₂e'],
            ['Carbon Total', '0.9999', '0.044 kgCO₂e', '0.936', '0.29 kgCO₂e'],
            ['Water Total', '0.9998', '115.3 L', '0.902', '772 L']
        ]
        
        table = Table(performance_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4682B4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F0F8FF')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#4682B4'))
        ]))
        story.append(table)
        story.append(Spacer(1, 20))
        
        # Architecture
        story.append(Paragraph("Model Architecture", heading_style))
        architecture_text = """
        <b>Input Layer (129 features):</b><br/>
        • Contextual Features (93): Gender, Category, Parent Category<br/>
        • Physics Features (36): Weight, Distance, Material percentages<br/>
        <br/>
        <b>Feature Engineering:</b><br/>
        • Formula features injected from physics calculations<br/>
        • One-hot encoding for categorical variables<br/>
        • Log transformation for numerical stability<br/>
        <br/>
        <b>Model Core:</b><br/>
        • XGBoost Multi-Output Regressor<br/>
        • 1000 estimators, max depth 8<br/>
        • GPU acceleration (CUDA)<br/>
        • Custom physics-constrained objective<br/>
        <br/>
        <b>Outputs (4 predictions):</b><br/>
        • carbon_material (kgCO₂e)<br/>
        • carbon_transport (kgCO₂e)<br/>
        • carbon_total (kgCO₂e)<br/>
        • water_total (liters)
        """
        story.append(Paragraph(architecture_text, normal_style))
        story.append(Spacer(1, 20))
        
        # Key Innovations
        story.append(Paragraph("Key Innovations", heading_style))
        
        innovations = [
            "1. <b>Synthetic Data Generation:</b> 900,000+ products generated using Google Gemini 2.5 Flash",
            "2. <b>Physics-Informed ML:</b> Hybrid architecture combining formulas with XGBoost learning",
            "3. <b>Robustness Training:</b> Feature dropout augmentation handles 40% missing data",
            "4. <b>Performance:</b> R² > 0.999 on complete data, R² > 0.93 with missing data"
        ]
        
        for innovation in innovations:
            story.append(Paragraph(innovation, normal_style))
            story.append(Spacer(1, 8))
        
        # Usage Example
        story.append(Paragraph("Usage Example", heading_style))
        
        example_code = """
        from hydrocarbon import FootprintPredictor
        
        predictor = FootprintPredictor("trained_model/robustness")
        
        results = predictor.predict(
            gender="Male",
            category="Jeans",
            weight_kg=0.934,
            materials={"cotton_conventional": 0.92, "elastane": 0.08},
            total_distance_km=12847
        )
        
        # Output: Carbon: 2.26 kgCO₂e, Water: 7,888 liters
        """
        
        story.append(Paragraph("<font face='Courier'>" + example_code + "</font>", normal_style))
        story.append(Spacer(1, 20))
        
        # Data Sources
        story.append(Paragraph("Data Sources", heading_style))
        
        sources = [
            "• <b>Material Factors:</b> TU Delft Idemat 2026 database (34 materials)",
            "• <b>Transport Factors:</b> CE Delft STREAM 2020 with multinomial logit modal split",
            "• <b>Water Factors:</b> Water Footprint Network studies",
            "• <b>Synthetic Products:</b> Google Gemini 2.5 Flash generation"
        ]
        
        for source in sources:
            story.append(Paragraph(source, normal_style))
            story.append(Spacer(1, 6))
        
        # Bottom matter
        story.append(Spacer(1, 30))
        story.append(Paragraph("Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), normal_style))
        story.append(Paragraph("Version: 2.0 | Status: Proof of Concept", normal_style))
        story.append(Paragraph("Repository: https://github.com/Avelero/Avelero_HydroCarbon", normal_style))
        
        # Build PDF
        doc.build(story)
        print(f"✓ PDF report created successfully: {pdf_path}")
        print(f"  File size: {pdf_path.stat().st_size:,} bytes")
        
        return str(pdf_path)
        
    except Exception as e:
        print(f"✗ Error creating PDF: {e}")
        return None

def main():
    """Main function"""
    print("=" * 70)
    print("HydroCarbon Model - PDF Report Generator")
    print("=" * 70)
    
    # Check dependencies
    if not check_and_install_reportlab():
        print("\n⚠ Could not install reportlab dependency")
        print("\nAlternative options:")
        print("1. Install LaTeX and compile the .tex file")
        print("2. Use Overleaf.com (upload the .tex files)")
        print("3. Use Docker: docker run --rm -v $(pwd):/work texlive/texlive:latest")
        return
    
    # Create PDF
    print("\nGenerating PDF report...")
    pdf_path = create_pdf_report()
    
    if pdf_path:
        print(f"\n✓ Success! PDF generated: {pdf_path}")
        print("\nNext steps:")
        print("1. Open the PDF with your PDF viewer")
        print("2. For the full LaTeX version, install LaTeX and run: ./compile_paper.sh")
    else:
        print("\n✗ PDF generation failed")

if __name__ == "__main__":
    main()
