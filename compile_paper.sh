#!/bin/bash

# HydroCarbon Research Paper Compilation Script
# This script compiles the LaTeX paper with all necessary passes

echo "============================================================"
echo "HydroCarbon Research Paper - LaTeX Compilation Script"
echo "============================================================"
echo ""

# Check if LaTeX is installed
if ! command -v pdflatex &> /dev/null; then
    echo "ERROR: pdflatex not found. Please install a LaTeX distribution."
    echo ""
    echo "Installation instructions:"
    echo "  Ubuntu/Debian: sudo apt-get install texlive-full"
    echo "  macOS: brew install --cask mactex"
    echo "  Windows: Download MiKTeX from https://miktex.org/download"
    exit 1
fi

echo "✓ LaTeX installation found"

# Check if the main tex file exists
if [ ! -f "paper_hydrocarbon_model.tex" ]; then
    echo "ERROR: paper_hydrocarbon_model.tex not found"
    exit 1
fi

echo "✓ Main paper file found"

# Clean previous build files
echo ""
echo "Cleaning previous build files..."
rm -f paper_hydrocarbon_model.aux
rm -f paper_hydrocarbon_model.bbl
rm -f paper_hydrocarbon_model.blg
rm -f paper_hydrocarbon_model.log
rm -f paper_hydrocarbon_model.out
rm -f paper_hydrocarbon_model.toc
rm -f paper_hydrocarbon_model.pdf

echo "✓ Cleaned build directory"

# First pass - process document structure
echo ""
echo "First compilation pass..."
pdflatex -interaction=nonstopmode paper_hydrocarbon_model.tex

if [ $? -ne 0 ]; then
    echo "ERROR: First LaTeX compilation failed"
    exit 1
fi

echo "✓ First pass complete"

# Check if bibliography needs to be processed
if grep -q "bibliography" paper_hydrocarbon_model.aux || grep -q "bibitem" paper_hydrocarbon_model.aux; then
    echo ""
    echo "Processing bibliography..."
    
    if command -v bibtex &> /dev/null; then
        bibtex paper_hydrocarbon_model
        echo "✓ Bibliography processed"
    else
        echo "⚠ bibtex not found, skipping bibliography"
    fi
else
    echo "⚠ No bibliography found, skipping bibtex"
fi

# Second pass - incorporate bibliography and references
echo ""
echo "Second compilation pass..."
pdflatex -interaction=nonstopmode paper_hydrocarbon_model.tex

if [ $? -ne 0 ]; then
    echo "ERROR: Second LaTeX compilation failed"
    exit 1
fi

echo "✓ Second pass complete"

# Third pass - resolve all references
echo ""
echo "Third compilation pass..."
pdflatex -interaction=nonstopmode paper_hydrocarbon_model.tex

if [ $? -ne 0 ]; then
    echo "ERROR: Third LaTeX compilation failed"
    exit 1
fi

echo "✓ Third pass complete"

# Check if PDF was created
if [ -f "paper_hydrocarbon_model.pdf" ]; then
    echo ""
    echo "============================================================"
    echo "✓ SUCCESS: Paper compiled successfully!"
    echo "============================================================"
    
    # Get file size
    file_size=$(ls -lh paper_hydrocarbon_model.pdf | awk '{print $5}')
    echo "Output file: paper_hydrocarbon_model.pdf ($file_size)"
    
    # Get page count (if available)
    if command -v pdfinfo &> /dev/null; then
        page_count=$(pdfinfo paper_hydrocarbon_model.pdf | grep Pages | awk '{print $2}')
        echo "Page count: $page_count pages"
    fi
    
    # Check if supplementary materials exist
    echo ""
    echo "Supplementary materials:"
    if [ -f "supplementary_material_factors.csv" ]; then
        echo "  ✓ Material factors data"
    fi
    if [ -f "supplementary_transport_params.csv" ]; then
        echo "  ✓ Transport parameters data"
    fi
    if [ -f "supplementary_model_performance.csv" ]; then
        echo "  ✓ Model performance data"
    fi
    if [ -f "feature_importance_plot.png" ]; then
        echo "  ✓ Feature importance chart"
    fi
    if [ -f "model_performance_chart.png" ]; then
        echo "  ✓ Performance comparison chart"
    fi
    
    echo ""
    echo "Next steps:"
    echo "  1. Review the PDF output"
    echo "  2. Check for any LaTeX warnings in the log file"
    echo "  3. Verify all figures and tables render correctly"
    echo "  4. Share the paper_hydrocarbon_model.pdf file"
    
else
    echo ""
    echo "============================================================"
    echo "✗ ERROR: PDF file was not created"
    echo "============================================================"
    exit 1
fi

# Optional: Open the PDF
echo ""
read -p "Would you like to open the PDF now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open paper_hydrocarbon_model.pdf
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        open paper_hydrocarbon_model.pdf
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        start paper_hydrocarbon_model.pdf
    fi
fi

echo ""
echo "============================================================"
echo "Compilation script finished!"
echo "============================================================"
echo ""

# Success indicator
exit 0
