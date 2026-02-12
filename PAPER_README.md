# HydroCarbon Research Paper - Compilation Guide

This directory contains the research paper and related materials for the HydroCarbon environmental footprint prediction model.

## Files Overview

- `paper_hydrocarbon_model.tex` - Main research paper in LaTeX format
- `paper_diagrams.tex` - TikZ diagrams and figures (can be compiled separately)
- `paper_preamble.sty` - Custom LaTeX style file with packages and commands
- `generate_paper_supplementary.py` - Python script to generate supplementary data

## Requirements

### LaTeX Distribution
You'll need a full LaTeX installation with the following packages:
- `pdflatex` or `xelatex`
- `bibtex` or `biber` (for references)
- TikZ and PGF (for diagrams)
- `standalone` package (for separate diagram compilation)

### Installing LaTeX

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install texlive-full
```

**macOS:**
```bash
brew install --cask mactex
```

**Windows:**
Download and install MiKTeX from: https://miktex.org/download

## Compiling the Paper

### Quick Compilation (Single Command)

```bash
# Compile the main paper
pdflatex paper_hydrocarbon_model.tex
bibtex paper_hydrocarbon_model
pdflatex paper_hydrocarbon_model.tex
pdflatex paper_hydrocarbon_model.tex
```

Or use the provided script:
```bash
./compile_paper.sh
```

### Detailed Compilation Steps

1. **First Pass**: Process document structure and references
```bash
pdflatex paper_hydrocarbon_model.tex
```

2. **Bibliography**: Process references
```bash
bibtex paper_hydrocarbon_model
```

3. **Second Pass**: Incorporate bibliography
```bash
pdflatex paper_hydrocarbon_model.tex
```

4. **Final Pass**: Resolve all references and generate final PDF
```bash
pdflatex paper_hydrocarbon_model.tex
```

## Compiling Diagrams Separately

The diagrams can be compiled separately as standalone PDFs:

```bash
# Compile all diagrams
pdflatex -shell-escape paper_diagrams.tex

# Or compile specific diagrams
texdoc tikz
```

## Overleaf/ShareLaTeX

For online compilation:

1. Create a new project on Overleaf.com
2. Upload all three files:
   - `paper_hydrocarbon_model.tex`
   - `paper_diagrams.tex`
   - `paper_preamble.sty`
3. Click "Recompile" - Overleaf handles multiple compilation passes automatically

## Troubleshooting

### Common Issues

**1. Missing Packages**
```
! LaTeX Error: File `somepackage.sty' not found.
```
**Solution**: Install the missing package via your TeX package manager or use `tlmgr`:
```bash
tlmgr install somepackage
```

**2. Bibliography not showing**
**Solution**: Ensure you run `bibtex` between LaTeX compilations

**3. References showing as ??**
**Solution**: You need to compile multiple times (at least 3) for all references to resolve

**4. TikZ diagrams not compiling**
**Solution**: Ensure you have the `tikz` and `pgf` packages installed

### Compilation Script

For convenience, use the `compile_paper.sh` script:

```bash
#!/bin/bash
echo "Compiling HydroCarbon Research Paper..."
pdflatex paper_hydrocarbon_model.tex
bibtex paper_hydrocarbon_model
pdflatex paper_hydrocarbon_model.tex
pdflatex paper_hydrocarbon_model.tex
echo "Compilation complete! Output: paper_hydrocarbon_model.pdf"
```

Make it executable:
```bash
chmod +x compile_paper.sh
./compile_paper.sh
```

## Generating Supplementary Materials

Run the Python script to extract model details and generate supplementary data:

```bash
python generate_paper_supplementary.py
```

This will generate:
- `supplementary_material_factors.csv` - Material emission factors
- `supplementary_transport_params.csv` - Transport parameters
- `supplementary_model_performance.csv` - Model performance metrics
- `supplementary_feature_importance.csv` - Feature importance analysis

## Viewing the PDF

After successful compilation:

**Linux:**
```bash
xdg-open paper_hydrocarbon_model.pdf
```

**macOS:**
```bash
open paper_hydrocarbon_model.pdf
```

**Windows:**
```bash
start paper_hydrocarbon_model.pdf
```

## Paper Structure

The research paper includes:

1. **Introduction**: Motivation and key contributions
2. **Mathematical Foundations**: Detailed formulas for carbon and water footprints
3. **Model Architecture**: Hybrid physics-ML design
4. **Robustness Training**: Handling missing data
5. **Implementation Details**: C and Python implementation
6. **Data Sources**: Material and transport emission factors
7. **Evaluation and Results**: Performance metrics
8. **Use Cases**: Suitable applications and limitations
9. **Production Deployment**: API and performance characteristics
10. **Future Work**: Enhancement roadmap

## Customization

### Modifying the Paper

- Edit `paper_hydrocarbon_model.tex` for content changes
- Update `paper_preamble.sty` for formatting adjustments
- Modify diagram parameters in `paper_diagrams.tex`
- Add references in the `thebibliography` section

### Adding New Sections

```latex
\section{New Section Name}
\label{sec:newsection}

Your content here...

\subsection{Subsection}
\label{subsec:subsection}

More content...
```

### Updating Performance Metrics

Edit the tables in the "Evaluation and Results" section with new metrics:

```latex
\begin{table}[h]
\centering
\caption{New Results}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Old} & \textbf{New} \\ 
\midrule
$R^2$ & 0.999 & 0.9999 \\ 
MAE & 0.05 & 0.044 \\ 
\bottomrule
\end{tabular}
\end{table}
```

## Citation

If you use this paper or the HydroCarbon model in your research, please cite:

```bibtex
@techreport{hydrocarbon2025,
  title={{HydroCarbon: A Physics-Informed Machine Learning Model for Environmental Footprint Prediction in Fashion Products}},
  author={{Avelero Project}},
  year={2025},
  institution={Open Source},
  url={https://github.com/Avelero/Avelero_HydroCarbon}
}
```

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the LaTeX logs for specific errors
3. Open an issue on the GitHub repository
4. Contact the project maintainers

## License

The paper content is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

## Acknowledgments

- TU Delft Idemat database for material factors
- CE Delft for transport methodology
- Water Footprint Network for water data
- XGBoost team for the excellent ML library
- Google for Gemini API for synthetic data

---

**Happy Research! 🌍**
