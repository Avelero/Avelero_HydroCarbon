#!/usr/bin/env python3
"""
HydroCarbon Footprint Predictor - Preview Mode

A terminal-based program that allows users to input product details and get
carbon and water footprint predictions from the trained XGBoost model.

Input Fields:
- product_name: Name of the product (for display only)
- gender: Female/Male/Unisex
- parent_category: Top-level category (e.g., Tops, Bottoms, Footwear)
- category: Specific category (e.g., T-Shirts, Maxi Skirts)
- manufacturer_country: 2-letter country code (e.g., CN, BD, VN)
- materials: JSON dict of material percentages (e.g., {"cotton_conventional": 0.95, "elastane": 0.05})
- weight_kg: Product weight in kilograms
- total_distance_km: Total transport distance in kilometers

Output:
- carbon_material: Carbon footprint from materials (kgCO2e)
- carbon_transport: Carbon footprint from transport (kgCO2e)  
- carbon_total: Total carbon footprint (kgCO2e)
- water_total: Total water footprint (liters)
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import logging
import io
import contextlib

# Suppress warnings and configure logging before importing model modules
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore')

# Configure logging to suppress INFO messages from trainer
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logging.getLogger('trainer').setLevel(logging.WARNING)

# Add the models directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
sys.path.insert(0, str(MODELS_DIR))

from src.preprocessor import FootprintPreprocessor, MATERIAL_COLUMNS
from src.trainer import FootprintModelTrainer
from src.formula_features import add_formula_features


# Valid material names (must match preprocessor expectations)
VALID_MATERIALS = set(MATERIAL_COLUMNS)

# Map common material aliases to canonical names
MATERIAL_ALIASES = {
    'cotton': 'cotton_conventional',
    'polyester': 'polyester_virgin',
    'nylon': 'polyamide_6',
    'viscose': 'viscose',  # Note: this maps to viscose in the model columns
    'spandex': 'elastane',
    'lycra': 'elastane',
    'tencel': 'lyocell_tencel',
    'wool': 'wool_generic',
    'leather': 'leather_bovine',
    'down': 'down_feather',
    'rubber': 'rubber_synthetic',
    'polyamide': 'polyamide_6',
}


class FootprintPredictor:
    """Wrapper class for loading model and making predictions."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the predictor by loading the trained model and preprocessor.
        
        Args:
            model_path: Path to the model directory. Defaults to baseline model.
        """
        if model_path is None:
            model_path = SCRIPT_DIR / "trained_model" / "baseline"
        
        self.model_path = Path(model_path)
        self._load_model()
    
    def _load_model(self):
        """Load the XGBoost model and preprocessor."""
        print(f"Loading model from {self.model_path}...")
        
        # Load preprocessor
        preprocessor_path = self.model_path / "preprocessor.pkl"
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
        
        # Temporarily redirect stdout to suppress verbose loading messages
        with contextlib.redirect_stdout(io.StringIO()):
            self.preprocessor = FootprintPreprocessor.load(str(preprocessor_path))
        
        # Load trainer (model)
        if not (self.model_path / "xgb_model.json").exists():
            raise FileNotFoundError(f"Model not found at {self.model_path / 'xgb_model.json'}")
        
        with contextlib.redirect_stdout(io.StringIO()):
            self.trainer = FootprintModelTrainer.load(str(self.model_path))
        
        print("Model loaded successfully!\n")
    
    def parse_materials(self, materials_input: str) -> dict:
        """
        Parse materials input from JSON string.
        
        Args:
            materials_input: JSON string like '{"cotton_conventional": 0.95, "elastane": 0.05}'
            
        Returns:
            Dictionary of {material_name: percentage}
        """
        try:
            materials = json.loads(materials_input)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format for materials: {e}")
        
        if not isinstance(materials, dict):
            raise ValueError("Materials must be a JSON object/dictionary")
        
        # Normalize material names and validate
        normalized = {}
        for material, percentage in materials.items():
            # Check for aliases
            mat_lower = material.lower().strip()
            if mat_lower in MATERIAL_ALIASES:
                mat_lower = MATERIAL_ALIASES[mat_lower]
            
            # Validate material name
            if mat_lower not in VALID_MATERIALS:
                print(f"  Warning: Unknown material '{material}', using default carbon factor")
            
            # Validate percentage
            if not isinstance(percentage, (int, float)) or percentage < 0 or percentage > 1:
                raise ValueError(f"Material percentage must be between 0 and 1, got {percentage} for {material}")
            
            normalized[mat_lower] = float(percentage)
        
        # Check percentages sum to approximately 1
        total = sum(normalized.values())
        if abs(total - 1.0) > 0.01:
            print(f"  Warning: Material percentages sum to {total:.2f}, expected ~1.0")
        
        return normalized
    
    def prepare_input(
        self,
        product_name: str,
        gender: str,
        parent_category: str,
        category: str,
        manufacturer_country: str,
        materials: dict,
        weight_kg: float,
        total_distance_km: float
    ) -> pd.DataFrame:
        """
        Prepare user input as a DataFrame in the format expected by the preprocessor.
        
        Args:
            product_name: Name of the product
            gender: Gender category (Female/Male/Unisex)
            parent_category: Top-level category
            category: Specific category
            manufacturer_country: 2-letter country code
            materials: Dict of {material: percentage}
            weight_kg: Weight in kg
            total_distance_km: Distance in km
            
        Returns:
            DataFrame ready for preprocessing
        """
        # Create base row with categorical and numerical features
        row = {
            'gender': gender,
            'parent_category': parent_category,
            'category': category,
            'weight_kg': weight_kg,
            'total_distance_km': total_distance_km,
        }
        
        # Add one-hot encoded materials (all zeros initially)
        for mat_col in MATERIAL_COLUMNS:
            row[mat_col] = 0.0
        
        # Fill in the provided material percentages
        for material, percentage in materials.items():
            if material in MATERIAL_COLUMNS:
                row[material] = percentage
        
        # Create DataFrame
        df = pd.DataFrame([row])
        
        # Add formula features (required by the model) - suppress verbose output
        with contextlib.redirect_stdout(io.StringIO()):
            df = add_formula_features(df, MATERIAL_COLUMNS)
        
        return df
    
    def predict(
        self,
        product_name: str,
        gender: str,
        parent_category: str,
        category: str,
        manufacturer_country: str,
        materials: dict,
        weight_kg: float,
        total_distance_km: float
    ) -> dict:
        """
        Make a footprint prediction for a single product.
        
        Returns:
            Dict with carbon_material, carbon_transport, carbon_total, water_total
        """
        # Prepare input DataFrame
        X = self.prepare_input(
            product_name=product_name,
            gender=gender,
            parent_category=parent_category,
            category=category,
            manufacturer_country=manufacturer_country,
            materials=materials,
            weight_kg=weight_kg,
            total_distance_km=total_distance_km
        )
        
        # Apply preprocessing (transform using fitted preprocessor) - suppress verbose output
        with contextlib.redirect_stdout(io.StringIO()):
            X_processed = self.preprocessor.transform(X)
        
        # Make prediction
        predictions = self.trainer.predict(X_processed)
        
        # Extract results
        return {
            'carbon_material': predictions['carbon_material'].iloc[0],
            'carbon_transport': predictions['carbon_transport'].iloc[0],
            'carbon_total': predictions['carbon_total'].iloc[0],
            'water_total': predictions['water_total'].iloc[0],
        }


def print_banner():
    """Print the application banner."""
    print("=" * 70)
    print("  HydroCarbon Footprint Predictor - Preview Mode")
    print("  Predict carbon and water footprints for fashion products")
    print("=" * 70)
    print()


def print_valid_materials():
    """Print list of valid material names."""
    print("\nValid material names:")
    materials_per_row = 4
    materials = sorted(VALID_MATERIALS)
    for i in range(0, len(materials), materials_per_row):
        row = materials[i:i + materials_per_row]
        print("  " + ", ".join(row))
    print()


def get_user_input() -> dict:
    """
    Interactively get product details from user.
    
    Returns:
        Dict with all product fields
    """
    print("\nEnter product details (press Enter for defaults where shown):\n")
    
    # Product name
    product_name = input("Product name: ").strip()
    if not product_name:
        product_name = "Sample Product"
    
    # Gender
    print("\nGender options: Female, Male, Unisex")
    gender = input("Gender [Female]: ").strip()
    if not gender:
        gender = "Female"
    
    # Parent category
    print("\nParent category examples: Tops, Bottoms, Dresses, Outerwear, Footwear, Accessories")
    parent_category = input("Parent category [Bottoms]: ").strip()
    if not parent_category:
        parent_category = "Bottoms"
    
    # Category
    print("\nCategory examples: T-Shirts, Maxi Skirts, Jeans, Sneakers, Jackets")
    category = input("Category [Maxi Skirts]: ").strip()
    if not category:
        category = "Maxi Skirts"
    
    # Manufacturer country
    print("\nManufacturer country (2-letter code, e.g., CN, BD, VN, IN, TR)")
    manufacturer_country = input("Manufacturer country [CN]: ").strip().upper()
    if not manufacturer_country:
        manufacturer_country = "CN"
    
    # Materials
    print("\nMaterials as JSON dict (e.g., {\"cotton_conventional\": 0.95, \"elastane\": 0.05})")
    print("Type 'list' to see valid material names")
    materials_input = input("Materials: ").strip()
    while materials_input.lower() == 'list':
        print_valid_materials()
        materials_input = input("Materials: ").strip()
    
    if not materials_input:
        materials_input = '{"viscose": 0.72, "polyester_virgin": 0.28}'
        print(f"  Using default: {materials_input}")
    
    # Weight
    weight_input = input("\nWeight in kg [0.5]: ").strip()
    if not weight_input:
        weight_kg = 0.5
    else:
        weight_kg = float(weight_input)
    
    # Distance
    distance_input = input("Total transport distance in km [15000]: ").strip()
    if not distance_input:
        total_distance_km = 15000.0
    else:
        total_distance_km = float(distance_input)
    
    return {
        'product_name': product_name,
        'gender': gender,
        'parent_category': parent_category,
        'category': category,
        'manufacturer_country': manufacturer_country,
        'materials_input': materials_input,
        'weight_kg': weight_kg,
        'total_distance_km': total_distance_km,
    }


def parse_csv_line(line: str) -> dict:
    """
    Parse a CSV-formatted line with the expected columns.
    
    Expected format:
    product_name,gender,parent_category,category,manufacturer_country,materials,weight_kg,total_distance_km
    
    Example:
    Boho Floral Print Maxi Skirt,Female,Bottoms,Maxi Skirts,MQ,"{""viscose"":0.72,""polyester_virgin"":0.28}",0.587,14321.63
    """
    import csv
    from io import StringIO
    
    reader = csv.reader(StringIO(line))
    values = next(reader)
    
    if len(values) < 8:
        raise ValueError(f"Expected 8 columns, got {len(values)}")
    
    return {
        'product_name': values[0],
        'gender': values[1],
        'parent_category': values[2],
        'category': values[3],
        'manufacturer_country': values[4],
        'materials_input': values[5],
        'weight_kg': float(values[6]),
        'total_distance_km': float(values[7]),
    }


# Model accuracy metrics (from evaluation_report.json)
MODEL_ACCURACY = {
    'carbon_material': {'r2': 0.9999, 'mae': 0.041},
    'carbon_transport': {'r2': 0.9998, 'mae': 0.001},
    'carbon_total': {'r2': 0.9999, 'mae': 0.044},
    'water_total': {'r2': 0.9998, 'mae': 115.25},
}


def print_results(product_name: str, results: dict):
    """Print prediction results in a formatted way."""
    print("\n" + "=" * 60)
    print(f"  Prediction Results for: {product_name}")
    print("=" * 60)
    print(f"  carbon_material:  {results['carbon_material']:>10.4f} kgCO2e  (R²: {MODEL_ACCURACY['carbon_material']['r2']:.2%})")
    print(f"  carbon_transport: {results['carbon_transport']:>10.4f} kgCO2e  (R²: {MODEL_ACCURACY['carbon_transport']['r2']:.2%})")
    print(f"  carbon_total:     {results['carbon_total']:>10.4f} kgCO2e  (R²: {MODEL_ACCURACY['carbon_total']['r2']:.2%})")
    print(f"  water_total:      {results['water_total']:>10.1f} liters  (R²: {MODEL_ACCURACY['water_total']['r2']:.2%})")
    print("=" * 60)
    print("  Note: R² indicates model accuracy on validation data")


def interactive_mode(predictor: FootprintPredictor):
    """Run the predictor in interactive mode."""
    print_banner()
    
    while True:
        try:
            # Get user input
            user_input = get_user_input()
            
            # Parse materials
            materials = predictor.parse_materials(user_input['materials_input'])
            
            print("\nProcessing prediction...")
            
            # Make prediction
            results = predictor.predict(
                product_name=user_input['product_name'],
                gender=user_input['gender'],
                parent_category=user_input['parent_category'],
                category=user_input['category'],
                manufacturer_country=user_input['manufacturer_country'],
                materials=materials,
                weight_kg=user_input['weight_kg'],
                total_distance_km=user_input['total_distance_km'],
            )
            
            # Print results
            print_results(user_input['product_name'], results)
            
            # Continue?
            print()
            continue_input = input("Make another prediction? (y/n) [y]: ").strip().lower()
            if continue_input == 'n':
                break
            print()
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")


def csv_mode(predictor: FootprintPredictor, csv_line: str):
    """Process a single CSV line and output results."""
    try:
        # Parse CSV line
        data = parse_csv_line(csv_line)
        
        # Parse materials
        materials = predictor.parse_materials(data['materials_input'])
        
        # Make prediction
        results = predictor.predict(
            product_name=data['product_name'],
            gender=data['gender'],
            parent_category=data['parent_category'],
            category=data['category'],
            manufacturer_country=data['manufacturer_country'],
            materials=materials,
            weight_kg=data['weight_kg'],
            total_distance_km=data['total_distance_km'],
        )
        
        # Output with named fields
        print(f"carbon_material: {results['carbon_material']:.6f} kgCO2e")
        print(f"carbon_transport: {results['carbon_transport']:.6f} kgCO2e")
        print(f"carbon_total: {results['carbon_total']:.6f} kgCO2e")
        print(f"water_total: {results['water_total']:.2f} liters")
        print(f"")
        print(f"Model Accuracy (R²): carbon_material={MODEL_ACCURACY['carbon_material']['r2']:.2%}, carbon_transport={MODEL_ACCURACY['carbon_transport']['r2']:.2%}, carbon_total={MODEL_ACCURACY['carbon_total']['r2']:.2%}, water_total={MODEL_ACCURACY['water_total']['r2']:.2%}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def batch_mode(predictor: FootprintPredictor, input_file: str, output_file: str = None):
    """Process a CSV file in batch mode."""
    import csv
    
    print(f"Processing batch file: {input_file}")
    
    results_data = []
    
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            try:
                # Parse materials
                materials = predictor.parse_materials(row['materials'])
                
                # Make prediction
                results = predictor.predict(
                    product_name=row['product_name'],
                    gender=row['gender'],
                    parent_category=row['parent_category'],
                    category=row['category'],
                    manufacturer_country=row['manufacturer_country'],
                    materials=materials,
                    weight_kg=float(row['weight_kg']),
                    total_distance_km=float(row['total_distance_km']),
                )
                
                # Add to results
                results_data.append({
                    'product_name': row['product_name'],
                    **results
                })
                
            except Exception as e:
                print(f"Error processing {row.get('product_name', 'unknown')}: {e}", file=sys.stderr)
    
    # Output results
    if output_file:
        df = pd.DataFrame(results_data)
        df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
    else:
        # Print to stdout
        print("\nResults:")
        print("product_name,carbon_material,carbon_transport,carbon_total,water_total")
        for r in results_data:
            print(f"{r['product_name']},{r['carbon_material']:.6f},{r['carbon_transport']:.6f},{r['carbon_total']:.6f},{r['water_total']:.2f}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="HydroCarbon Footprint Predictor - Preview Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode:
    python preview.py
    
  Single prediction from CSV line:
    python preview.py --csv 'Boho Skirt,Female,Bottoms,Maxi Skirts,MQ,"{""viscose"":0.72,""polyester_virgin"":0.28}",0.587,14321.63'
    
  Quiet mode (only output the prediction):
    python preview.py -q --csv 'Boho Skirt,Female,Bottoms,Maxi Skirts,MQ,"{""viscose"":0.72,""polyester_virgin"":0.28}",0.587,14321.63'
    
  Batch processing:
    python preview.py --batch input.csv --output results.csv
    
  Use robustness model:
    python preview.py --model trained_model/robustness
"""
    )
    
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model directory (default: trained_model/baseline)')
    parser.add_argument('--csv', type=str, default=None,
                       help='Single CSV line to process')
    parser.add_argument('--batch', type=str, default=None,
                       help='CSV file for batch processing')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file for batch processing')
    parser.add_argument('--list-materials', action='store_true',
                       help='List valid material names and exit')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Quiet mode - suppress loading messages')
    
    args = parser.parse_args()
    
    if args.list_materials:
        print_valid_materials()
        return
    
    # Load model (suppress messages in quiet mode)
    if args.quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            predictor = FootprintPredictor(model_path=args.model)
    else:
        predictor = FootprintPredictor(model_path=args.model)
    
    if args.csv:
        # Single CSV line mode
        csv_mode(predictor, args.csv)
    elif args.batch:
        # Batch file mode
        batch_mode(predictor, args.batch, args.output)
    else:
        # Interactive mode
        interactive_mode(predictor)


if __name__ == '__main__':
    main()
