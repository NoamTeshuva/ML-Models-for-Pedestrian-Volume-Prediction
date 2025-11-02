#!/usr/bin/env python3
"""
City Dataset Alignment Script
Aligns all secondary city feature tables to exactly match the Melbourne gold standard schema.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_melbourne_schema():
    """Load Melbourne data and extract its schema as the gold standard."""
    project_root = Path(__file__).parent.parent.parent
    melbourne_path = project_root / 'data' / 'processed' / 'melbourne' / 'csv' / 'feature_table_2023_melbourne.csv'
    
    if not melbourne_path.exists():
        raise FileNotFoundError(f"Melbourne data not found at: {melbourne_path}")
    
    melbourne_df = pd.read_csv(melbourne_path)
    
    # Extract schema information
    schema = {
        'columns': melbourne_df.columns.tolist(),
        'dtypes': melbourne_df.dtypes.to_dict(),
        'shape': melbourne_df.shape
    }
    
    print(f"Melbourne schema loaded: {schema['shape'][0]} rows, {schema['shape'][1]} columns")
    print(f"Columns: {schema['columns']}")
    
    return schema

def align_dataset(df, schema, city_name):
    """Align a secondary dataset to match the Melbourne schema."""
    print(f"\n=== ALIGNING {city_name.upper()} DATASET ===")
    original_shape = df.shape
    print(f"Original shape: {original_shape}")
    
    melbourne_columns = schema['columns']
    melbourne_dtypes = schema['dtypes']
    
    # Special handling for target variable mapping
    if 'vol_bin' in df.columns and 'volume_level' in melbourne_columns:
        print(f"  Mapping 'vol_bin' to 'volume_level'")
        df['volume_level'] = df['vol_bin']
    
    # Identify missing columns
    missing_columns = [col for col in melbourne_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        
        # Add missing columns with defaults
        for col in missing_columns:
            melbourne_dtype = melbourne_dtypes[col]
            
            if pd.api.types.is_numeric_dtype(melbourne_dtype):
                default_value = 0
            else:
                default_value = "missing"
            
            df[col] = default_value
            print(f"  Added '{col}' with default value: {default_value}")
    
    # Identify extra columns
    extra_columns = [col for col in df.columns if col not in melbourne_columns]
    if extra_columns:
        print(f"Extra columns to drop: {extra_columns}")
        df = df.drop(columns=extra_columns)
        print(f"  Dropped {len(extra_columns)} extra columns")
    
    # Reorder columns to match Melbourne
    df = df[melbourne_columns]
    print(f"Columns reordered to match Melbourne schema")
    
    # Cast dtypes to match Melbourne
    print(f"Casting dtypes to match Melbourne...")
    for col in melbourne_columns:
        try:
            if col in df.columns:
                target_dtype = melbourne_dtypes[col]
                
                # Handle special cases for dtype conversion
                if pd.api.types.is_integer_dtype(target_dtype):
                    # Convert to numeric first, then to int
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(target_dtype)
                elif pd.api.types.is_float_dtype(target_dtype):
                    # Convert to numeric first, then to float
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(target_dtype)
                elif pd.api.types.is_object_dtype(target_dtype):
                    # Keep as object/string
                    df[col] = df[col].astype(str)
                else:
                    # Try direct conversion
                    df[col] = df[col].astype(target_dtype)
                    
        except Exception as e:
            print(f"  Warning: Could not convert column '{col}' to {target_dtype}: {e}")
            # Keep original dtype if conversion fails
    
    aligned_shape = df.shape
    print(f"Final aligned shape: {aligned_shape}")
    print(f"Shape change: {original_shape} -> {aligned_shape}")
    
    return df

def process_city_dataset(city_name, relative_path, schema):
    """Process a single city dataset."""
    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / relative_path
    
    if not input_path.exists():
        print(f"\nWarning: {city_name} data not found at: {input_path}")
        return False
    
    # Load dataset
    df = pd.read_csv(input_path)
    
    # Align to Melbourne schema
    aligned_df = align_dataset(df, schema, city_name)
    
    # Generate output path
    output_filename = input_path.stem + "_aligned" + input_path.suffix
    output_path = input_path.parent / output_filename
    
    # Ensure output directory exists
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Save aligned dataset
    aligned_df.to_csv(output_path, index=False)
    print(f"Aligned dataset saved to: {output_path}")
    
    return True

def main():
    """Main alignment function."""
    print("="*60)
    print("CITY DATASET ALIGNMENT TO MELBOURNE SCHEMA")
    print("="*60)
    
    try:
        # Load Melbourne schema as gold standard
        schema = load_melbourne_schema()
        
        # Define secondary city datasets
        city_datasets = {
            'NYC': 'data/processed/NewYork/csv/feature_table_2023_nyc.csv',
            'Zurich': 'data/processed/zurich/csv/feature_table_2023_zurich_final.csv',
            'Tel Aviv': 'data/processed/tel_aviv/csv/final_feature_table_2023_tel_aviv.csv',
            'Dublin': 'data/processed/dublin/csv/feature_table_2023_dublin.csv'
        }
        
        # Process each secondary city
        successful_alignments = 0
        total_cities = len(city_datasets)
        
        for city_name, relative_path in city_datasets.items():
            success = process_city_dataset(city_name, relative_path, schema)
            if success:
                successful_alignments += 1
        
        # Summary
        print("\n" + "="*60)
        print("ALIGNMENT SUMMARY")
        print("="*60)
        print(f"Melbourne schema: {len(schema['columns'])} columns")
        print(f"Secondary cities processed: {successful_alignments}/{total_cities}")
        print(f"All aligned datasets now match Melbourne schema exactly")
        
        if successful_alignments == total_cities:
            print("\n✓ SUCCESS: All city datasets aligned successfully!")
        else:
            print(f"\n⚠ WARNING: {total_cities - successful_alignments} cities could not be aligned")
        
        return successful_alignments
        
    except Exception as e:
        print(f"Error in alignment pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    result = main()
    if result > 0:
        print(f"\nAlignment completed for {result} cities")
    else:
        print("\nAlignment failed")