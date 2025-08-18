#!/usr/bin/env python3
"""
Diagnostic Runner for Hyperblock Classification Algorithm

This program runs the hyperblock classification algorithm on multiple datasets
and generates a performance summary table.
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime

# Dataset configurations
DATASETS = {
    'Fisher Iris': 'datasets/fisher_iris.csv',
    'Wisconsin Breast Cancer': 'datasets/wbc9.csv',
    'Wisconsin Breast Cancer Diagnostic': 'datasets/wbc30.csv',
    'Wheat Seeds': 'datasets/wheat_seeds.csv',
    'Diabetes': 'datasets/diabetes.csv',
    'Heart Disease': 'datasets/heart_disease.csv',
    'Glass': 'datasets/glass.csv'
}

def run_dataset_analysis(dataset_name, dataset_path, k_folds=10):
    """
    Run hyperblock analysis on a single dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        dataset_path (str): Path to the dataset CSV file
        k_folds (int): Number of cross-validation folds
    
    Returns:
        dict: Analysis results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {dataset_name}")
    print(f"Dataset: {dataset_path}")
    print(f"{'='*60}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found: {dataset_path}")
        return None
    
    # Run the hyperblock analysis
    try:
        cmd = [
            sys.executable, 'hb_geo.py',
            '--dataset', dataset_path,
            '--k-folds', str(k_folds),
            '--diagnostic'
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"ERROR: Command failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None
        
        # Parse the output to extract metrics
        output = result.stdout
        return parse_output(output, dataset_name)
        
    except Exception as e:
        print(f"ERROR: Failed to analyze {dataset_name}: {str(e)}")
        return None

def parse_output(output, dataset_name):
    """
    Parse the output from hb_geo.py to extract performance metrics.
    
    Args:
        output (str): Output from hb_geo.py
        dataset_name (str): Name of the dataset
    
    Returns:
        dict: Parsed metrics
    """
    try:
        # Initialize default values
        metrics = {
            'dataset': dataset_name,
            'avg_accuracy': 0.0,
            'std_accuracy': 0.0,
            'avg_blocks': 0.0,
            'std_blocks': 0.0,
            'avg_contained': 0.0,
            'std_contained': 0.0,
            'avg_knn': 0.0,
            'std_knn': 0.0
        }
        
        # Extract accuracy
        accuracy_match = re.search(r'Average accuracy across all folds: ([\d.]+) \+\/- ([\d.]+)', output)
        if accuracy_match:
            metrics['avg_accuracy'] = float(accuracy_match.group(1))
            metrics['std_accuracy'] = float(accuracy_match.group(2))
        
        # Extract blocks
        blocks_match = re.search(r'Average blocks per fold: ([\d.]+) \+\/- ([\d.]+)', output)
        if blocks_match:
            metrics['avg_blocks'] = float(blocks_match.group(1))
            metrics['std_blocks'] = float(blocks_match.group(2))
        
        # Extract contained cases
        contained_match = re.search(r'Average contained cases per fold: ([\d.]+) \+\/- ([\d.]+)', output)
        if contained_match:
            metrics['avg_contained'] = float(contained_match.group(1))
            metrics['std_contained'] = float(contained_match.group(2))
        
        # Extract k-NN cases
        knn_match = re.search(r'Average k-NN cases per fold: ([\d.]+) \+\/- ([\d.]+)', output)
        if knn_match:
            metrics['avg_knn'] = float(knn_match.group(1))
            metrics['std_knn'] = float(knn_match.group(2))
        
        return metrics
        
    except Exception as e:
        print(f"ERROR: Failed to parse output for {dataset_name}: {str(e)}")
        return None

def generate_summary_table(results):
    """
    Generate a formatted summary table from the results.
    
    Args:
        results (list): List of result dictionaries
    
    Returns:
        str: Formatted table
    """
    if not results:
        return "No results to display."
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Format the table
    table = []
    table.append("=" * 120)
    table.append("HYPERBLOCK CLASSIFICATION PERFORMANCE SUMMARY")
    table.append("=" * 120)
    table.append("")
    
    # Header
    header = f"{'Dataset':<25} {'Avg Accuracy':<15} {'Avg Blocks':<12} {'Avg Contained':<15} {'Avg k-NN':<12}"
    table.append(header)
    table.append("-" * 120)
    
    # Data rows
    for _, row in df.iterrows():
        accuracy_str = f"{row['avg_accuracy']:.4f} +/- {row['std_accuracy']:.4f}"
        blocks_str = f"{row['avg_blocks']:.1f} +/- {row['std_blocks']:.1f}"
        contained_str = f"{row['avg_contained']:.1f} +/- {row['std_contained']:.1f}"
        knn_str = f"{row['avg_knn']:.1f} +/- {row['std_knn']:.1f}"
        
        data_row = f"{row['dataset']:<25} {accuracy_str:<15} {blocks_str:<12} {contained_str:<15} {knn_str:<12}"
        table.append(data_row)
    
    table.append("-" * 120)
    
    return "\n".join(table)

def save_results_to_csv(results, filename=None):
    """
    Save results to CSV file.
    
    Args:
        results (list): List of result dictionaries
        filename (str): Output filename
    """
    if not results:
        return
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hyperblock_diagnostic_results_{timestamp}.csv"
    
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"\nResults saved to: {filename}")

def main():
    """Main function to run diagnostic analysis."""
    print("Hyperblock Classification Diagnostic Runner")
    print("=" * 50)
    
    # Check if hb_geo.py exists
    if not os.path.exists('hb_geo.py'):
        print("ERROR: hb_geo.py not found in current directory")
        sys.exit(1)
    
    # Run analysis on each dataset
    results = []
    successful_runs = 0
    total_datasets = len(DATASETS)
    
    for dataset_name, dataset_path in DATASETS.items():
        print(f"\nProcessing {successful_runs + 1}/{total_datasets}: {dataset_name}")
        
        result = run_dataset_analysis(dataset_name, dataset_path)
        
        if result is not None:
            results.append(result)
            successful_runs += 1
            print(f"✓ Successfully analyzed {dataset_name}")
        else:
            print(f"✗ Failed to analyze {dataset_name}")
    
    # Generate and display summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Successfully analyzed: {successful_runs}/{total_datasets} datasets")
    
    if results:
        # Display summary table
        summary_table = generate_summary_table(results)
        print("\n" + summary_table)
        
        # Save results to CSV
        save_results_to_csv(results)
        
        # Create detailed results file
        create_detailed_report(results)
    else:
        print("No successful analyses to report.")

def create_detailed_report(results):
    """
    Create a detailed report with additional analysis.
    
    Args:
        results (list): List of result dictionaries
    """
    if not results:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"hyperblock_diagnostic_report_{timestamp}.txt"
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("HYPERBLOCK CLASSIFICATION DIAGNOSTIC REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Datasets analyzed: {len(results)}\n\n")
        
        # Summary table
        summary_table = generate_summary_table(results)
        f.write(summary_table + "\n\n")

    print(f"Report saved to: {report_filename}")

if __name__ == "__main__":
    main()
