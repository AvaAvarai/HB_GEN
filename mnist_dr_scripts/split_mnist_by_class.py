import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
import os
from pathlib import Path

def download_mnist():
    """
    Download MNIST dataset from sklearn.
    Returns:
        X: images as numpy array
        y: labels as numpy array
    """
    print("Downloading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target
    print(f"Downloaded {len(X)} images with shape {X.shape}")
    return X, y

def create_feature_names(n_features):
    """
    Create feature names for the pixel columns.
    
    Args:
        n_features: number of features (784 for MNIST)
    
    Returns:
        list of feature names
    """
    return [f'pixel_{i}' for i in range(n_features)]

def split_and_save_by_class(X_train, y_train, X_test, y_test, output_dir):
    """
    Split the dataset by class and save each class to separate CSV files for training and test sets.
    
    Args:
        X_train: training features array (n_samples, n_features)
        y_train: training labels array (n_samples,)
        X_test: test features array (n_samples, n_features)
        y_test: test labels array (n_samples,)
        output_dir: directory to save the CSV files
    """
    # Create output directories if they don't exist
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get unique classes
    unique_classes = np.unique(np.concatenate([y_train, y_test]))
    print(f"Found {len(unique_classes)} unique classes: {sorted(unique_classes)}")
    
    # Create feature names
    feature_names = create_feature_names(X_train.shape[1])
    
    # Process each class
    for class_label in unique_classes:
        print(f"\nProcessing class {class_label}...")
        
        # Filter training data for this class
        train_mask = y_train == class_label
        X_train_class = X_train[train_mask]
        y_train_class = y_train[train_mask]
        
        # Filter test data for this class
        test_mask = y_test == class_label
        X_test_class = X_test[test_mask]
        y_test_class = y_test[test_mask]
        
        print(f"  - Training: {len(X_train_class)} samples")
        print(f"  - Test: {len(X_test_class)} samples")
        
        # Create training DataFrame with features first, class last
        if len(X_train_class) > 0:
            df_train_class = pd.DataFrame(X_train_class, columns=feature_names)
            df_train_class['class'] = y_train_class
            
            # Save training CSV
            train_filename = f"mnist_class_{class_label}_train.csv"
            train_filepath = os.path.join(train_dir, train_filename)
            df_train_class.to_csv(train_filepath, index=False)
            print(f"  - Training saved to {train_filepath}")
        
        # Create test DataFrame with features first, class last
        if len(X_test_class) > 0:
            df_test_class = pd.DataFrame(X_test_class, columns=feature_names)
            df_test_class['class'] = y_test_class
            
            # Save test CSV
            test_filename = f"mnist_class_{class_label}_test.csv"
            test_filepath = os.path.join(test_dir, test_filename)
            df_test_class.to_csv(test_filepath, index=False)
            print(f"  - Test saved to {test_filepath}")
        
        # Display some statistics
        if len(X_train_class) > 0:
            print(f"  - Training feature range: [{X_train_class.min():.2f}, {X_train_class.max():.2f}]")
            print(f"  - Training mean pixel value: {X_train_class.mean():.2f}")
        if len(X_test_class) > 0:
            print(f"  - Test feature range: [{X_test_class.min():.2f}, {X_test_class.max():.2f}]")
            print(f"  - Test mean pixel value: {X_test_class.mean():.2f}")

def create_summary_file(output_dir, X_train, y_train, X_test, y_test):
    """
    Create summary files with statistics about each class for training and test sets.
    
    Args:
        output_dir: directory where CSV files are saved
        X_train: training features array
        y_train: training labels array
        X_test: test features array
        y_test: test labels array
    """
    unique_classes = np.unique(np.concatenate([y_train, y_test]))
    
    # Training summary
    train_summary_data = []
    for class_label in unique_classes:
        mask = y_train == class_label
        X_class = X_train[mask]
        
        if len(X_class) > 0:
            train_summary_data.append({
                'class': class_label,
                'count': len(X_class),
                'percentage': len(X_class) / len(X_train) * 100,
                'mean_pixel_value': X_class.mean(),
                'std_pixel_value': X_class.std(),
                'min_pixel_value': X_class.min(),
                'max_pixel_value': X_class.max()
            })
    
    train_summary_df = pd.DataFrame(train_summary_data)
    train_summary_file = os.path.join(output_dir, 'train', 'class_summary.csv')
    train_summary_df.to_csv(train_summary_file, index=False)
    
    # Test summary
    test_summary_data = []
    for class_label in unique_classes:
        mask = y_test == class_label
        X_class = X_test[mask]
        
        if len(X_class) > 0:
            test_summary_data.append({
                'class': class_label,
                'count': len(X_class),
                'percentage': len(X_class) / len(X_test) * 100,
                'mean_pixel_value': X_class.mean(),
                'std_pixel_value': X_class.std(),
                'min_pixel_value': X_class.min(),
                'max_pixel_value': X_class.max()
            })
    
    test_summary_df = pd.DataFrame(test_summary_data)
    test_summary_file = os.path.join(output_dir, 'test', 'class_summary.csv')
    test_summary_df.to_csv(test_summary_file, index=False)
    
    print(f"\n✓ Training summary saved to {train_summary_file}")
    print(f"✓ Test summary saved to {test_summary_file}")
    
    print("\nTraining set class distribution:")
    for _, row in train_summary_df.iterrows():
        print(f"  - Class {row['class']}: {row['count']} samples ({row['percentage']:.1f}%)")
    
    print("\nTest set class distribution:")
    for _, row in test_summary_df.iterrows():
        print(f"  - Class {row['class']}: {row['count']} samples ({row['percentage']:.1f}%)")

def main():
    """
    Main function to download MNIST and split by class into training and test sets.
    """
    print("Starting MNIST dataset download and class splitting...")
    
    # Step 1: Download MNIST dataset
    X, y = download_mnist()
    
    # Step 2: Split into training and test sets
    print("Splitting into training and test sets...")
    # MNIST is already split: first 60,000 are training, last 10,000 are test
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Step 3: Create output directory
    output_dir = 'datasets/mnist_by_class'
    print(f"\nOutput directory: {output_dir}")
    
    # Step 4: Split and save by class
    split_and_save_by_class(X_train, y_train, X_test, y_test, output_dir)
    
    # Step 5: Create summary
    create_summary_file(output_dir, X_train, y_train, X_test, y_test)
    
    # Step 6: Verify files were created
    print(f"\nVerifying created files...")
    train_csv_files = list(Path(output_dir, 'train').glob("*.csv"))
    test_csv_files = list(Path(output_dir, 'test').glob("*.csv"))
    
    print(f"Created {len(train_csv_files)} training CSV files:")
    for file in train_csv_files:
        file_size = file.stat().st_size / 1024  # Size in KB
        print(f"  - {file.name} ({file_size:.1f} KB)")
    
    print(f"\nCreated {len(test_csv_files)} test CSV files:")
    for file in test_csv_files:
        file_size = file.stat().st_size / 1024  # Size in KB
        print(f"  - {file.name} ({file_size:.1f} KB)")
    
    print(f"\n✓ MNIST dataset successfully split by class!")
    print(f"  - Total samples: {len(X)}")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")
    print(f"  - Features per sample: {X.shape[1]} (784 pixels)")
    print(f"  - Classes: {len(np.unique(y))}")
    print(f"  - Output directory: {output_dir}")

if __name__ == "__main__":
    main() 