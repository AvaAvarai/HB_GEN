import numpy as np
import pandas as pd
import os
from pathlib import Path
import glob

class HyperBlockPredictor:
    """
    Load hyperblocks from CSV and make predictions.
    """
    def __init__(self, hyperblocks_file):
        """
        Initialize with hyperblocks CSV file.
        
        Args:
            hyperblocks_file: Path to the hyperblocks CSV file
        """
        self.hyperblocks_file = hyperblocks_file
        self.blocks = self._load_hyperblocks()
        
    def _load_hyperblocks(self):
        """
        Load hyperblocks from CSV file.
        
        Returns:
            List of hyperblock dictionaries
        """
        print(f"Loading hyperblocks from {self.hyperblocks_file}...")
        
        df = pd.read_csv(self.hyperblocks_file)
        blocks = []
        
        for _, row in df.iterrows():
            # Extract basic block info
            block = {
                'model': row['Model'],
                'block_id': row['Block_ID'],
                'class': row['Class']
            }
            
            # Extract attribute bounds
            bounds = {}
            feature_count = 0
            
            # Find all attribute columns
            attr_columns = [col for col in df.columns if col.startswith('Attr_') and col.endswith('_Min')]
            feature_count = len(attr_columns)
            
            for i in range(feature_count):
                min_col = f'Attr_{i}_Min'
                max_col = f'Attr_{i}_Max'
                
                if min_col in df.columns and max_col in df.columns:
                    bounds[i] = (row[min_col], row[max_col])
            
            block['bounds'] = bounds
            block['feature_count'] = feature_count
            blocks.append(block)
            
        print(f"  - Loaded {len(blocks)} hyperblocks")
        print(f"  - Features per block: {feature_count}")
        print(f"  - Models: {df['Model'].unique()}")
        
        return blocks
    
    def predict(self, X):
        """
        Predict classes using loaded hyperblocks.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            predictions: Predicted class labels
        """
        predictions = []
        
        for sample in X:
            best_class = None
            best_score = -1
            
            for block in self.blocks:
                score = self._calculate_block_score(sample, block['bounds'])
                if score > best_score:
                    best_score = score
                    best_class = block['class']
            
            predictions.append(best_class)
            
        return np.array(predictions)
    
    def _calculate_block_score(self, sample, bounds):
        """
        Calculate how well a sample fits within the hyperblock bounds.
        
        Args:
            sample: Input sample
            bounds: Dictionary of feature bounds {feature_idx: (min, max)}
            
        Returns:
            score: Fit score (0 to 1)
        """
        score = 0
        total_features = len(bounds)
        
        for feature_idx, (min_val, max_val) in bounds.items():
            if min_val <= sample[feature_idx] <= max_val:
                score += 1
        
        return score / total_features if total_features > 0 else 0
    
    def evaluate(self, X, y):
        """
        Evaluate hyperblock performance.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            accuracy: Classification accuracy
            predictions: Model predictions
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy, predictions

def load_test_data(dataset_name):
    """
    Load test data based on dataset name.
    Use test data if available, otherwise use full data.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        X_test, y_test: Test features and labels
    """
    print(f"Loading data for {dataset_name}...")
    
    if "Fisher Iris" in dataset_name:
        # No separate test data - use full dataset
        df = pd.read_csv('datasets/fisher_iris.csv')
        X_test = df.iloc[:, :-1].values
        y_test = df.iloc[:, -1].values
        print("  - Using full dataset (no separate test data)")
        
    elif "WBC (9-D)" in dataset_name:
        # No separate test data - use full dataset
        df = pd.read_csv('datasets/wbc9.csv')
        X_test = df.iloc[:, :-1].values
        y_test = df.iloc[:, -1].values
        print("  - Using full dataset (no separate test data)")
        
    elif "WBC (30-D)" in dataset_name:
        # No separate test data - use full dataset
        df = pd.read_csv('datasets/wbc30.csv')
        X_test = df.iloc[:, :-1].values
        y_test = df.iloc[:, -1].values
        print("  - Using full dataset (no separate test data)")
        
    elif "Dimensionally Reduced MNIST" in dataset_name:
        # Use the separate test split
        test_df = pd.read_csv('datasets/mnist_dr_test.csv')
        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values
        print("  - Using separate test data")
        
    elif "Full MNIST" in dataset_name:
        # Use the separate test files from mnist_by_class
        test_files = []
        for class_label in range(10):
            test_file = f'datasets/mnist_by_class/test/mnist_class_{class_label}_test.csv'
            if os.path.exists(test_file):
                test_files.append(pd.read_csv(test_file))
        
        if test_files:
            test_df = pd.concat(test_files, ignore_index=True)
            X_test = test_df.iloc[:, :-1].values
            y_test = test_df.iloc[:, -1].values
            print("  - Using separate test data")
        else:
            raise ValueError("No MNIST test files found")
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"  - Samples: {len(X_test)}")
    print(f"  - Features: {X_test.shape[1]}")
    print(f"  - Classes: {len(np.unique(y_test))}")
    
    return X_test, y_test

def main():
    """
    Main function to test hyperblocks.
    """
    print("=" * 60)
    print("HyperBlock Testing System")
    print("=" * 60)
    
    # Find hyperblock CSV files
    hyperblock_files = glob.glob("hyperblocks_*.csv")
    
    if not hyperblock_files:
        print("No hyperblock CSV files found!")
        print("Please run hb_gen.py first to generate hyperblocks.")
        return
    
    print(f"\nFound {len(hyperblock_files)} hyperblock file(s):")
    for i, file in enumerate(hyperblock_files):
        print(f"{i+1}. {file}")
    
    # Let user select file
    if len(hyperblock_files) == 1:
        selected_file = hyperblock_files[0]
        print(f"\nUsing: {selected_file}")
    else:
        while True:
            try:
                choice = int(input(f"\nSelect file (1-{len(hyperblock_files)}): "))
                if 1 <= choice <= len(hyperblock_files):
                    selected_file = hyperblock_files[choice - 1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(hyperblock_files)}.")
            except ValueError:
                print("Please enter a valid number.")
    
    try:
        # Load hyperblocks
        predictor = HyperBlockPredictor(selected_file)
        
        # Extract dataset name from filename
        # Example: hyperblocks_Fisher_Iris_20250707_195037.csv
        filename = Path(selected_file).stem
        parts = filename.split('_')
        
        # Find dataset name (everything between 'hyperblocks' and timestamp)
        dataset_parts = []
        for part in parts[1:]:
            if part.isdigit() and len(part) >= 8:  # Timestamp
                break
            dataset_parts.append(part)
        
        dataset_name = ' '.join(dataset_parts)
        print(f"\nDetected dataset: {dataset_name}")
        
        # Load test data
        X_test, y_test = load_test_data(dataset_name)
        
        # Test hyperblocks
        print(f"\n{'='*40}")
        print("Testing Hyperblocks")
        print(f"{'='*40}")
        
        accuracy, predictions = predictor.evaluate(X_test, y_test)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Correct Predictions: {np.sum(predictions == y_test)}/{len(y_test)}")
        
        # Show per-class accuracy
        unique_classes = np.unique(y_test)
        print(f"\nPer-class accuracy:")
        for class_label in unique_classes:
            mask = y_test == class_label
            class_accuracy = np.mean(predictions[mask] == y_test[mask])
            class_count = np.sum(mask)
            print(f"  - Class {class_label}: {class_accuracy:.4f} ({np.sum(predictions[mask] == y_test[mask])}/{class_count})")
        
        # Save results
        results_file = f"test_results_{dataset_name.replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        results_data = []
        for i, (true_label, pred_label) in enumerate(zip(y_test, predictions)):
            results_data.append({
                'Sample_ID': i,
                'True_Class': true_label,
                'Predicted_Class': pred_label,
                'Correct': true_label == pred_label
            })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(results_file, index=False)
        
        print(f"\n✓ Test results saved to {results_file}")
        print(f"✓ Overall accuracy: {accuracy:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check that the hyperblock file and dataset files exist.")

if __name__ == "__main__":
    main()
