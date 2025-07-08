import numpy as np
import pandas as pd
import os
from pathlib import Path
import glob

class HyperBlockPredictor:
    """
    Load hyperblocks from CSV and make predictions with k-nearest hyperblock classification.
    """
    def __init__(self, hyperblocks_file):
        """
        Initialize with hyperblocks CSV file.
        
        Args:
            hyperblocks_file: Path to the hyperblocks CSV file
        """
        self.hyperblocks_file = hyperblocks_file
        self.blocks_by_model = self._load_hyperblocks()
        self.optimal_k = {}  # Store optimal k for each model
        
    def _load_hyperblocks(self):
        """
        Load hyperblocks from CSV file, organized by model.
        
        Returns:
            Dictionary of {model_name: list_of_blocks}
        """
        print(f"Loading hyperblocks from {self.hyperblocks_file}...")
        
        df = pd.read_csv(self.hyperblocks_file)
        blocks_by_model = {}
        
        for model_name in df['Model'].unique():
            model_df = df[df['Model'] == model_name]
            blocks = []
            
            for _, row in model_df.iterrows():
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
            
            blocks_by_model[model_name] = blocks
            print(f"  - {model_name}: {len(blocks)} blocks")
        
        return blocks_by_model
    
    def _is_point_inside_block(self, sample, bounds):
        """
        Check if a point is completely inside a hyperblock.
        
        Args:
            sample: Input sample
            bounds: Dictionary of feature bounds {feature_idx: (min, max)}
            
        Returns:
            bool: True if point is inside, False otherwise
        """
        for feature_idx, (min_val, max_val) in bounds.items():
            if not (min_val <= sample[feature_idx] <= max_val):
                return False
        return True
    
    def _calculate_distance_to_block(self, sample, bounds):
        """
        Calculate distance from point to hyperblock boundary.
        
        Args:
            sample: Input sample
            bounds: Dictionary of feature bounds {feature_idx: (min, max)}
            
        Returns:
            float: Distance to block boundary
        """
        total_distance = 0
        
        for feature_idx, (min_val, max_val) in bounds.items():
            if sample[feature_idx] < min_val:
                total_distance += (min_val - sample[feature_idx]) ** 2
            elif sample[feature_idx] > max_val:
                total_distance += (sample[feature_idx] - max_val) ** 2
            # If inside, distance is 0 for this feature
        
        return np.sqrt(total_distance)
    
    def _learn_optimal_k(self, X_train, y_train, model_name, max_k=10):
        """
        Learn optimal k value for k-nearest hyperblock classification.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_name: Name of the model
            max_k: Maximum k to try
            
        Returns:
            int: Optimal k value
        """
        print(f"Learning optimal k for {model_name}...")
        
        best_k = 1
        best_accuracy = 0
        
        for k in range(1, min(max_k + 1, len(self.blocks_by_model[model_name]) + 1)):
            predictions = self._predict_with_k(X_train, model_name, k)
            accuracy = np.mean(predictions == y_train)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
        
        print(f"  - Optimal k: {best_k} (accuracy: {best_accuracy:.4f})")
        return best_k
    
    def _apply_copeland_rule(self, class_counts):
        """
        Apply Copeland rule to break ties in class voting.
        
        Args:
            class_counts: Counter object with class counts
            
        Returns:
            predicted_class: Class with highest Copeland score
        """
        classes = list(class_counts.keys())
        copeland_scores = {}
        
        for class_i in classes:
            score = 0
            for class_j in classes:
                if class_i != class_j:
                    if class_counts[class_i] > class_counts[class_j]:
                        score += 1
                    elif class_counts[class_i] == class_counts[class_j]:
                        score += 0.5  # Tie gets 0.5 points
            copeland_scores[class_i] = score
        
        # Return class with highest Copeland score
        return max(copeland_scores.keys(), key=lambda x: copeland_scores[x])
    
    def _predict_with_k(self, X, model_name, k):
        """
        Predict using k-nearest hyperblock classification with Copeland rule for ties.
        
        Args:
            X: Input features
            model_name: Name of the model to use
            k: Number of nearest hyperblocks
            
        Returns:
            predictions: Predicted class labels
        """
        blocks = self.blocks_by_model[model_name]
        predictions = []
        
        for sample in X:
            # Check if point is inside any block
            inside_blocks = []
            for block in blocks:
                if self._is_point_inside_block(sample, block['bounds']):
                    inside_blocks.append(block)
            
            if inside_blocks:
                # Point is inside one or more blocks - use majority class with Copeland rule
                classes = [block['class'] for block in inside_blocks]
                from collections import Counter
                class_counts = Counter(classes)
                
                # Check if there's a tie
                max_count = max(class_counts.values())
                max_classes = [cls for cls, count in class_counts.items() if count == max_count]
                
                if len(max_classes) == 1:
                    # No tie, use simple majority
                    prediction = max_classes[0]
                else:
                    # Tie exists, use Copeland rule
                    prediction = self._apply_copeland_rule(class_counts)
            else:
                # Point is outside all blocks - use k-nearest hyperblock with Copeland rule
                distances = []
                for block in blocks:
                    distance = self._calculate_distance_to_block(sample, block['bounds'])
                    distances.append((distance, block))
                
                # Sort by distance and take k nearest
                distances.sort(key=lambda x: x[0])
                k_nearest = distances[:k]
                
                # Majority vote among k nearest with Copeland rule for ties
                classes = [block['class'] for _, block in k_nearest]
                from collections import Counter
                class_counts = Counter(classes)
                
                # Check if there's a tie
                max_count = max(class_counts.values())
                max_classes = [cls for cls, count in class_counts.items() if count == max_count]
                
                if len(max_classes) == 1:
                    # No tie, use simple majority
                    prediction = max_classes[0]
                else:
                    # Tie exists, use Copeland rule
                    prediction = self._apply_copeland_rule(class_counts)
            
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict(self, X, model_name):
        """
        Predict classes using specified model with learned optimal k.
        
        Args:
            X: Input features (n_samples, n_features)
            model_name: Name of the model to use
            
        Returns:
            predictions: Predicted class labels
        """
        if model_name not in self.optimal_k:
            raise ValueError(f"Model {model_name} not trained. Call learn_optimal_k first.")
        
        return self._predict_with_k(X, model_name, self.optimal_k[model_name])
    
    def learn_optimal_k(self, X_train, y_train, max_k=10):
        """
        Learn optimal k values for all models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            max_k: Maximum k to try
        """
        print("Learning optimal k values for all models...")
        
        for model_name in self.blocks_by_model.keys():
            self.optimal_k[model_name] = self._learn_optimal_k(X_train, y_train, model_name, max_k)
    
    def evaluate_model(self, X, y, model_name):
        """
        Evaluate specific model performance.
        
        Args:
            X: Test features
            y: True labels
            model_name: Name of the model to evaluate
            
        Returns:
            accuracy: Classification accuracy
            predictions: Model predictions
        """
        if model_name not in self.optimal_k:
            raise ValueError(f"Model {model_name} not trained. Call learn_optimal_k first.")
        
        predictions = self.predict(X, model_name)
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
        
        # Load data
        X_data, y_data = load_test_data(dataset_name)
        
        # Use ALL training data to learn optimal k
        print(f"  - Using all {len(X_data)} samples for k-learning")
        
        # Learn optimal k values for each model using all data
        predictor.learn_optimal_k(X_data, y_data, max_k=10)
        
        # Test each algorithm independently
        print(f"\n{'='*60}")
        print("Testing Each Algorithm Independently")
        print(f"{'='*60}")
        
        results = {}
        
        for model_name in predictor.blocks_by_model.keys():
            print(f"\n{'='*40}")
            print(f"Testing {model_name}")
            print(f"{'='*40}")
            
            # Evaluate model using all data
            accuracy, predictions = predictor.evaluate_model(X_data, y_data, model_name)
            
            # Store results
            results[model_name] = {
                'accuracy': accuracy,
                'predictions': predictions,
                'optimal_k': predictor.optimal_k[model_name]
            }
            
            print(f"Optimal k: {predictor.optimal_k[model_name]}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Correct Predictions: {np.sum(predictions == y_data)}/{len(y_data)}")
            
            # Show per-class accuracy
            unique_classes = np.unique(y_data)
            print(f"Per-class accuracy:")
            for class_label in unique_classes:
                mask = y_data == class_label
                class_accuracy = np.mean(predictions[mask] == y_data[mask])
                class_count = np.sum(mask)
                print(f"  - Class {class_label}: {class_accuracy:.4f} ({np.sum(predictions[mask] == y_data[mask])}/{class_count})")
        
        # Summary comparison
        print(f"\n{'='*60}")
        print("SUMMARY COMPARISON")
        print(f"{'='*60}")
        for model_name, result in results.items():
            print(f"{model_name}:")
            print(f"  - Optimal k: {result['optimal_k']}")
            print(f"  - Accuracy: {result['accuracy']:.4f}")
        
        # Find best model
        best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
        print(f"\nBest Model: {best_model} (Accuracy: {results[best_model]['accuracy']:.4f})")
        
        # Save detailed results
        results_file = f"test_results_{dataset_name.replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        results_data = []
        for i, (true_label, pred_label) in enumerate(zip(y_data, results[best_model]['predictions'])):
            results_data.append({
                'Sample_ID': i,
                'True_Class': true_label,
                'Predicted_Class': pred_label,
                'Correct': true_label == pred_label
            })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(results_file, index=False)
        
        # Save model comparison
        comparison_file = f"model_comparison_{dataset_name.replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        comparison_data = []
        for model_name, result in results.items():
            comparison_data.append({
                'Model': model_name,
                'Optimal_k': result['optimal_k'],
                'Accuracy': result['accuracy'],
                'Correct_Predictions': np.sum(result['predictions'] == y_data),
                'Total_Samples': len(y_data)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(comparison_file, index=False)
        
        print(f"\n✓ Detailed results saved to {results_file}")
        print(f"✓ Model comparison saved to {comparison_file}")
        print(f"✓ Best model: {best_model} with accuracy {results[best_model]['accuracy']:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check that the hyperblock file and dataset files exist.")

if __name__ == "__main__":
    main()
