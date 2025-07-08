import numpy as np
import pandas as pd
import os
from pathlib import Path
import time
from datetime import datetime

class HyperBlock:
    """
    Base class for HyperBlock algorithms.
    """
    def __init__(self, name):
        self.name = name
        self.blocks = []
        self.training_time = 0
        
    def train(self, X, y):
        """Train the hyperblock model."""
        start_time = time.time()
        self._train_implementation(X, y)
        self.training_time = time.time() - start_time
        
    def _train_implementation(self, X, y):
        """Implementation to be overridden by subclasses."""
        raise NotImplementedError
        
    def predict(self, X):
        """Predict classes for input data."""
        raise NotImplementedError
        
    def get_blocks(self):
        """Return the trained blocks."""
        return self.blocks

class IHyper(HyperBlock):
    """
    Interval Hyper (IHyper) algorithm.
    Creates hyperblocks based on interval constraints.
    """
    def __init__(self):
        super().__init__("IHyper")
        
    def _train_implementation(self, X, y):
        """Train IHyper model using interval-based hyperblocks."""
        print(f"Training {self.name}...")
        
        unique_classes = np.unique(y)
        self.blocks = []
        
        for class_label in unique_classes:
            # Get samples for this class
            class_mask = y == class_label
            X_class = X[class_mask]
            
            if len(X_class) == 0:
                continue
                
            # Calculate intervals for each feature
            feature_intervals = []
            for feature_idx in range(X_class.shape[1]):
                feature_values = X_class[:, feature_idx]
                min_val = np.min(feature_values)
                max_val = np.max(feature_values)
                feature_intervals.append((min_val, max_val))
            
            # Create hyperblock for this class
            block = {
                'class': class_label,
                'intervals': feature_intervals,
                'support': len(X_class),
                'type': 'interval'
            }
            self.blocks.append(block)
            
        print(f"  - Created {len(self.blocks)} interval blocks")
        
    def predict(self, X):
        """Predict using interval-based classification."""
        predictions = []
        
        for sample in X:
            best_class = None
            best_score = -1
            
            for block in self.blocks:
                score = self._calculate_interval_score(sample, block['intervals'])
                if score > best_score:
                    best_score = score
                    best_class = block['class']
            
            predictions.append(best_class)
            
        return np.array(predictions)
    
    def _calculate_interval_score(self, sample, intervals):
        """Calculate how well a sample fits within the intervals."""
        score = 0
        for i, (min_val, max_val) in enumerate(intervals):
            if min_val <= sample[i] <= max_val:
                score += 1
        return score / len(intervals)

class MHyper(HyperBlock):
    """
    Merger Hyper (MHyper) algorithm.
    Creates hyperblocks by merging overlapping regions.
    """
    def __init__(self):
        super().__init__("MHyper")
        
    def _train_implementation(self, X, y):
        """Train MHyper model using merging strategy."""
        print(f"Training {self.name}...")
        
        unique_classes = np.unique(y)
        self.blocks = []
        
        for class_label in unique_classes:
            # Get samples for this class
            class_mask = y == class_label
            X_class = X[class_mask]
            
            if len(X_class) == 0:
                continue
            
            # Create initial blocks from individual samples
            initial_blocks = []
            for sample in X_class:
                block = {
                    'center': sample.copy(),
                    'radius': np.std(X_class, axis=0) * 0.5,  # Adaptive radius
                    'support': 1
                }
                initial_blocks.append(block)
            
            # Merge overlapping blocks
            merged_blocks = self._merge_blocks(initial_blocks)
            
            # Convert to hyperblock format
            for merged_block in merged_blocks:
                block = {
                    'class': class_label,
                    'center': merged_block['center'],
                    'radius': merged_block['radius'],
                    'support': merged_block['support'],
                    'type': 'merger'
                }
                self.blocks.append(block)
                
        print(f"  - Created {len(self.blocks)} merger blocks")
    
    def _merge_blocks(self, blocks, threshold=0.7):
        """Merge blocks that have sufficient overlap."""
        if len(blocks) <= 1:
            return blocks
            
        merged = True
        while merged:
            merged = False
            new_blocks = []
            used = set()
            
            for i, block1 in enumerate(blocks):
                if i in used:
                    continue
                    
                merged_block = block1.copy()
                used.add(i)
                
                for j, block2 in enumerate(blocks[i+1:], i+1):
                    if j in used:
                        continue
                        
                    if self._blocks_overlap(block1, block2, threshold):
                        merged_block = self._merge_two_blocks(merged_block, block2)
                        used.add(j)
                        merged = True
                
                new_blocks.append(merged_block)
            
            blocks = new_blocks
            
        return blocks
    
    def _blocks_overlap(self, block1, block2, threshold):
        """Check if two blocks overlap significantly."""
        distance = np.linalg.norm(block1['center'] - block2['center'])
        avg_radius = (np.mean(block1['radius']) + np.mean(block2['radius'])) / 2
        return distance < avg_radius * threshold
    
    def _merge_two_blocks(self, block1, block2):
        """Merge two blocks into one."""
        total_support = block1['support'] + block2['support']
        
        # Weighted average of centers
        new_center = (block1['center'] * block1['support'] + 
                     block2['center'] * block2['support']) / total_support
        
        # Average radius
        new_radius = (block1['radius'] + block2['radius']) / 2
        
        return {
            'center': new_center,
            'radius': new_radius,
            'support': total_support
        }
    
    def predict(self, X):
        """Predict using merger-based classification."""
        predictions = []
        
        for sample in X:
            best_class = None
            best_score = -1
            
            for block in self.blocks:
                score = self._calculate_merger_score(sample, block['center'], block['radius'])
                if score > best_score:
                    best_score = score
                    best_class = block['class']
            
            predictions.append(best_class)
            
        return np.array(predictions)
    
    def _calculate_merger_score(self, sample, center, radius):
        """Calculate how well a sample fits within a merger block."""
        distance = np.linalg.norm(sample - center)
        avg_radius = np.mean(radius)
        return 1.0 / (1.0 + distance / avg_radius)

class IMHyper(HyperBlock):
    """
    Interval Merger Hyper (IMHyper) algorithm.
    Combines interval and merger approaches.
    """
    def __init__(self):
        super().__init__("IMHyper")
        
    def _train_implementation(self, X, y):
        """Train IMHyper model combining interval and merger strategies."""
        print(f"Training {self.name}...")
        
        # Train both IHyper and MHyper
        ihyper = IHyper()
        mhyper = MHyper()
        
        ihyper.train(X, y)
        mhyper.train(X, y)
        
        # Combine blocks from both approaches
        self.blocks = []
        
        # Add interval blocks
        for block in ihyper.get_blocks():
            block['algorithm'] = 'interval'
            self.blocks.append(block)
        
        # Add merger blocks
        for block in mhyper.get_blocks():
            block['algorithm'] = 'merger'
            self.blocks.append(block)
            
        print(f"  - Created {len(self.blocks)} combined blocks")
    
    def predict(self, X):
        """Predict using combined interval and merger classification."""
        predictions = []
        
        for sample in X:
            best_class = None
            best_score = -1
            
            for block in self.blocks:
                if block['algorithm'] == 'interval':
                    score = self._calculate_interval_score(sample, block['intervals'])
                else:  # merger
                    score = self._calculate_merger_score(sample, block['center'], block['radius'])
                
                if score > best_score:
                    best_score = score
                    best_class = block['class']
            
            predictions.append(best_class)
            
        return np.array(predictions)
    
    def _calculate_interval_score(self, sample, intervals):
        """Calculate interval score (same as IHyper)."""
        score = 0
        for i, (min_val, max_val) in enumerate(intervals):
            if min_val <= sample[i] <= max_val:
                score += 1
        return score / len(intervals)
    
    def _calculate_merger_score(self, sample, center, radius):
        """Calculate merger score (same as MHyper)."""
        distance = np.linalg.norm(sample - center)
        avg_radius = np.mean(radius)
        return 1.0 / (1.0 + distance / avg_radius)

def load_dataset(choice):
    """
    Load dataset based on user choice.
    
    Args:
        choice: Integer 1-5 representing dataset choice
    
    Returns:
        X_train, X_test, y_train, y_test: Training and test data
        dataset_name: Name of the dataset
    """
    if choice == 1:
        # Fisher Iris
        df = pd.read_csv('datasets/fisher_iris.csv')
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        dataset_name = "Fisher Iris"
        
    elif choice == 2:
        # WBC 9-D
        df = pd.read_csv('datasets/wbc9.csv')
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        dataset_name = "WBC (9-D)"
        
    elif choice == 3:
        # WBC 30-D
        df = pd.read_csv('datasets/wbc30.csv')
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        dataset_name = "WBC (30-D)"
        
    elif choice == 4:
        # Dimensionally Reduced MNIST (121-D)
        train_df = pd.read_csv('datasets/mnist_dr_train.csv')
        test_df = pd.read_csv('datasets/mnist_dr_test.csv')
        
        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values
        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values
        
        dataset_name = "Dimensionally Reduced MNIST (121-D)"
        return X_train, X_test, y_train, y_test, dataset_name
        
    elif choice == 5:
        # Full MNIST (784-D) - using mnist_by_class files
        print("Loading full MNIST dataset from individual class files...")
        
        # Load training data from all classes
        train_files = []
        test_files = []
        
        for class_label in range(10):  # Classes 0-9
            train_file = f'datasets/mnist_by_class/train/mnist_class_{class_label}_train.csv'
            test_file = f'datasets/mnist_by_class/test/mnist_class_{class_label}_test.csv'
            
            if os.path.exists(train_file) and os.path.exists(test_file):
                train_files.append(pd.read_csv(train_file))
                test_files.append(pd.read_csv(test_file))
            else:
                print(f"Warning: Missing files for class {class_label}")
        
        if not train_files:
            raise ValueError("No MNIST class files found")
        
        # Combine all training and test data
        train_df = pd.concat(train_files, ignore_index=True)
        test_df = pd.concat(test_files, ignore_index=True)
        
        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values
        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values
        
        dataset_name = "Full MNIST (784-D)"
        return X_train, X_test, y_train, y_test, dataset_name
    
    else:
        raise ValueError("Invalid choice")
    
    # For datasets without train/test split, create one
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, dataset_name

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Args:
        model: Trained hyperblock model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        accuracy: Classification accuracy
        predictions: Model predictions
    """
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    return accuracy, predictions

def export_hyperblocks(results, dataset_name, output_file):
    """
    Export clean hyperblock bounds to CSV.
    
    Args:
        results: Dictionary containing results with trained models
        dataset_name: Name of the dataset
        output_file: Output CSV file path
    """
    # Prepare data for export
    export_data = []
    
    for model_name, result in results.items():
        model = result['model']  # Get the actual trained model
        blocks = model.get_blocks()
        
        for block_idx, block in enumerate(blocks):
            # Base block info
            block_data = {
                'Model': model_name,
                'Block_ID': block_idx,
                'Class': block['class']
            }
            
            # Add bounds per attribute
            if block['type'] == 'interval':
                # For interval blocks: min/max bounds
                for feature_idx, (min_val, max_val) in enumerate(block['intervals']):
                    block_data[f'Attr_{feature_idx}_Min'] = min_val
                    block_data[f'Attr_{feature_idx}_Max'] = max_val
                    
            elif block['type'] == 'merger':
                # For merger blocks: center ± radius bounds
                for feature_idx in range(len(block['center'])):
                    center = block['center'][feature_idx]
                    radius = block['radius'][feature_idx]
                    block_data[f'Attr_{feature_idx}_Min'] = center - radius
                    block_data[f'Attr_{feature_idx}_Max'] = center + radius
            
            export_data.append(block_data)
    
    # Create DataFrame and save
    df = pd.DataFrame(export_data)
    df.to_csv(output_file, index=False)
    print(f"✓ Hyperblock bounds exported to {output_file}")
    print(f"  - Total blocks: {len(export_data)}")
    print(f"  - Models: {list(results.keys())}")

def main():
    """
    Main function to run hyperblock training and evaluation.
    """
    print("=" * 60)
    print("HyperBlock Generation System")
    print("=" * 60)
    
    # Display dataset options
    print("\nAvailable Datasets:")
    print("1. Fisher Iris (4-D): Quick testing")
    print("2. WBC (9-D): Benchmark testing")
    print("3. WBC (30-D): Extended benchmark testing")
    print("4. Dimensionally Reduced MNIST (121-D): Initial goal")
    print("5. Full MNIST (784-D): Top-level goal")
    
    # Get user choice
    while True:
        try:
            choice = int(input("\nSelect dataset (1-5): "))
            if 1 <= choice <= 5:
                break
            else:
                print("Please enter a number between 1 and 5.")
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"\nLoading dataset {choice}...")
    
    try:
        # Load dataset
        X_train, X_test, y_train, y_test, dataset_name = load_dataset(choice)
        
        print(f"✓ Loaded {dataset_name}")
        print(f"  - Training samples: {len(X_train)}")
        print(f"  - Test samples: {len(X_test)}")
        print(f"  - Features: {X_train.shape[1]}")
        print(f"  - Classes: {len(np.unique(y_train))}")
        
        # Initialize models
        models = {
            'IHyper': IHyper(),
            'MHyper': MHyper(),
            'IMHyper': IMHyper()
        }
        
        # Train and evaluate models
        results = {}
        
        for model_name, model in models.items():
            print(f"\n{'='*40}")
            print(f"Training {model_name}")
            print(f"{'='*40}")
            
            # Train model
            model.train(X_train, y_train)
            
            # Evaluate model
            accuracy, predictions = evaluate_model(model, X_test, y_test)
            
            # Store results
            results[model_name] = {
                'model': model,  # Store the actual trained model
                'training_time': model.training_time,
                'accuracy': accuracy,
                'num_blocks': len(model.get_blocks()),
                'predictions': predictions
            }
            
            print(f"  - Training time: {model.training_time:.3f} seconds")
            print(f"  - Test accuracy: {accuracy:.4f}")
            print(f"  - Number of blocks: {len(model.get_blocks())}")
        
        # Export hyperblocks
        output_file = f"hyperblocks_{dataset_name.replace(' ', '_').replace('(', '').replace(')', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        export_hyperblocks(results, dataset_name, output_file)
        
        # Display summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for model_name, result in results.items():
            print(f"{model_name}:")
            print(f"  - Accuracy: {result['accuracy']:.4f}")
            print(f"  - Training Time: {result['training_time']:.3f}s")
            print(f"  - Blocks: {result['num_blocks']}")
        
        print(f"\n✓ HyperBlock training and evaluation complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check that the dataset files exist and are properly formatted.")

if __name__ == "__main__":
    main()
