import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from collections import Counter
from sklearn.model_selection import KFold

# --- HyperBlock Base Class ---
class HyperBlock:
    """Base class for all HyperBlock algorithms."""
    
    def __init__(self, name):
        self.name = name
        self.blocks = []
        self.training_time = 0
        
    def train(self, X, y):
        """Train the hyperblock model."""
        import time
        start_time = time.time()
        self._train_implementation(X, y)
        self.training_time = time.time() - start_time
        
    def _train_implementation(self, X, y):
        """To be implemented by subclasses."""
        pass
        
    def predict(self, X):
        """To be implemented by subclasses."""
        pass
        
    def get_blocks(self):
        """Return the trained hyperblocks."""
        return self.blocks

# --- IHyper Implementation ---
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

# --- MHyper Implementation ---
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

# --- IMHyper Implementation ---
class IMHyper(HyperBlock):
    """
    Interval Merger Hyper (IMHyper) algorithm.
    Combines interval and merger approaches.
    """
    def __init__(self):
        super().__init__("IMHyper")
        
    def _train_implementation(self, X, y):
        """Train IMHyper model using combined interval and merger strategy."""
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
            
            # Calculate center and radius
            center = np.mean(X_class, axis=0)
            radius = np.std(X_class, axis=0) * 0.5
            
            # Create hyperblock for this class
            block = {
                'class': class_label,
                'intervals': feature_intervals,
                'center': center,
                'radius': radius,
                'support': len(X_class),
                'type': 'imhyper'
            }
            self.blocks.append(block)
            
        print(f"  - Created {len(self.blocks)} IMHyper blocks")
        
    def predict(self, X):
        """Predict using combined interval and merger classification."""
        predictions = []
        
        for sample in X:
            best_class = None
            best_score = -1
            
            for block in self.blocks:
                # Combine interval and merger scores
                interval_score = self._calculate_interval_score(sample, block['intervals'])
                merger_score = self._calculate_merger_score(sample, block['center'], block['radius'])
                combined_score = (interval_score + merger_score) / 2
                
                if combined_score > best_score:
                    best_score = combined_score
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
    
    def _calculate_merger_score(self, sample, center, radius):
        """Calculate how well a sample fits within a merger block."""
        distance = np.linalg.norm(sample - center)
        avg_radius = np.mean(radius)
        return 1.0 / (1.0 + distance / avg_radius)

# --- HyperBlock Predictor ---
class HyperBlockPredictor:
    """
    Predictor class for k-nearest hyperblock classification with Copeland rule.
    """
    
    def __init__(self, hyperblocks_file):
        """
        Initialize predictor.
        
        Args:
            hyperblocks_file: Path to hyperblocks CSV file (not used in unified version)
        """
        self.blocks_by_model = {}
        self.optimal_k = {}
        
    def _is_point_inside_block(self, sample, block):
        """
        Check if a point is completely inside a hyperblock.
        
        Args:
            sample: Input sample
            block: Hyperblock with either 'intervals', 'center'/'radius', or both
            
        Returns:
            bool: True if point is inside, False otherwise
        """
        if block['type'] == 'interval':
            intervals = block['intervals']
            for i, (min_val, max_val) in enumerate(intervals):
                if not (min_val <= sample[i] <= max_val):
                    return False
            return True
        elif block['type'] == 'merger':
            center = block['center']
            radius = block['radius']
            distance = np.linalg.norm(sample - center)
            return distance <= np.mean(radius)
        else:  # IMHyper
            # Check both interval and merger conditions
            intervals = block['intervals']
            center = block['center']
            radius = block['radius']
            
            # Interval check
            interval_inside = True
            for i, (min_val, max_val) in enumerate(intervals):
                if not (min_val <= sample[i] <= max_val):
                    interval_inside = False
                    break
            
            # Merger check
            distance = np.linalg.norm(sample - center)
            merger_inside = distance <= np.mean(radius)
            
            return interval_inside and merger_inside
    
    def _calculate_distance_to_block(self, sample, block):
        """
        Calculate distance from point to hyperblock boundary.
        
        Args:
            sample: Input sample
            block: Hyperblock with either 'intervals', 'center'/'radius', or both
            
        Returns:
            float: Distance to block boundary
        """
        if block['type'] == 'interval':
            total_distance = 0
            intervals = block['intervals']
            for i, (min_val, max_val) in enumerate(intervals):
                if sample[i] < min_val:
                    total_distance += (min_val - sample[i]) ** 2
                elif sample[i] > max_val:
                    total_distance += (sample[i] - max_val) ** 2
                # If inside, distance is 0 for this feature
            return np.sqrt(total_distance)
        elif block['type'] == 'merger':
            center = block['center']
            radius = block['radius']
            distance = np.linalg.norm(sample - center)
            avg_radius = np.mean(radius)
            return max(0, distance - avg_radius)
        else:  # IMHyper
            # Use the minimum of interval and merger distances
            intervals = block['intervals']
            center = block['center']
            radius = block['radius']
            
            # Interval distance
            interval_distance = 0
            for i, (min_val, max_val) in enumerate(intervals):
                if sample[i] < min_val:
                    interval_distance += (min_val - sample[i]) ** 2
                elif sample[i] > max_val:
                    interval_distance += (sample[i] - max_val) ** 2
            interval_distance = np.sqrt(interval_distance)
            
            # Merger distance
            merger_distance = np.linalg.norm(sample - center)
            avg_radius = np.mean(radius)
            merger_distance = max(0, merger_distance - avg_radius)
            
            return min(interval_distance, merger_distance)
    
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
                if self._is_point_inside_block(sample, block):
                    inside_blocks.append(block)
            
            if inside_blocks:
                # Point is inside one or more blocks - use majority class with Copeland rule
                classes = [block['class'] for block in inside_blocks]
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
                    distance = self._calculate_distance_to_block(sample, block)
                    distances.append((distance, block))
                
                # Sort by distance and take k nearest
                distances.sort(key=lambda x: x[0])
                k_nearest = distances[:k]
                
                # Majority vote among k nearest with Copeland rule for ties
                classes = [block['class'] for _, block in k_nearest]
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
            predictions: Predicted labels
        """
        predictions = self.predict(X, model_name)
        accuracy = np.mean(predictions == y)
        return accuracy, predictions

# --- Dataset Loading ---
def load_dataset_and_folds(choice):
    """
    Load dataset and, if needed, generate cross-validation folds.
    Returns:
        - X_train, X_test, y_train, y_test, dataset_name (for split datasets)
        - all_folds, X_full, y_full, dataset_name (for CV datasets)
    """
    if choice == 1:
        # Fisher Iris - 10-fold CV
        df = pd.read_csv('datasets/fisher_iris.csv')
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        dataset_name = "Fisher Iris"
        # Remove duplicates
        unique_indices = []
        seen_samples = set()
        for i, sample in enumerate(X):
            sample_tuple = tuple(sample)
            if sample_tuple not in seen_samples:
                seen_samples.add(sample_tuple)
                unique_indices.append(i)
        X_unique = X[unique_indices]
        y_unique = y[unique_indices]
        print(f"  - Original samples: {len(X)}")
        print(f"  - Unique samples: {len(X_unique)}")
        print(f"  - Removed {len(X) - len(X_unique)} duplicates")
        # Generate folds
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        all_folds = []
        for fold_num, (train_idx, test_idx) in enumerate(kf.split(X_unique)):
            original_train_idx = [unique_indices[i] for i in train_idx]
            original_test_idx = [unique_indices[i] for i in test_idx]
            fold_info = {
                'fold_number': fold_num,
                'fold_indices': original_train_idx,
                'test_indices': original_test_idx,
                'total_folds': 10
            }
            all_folds.append(fold_info)
        print(f"  - Generated 10-fold CV splits (duplicate-safe)")
        return all_folds, X, y, dataset_name
    elif choice == 2:
        # WBC 9-D - 10-fold CV
        df = pd.read_csv('datasets/wbc9.csv')
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        dataset_name = "WBC (9-D)"
        unique_indices = []
        seen_samples = set()
        for i, sample in enumerate(X):
            sample_tuple = tuple(sample)
            if sample_tuple not in seen_samples:
                seen_samples.add(sample_tuple)
                unique_indices.append(i)
        X_unique = X[unique_indices]
        y_unique = y[unique_indices]
        print(f"  - Original samples: {len(X)}")
        print(f"  - Unique samples: {len(X_unique)}")
        print(f"  - Removed {len(X) - len(X_unique)} duplicates")
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        all_folds = []
        for fold_num, (train_idx, test_idx) in enumerate(kf.split(X_unique)):
            original_train_idx = [unique_indices[i] for i in train_idx]
            original_test_idx = [unique_indices[i] for i in test_idx]
            fold_info = {
                'fold_number': fold_num,
                'fold_indices': original_train_idx,
                'test_indices': original_test_idx,
                'total_folds': 10
            }
            all_folds.append(fold_info)
        print(f"  - Generated 10-fold CV splits (duplicate-safe)")
        return all_folds, X, y, dataset_name
    elif choice == 3:
        # WBC 30-D - 10-fold CV
        df = pd.read_csv('datasets/wbc30.csv')
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        dataset_name = "WBC (30-D)"
        unique_indices = []
        seen_samples = set()
        for i, sample in enumerate(X):
            sample_tuple = tuple(sample)
            if sample_tuple not in seen_samples:
                seen_samples.add(sample_tuple)
                unique_indices.append(i)
        X_unique = X[unique_indices]
        y_unique = y[unique_indices]
        print(f"  - Original samples: {len(X)}")
        print(f"  - Unique samples: {len(X_unique)}")
        print(f"  - Removed {len(X) - len(X_unique)} duplicates")
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        all_folds = []
        for fold_num, (train_idx, test_idx) in enumerate(kf.split(X_unique)):
            original_train_idx = [unique_indices[i] for i in train_idx]
            original_test_idx = [unique_indices[i] for i in test_idx]
            fold_info = {
                'fold_number': fold_num,
                'fold_indices': original_train_idx,
                'test_indices': original_test_idx,
                'total_folds': 10
            }
            all_folds.append(fold_info)
        print(f"  - Generated 10-fold CV splits (duplicate-safe)")
        return all_folds, X, y, dataset_name
    elif choice == 4:
        # DR MNIST - train/test split
        train_df = pd.read_csv('datasets/mnist_dr_train.csv')
        test_df = pd.read_csv('datasets/mnist_dr_test.csv')
        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values
        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values
        dataset_name = "Dimensionally Reduced MNIST (121-D)"
        return X_train, X_test, y_train, y_test, dataset_name
    elif choice == 5:
        # Full MNIST - train/test split
        train_files = []
        test_files = []
        for class_label in range(10):
            train_file = f'datasets/mnist_by_class/train/mnist_class_{class_label}_train.csv'
            test_file = f'datasets/mnist_by_class/test/mnist_class_{class_label}_test.csv'
            if os.path.exists(train_file) and os.path.exists(test_file):
                train_files.append(pd.read_csv(train_file))
                test_files.append(pd.read_csv(test_file))
        if not train_files:
            raise ValueError("No MNIST class files found")
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

# --- Main unified workflow ---
def main():
    print("=" * 60)
    print("Unified HyperBlock System")
    print("=" * 60)
    print("\nAvailable Datasets:")
    print("1. Fisher Iris (4-D): Quick testing")
    print("2. WBC (9-D): Benchmark testing")
    print("3. WBC (30-D): Extended benchmark testing")
    print("4. Dimensionally Reduced MNIST (121-D): Initial goal")
    print("5. Full MNIST (784-D): Top-level goal")
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
    data = load_dataset_and_folds(choice)
    
    # --- Cross-validation datasets ---
    if choice in [1, 2, 3]:
        all_folds, X_full, y_full, dataset_name = data
        print(f"  - Running 10-fold cross-validation on {dataset_name}")
        
        # For each fold, train and test
        all_fold_results = {}
        for fold_info in all_folds:
            fold_num = fold_info['fold_number']
            train_idx = np.array(fold_info['fold_indices'])
            test_idx = np.array(fold_info['test_indices'])
            X_train, X_test = X_full[train_idx], X_full[test_idx]
            y_train, y_test = y_full[train_idx], y_full[test_idx]
            print(f"\n  - Fold {fold_num}: {len(X_train)} train, {len(X_test)} test")
            
            # Train models
            models = {
                'IHyper': IHyper(),
                'MHyper': MHyper(),
                'IMHyper': IMHyper()
            }
            fold_results = {}
            for model_name, model in models.items():
                model.train(X_train, y_train)
                # Learn optimal k
                predictor = HyperBlockPredictor(None)
                predictor.blocks_by_model = {model_name: model.get_blocks()}
                predictor.optimal_k = {}
                predictor.optimal_k[model_name] = predictor._learn_optimal_k(X_train, y_train, model_name, max_k=10)
                # Evaluate
                accuracy, predictions = predictor.evaluate_model(X_test, y_test, model_name)
                fold_results[model_name] = {
                    'accuracy': accuracy,
                    'predictions': predictions,
                    'optimal_k': predictor.optimal_k[model_name],
                    'num_blocks': len(model.get_blocks()),
                }
            all_fold_results[fold_num] = fold_results
        
        # Aggregate results
        avg_results = {}
        for model_name in models.keys():
            accuracies = [all_fold_results[fold][model_name]['accuracy'] for fold in range(10)]
            avg_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            avg_results[model_name] = {
                'mean_accuracy': avg_accuracy,
                'std_accuracy': std_accuracy,
                'fold_accuracies': accuracies,
                'optimal_k': all_fold_results[0][model_name]['optimal_k'],
                'num_blocks': np.mean([all_fold_results[fold][model_name]['num_blocks'] for fold in range(10)])
            }
        
        # Print results
        print("\n" + "="*60)
        print("10-FOLD CROSS-VALIDATION RESULTS")
        print("="*60)
        for model_name, result in avg_results.items():
            print(f"\n{'='*40}")
            print(f"{model_name}")
            print(f"{'='*40}")
            print(f"Optimal k: {result['optimal_k']}")
            print(f"Mean Accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
            print(f"Fold Accuracies: {[f'{acc:.4f}' for acc in result['fold_accuracies']]}")
            print(f"Avg. Number of Blocks: {result['num_blocks']:.2f}")
        
        print("\nDone.")
    
    # --- Train/test split datasets ---
    else:
        X_train, X_test, y_train, y_test, dataset_name = data
        print(f"  - Training samples: {len(X_train)}")
        print(f"  - Test samples: {len(X_test)}")
        print(f"  - Features: {X_train.shape[1]}")
        print(f"  - Classes: {len(np.unique(y_train))}")
        
        # Train models
        models = {
            'IHyper': IHyper(),
            'MHyper': MHyper(),
            'IMHyper': IMHyper()
        }
        results = {}
        for model_name, model in models.items():
            print(f"\n{'='*40}")
            print(f"Training {model_name}")
            print(f"{'='*40}")
            model.train(X_train, y_train)
            # Learn optimal k
            predictor = HyperBlockPredictor(None)
            predictor.blocks_by_model = {model_name: model.get_blocks()}
            predictor.optimal_k = {}
            predictor.optimal_k[model_name] = predictor._learn_optimal_k(X_train, y_train, model_name, max_k=10)
            # Evaluate
            accuracy, predictions = predictor.evaluate_model(X_test, y_test, model_name)
            results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'optimal_k': predictor.optimal_k[model_name],
                'num_blocks': len(model.get_blocks()),
                'predictions': predictions
            }
            print(f"  - Training time: {model.training_time:.3f} seconds")
            print(f"  - Test accuracy: {accuracy:.4f}")
            print(f"  - Number of blocks: {len(model.get_blocks())}")
        
        print("\nDone.")

if __name__ == "__main__":
    main() 