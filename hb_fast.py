import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
import multiprocessing
import time
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree

# --- Configuration ---
N_JOBS = multiprocessing.cpu_count()  # Use all available CPU cores
PARALLEL_THRESHOLD = 1000  # Minimum dataset size for parallel prediction
VERBOSE_PARALLEL = 0  # Set to 1 for verbose parallel processing output
ENABLE_TIMING = True

# --- HyperBlock Base Class ---
class HyperBlock:
    """Base class for all efficient HyperBlock algorithms."""
    
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
        """To be implemented by subclasses."""
        pass
        
    def predict(self, X):
        """To be implemented by subclasses."""
        pass
        
    def get_blocks(self):
        """Return the trained hyperblocks."""
        return self.blocks

# --- Efficient IHyper Implementation ---
class EIHyper(HyperBlock):
    """
    Efficient Interval Hyper (EIHyper) algorithm.
    Uses pre-sorted arrays, binary search, and batch processing for O(n log n) complexity.
    """
    def __init__(self, purity_threshold=0.8):
        super().__init__("EIHyper")
        self.purity_threshold = purity_threshold
        self.learned_purity_threshold = None
        
    def _train_implementation(self, X, y):
        """Train EIHyper model using efficient interval-based hyperblocks."""
        print(f"Training {self.name}...")
        
        # Learn optimal purity threshold if needed
        if self.purity_threshold == 'auto':
            print("  - Learning optimal purity threshold...")
            self.learned_purity_threshold = self._learn_optimal_purity_threshold(X, y)
            print(f"  - Learned purity threshold: {self.learned_purity_threshold:.3f}")
        else:
            self.learned_purity_threshold = self.purity_threshold
            print(f"  - Using purity threshold: {self.learned_purity_threshold}")
        
        self.blocks = []
        remaining_mask = np.ones(len(X), dtype=bool)
        
        # Pre-sort all attributes for efficient interval generation
        sorted_attributes = self._preprocess_attributes(X, y)
        
        while np.sum(remaining_mask) > 0:
            # Find best interval across all attributes using batch processing
            best_interval = self._find_best_interval_batch(X, y, remaining_mask, sorted_attributes)
            
            if best_interval is None:
                break
                
            # Create hyperblock from best interval
            attr_idx, b_i, d_i, interval_points = best_interval
            interval_classes = y[interval_points]
            
            # Determine dominant class
            unique_classes, class_counts = np.unique(interval_classes, return_counts=True)
            dominant_class = unique_classes[np.argmax(class_counts)]
            
            # Create intervals for all attributes
            feature_intervals = self._create_feature_intervals(X[interval_points])
            
            # Create the hyperblock
            block = {
                'class': dominant_class,
                'intervals': feature_intervals,
                'support': len(interval_points),
                'type': 'interval',
                'primary_attribute': attr_idx,
                'primary_interval': (b_i, d_i)
            }
            self.blocks.append(block)
            
            # Remove covered points
            remaining_mask[interval_points] = False
            
        print(f"  - Created {len(self.blocks)} interval blocks")
    
    def _preprocess_attributes(self, X, y):
        """Pre-sort all attributes for efficient interval generation."""
        sorted_attributes = []
        for attr_idx in range(X.shape[1]):
            attr_values = X[:, attr_idx]
            sorted_indices = np.argsort(attr_values)
            sorted_attributes.append({
                'values': attr_values[sorted_indices],
                'indices': sorted_indices,
                'classes': y[sorted_indices]
            })
        return sorted_attributes
    
    def _find_best_interval_batch(self, X, y, remaining_mask, sorted_attributes):
        """Find best interval using batch processing and binary search."""
        best_interval = None
        best_support = 0
        
        # Process attributes in parallel
        def process_attribute(attr_idx):
            attr_data = sorted_attributes[attr_idx]
            attr_values = attr_data['values']
            attr_indices = attr_data['indices']
            attr_classes = attr_data['classes']
            
            # Only consider remaining points
            remaining_indices = np.where(remaining_mask)[0]
            remaining_mask_attr = np.isin(attr_indices, remaining_indices)
            
            if np.sum(remaining_mask_attr) == 0:
                return None
            
            # Find best interval for this attribute
            best_attr_interval = None
            best_attr_support = 0
            
            # Use binary search for efficient interval expansion
            for seed_idx in range(len(attr_values)):
                if not remaining_mask_attr[seed_idx]:
                    continue
                    
                seed_value = attr_values[seed_idx]
                seed_class = attr_classes[seed_idx]
                
                # Binary search for right expansion boundary
                right_bound = self._binary_search_expansion(
                    attr_values, attr_classes, seed_idx, seed_class, 'right', remaining_mask_attr
                )
                
                # Binary search for left expansion boundary
                left_bound = self._binary_search_expansion(
                    attr_values, attr_classes, seed_idx, seed_class, 'left', remaining_mask_attr
                )
                
                # Calculate interval support
                interval_mask = (attr_values >= attr_values[left_bound]) & (attr_values <= attr_values[right_bound])
                interval_support = np.sum(interval_mask & remaining_mask_attr)
                
                if interval_support > best_attr_support:
                    best_attr_support = interval_support
                    best_attr_interval = (
                        attr_values[left_bound],
                        attr_values[right_bound],
                        attr_indices[left_bound:right_bound+1]
                    )
            
            return (attr_idx, best_attr_interval[0], best_attr_interval[1], best_attr_interval[2]) if best_attr_interval else None
        
        # Parallelize attribute processing
        n_jobs = min(N_JOBS, X.shape[1])
        results = Parallel(n_jobs=n_jobs, verbose=VERBOSE_PARALLEL)(
            delayed(process_attribute)(attr_idx) for attr_idx in range(X.shape[1])
        )
        
        # Find best interval across all attributes
        for result in results:
            if result is not None:
                attr_idx, b_i, d_i, interval_indices = result
                interval_support = len(interval_indices)
                if interval_support > best_support:
                    best_support = interval_support
                    best_interval = (attr_idx, b_i, d_i, interval_indices)
        
        return best_interval
    
    def _binary_search_expansion(self, values, classes, seed_idx, seed_class, direction, mask):
        """Use binary search to find expansion boundary efficiently."""
        if direction == 'right':
            left, right = seed_idx, len(values) - 1
            while left < right:
                mid = (left + right + 1) // 2
                if mid > right:
                    break
                # Check if we can expand to mid
                if self._check_purity_batch(classes[seed_idx:mid+1], seed_class, mask[seed_idx:mid+1]):
                    left = mid
                else:
                    right = mid - 1
            return left
        else:  # left
            left, right = 0, seed_idx
            while left < right:
                mid = (left + right) // 2
                # Check if we can expand to mid
                if self._check_purity_batch(classes[mid:seed_idx+1], seed_class, mask[mid:seed_idx+1]):
                    right = mid
                else:
                    left = mid + 1
            return right
    
    def _check_purity_batch(self, classes, target_class, mask):
        """Check purity using vectorized operations."""
        if len(classes) == 0:
            return False
        valid_classes = classes[mask]
        if len(valid_classes) == 0:
            return False
        purity = np.sum(valid_classes == target_class) / len(valid_classes)
        return purity >= self.learned_purity_threshold
    
    def _create_feature_intervals(self, points):
        """Create intervals for all features based on point ranges."""
        feature_intervals = []
        for feature_idx in range(points.shape[1]):
            feature_values = points[:, feature_idx]
            min_val = np.min(feature_values)
            max_val = np.max(feature_values)
            feature_intervals.append((min_val, max_val))
        return feature_intervals
    
    def _learn_optimal_purity_threshold(self, X, y, cv_folds=10):
        """Learn optimal purity threshold using efficient cross-validation."""
        thresholds = np.arange(0.6, 1.0, 0.05)
        best_threshold = 0.8
        best_accuracy = 0
        
        def evaluate_threshold(threshold):
            fold_accuracies = []
            kf = KFold(n_splits=min(cv_folds, len(X)), shuffle=True, random_state=42)
            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                temp_model = EIHyper(purity_threshold=threshold)
                temp_model._train_implementation(X_train_fold, y_train_fold)
                predictions = temp_model.predict(X_val_fold)
                accuracy = np.mean(predictions == y_val_fold)
                fold_accuracies.append(accuracy)
            return threshold, np.mean(fold_accuracies)
        
        n_jobs = min(N_JOBS, len(thresholds))
        results = Parallel(n_jobs=n_jobs, verbose=VERBOSE_PARALLEL)(
            delayed(evaluate_threshold)(threshold) for threshold in thresholds
        )
        
        for threshold, avg_accuracy in results:
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_threshold = threshold
        
        return best_threshold
    
    def predict(self, X):
        """Predict using efficient interval-based classification."""
        if not self.blocks:
            return np.zeros(len(X), dtype=int)
        
        predictions = np.zeros(len(X), dtype=int)
        
        # Vectorized prediction for efficiency
        for sample_idx, sample in enumerate(X):
            best_score = -1
            best_class = 0
            
            for block in self.blocks:
                score = self._calculate_interval_score_vectorized(sample, block['intervals'])
                if score > best_score:
                    best_score = score
                    best_class = block['class']
            
            predictions[sample_idx] = best_class
        
        return predictions
    
    def _calculate_interval_score_vectorized(self, sample, intervals):
        """Calculate interval score using vectorized operations."""
        score = 0
        for i, (min_val, max_val) in enumerate(intervals):
            if min_val <= sample[i] <= max_val:
                score += 1
        return score / len(intervals)

# --- Efficient MHyper Implementation ---
class EMHyper(HyperBlock):
    """
    Efficient Merger Hyper (EMHyper) algorithm.
    Uses hierarchical clustering, spatial indexing, and batch merging for O(n log n) complexity.
    """
    def __init__(self, impurity_threshold=None):
        super().__init__("EMHyper")
        self.impurity_threshold = impurity_threshold
        self.learned_threshold = None
        
    def _train_implementation(self, X, y):
        """Train EMHyper model using efficient merging with spatial indexing."""
        print(f"Training {self.name}...")
        
        if self.impurity_threshold is None:
            print("  - Learning optimal impurity threshold...")
            self.learned_threshold = self._learn_optimal_impurity_threshold(X, y)
            print(f"  - Learned impurity threshold: {self.learned_threshold:.3f}")
        else:
            self.learned_threshold = self.impurity_threshold
            print(f"  - Using impurity threshold: {self.learned_threshold}")
        
        self.blocks = []
        
        # Use hierarchical clustering for initial block formation
        initial_blocks = self._create_initial_blocks_hierarchical(X, y)
        
        # Efficient pure merging phase
        pure_blocks = self._efficient_pure_merge_phase(initial_blocks, X, y)
        
        # Efficient impure merging phase
        final_blocks = self._efficient_impure_merge_phase(pure_blocks, X, y)
        
        # Convert to standard format
        for block in final_blocks:
            if len(block['points']) > 0:
                block_indices = block['points']
                block_points = X[block_indices]
                block_classes = y[block_indices]
                unique_classes, class_counts = np.unique(block_classes, return_counts=True)
                dominant_class = unique_classes[np.argmax(class_counts)]
                center = np.mean(block_points, axis=0)
                radius = np.std(block_points, axis=0) * 0.5
                final_block = {
                    'class': dominant_class,
                    'center': center,
                    'radius': radius,
                    'support': len(block_points),
                    'type': 'merger',
                    'envelope': block['envelope']
                }
                self.blocks.append(final_block)
        
        print(f"  - Created {len(self.blocks)} merger blocks")
    
    def _create_initial_blocks_hierarchical(self, X, y):
        """Create initial blocks using hierarchical clustering for efficiency."""
        from sklearn.cluster import AgglomerativeClustering
        
        # Use hierarchical clustering to create initial pure blocks
        n_clusters = min(len(X) // 2, 100)  # Adaptive number of clusters
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward',
            compute_full_tree=True
        )
        cluster_labels = clustering.fit_predict(X)
        
        # Create blocks from clusters
        initial_blocks = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_points = np.where(cluster_mask)[0]
            cluster_classes = y[cluster_points]
            
            # Check if cluster is pure
            unique_classes = np.unique(cluster_classes)
            if len(unique_classes) == 1:
                # Pure cluster
                block = {
                    'points': cluster_points.tolist(),
                    'class': unique_classes[0],
                    'envelope': self._create_envelope(X[cluster_points]),
                    'type': 'merger'
                }
                initial_blocks.append(block)
            else:
                # Impure cluster - split into individual points
                for point_idx in cluster_points:
                    block = {
                        'points': [point_idx],
                        'class': y[point_idx],
                        'envelope': self._create_envelope([X[point_idx]]),
                        'type': 'merger'
                    }
                    initial_blocks.append(block)
        
        return initial_blocks
    
    def _efficient_pure_merge_phase(self, initial_blocks, X, y):
        """Efficient pure merging phase using spatial indexing."""
        blocks = initial_blocks.copy()
        
        # Build spatial index for fast overlap detection
        spatial_index = self._build_spatial_index(blocks, X)
        
        merged = True
        iteration = 0
        max_iterations = len(blocks)  # Prevent infinite loops
        
        while merged and iteration < max_iterations:
            merged = False
            iteration += 1
            
            # Find mergeable pairs using spatial index
            merge_pairs = self._find_mergeable_pairs_spatial(blocks, spatial_index, X, y, pure_only=True)
            
            if merge_pairs:
                # Perform batch merge
                for i, j in merge_pairs:
                    if blocks[i] is not None and blocks[j] is not None:
                        # Merge blocks
                        joint_points = blocks[i]['points'] + blocks[j]['points']
                        joint_envelope = self._create_envelope([X[idx] for idx in joint_points])
                        
                        # Check if any other points fall within this envelope
                        additional_points = self._find_points_in_envelope(joint_envelope, X, joint_points)
                        joint_points.extend(additional_points)
                        
                        # Check purity
                        joint_classes = y[joint_points]
                        if len(np.unique(joint_classes)) == 1:  # Pure
                            new_block = {
                                'points': joint_points,
                                'class': blocks[i]['class'],
                                'envelope': self._create_envelope([X[idx] for idx in joint_points]),
                                'type': 'merger'
                            }
                            blocks[i] = new_block
                            blocks[j] = None
                            merged = True
                
                # Update spatial index after merges
                spatial_index = self._build_spatial_index([b for b in blocks if b is not None], X)
        
        return [block for block in blocks if block is not None]
    
    def _efficient_impure_merge_phase(self, pure_blocks, X, y):
        """Efficient impure merging phase using priority queue and batch processing."""
        blocks = pure_blocks.copy()
        
        # Build spatial index
        spatial_index = self._build_spatial_index(blocks, X)
        
        merged = True
        iteration = 0
        max_iterations = len(blocks)
        
        while merged and iteration < max_iterations:
            merged = False
            iteration += 1
            
            # Find best merge candidates using spatial index
            merge_candidates = self._find_mergeable_pairs_spatial(blocks, spatial_index, X, y, pure_only=False)
            
            if merge_candidates:
                # Evaluate all candidates in batch
                best_merge = None
                best_impurity = float('inf')
                
                for i, j in merge_candidates:
                    if blocks[i] is not None and blocks[j] is not None:
                        # Calculate joint block
                        joint_points = blocks[i]['points'] + blocks[j]['points']
                        joint_envelope = self._create_envelope([X[idx] for idx in joint_points])
                        
                        # Find additional points
                        additional_points = self._find_points_in_envelope(joint_envelope, X, joint_points)
                        joint_points.extend(additional_points)
                        
                        # Calculate impurity
                        joint_classes = y[joint_points]
                        unique_classes, class_counts = np.unique(joint_classes, return_counts=True)
                        impurity = 1 - (np.max(class_counts) / len(joint_classes))
                        
                        if impurity < best_impurity:
                            best_impurity = impurity
                            best_merge = (i, j, joint_points, joint_envelope, impurity)
                
                # Perform best merge if below threshold
                if best_merge and best_merge[4] < self.learned_threshold:
                    i, j, joint_points, joint_envelope, impurity = best_merge
                    joint_classes = y[joint_points]
                    unique_classes, class_counts = np.unique(joint_classes, return_counts=True)
                    dominant_class = unique_classes[np.argmax(class_counts)]
                    
                    new_block = {
                        'points': joint_points,
                        'class': dominant_class,
                        'envelope': joint_envelope,
                        'type': 'merger'
                    }
                    blocks[i] = new_block
                    blocks[j] = None
                    merged = True
                    
                    # Update spatial index
                    spatial_index = self._build_spatial_index([b for b in blocks if b is not None], X)
        
        return [block for block in blocks if block is not None]
    
    def _build_spatial_index(self, blocks, X):
        """Build spatial index for fast overlap detection."""
        if not blocks:
            return None
        
        # Create envelope centers and radii for spatial indexing
        centers = []
        radii = []
        for block in blocks:
            if block is not None and len(block['points']) > 0:
                envelope = block['envelope']
                center = (envelope['min'] + envelope['max']) / 2
                radius = np.linalg.norm(envelope['max'] - envelope['min']) / 2
                centers.append(center)
                radii.append(radius)
        
        if not centers:
            return None
        
        centers = np.array(centers)
        radii = np.array(radii)
        
        # Use BallTree for efficient spatial queries
        tree = BallTree(centers, metric='euclidean')
        
        return {
            'tree': tree,
            'centers': centers,
            'radii': radii,
            'blocks': blocks
        }
    
    def _find_mergeable_pairs_spatial(self, blocks, spatial_index, X, y, pure_only=True):
        """Find mergeable pairs using spatial indexing."""
        if spatial_index is None:
            return []
        
        tree = spatial_index['tree']
        centers = spatial_index['centers']
        radii = spatial_index['radii']
        
        merge_pairs = []
        
        # Create a mapping from block index to spatial index
        valid_blocks = [i for i, block in enumerate(blocks) if block is not None]
        block_to_spatial = {block_idx: spatial_idx for spatial_idx, block_idx in enumerate(valid_blocks)}
        
        for i, block in enumerate(blocks):
            if block is None:
                continue
            
            # Get the corresponding spatial index
            spatial_idx = block_to_spatial[i]
            query_center = centers[spatial_idx]
            query_radius = radii[spatial_idx] * 2  # Search radius
            
            # Query spatial index
            indices, distances = tree.query_radius([query_center], r=query_radius, return_distance=True)
            nearby_spatial_indices = indices[0]
            
            for spatial_j in nearby_spatial_indices:
                # Convert spatial index back to block index
                j = valid_blocks[spatial_j]
                if i != j and blocks[j] is not None:
                    # Check if blocks can be merged
                    if pure_only:
                        if block['class'] == blocks[j]['class']:
                            merge_pairs.append((i, j))
                    else:
                        # For impure merging, check all pairs
                        merge_pairs.append((i, j))
        
        return merge_pairs
    
    def _find_points_in_envelope(self, envelope, X, exclude_points):
        """Find points that fall within an envelope."""
        exclude_set = set(exclude_points)
        points_in_envelope = []
        
        for i, point in enumerate(X):
            if i not in exclude_set and self._point_in_envelope(point, envelope):
                points_in_envelope.append(i)
        
        return points_in_envelope
    
    def _point_in_envelope(self, point, envelope):
        """Check if a point is within an envelope."""
        return np.all(point >= envelope['min']) and np.all(point <= envelope['max'])
    
    def _create_envelope(self, points):
        """Create envelope (min/max bounds) for a set of points."""
        if len(points) == 0:
            return {'min': None, 'max': None}
        points_array = np.array(points)
        return {
            'min': np.min(points_array, axis=0),
            'max': np.max(points_array, axis=0)
        }
    
    def _learn_optimal_impurity_threshold(self, X, y, cv_folds=10):
        """Learn optimal impurity threshold using efficient cross-validation."""
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.3
        best_accuracy = 0
        
        def evaluate_threshold(threshold):
            fold_accuracies = []
            kf = KFold(n_splits=min(cv_folds, len(X)), shuffle=True, random_state=42)
            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                temp_model = EMHyper(impurity_threshold=threshold)
                temp_model._train_implementation(X_train_fold, y_train_fold)
                predictions = temp_model.predict(X_val_fold)
                accuracy = np.mean(predictions == y_val_fold)
                fold_accuracies.append(accuracy)
            return threshold, np.mean(fold_accuracies)
        
        n_jobs = min(N_JOBS, len(thresholds))
        results = Parallel(n_jobs=n_jobs, verbose=VERBOSE_PARALLEL)(
            delayed(evaluate_threshold)(threshold) for threshold in thresholds
        )
        
        for threshold, avg_accuracy in results:
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_threshold = threshold
        
        return best_threshold
    
    def predict(self, X):
        """Predict using efficient distance-based classification."""
        if not self.blocks:
            return np.zeros(len(X), dtype=int)
        
        predictions = np.zeros(len(X), dtype=int)
        
        for sample_idx, sample in enumerate(X):
            best_score = -1
            best_class = 0
            
            for block in self.blocks:
                # Calculate distance to block center
                distance = np.linalg.norm(sample - block['center'])
                score = 1.0 / (1.0 + distance)
                
                if score > best_score:
                    best_score = score
                    best_class = block['class']
            
            predictions[sample_idx] = best_class
        
        return predictions

# --- Efficient IMHyper Implementation ---
class EIMHyper(HyperBlock):
    """
    Efficient Interval Merger Hyper (EIMHyper) algorithm.
    Combines EIHyper and EMHyper with shared spatial indexing and pipelined processing.
    """
    def __init__(self, purity_threshold=0.8, impurity_threshold=None):
        super().__init__("EIMHyper")
        self.purity_threshold = purity_threshold
        self.impurity_threshold = impurity_threshold
        self.learned_impurity_threshold = None
        self.learned_purity_threshold = None
        
    def _train_implementation(self, X, y):
        """Train EIMHyper model using pipelined IHyper and MHyper phases."""
        print(f"Training {self.name}...")
        
        # Learn optimal thresholds if needed
        if self.purity_threshold == 'auto':
            print("  - Learning optimal purity threshold...")
            self.learned_purity_threshold = self._learn_optimal_purity_threshold(X, y)
            print(f"  - Learned purity threshold: {self.learned_purity_threshold:.3f}")
        else:
            self.learned_purity_threshold = self.purity_threshold
            print(f"  - Using purity threshold: {self.learned_purity_threshold}")
            
        if self.impurity_threshold is None:
            print("  - Learning optimal impurity threshold...")
            self.learned_impurity_threshold = self._learn_optimal_impurity_threshold(X, y)
            print(f"  - Learned impurity threshold: {self.learned_impurity_threshold:.3f}")
        else:
            self.learned_impurity_threshold = self.impurity_threshold
            print(f"  - Using impurity threshold: {self.learned_impurity_threshold}")
        
        self.blocks = []
        
        # Phase 1: Efficient IHyper phase
        print("  - Phase 1: Running efficient IHyper...")
        ihyper_blocks = self._run_efficient_ihyper_phase(X, y)
        
        # Phase 2: Find remaining points efficiently
        remaining_indices = self._find_remaining_points_efficient(X, ihyper_blocks)
        
        # Phase 3: Efficient MHyper phase on remaining points
        if len(remaining_indices) > 0:
            print("  - Phase 2: Running efficient MHyper on remaining points...")
            mhyper_blocks = self._run_efficient_mhyper_phase(X, y, remaining_indices, ihyper_blocks)
            self.blocks = ihyper_blocks + mhyper_blocks
        else:
            self.blocks = ihyper_blocks
        
        print(f"  - Created {len(self.blocks)} combined blocks ({len(ihyper_blocks)} interval + {len(self.blocks) - len(ihyper_blocks)} merger)")
    
    def _run_efficient_ihyper_phase(self, X, y):
        """Run efficient IHyper phase with pre-sorted arrays and binary search."""
        blocks = []
        remaining_mask = np.ones(len(X), dtype=bool)
        
        # Pre-sort all attributes for efficient interval generation
        sorted_attributes = self._preprocess_attributes(X, y)
        
        while np.sum(remaining_mask) > 0:
            # Find best interval across all attributes using batch processing
            best_interval = self._find_best_interval_batch(X, y, remaining_mask, sorted_attributes)
            
            if best_interval is None:
                break
                
            # Create hyperblock from best interval
            attr_idx, b_i, d_i, interval_points = best_interval
            interval_classes = y[interval_points]
            
            # Determine dominant class
            unique_classes, class_counts = np.unique(interval_classes, return_counts=True)
            dominant_class = unique_classes[np.argmax(class_counts)]
            
            # Create intervals for all attributes
            feature_intervals = self._create_feature_intervals(X[interval_points])
            
            # Create the hyperblock
            block = {
                'class': dominant_class,
                'intervals': feature_intervals,
                'support': len(interval_points),
                'type': 'imhyper',
                'primary_attribute': attr_idx,
                'primary_interval': (b_i, d_i)
            }
            blocks.append(block)
            
            # Remove covered points
            remaining_mask[interval_points] = False
        
        return blocks
    
    def _preprocess_attributes(self, X, y):
        """Pre-sort all attributes for efficient interval generation."""
        sorted_attributes = []
        for attr_idx in range(X.shape[1]):
            attr_values = X[:, attr_idx]
            sorted_indices = np.argsort(attr_values)
            sorted_attributes.append({
                'values': attr_values[sorted_indices],
                'indices': sorted_indices,
                'classes': y[sorted_indices]
            })
        return sorted_attributes
    
    def _find_best_interval_batch(self, X, y, remaining_mask, sorted_attributes):
        """Find best interval using batch processing and binary search."""
        best_interval = None
        best_support = 0
        
        # Process attributes in parallel
        def process_attribute(attr_idx):
            attr_data = sorted_attributes[attr_idx]
            attr_values = attr_data['values']
            attr_indices = attr_data['indices']
            attr_classes = attr_data['classes']
            
            # Only consider remaining points
            remaining_indices = np.where(remaining_mask)[0]
            remaining_mask_attr = np.isin(attr_indices, remaining_indices)
            
            if np.sum(remaining_mask_attr) == 0:
                return None
            
            # Find best interval for this attribute
            best_attr_interval = None
            best_attr_support = 0
            
            # Use binary search for efficient interval expansion
            for seed_idx in range(len(attr_values)):
                if not remaining_mask_attr[seed_idx]:
                    continue
                    
                seed_value = attr_values[seed_idx]
                seed_class = attr_classes[seed_idx]
                
                # Binary search for right expansion boundary
                right_bound = self._binary_search_expansion(
                    attr_values, attr_classes, seed_idx, seed_class, 'right', remaining_mask_attr
                )
                
                # Binary search for left expansion boundary
                left_bound = self._binary_search_expansion(
                    attr_values, attr_classes, seed_idx, seed_class, 'left', remaining_mask_attr
                )
                
                # Calculate interval support
                interval_mask = (attr_values >= attr_values[left_bound]) & (attr_values <= attr_values[right_bound])
                interval_support = np.sum(interval_mask & remaining_mask_attr)
                
                if interval_support > best_attr_support:
                    best_attr_support = interval_support
                    best_attr_interval = (
                        attr_values[left_bound],
                        attr_values[right_bound],
                        attr_indices[left_bound:right_bound+1]
                    )
            
            return (attr_idx, best_attr_interval[0], best_attr_interval[1], best_attr_interval[2]) if best_attr_interval else None
        
        # Parallelize attribute processing
        n_jobs = min(N_JOBS, X.shape[1])
        results = Parallel(n_jobs=n_jobs, verbose=VERBOSE_PARALLEL)(
            delayed(process_attribute)(attr_idx) for attr_idx in range(X.shape[1])
        )
        
        # Find best interval across all attributes
        for result in results:
            if result is not None:
                attr_idx, b_i, d_i, interval_indices = result
                interval_support = len(interval_indices)
                if interval_support > best_support:
                    best_support = interval_support
                    best_interval = (attr_idx, b_i, d_i, interval_indices)
        
        return best_interval
    
    def _binary_search_expansion(self, values, classes, seed_idx, seed_class, direction, mask):
        """Use binary search to find expansion boundary efficiently."""
        if direction == 'right':
            left, right = seed_idx, len(values) - 1
            while left < right:
                mid = (left + right + 1) // 2
                if mid > right:
                    break
                # Check if we can expand to mid
                if self._check_purity_batch(classes[seed_idx:mid+1], seed_class, mask[seed_idx:mid+1]):
                    left = mid
                else:
                    right = mid - 1
            return left
        else:  # left
            left, right = 0, seed_idx
            while left < right:
                mid = (left + right) // 2
                # Check if we can expand to mid
                if self._check_purity_batch(classes[mid:seed_idx+1], seed_class, mask[mid:seed_idx+1]):
                    right = mid
                else:
                    left = mid + 1
            return right
    
    def _check_purity_batch(self, classes, target_class, mask):
        """Check purity using vectorized operations."""
        if len(classes) == 0:
            return False
        valid_classes = classes[mask]
        if len(valid_classes) == 0:
            return False
        purity = np.sum(valid_classes == target_class) / len(valid_classes)
        return purity >= self.learned_purity_threshold
    
    def _create_feature_intervals(self, points):
        """Create intervals for all features based on point ranges."""
        feature_intervals = []
        for feature_idx in range(points.shape[1]):
            feature_values = points[:, feature_idx]
            min_val = np.min(feature_values)
            max_val = np.max(feature_values)
            feature_intervals.append((min_val, max_val))
        return feature_intervals
    
    def _find_remaining_points_efficient(self, X, ihyper_blocks):
        """Find points not covered by IHyper blocks using vectorized operations."""
        covered_mask = np.zeros(len(X), dtype=bool)
        
        for block in ihyper_blocks:
            if 'intervals' in block:
                # Vectorized point-in-interval check
                for i in range(len(X)):
                    if not covered_mask[i] and self._point_in_intervals_vectorized(X[i], block['intervals']):
                        covered_mask[i] = True
        
        return np.where(~covered_mask)[0]
    
    def _point_in_intervals_vectorized(self, point, intervals):
        """Check if a point is within given intervals using vectorized operations."""
        for i, (min_val, max_val) in enumerate(intervals):
            if not (min_val <= point[i] <= max_val):
                return False
        return True
    
    def _run_efficient_mhyper_phase(self, X, y, remaining_indices, existing_blocks):
        """Run efficient MHyper phase on remaining points with shared spatial indexing."""
        # Create initial blocks from remaining points
        initial_blocks = []
        for i in remaining_indices:
            block = {
                'points': [i],
                'class': y[i],
                'envelope': self._create_envelope([X[i]]),
                'type': 'merger'
            }
            initial_blocks.append(block)
        
        # Add existing IHyper blocks as merger blocks
        for block in existing_blocks:
            if 'intervals' in block:
                # Convert interval block to envelope
                envelope = {
                    'min': np.array([intervals[0] for intervals in block['intervals']]),
                    'max': np.array([intervals[1] for intervals in block['intervals']])
                }
                merger_block = {
                    'points': [],  # Will be filled during merging
                    'class': block['class'],
                    'envelope': envelope,
                    'type': 'merger'
                }
                initial_blocks.append(merger_block)
        
        # Run efficient pure merge phase
        pure_blocks = self._efficient_pure_merge_phase(initial_blocks, X, y)
        
        # Run efficient impure merge phase
        final_blocks = self._efficient_impure_merge_phase(pure_blocks, X, y)
        
        # Convert to standard format
        mhyper_blocks = []
        for block in final_blocks:
            if len(block['points']) > 0:
                block_indices = block['points']
                block_points = X[block_indices]
                block_classes = y[block_indices]
                unique_classes, class_counts = np.unique(block_classes, return_counts=True)
                dominant_class = unique_classes[np.argmax(class_counts)]
                center = np.mean(block_points, axis=0)
                radius = np.std(block_points, axis=0) * 0.5
                final_block = {
                    'class': dominant_class,
                    'center': center,
                    'radius': radius,
                    'support': len(block_points),
                    'type': 'imhyper',
                    'envelope': block['envelope']
                }
                mhyper_blocks.append(final_block)
        
        return mhyper_blocks
    
    def _efficient_pure_merge_phase(self, initial_blocks, X, y):
        """Efficient pure merging phase using spatial indexing."""
        blocks = initial_blocks.copy()
        
        # Build spatial index for fast overlap detection
        spatial_index = self._build_spatial_index(blocks, X)
        
        merged = True
        iteration = 0
        max_iterations = len(blocks)
        
        while merged and iteration < max_iterations:
            merged = False
            iteration += 1
            
            # Find mergeable pairs using spatial index
            merge_pairs = self._find_mergeable_pairs_spatial(blocks, spatial_index, X, y, pure_only=True)
            
            if merge_pairs:
                # Perform batch merge
                for i, j in merge_pairs:
                    if blocks[i] is not None and blocks[j] is not None:
                        # Merge blocks
                        joint_points = blocks[i]['points'] + blocks[j]['points']
                        joint_envelope = self._create_envelope([X[idx] for idx in joint_points])
                        
                        # Check if any other points fall within this envelope
                        additional_points = self._find_points_in_envelope(joint_envelope, X, joint_points)
                        joint_points.extend(additional_points)
                        
                        # Check purity
                        joint_classes = y[joint_points]
                        if len(np.unique(joint_classes)) == 1:  # Pure
                            new_block = {
                                'points': joint_points,
                                'class': blocks[i]['class'],
                                'envelope': self._create_envelope([X[idx] for idx in joint_points]),
                                'type': 'merger'
                            }
                            blocks[i] = new_block
                            blocks[j] = None
                            merged = True
                
                # Update spatial index after merges
                spatial_index = self._build_spatial_index([b for b in blocks if b is not None], X)
        
        return [block for block in blocks if block is not None]
    
    def _efficient_impure_merge_phase(self, pure_blocks, X, y):
        """Efficient impure merging phase using priority queue and batch processing."""
        blocks = pure_blocks.copy()
        
        # Build spatial index
        spatial_index = self._build_spatial_index(blocks, X)
        
        merged = True
        iteration = 0
        max_iterations = len(blocks)
        
        while merged and iteration < max_iterations:
            merged = False
            iteration += 1
            
            # Find best merge candidates using spatial index
            merge_candidates = self._find_mergeable_pairs_spatial(blocks, spatial_index, X, y, pure_only=False)
            
            if merge_candidates:
                # Evaluate all candidates in batch
                best_merge = None
                best_impurity = float('inf')
                
                for i, j in merge_candidates:
                    if blocks[i] is not None and blocks[j] is not None:
                        # Calculate joint block
                        joint_points = blocks[i]['points'] + blocks[j]['points']
                        joint_envelope = self._create_envelope([X[idx] for idx in joint_points])
                        
                        # Find additional points
                        additional_points = self._find_points_in_envelope(joint_envelope, X, joint_points)
                        joint_points.extend(additional_points)
                        
                        # Calculate impurity
                        joint_classes = y[joint_points]
                        unique_classes, class_counts = np.unique(joint_classes, return_counts=True)
                        impurity = 1 - (np.max(class_counts) / len(joint_classes))
                        
                        if impurity < best_impurity:
                            best_impurity = impurity
                            best_merge = (i, j, joint_points, joint_envelope, impurity)
                
                # Perform best merge if below threshold
                if best_merge and best_merge[4] < self.learned_impurity_threshold:
                    i, j, joint_points, joint_envelope, impurity = best_merge
                    joint_classes = y[joint_points]
                    unique_classes, class_counts = np.unique(joint_classes, return_counts=True)
                    dominant_class = unique_classes[np.argmax(class_counts)]
                    
                    new_block = {
                        'points': joint_points,
                        'class': dominant_class,
                        'envelope': joint_envelope,
                        'type': 'merger'
                    }
                    blocks[i] = new_block
                    blocks[j] = None
                    merged = True
                    
                    # Update spatial index
                    spatial_index = self._build_spatial_index([b for b in blocks if b is not None], X)
        
        return [block for block in blocks if block is not None]
    
    def _build_spatial_index(self, blocks, X):
        """Build spatial index for fast overlap detection."""
        if not blocks:
            return None
        
        # Create envelope centers and radii for spatial indexing
        centers = []
        radii = []
        for block in blocks:
            if block is not None and len(block['points']) > 0:
                envelope = block['envelope']
                center = (envelope['min'] + envelope['max']) / 2
                radius = np.linalg.norm(envelope['max'] - envelope['min']) / 2
                centers.append(center)
                radii.append(radius)
        
        if not centers:
            return None
        
        centers = np.array(centers)
        radii = np.array(radii)
        
        # Use BallTree for efficient spatial queries
        from sklearn.neighbors import BallTree
        tree = BallTree(centers, metric='euclidean')
        
        return {
            'tree': tree,
            'centers': centers,
            'radii': radii,
            'blocks': blocks
        }
    
    def _find_mergeable_pairs_spatial(self, blocks, spatial_index, X, y, pure_only=True):
        """Find mergeable pairs using spatial indexing."""
        if spatial_index is None:
            return []
        
        tree = spatial_index['tree']
        centers = spatial_index['centers']
        radii = spatial_index['radii']
        
        merge_pairs = []
        
        # Create a mapping from block index to spatial index
        valid_blocks = [i for i, block in enumerate(blocks) if block is not None]
        block_to_spatial = {block_idx: spatial_idx for spatial_idx, block_idx in enumerate(valid_blocks)}
        
        for i, block in enumerate(blocks):
            if block is None:
                continue
            
            # Get the corresponding spatial index
            spatial_idx = block_to_spatial[i]
            query_center = centers[spatial_idx]
            query_radius = radii[spatial_idx] * 2  # Search radius
            
            # Query spatial index
            indices, distances = tree.query_radius([query_center], r=query_radius, return_distance=True)
            nearby_spatial_indices = indices[0]
            
            for spatial_j in nearby_spatial_indices:
                # Convert spatial index back to block index
                j = valid_blocks[spatial_j]
                if i != j and blocks[j] is not None:
                    # Check if blocks can be merged
                    if pure_only:
                        if block['class'] == blocks[j]['class']:
                            merge_pairs.append((i, j))
                    else:
                        # For impure merging, check all pairs
                        merge_pairs.append((i, j))
        
        return merge_pairs
    
    def _find_points_in_envelope(self, envelope, X, exclude_points):
        """Find points that fall within an envelope."""
        exclude_set = set(exclude_points)
        points_in_envelope = []
        
        for i, point in enumerate(X):
            if i not in exclude_set and self._point_in_envelope(point, envelope):
                points_in_envelope.append(i)
        
        return points_in_envelope
    
    def _point_in_envelope(self, point, envelope):
        """Check if a point is within an envelope."""
        return np.all(point >= envelope['min']) and np.all(point <= envelope['max'])
    
    def _create_envelope(self, points):
        """Create envelope (min/max bounds) for a set of points."""
        if len(points) == 0:
            return {'min': None, 'max': None}
        points_array = np.array(points)
        return {
            'min': np.min(points_array, axis=0),
            'max': np.max(points_array, axis=0)
        }
    
    def _learn_optimal_impurity_threshold(self, X, y, cv_folds=10):
        """Learn optimal impurity threshold using efficient cross-validation."""
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.3
        best_accuracy = 0
        
        def evaluate_threshold(threshold):
            fold_accuracies = []
            kf = KFold(n_splits=min(cv_folds, len(X)), shuffle=True, random_state=42)
            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                temp_model = EIMHyper(purity_threshold=self.purity_threshold, impurity_threshold=threshold)
                temp_model._train_implementation(X_train_fold, y_train_fold)
                predictions = temp_model.predict(X_val_fold)
                accuracy = np.mean(predictions == y_val_fold)
                fold_accuracies.append(accuracy)
            return threshold, np.mean(fold_accuracies)
        
        n_jobs = min(N_JOBS, len(thresholds))
        results = Parallel(n_jobs=n_jobs, verbose=VERBOSE_PARALLEL)(
            delayed(evaluate_threshold)(threshold) for threshold in thresholds
        )
        
        for threshold, avg_accuracy in results:
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_threshold = threshold
        
        return best_threshold
    
    def _learn_optimal_purity_threshold(self, X, y, cv_folds=10):
        """Learn optimal purity threshold using efficient cross-validation."""
        thresholds = np.arange(0.6, 1.0, 0.05)
        best_threshold = 0.8
        best_accuracy = 0
        
        def evaluate_threshold(threshold):
            fold_accuracies = []
            kf = KFold(n_splits=min(cv_folds, len(X)), shuffle=True, random_state=42)
            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                temp_model = EIMHyper(purity_threshold=threshold, impurity_threshold=self.impurity_threshold)
                temp_model._train_implementation(X_train_fold, y_train_fold)
                predictions = temp_model.predict(X_val_fold)
                accuracy = np.mean(predictions == y_val_fold)
                fold_accuracies.append(accuracy)
            return threshold, np.mean(fold_accuracies)
        
        n_jobs = min(N_JOBS, len(thresholds))
        results = Parallel(n_jobs=n_jobs, verbose=VERBOSE_PARALLEL)(
            delayed(evaluate_threshold)(threshold) for threshold in thresholds
        )
        
        for threshold, avg_accuracy in results:
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_threshold = threshold
        
        return best_threshold
    
    def predict(self, X):
        """Predict using efficient combined classification."""
        if not self.blocks:
            return np.zeros(len(X), dtype=int)
        
        predictions = np.zeros(len(X), dtype=int)
        
        for sample_idx, sample in enumerate(X):
            best_score = -1
            best_class = 0
            
            for block in self.blocks:
                if block['type'] == 'imhyper':
                    if 'intervals' in block:
                        score = self._calculate_interval_score_vectorized(sample, block['intervals'])
                    else:
                        # Fall back to distance-based scoring
                        center = block['center']
                        radius = block['radius']
                        distance = np.linalg.norm(sample - center)
                        score = 1.0 / (1.0 + distance)
                    
                    if score > best_score:
                        best_score = score
                        best_class = block['class']
            
            predictions[sample_idx] = best_class
        
        return predictions
    
    def _calculate_interval_score_vectorized(self, sample, intervals):
        """Calculate interval score using vectorized operations."""
        total_distance = 0
        for i, (min_val, max_val) in enumerate(intervals):
            if sample[i] < min_val:
                total_distance += (min_val - sample[i]) ** 2
            elif sample[i] > max_val:
                total_distance += (sample[i] - max_val) ** 2
        return 1.0 / (1.0 + np.sqrt(total_distance))

# --- Efficient HyperBlock Predictor ---
class HyperBlockPredictor:
    """
    Efficient predictor class for k-nearest hyperblock classification with Copeland rule.
    Uses vectorized operations and parallel processing for speed.
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
        Check if a point is completely inside a hyperblock using vectorized operations.
        
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
        elif block['type'] == 'imhyper':
            # For IMHyper blocks, check if intervals are available
            if 'intervals' in block:
                intervals = block['intervals']
                for i, (min_val, max_val) in enumerate(intervals):
                    if not (min_val <= sample[i] <= max_val):
                        return False
                return True
            elif 'center' in block and 'radius' in block:
                # Fall back to merger-style check
                center = block['center']
                radius = block['radius']
                distance = np.linalg.norm(sample - center)
                return distance <= np.mean(radius)
            else:
                return False
        else:
            return False
    
    def _calculate_distance_to_block(self, sample, block):
        """
        Calculate distance from point to hyperblock boundary using vectorized operations.
        
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
        elif block['type'] == 'imhyper':
            # For IMHyper blocks, use interval distance if available, otherwise merger distance
            if 'intervals' in block:
                intervals = block['intervals']
                interval_distance = 0
                for i, (min_val, max_val) in enumerate(intervals):
                    if sample[i] < min_val:
                        interval_distance += (min_val - sample[i]) ** 2
                    elif sample[i] > max_val:
                        interval_distance += (sample[i] - max_val) ** 2
                return np.sqrt(interval_distance)
            elif 'center' in block and 'radius' in block:
                # Fall back to merger distance
                center = block['center']
                radius = block['radius']
                distance = np.linalg.norm(sample - center)
                avg_radius = np.mean(radius)
                return max(0, distance - avg_radius)
            else:
                return float('inf')
        else:
            return float('inf')
    
    def _learn_optimal_k(self, X_train, y_train, model_name, max_k=10):
        """
        Learn optimal k value for k-nearest hyperblock classification using parallel processing.
        
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
        
        def evaluate_k(k):
            """Evaluate a single k value."""
            predictions = self._predict_with_k(X_train, model_name, k)
            accuracy = np.mean(predictions == y_train)
            return k, accuracy
        
        # Parallelize k evaluation
        k_values = range(1, min(max_k + 1, len(self.blocks_by_model[model_name]) + 1))
        n_jobs = min(multiprocessing.cpu_count(), len(k_values))
        
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(evaluate_k)(k) for k in k_values
        )
        
        # Find best k
        for k, accuracy in results:
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
        Uses parallel processing for large datasets.
        
        Args:
            X: Input features
            model_name: Name of the model to use
            k: Number of nearest hyperblocks
            
        Returns:
            predictions: Predicted class labels
        """
        blocks = self.blocks_by_model[model_name]
        
        def predict_single_sample(sample_idx):
            """Predict for a single sample."""
            sample = X[sample_idx]
            
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
            
            return prediction
        
        # Parallelize prediction for large datasets
        if len(X) > PARALLEL_THRESHOLD:
            n_jobs = min(multiprocessing.cpu_count(), len(X))
            predictions = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(predict_single_sample)(i) for i in range(len(X))
            )
        else:
            # Sequential processing for small datasets
            predictions = [predict_single_sample(i) for i in range(len(X))]
        
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
    
    def learn_optimal_k(self, X_train, y_train, max_k=21):
        """
        Learn optimal k values for all models using parallel processing.
        
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
        y_raw = df.iloc[:, -1].values
        
        # Encode string labels to integers
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        
        dataset_name = "Fisher Iris"
        print(f"  - Classes: {label_encoder.classes_} -> {list(range(len(label_encoder.classes_)))}")
        
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
        y_raw = df.iloc[:, -1].values
        
        # Encode string labels to integers
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        
        dataset_name = "WBC (9-D)"
        print(f"  - Classes: {label_encoder.classes_} -> {list(range(len(label_encoder.classes_)))}")
        
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
        y_raw = df.iloc[:, -1].values
        
        # Encode string labels to integers
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        
        dataset_name = "WBC (30-D)"
        print(f"  - Classes: {label_encoder.classes_} -> {list(range(len(label_encoder.classes_)))}")
        
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
        y_train_raw = train_df.iloc[:, -1].values
        X_test = test_df.iloc[:, :-1].values
        y_test_raw = test_df.iloc[:, -1].values
        
        # Encode string labels to integers
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train_raw)
        y_test = label_encoder.transform(y_test_raw)
        
        dataset_name = "Dimensionally Reduced MNIST (121-D)"
        print(f"  - Classes: {label_encoder.classes_} -> {list(range(len(label_encoder.classes_)))}")
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
        y_train_raw = train_df.iloc[:, -1].values
        X_test = test_df.iloc[:, :-1].values
        y_test_raw = test_df.iloc[:, -1].values
        
        # Encode string labels to integers
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train_raw)
        y_test = label_encoder.transform(y_test_raw)
        
        dataset_name = "Full MNIST (784-D)"
        print(f"  - Classes: {label_encoder.classes_} -> {list(range(len(label_encoder.classes_)))}")
        return X_train, X_test, y_train, y_test, dataset_name
    else:
        raise ValueError("Invalid choice") 

# --- Main unified workflow ---
def main():
    print("=" * 60)
    print("Efficient HyperBlock System")
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
        
        def process_fold(fold_info):
            """Process a single fold - train models and evaluate."""
            fold_num = fold_info['fold_number']
            train_idx = np.array(fold_info['fold_indices'])
            test_idx = np.array(fold_info['test_indices'])
            X_train, X_test = X_full[train_idx], X_full[test_idx]
            y_train, y_test = y_full[train_idx], y_full[test_idx]
            
            # Train models
            models = {
                'EIHyper': EIHyper(purity_threshold='auto'),
                'EMHyper': EMHyper(impurity_threshold=None),  # Will be learned
                'EIMHyper': EIMHyper(purity_threshold='auto', impurity_threshold=None)  # Both will be learned
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
            
            return fold_num, fold_results
        
        # Parallelize fold processing
        n_jobs = min(multiprocessing.cpu_count(), len(all_folds))
        print(f"  - Processing {len(all_folds)} folds using {n_jobs} parallel processes...")
        
        fold_results_list = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(process_fold)(fold_info) for fold_info in all_folds
        )
        
        # Collect results
        for fold_num, fold_results in fold_results_list:
            all_fold_results[fold_num] = fold_results
        
        # Aggregate results
        model_names = ['EIHyper', 'EMHyper', 'EIMHyper']
        avg_results = {}
        blocks_per_fold = {model_name: [] for model_name in model_names}
        for fold in range(10):
            for model_name in model_names:
                blocks_per_fold[model_name].append(all_fold_results[fold][model_name]['num_blocks'])
        for model_name in model_names:
            accuracies = [all_fold_results[fold][model_name]['accuracy'] for fold in range(10)]
            avg_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            avg_results[model_name] = {
                'mean_accuracy': avg_accuracy,
                'std_accuracy': std_accuracy,
                'fold_accuracies': accuracies,
                'optimal_k': all_fold_results[0][model_name]['optimal_k'],
                'blocks_per_fold': blocks_per_fold[model_name]
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
            print(f"Number of Blocks per Fold: {result['blocks_per_fold']}")
        
        print("\nDone.")
    
    # --- Train/test split datasets ---
    else:
        X_train, X_test, y_train, y_test, dataset_name = data
        print(f"  - Training samples: {len(X_train)}")
        print(f"  - Test samples: {len(X_test)}")
        print(f"  - Features: {X_train.shape[1]}")
        print(f"  - Classes: {len(np.unique(y_train))}")
        
        # Train models with algorithm parameters (thresholds will be learned)
        models = {
            'EIHyper': EIHyper(purity_threshold='auto'),
            'EMHyper': EMHyper(impurity_threshold=None),  # Will be learned
            'EIMHyper': EIMHyper(purity_threshold='auto', impurity_threshold=None)  # Both will be learned
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