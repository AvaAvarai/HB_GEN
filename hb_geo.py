"""
Hyperblock Geometric Classification Algorithm

This module implements a hyperblock-based classification algorithm that constructs
geometric regions (hyperblocks) around class points and uses distance-based
classification with k-NN fallback.

Author: Alice Williams
"""

import os
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Global diagnostic flag for controlling logging output
DIAGNOSTIC_MODE = False

# Hyperblock parameters
DEFAULT_VOLUME_THRESHOLD = 1e-10
DEFAULT_DUPLICATE_TOLERANCE = 1e-6
DEFAULT_MISCLASSIFICATION_THRESHOLD = 0.1
DEFAULT_SHRINK_EPSILON = 0.1
DEFAULT_MIN_POINTS_FOR_PENALTY = 3
DEFAULT_MAX_SPLITS = 3
DEFAULT_MIN_POINTS_PER_SPLIT = 2

# Cross-validation parameters
DEFAULT_K_FOLDS = 10
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_PERCENTAGE = 0.3  # 30% default test split

# =============================================================================
# LOGGING UTILITY FUNCTIONS
# =============================================================================

def diagnostic_print(*args, **kwargs):
    """
    Print function that only outputs when diagnostic mode is enabled.
    
    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print function
    """
    if DIAGNOSTIC_MODE:
        print(*args, **kwargs)

# =============================================================================
# CORE GEOMETRIC FUNCTIONS
# =============================================================================

def distance_to_hyperblock(point, bounds, norm=2):
    """
    Calculate the distance from a point to a hyperblock (box) using the given norm.
    
    Args:
        point (np.ndarray): Point coordinates
        bounds (np.ndarray): Hyperblock bounds as (n_features, 2) array [min, max]
        norm (int): Distance norm (1 for L1, 2 for L2, 'inf' for Linf)
    
    Returns:
        float: Distance from point to hyperblock
        
    Note:
        Linf (Chebyshev) distance is particularly natural for axis-aligned boxes as it
        equals the maximum componentwise violation, reducing "almost contained" ties
        that L1 distance tends to favor.
    """
    projected = np.clip(point, bounds[:, 0], bounds[:, 1])
    diff = point - projected
    
    if norm == 1:
        return np.sum(np.abs(diff))
    elif norm == 2:
        return np.sqrt(np.sum(diff * diff))
    elif norm == 'inf' or norm == float('inf'):
        return np.max(np.abs(diff))
    else:
        return np.sqrt(np.sum(diff * diff))


def bounds_calculation(points, num_features):
    """
    Calculate min/max bounds for a set of points.
    
    Args:
        points (np.ndarray): Array of points
        num_features (int): Number of features
    
    Returns:
        np.ndarray: Bounds as (n_features, 2) array [min, max]
    """
    if len(points) == 0:
        return np.full((num_features, 2), [np.inf, -np.inf])
    
    bounds = np.column_stack([
        np.min(points, axis=0),
        np.max(points, axis=0)
    ])
    return bounds


def intersection_refinement(class_points, other_points, sorted_features, num_features):
    """
    Refine bounds by checking for intersection patterns with other classes.
    
    Args:
        class_points (np.ndarray): Points from the target class
        other_points (np.ndarray): Points from other classes
        sorted_features (np.ndarray): Feature indices sorted by importance
        num_features (int): Number of features
    
    Returns:
        tuple: (refined_bounds, has_intersection)
    """
    if len(other_points) == 0:
        return bounds_calculation(class_points, num_features), False
    
    refined_bounds = np.full((num_features, 2), [np.inf, -np.inf])
    has_intersection = False
    
    for i in range(len(sorted_features)):
        left_idx = sorted_features[i - 1] if i > 0 else sorted_features[-1]
        right_idx = sorted_features[i + 1] if i < len(sorted_features) - 1 else sorted_features[0]
        curr_idx = sorted_features[i]
        
        # Extract values for current and adjacent features
        p_curr_vals = class_points[:, curr_idx]
        p_left_vals = class_points[:, left_idx]
        p_right_vals = class_points[:, right_idx]
        q_curr_vals = other_points[:, curr_idx]
        q_left_vals = other_points[:, left_idx]
        q_right_vals = other_points[:, right_idx]
        
        for p_idx in range(len(class_points)):
            p_curr = p_curr_vals[p_idx]
            p_left = p_left_vals[p_idx]
            p_right = p_right_vals[p_idx]
            
            curr_diff = p_curr - q_curr_vals
            left_diff = p_left - q_left_vals
            right_diff = p_right - q_right_vals
            
            intersection_mask = (curr_diff * left_diff < 0) | (curr_diff * right_diff < 0)
            
            if np.any(intersection_mask):
                refined_bounds[curr_idx, 0] = min(refined_bounds[curr_idx, 0], p_curr)
                refined_bounds[curr_idx, 1] = max(refined_bounds[curr_idx, 1], p_curr)
                refined_bounds[left_idx, 0] = min(refined_bounds[left_idx, 0], p_left)
                refined_bounds[left_idx, 1] = max(refined_bounds[left_idx, 1], p_left)
                refined_bounds[right_idx, 0] = min(refined_bounds[right_idx, 0], p_right)
                refined_bounds[right_idx, 1] = max(refined_bounds[right_idx, 1], p_right)
                has_intersection = True
    
    return refined_bounds, has_intersection


# =============================================================================
# HYPERBLOCK CONSTRUCTION
# =============================================================================

def create_interval_groups(class_points, num_intervals_per_dim=2, min_points_per_group=2):
    """
    Create interval-based groups by dividing each dimension into intervals and 
    grouping points that fall into the same intervals across all dimensions.
    
    Args:
        class_points (np.ndarray): Points from the target class
        num_intervals_per_dim (int): Number of intervals to create per dimension
        min_points_per_group (int): Minimum points required for a valid group
    
    Returns:
        list: List of group indices for each point, or None if no valid groups found
    """
    if len(class_points) < min_points_per_group:
        return None
    
    num_features = class_points.shape[1]
    num_points = len(class_points)
    
    # Calculate adaptive intervals for each dimension based on data distribution
    intervals = []
    for dim in range(num_features):
        dim_values = class_points[:, dim]
        min_val = np.min(dim_values)
        max_val = np.max(dim_values)
        
        # Create adaptive intervals based on data distribution
        if max_val > min_val and num_intervals_per_dim > 1:
            # Use percentiles for more balanced intervals
            percentiles = np.linspace(0, 100, num_intervals_per_dim + 1)
            interval_boundaries = np.percentile(dim_values, percentiles)
            
            dim_intervals = []
            for i in range(num_intervals_per_dim):
                lower = interval_boundaries[i]
                upper = interval_boundaries[i + 1]
                # Add small overlap to handle boundary cases
                if i > 0:
                    lower -= (upper - lower) * 0.05  # 5% overlap
                if i < num_intervals_per_dim - 1:
                    upper += (upper - lower) * 0.05  # 5% overlap
                dim_intervals.append((lower, upper))
        else:
            # All values are the same or single interval - create single interval
            dim_intervals = [(min_val, max_val)]
        
        intervals.append(dim_intervals)
    
    # Assign each point to an interval group
    group_assignments = []
    for point_idx in range(num_points):
        point = class_points[point_idx]
        group_coords = []
        
        for dim in range(num_features):
            dim_val = point[dim]
            # Find which interval this value falls into
            for interval_idx, (lower, upper) in enumerate(intervals[dim]):
                if lower <= dim_val <= upper:
                    group_coords.append(interval_idx)
                    break
            else:
                # Should not happen, but fallback
                group_coords.append(0)
        
        group_assignments.append(tuple(group_coords))
    
    # Count points in each group
    group_counts = {}
    for group_coords in group_assignments:
        group_counts[group_coords] = group_counts.get(group_coords, 0) + 1
    
    # Filter groups with too few points
    valid_groups = {coords for coords, count in group_counts.items() 
                   if count >= min_points_per_group}
    
    if not valid_groups:
        # If no valid groups, try with fewer intervals
        if num_intervals_per_dim > 1:
            return create_interval_groups(class_points, num_intervals_per_dim - 1, min_points_per_group)
        else:
            return None
    
    # Create group labels
    group_to_label = {coords: label for label, coords in enumerate(valid_groups)}
    group_labels = [group_to_label.get(coords, -1) for coords in group_assignments]
    
    # Check if we have valid groups
    valid_labels = [label for label in group_labels if label != -1]
    if len(valid_labels) < min_points_per_group:
        return None
    
    return group_labels


def find_optimal_interval_groups(class_points, max_intervals_per_dim=4):
    """
    Find optimal number of intervals per dimension for interval-based grouping.
    
    Args:
        class_points (np.ndarray): Points from the target class
        max_intervals_per_dim (int): Maximum intervals per dimension to consider
    
    Returns:
        int: Optimal number of intervals per dimension
    """
    if len(class_points) < 2:
        return 1
    
    best_num_intervals = 1
    best_group_count = 1
    best_coverage = 1.0  # Percentage of points covered by valid groups
    
    for num_intervals in range(1, max_intervals_per_dim + 1):
        group_labels = create_interval_groups(class_points, num_intervals, min_points_per_group=2)
        
        if group_labels is not None:
            unique_groups = len(set(group_labels))
            coverage = len([label for label in group_labels if label != -1]) / len(class_points)
            
            # Prefer more groups with good coverage, but avoid too many tiny groups
            if unique_groups > best_group_count and coverage >= 0.8:
                best_group_count = unique_groups
                best_num_intervals = num_intervals
                best_coverage = coverage
            elif unique_groups == best_group_count and coverage > best_coverage:
                best_num_intervals = num_intervals
                best_coverage = coverage
        else:
            # If grouping fails, stop here
            break
    
    return best_num_intervals


def compute_envelope_blocks_for_class(args):
    """
    Compute envelope blocks for a single class using interval-based grouping.
    
    Args:
        args (tuple): (class_label, X, y, feature_indices)
    
    Returns:
        list: List of hyperblock dictionaries
    """
    class_label, X, y, feature_indices = args
    
    # Separate class points from other points
    class_mask = y == class_label
    other_mask = y != class_label
    class_points = X[class_mask]
    other_points = X[other_mask]
    
    if class_points.shape[0] == 0:
        return []
    
    num_features = X.shape[1]
    blocks = []
    
    # Use sequential feature ordering instead of LDA-based importance
    sorted_features = np.arange(num_features)
    
    # Find optimal number of intervals per dimension
    best_num_intervals = find_optimal_interval_groups(class_points, max_intervals_per_dim=4)
    
    # Create interval groups
    group_labels = create_interval_groups(class_points, best_num_intervals, min_points_per_group=2)
    
    if group_labels is None or len(set(group_labels)) <= 1:
        # Single group case - create one hyperblock
        bounds = bounds_calculation(class_points, num_features)
        
        if other_points.shape[0] > 0:
            refined_bounds, has_intersection = intersection_refinement(
                class_points, other_points, sorted_features, num_features
            )
            if has_intersection:
                bounds = refined_bounds
        
        blocks.append({
            'class': class_label,
            'bounds': bounds.tolist(),
            'feature_order': sorted_features.tolist(),
            'grouping_method': 'interval',
            'num_intervals': best_num_intervals
        })
    else:
        # Multiple groups case - create hyperblock for each group
        unique_groups = set(group_labels)
        
        for group_id in unique_groups:
            group_mask = np.array(group_labels) == group_id
            group_points = class_points[group_mask]
            
            if group_points.shape[0] == 0:
                continue
            
            bounds = bounds_calculation(group_points, num_features)
            
            # Add small margin to reduce boundary misclassifications
            margin = 0.02  # 2% margin
            for dim in range(num_features):
                range_size = bounds[dim, 1] - bounds[dim, 0]
                margin_size = range_size * margin
                bounds[dim, 0] = max(0, bounds[dim, 0] - margin_size)
                bounds[dim, 1] = min(1, bounds[dim, 1] + margin_size)
            
            if other_points.shape[0] > 0:
                refined_bounds, has_intersection = intersection_refinement(
                    group_points, other_points, sorted_features, num_features
                )
                if has_intersection:
                    bounds = refined_bounds
            
            blocks.append({
                'class': class_label,
                'bounds': bounds.tolist(),
                'feature_order': sorted_features.tolist(),
                'group_id': group_id,
                'grouping_method': 'interval',
                'num_intervals': best_num_intervals,
                'group_size': len(group_points)
            })
    
    return blocks


# =============================================================================
# HYPERBLOCK POST-PROCESSING
# =============================================================================

def check_block_purity_and_refine(blocks, X, y, max_splits=3, min_points_per_split=2):
    """
    Check if blocks contain opposite-class points and refine them until pure.
    
    Args:
        blocks (list): List of hyperblock dictionaries
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target labels
        max_splits (int): Maximum number of splits per block
        min_points_per_split (int): Minimum points required for a split
    
    Returns:
        list: Refined blocks that are pure (contain only target class points)
    """
    if not blocks:
        return blocks
    
    refined_blocks = []
    total_original_blocks = len(blocks)
    blocks_refined = 0
    
    for block in blocks:
        bounds = np.array(block['bounds'])
        block_class = block['class']
        
        # Find all points within this block
        block_mask = np.ones(len(X), dtype=bool)
        for dim in range(len(bounds)):
            block_mask &= (X[:, dim] >= bounds[dim, 0]) & (X[:, dim] <= bounds[dim, 1])
        
        block_points = X[block_mask]
        block_point_classes = y[block_mask]
        
        if len(block_points) == 0:
            # Empty block - skip
            continue
        
        # Check if block is pure (contains only target class points)
        other_class_mask = block_point_classes != block_class
        if not np.any(other_class_mask):
            # Block is pure - keep as is
            refined_blocks.append(block)
            continue
        
        # Block contains opposite-class points - need to refine
        blocks_refined += 1
        refined_sub_blocks = _split_block_until_pure(
            block, block_points, block_point_classes, X, y, 
            max_splits, min_points_per_split
        )
        refined_blocks.extend(refined_sub_blocks)
    
    if blocks_refined > 0:
        diagnostic_print(f"Purity refinement: {blocks_refined}/{total_original_blocks} blocks refined "
              f"({len(refined_blocks)} total blocks after refinement)")
    
    return refined_blocks


def _split_block_until_pure(block, block_points, block_point_classes, X, y, 
                           max_splits, min_points_per_split):
    """
    Recursively split a block until it's pure or max splits reached.
    
    Args:
        block (dict): Original block dictionary
        block_points (np.ndarray): Points within the block
        block_point_classes (np.ndarray): Classes of points within the block
        X (np.ndarray): Full feature matrix
        y (np.ndarray): Full target labels
        max_splits (int): Maximum number of splits
        min_points_per_split (int): Minimum points required for a split
    
    Returns:
        list: List of pure sub-blocks
    """
    block_class = block['class']
    bounds = np.array(block['bounds'])
    
    # Check if block is pure
    other_class_mask = block_point_classes != block_class
    if not np.any(other_class_mask):
        return [block]
    
    # Check if we've reached max splits or too few points
    if max_splits <= 0 or len(block_points) < min_points_per_split * 2:
        # Can't split further - try shrinking instead
        return _shrink_block_to_purity(block, block_points, block_point_classes, X, y)
    
    # Find the best feature to split on
    best_feature, best_split_value = _find_best_split_feature(
        block_points, block_point_classes, block_class, bounds
    )
    
    if best_feature is None:
        # No good split found - try shrinking
        return _shrink_block_to_purity(block, block_points, block_point_classes, X, y)
    
    # Log the split decision
    other_class_count = np.sum(block_point_classes != block_class)
    target_class_count = np.sum(block_point_classes == block_class)
    diagnostic_print(f"  Splitting block (class {block_class}): {target_class_count} target, {other_class_count} other points")
    
    # Create two sub-blocks
    sub_blocks = []
    
    # Left sub-block (feature <= split_value)
    left_bounds = bounds.copy()
    left_bounds[best_feature, 1] = best_split_value
    
    left_mask = block_points[:, best_feature] <= best_split_value
    left_points = block_points[left_mask]
    left_classes = block_point_classes[left_mask]
    
    if len(left_points) >= min_points_per_split:
        left_block = block.copy()
        left_block['bounds'] = left_bounds.tolist()
        left_block['split_feature'] = best_feature
        left_block['split_value'] = best_split_value
        left_block['split_direction'] = 'left'
        
        left_sub_blocks = _split_block_until_pure(
            left_block, left_points, left_classes, X, y, 
            max_splits - 1, min_points_per_split
        )
        sub_blocks.extend(left_sub_blocks)
    
    # Right sub-block (feature > split_value)
    right_bounds = bounds.copy()
    right_bounds[best_feature, 0] = best_split_value
    
    right_mask = block_points[:, best_feature] > best_split_value
    right_points = block_points[right_mask]
    right_classes = block_point_classes[right_mask]
    
    if len(right_points) >= min_points_per_split:
        right_block = block.copy()
        right_block['bounds'] = right_bounds.tolist()
        right_block['split_feature'] = best_feature
        right_block['split_value'] = best_split_value
        right_block['split_direction'] = 'right'
        
        right_sub_blocks = _split_block_until_pure(
            right_block, right_points, right_classes, X, y, 
            max_splits - 1, min_points_per_split
        )
        sub_blocks.extend(right_sub_blocks)
    
    return sub_blocks


def _find_best_split_feature(block_points, block_point_classes, block_class, bounds):
    """
    Find the best feature and split value to separate target class from others.
    
    Args:
        block_points (np.ndarray): Points within the block
        block_point_classes (np.ndarray): Classes of points within the block
        block_class (int): Target class
        bounds (np.ndarray): Current block bounds
    
    Returns:
        tuple: (best_feature, best_split_value) or (None, None) if no good split
    """
    num_features = block_points.shape[1]
    best_feature = None
    best_split_value = None
    best_purity_gain = -1
    
    target_mask = block_point_classes == block_class
    other_mask = block_point_classes != block_class
    
    if not np.any(target_mask) or not np.any(other_mask):
        return None, None
    
    for feature in range(num_features):
        feature_values = block_points[:, feature]
        target_values = feature_values[target_mask]
        other_values = feature_values[other_mask]
        
        if len(target_values) == 0 or len(other_values) == 0:
            continue
        
        # Find potential split points
        all_values = np.concatenate([target_values, other_values])
        unique_values = np.unique(all_values)
        
        if len(unique_values) < 2:
            continue
        
        # Sort values and try midpoints as split points
        sorted_values = np.sort(unique_values)
        split_candidates = (sorted_values[:-1] + sorted_values[1:]) / 2
        
        for split_value in split_candidates:
            # Skip if split is outside current bounds
            if split_value <= bounds[feature, 0] or split_value >= bounds[feature, 1]:
                continue
            
            # Calculate purity gain
            left_mask = feature_values <= split_value
            right_mask = feature_values > split_value
            
            if not np.any(left_mask) or not np.any(right_mask):
                continue
            
            left_target = np.sum(target_mask & left_mask)
            left_other = np.sum(other_mask & left_mask)
            right_target = np.sum(target_mask & right_mask)
            right_other = np.sum(other_mask & right_mask)
            
            # Calculate purity for each side
            left_purity = left_target / (left_target + left_other) if (left_target + left_other) > 0 else 0
            right_purity = right_target / (right_target + right_other) if (right_target + right_other) > 0 else 0
            
            # Overall purity gain (weighted average)
            left_weight = (left_target + left_other) / len(block_points)
            right_weight = (right_target + right_other) / len(block_points)
            weighted_purity = left_weight * left_purity + right_weight * right_purity
            
            if weighted_purity > best_purity_gain:
                best_purity_gain = weighted_purity
                best_feature = feature
                best_split_value = split_value
    
    return best_feature, best_split_value


def _shrink_block_to_purity(block, block_points, block_point_classes, X, y):
    """
    Shrink block bounds to exclude opposite-class points.
    
    Args:
        block (dict): Block dictionary
        block_points (np.ndarray): Points within the block
        block_point_classes (np.ndarray): Classes of points within the block
        X (np.ndarray): Full feature matrix
        y (np.ndarray): Full target labels
    
    Returns:
        list: List containing the shrunk block (if any points remain)
    """
    block_class = block['class']
    bounds = np.array(block['bounds'])
    
    # Find target class points
    target_mask = block_point_classes == block_class
    target_points = block_points[target_mask]
    
    if len(target_points) == 0:
        return []  # No target class points - discard block
    
    # Calculate new bounds based only on target class points
    new_bounds = bounds_calculation(target_points, bounds.shape[0])
    
    # Check if new bounds are significantly different
    bounds_diff = np.sum(np.abs(new_bounds - bounds))
    if bounds_diff < 1e-6:
        # No significant change - return original block
        return [block]
    
    # Create shrunk block
    shrunk_block = block.copy()
    shrunk_block['bounds'] = new_bounds.tolist()
    shrunk_block['shrunk_for_purity'] = True
    shrunk_block['original_bounds'] = bounds.tolist()
    
    # Log the shrinking
    other_class_count = np.sum(block_point_classes != block_class)
    target_class_count = np.sum(block_point_classes == block_class)
    diagnostic_print(f"  Shrunk block (class {block_class}): {target_class_count} target, {other_class_count} other points")
    
    return [shrunk_block]


def prune_blocks(blocks_bounds, blocks_classes, volume_threshold=DEFAULT_VOLUME_THRESHOLD):
    """
    Prune blocks with non-finite or tiny volume bounds.
    
    Args:
        blocks_bounds (list): List of block bounds
        blocks_classes (list): List of block class labels
        volume_threshold (float): Minimum volume threshold
    
    Returns:
        tuple: (pruned_bounds, pruned_classes)
    """
    pruned_bounds = []
    pruned_classes = []
    
    for i, bounds in enumerate(blocks_bounds):
        if np.all(np.isfinite(bounds)):
            volume = np.prod(bounds[:, 1] - bounds[:, 0])
            
            if volume > volume_threshold:
                pruned_bounds.append(bounds)
                pruned_classes.append(blocks_classes[i])
            else:
                # Expand tiny bounds
                expanded_bounds = bounds.copy()
                for j in range(bounds.shape[0]):
                    if bounds[j, 1] - bounds[j, 0] < 1e-6:
                        mid = (bounds[j, 0] + bounds[j, 1]) / 2
                        expanded_bounds[j, 0] = mid - 1e-6
                        expanded_bounds[j, 1] = mid + 1e-6
                
                pruned_bounds.append(expanded_bounds)
                pruned_classes.append(blocks_classes[i])
    
    return pruned_bounds, pruned_classes


def prune_and_merge_blocks(blocks):
    """
    Prune and merge blocks.
    
    Args:
        blocks (list): List of hyperblock dictionaries
    
    Returns:
        list: Pruned blocks
    """
    if not blocks:
        return []
    
    blocks_bounds = [np.array(b['bounds']) for b in blocks]
    blocks_classes = [b['class'] for b in blocks]
    pruned_bounds, pruned_classes = prune_blocks(blocks_bounds, blocks_classes)
    
    pruned = []
    for i, (bounds, class_label) in enumerate(zip(pruned_bounds, pruned_classes)):
        block = blocks[i].copy()
        block['bounds'] = bounds.tolist()
        pruned.append(block)
    
    return pruned


def deduplicate_blocks(blocks, tolerance=DEFAULT_DUPLICATE_TOLERANCE):
    """
    Remove duplicate or nearly identical blocks based on bounds.
    
    Args:
        blocks (list): List of hyperblock dictionaries
        tolerance (float): Tolerance for considering blocks identical
    
    Returns:
        list: Deduplicated blocks
    """
    if not blocks:
        return []
    
    unique = []
    seen = set()
    
    for block in blocks:
        bounds = np.round(np.array(block['bounds']), decimals=6)
        key = tuple(bounds.flatten())
        
        if key not in seen:
            seen.add(key)
            unique.append(block)
    
    return unique


def merge_overlapping_blocks(blocks, overlap_threshold=0.5):
    """
    Merge blocks of the same class that have significant overlap.
    
    Args:
        blocks (list): List of hyperblock dictionaries
        overlap_threshold (float): Minimum overlap ratio to trigger merging
    
    Returns:
        list: Blocks with overlapping blocks merged
    """
    if not blocks:
        return blocks
    
    # Group blocks by class
    class_blocks = {}
    for i, block in enumerate(blocks):
        class_label = block['class']
        if class_label not in class_blocks:
            class_blocks[class_label] = []
        class_blocks[class_label].append((i, np.array(block['bounds'])))
    
    # Merge overlapping blocks within each class
    merged_blocks = []
    processed_indices = set()
    
    for class_label, class_block_list in class_blocks.items():
        if len(class_block_list) <= 1:
            # No merging needed for single block
            for idx, bounds in class_block_list:
                merged_blocks.append(blocks[idx])
                processed_indices.add(idx)
            continue
        
        # Find overlapping blocks
        merged_groups = []
        remaining_blocks = class_block_list.copy()
        
        while remaining_blocks:
            current_idx, current_bounds = remaining_blocks.pop(0)
            current_group = [current_idx]
            
            # Find all blocks that overlap significantly with current block
            i = 0
            while i < len(remaining_blocks):
                other_idx, other_bounds = remaining_blocks[i]
                
                # Calculate overlap
                overlap_volume = 1.0
                for dim in range(len(current_bounds)):
                    lower_overlap = max(current_bounds[dim, 0], other_bounds[dim, 0])
                    upper_overlap = min(current_bounds[dim, 1], other_bounds[dim, 1])
                    
                    if lower_overlap >= upper_overlap:
                        overlap_volume = 0.0
                        break
                    
                    overlap_volume *= (upper_overlap - lower_overlap)
                
                if overlap_volume > 0:
                    # Calculate overlap ratio
                    current_volume = np.prod(current_bounds[:, 1] - current_bounds[:, 0])
                    other_volume = np.prod(other_bounds[:, 1] - other_bounds[:, 0])
                    min_volume = min(current_volume, other_volume)
                    
                    if min_volume > 0 and overlap_volume / min_volume >= overlap_threshold:
                        current_group.append(other_idx)
                        remaining_blocks.pop(i)
                        # Update current bounds to include the merged block
                        for dim in range(len(current_bounds)):
                            current_bounds[dim, 0] = min(current_bounds[dim, 0], other_bounds[dim, 0])
                            current_bounds[dim, 1] = max(current_bounds[dim, 1], other_bounds[dim, 1])
                    else:
                        i += 1
                else:
                    i += 1
            
            merged_groups.append(current_group)
        
        # Create merged blocks
        for group in merged_groups:
            if len(group) == 1:
                # Single block, no merging needed
                merged_blocks.append(blocks[group[0]])
                processed_indices.add(group[0])
            else:
                # Merge multiple blocks
                merged_bounds = np.array(blocks[group[0]]['bounds'])
                for idx in group[1:]:
                    other_bounds = np.array(blocks[idx]['bounds'])
                    for dim in range(len(merged_bounds)):
                        merged_bounds[dim, 0] = min(merged_bounds[dim, 0], other_bounds[dim, 0])
                        merged_bounds[dim, 1] = max(merged_bounds[dim, 1], other_bounds[dim, 1])
                
                # Create merged block
                merged_block = blocks[group[0]].copy()
                merged_block['bounds'] = merged_bounds.tolist()
                merged_block['merged_from'] = group
                merged_block['original_count'] = len(group)
                
                merged_blocks.append(merged_block)
                processed_indices.update(group)
    
    # Add any unprocessed blocks (shouldn't happen, but safety check)
    for i, block in enumerate(blocks):
        if i not in processed_indices:
            merged_blocks.append(block)
    
    merge_count = len(blocks) - len(merged_blocks)
    if merge_count > 0:
        diagnostic_print(f"Merged {merge_count} overlapping blocks")
    
    return merged_blocks


def remove_contained_blocks(blocks):
    """
    Remove blocks that are completely contained within other blocks of the same class.
    
    Args:
        blocks (list): List of hyperblock dictionaries
    
    Returns:
        list: Blocks with contained blocks removed
    """
    if not blocks:
        return blocks
    
    block_bounds = [np.array(b['bounds']) for b in blocks]
    block_classes = [b['class'] for b in blocks]
    
    # Group blocks by class
    class_blocks = {}
    for i, class_label in enumerate(block_classes):
        if class_label not in class_blocks:
            class_blocks[class_label] = []
        class_blocks[class_label].append((i, block_bounds[i]))
    
    # Find contained blocks within each class
    contained_indices = set()
    
    for class_label, class_block_list in class_blocks.items():
        for i, (idx_i, bounds_i) in enumerate(class_block_list):
            for j, (idx_j, bounds_j) in enumerate(class_block_list):
                if i == j:
                    continue
                
                # Check if bounds_i is contained within bounds_j
                is_contained = True
                for dim in range(len(bounds_i)):
                    # bounds_i is contained if its lower bound >= bounds_j lower bound
                    # and its upper bound <= bounds_j upper bound
                    if bounds_i[dim, 0] < bounds_j[dim, 0] or bounds_i[dim, 1] > bounds_j[dim, 1]:
                        is_contained = False
                        break
                
                if is_contained:
                    contained_indices.add(idx_i)
                    break  # No need to check against other blocks for this one
    
    # Remove contained blocks
    filtered_blocks = []
    for i, block in enumerate(blocks):
        if i not in contained_indices:
            filtered_blocks.append(block)
    
    removed_count = len(blocks) - len(filtered_blocks)
    if removed_count > 0:
        diagnostic_print(f"Removed {removed_count} contained blocks")
    
    return filtered_blocks


def compute_block_confidence_scores(blocks, X, y):
    """
    Compute confidence scores for blocks based on overlap with other classes.
    Enhanced version with better boundary handling and overlap detection.
    
    Args:
        blocks (list): List of hyperblock dictionaries
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target labels
    
    Returns:
        list: Blocks with confidence scores added
    """
    if not blocks:
        return blocks
    
    block_bounds = [np.array(b['bounds']) for b in blocks]
    block_classes = [b['class'] for b in blocks]
    
    for i, block in enumerate(blocks):
        bounds = block_bounds[i]
        block_class = block_classes[i]
        
        # Find points that fall within this block
        block_mask = np.ones(len(X), dtype=bool)
        for dim in range(len(bounds)):
            block_mask &= (X[:, dim] >= bounds[dim, 0]) & (X[:, dim] <= bounds[dim, 1])
        
        block_points = X[block_mask]
        block_point_classes = y[block_mask]
        
        if len(block_points) == 0:
            block['confidence_score'] = 0.0
            continue
        
        # Count points of different classes within this block
        unique_classes, class_counts = np.unique(block_point_classes, return_counts=True)
        class_point_dict = dict(zip(unique_classes, class_counts))
        
        # Calculate overlap score
        total_points = len(block_points)
        own_class_points = class_point_dict.get(block_class, 0)
        other_class_points = sum(count for class_label, count in class_point_dict.items() 
                               if class_label != block_class)
        
        # Enhanced confidence calculation
        if total_points > 0:
            # Base overlap score
            overlap_score = other_class_points / total_points
            base_confidence = 1.0 - overlap_score
            
            # Volume-based penalty for large blocks (more likely to contain other classes)
            volume = np.prod(bounds[:, 1] - bounds[:, 0])
            volume_penalty = min(0.3, volume * 0.5)  # Cap at 30% penalty
            
            # Density bonus for compact blocks
            density = own_class_points / volume if volume > 0 else 0
            density_bonus = min(0.2, density * 0.1)  # Cap at 20% bonus
            
            # Boundary penalty for blocks near other classes
            boundary_penalty = 0.0
            if other_class_points > 0:
                # Check if other-class points are near the boundaries
                other_class_mask = block_point_classes != block_class
                other_points = block_points[other_class_mask]
                
                # Calculate average distance to boundaries
                boundary_distances = []
                for point in other_points:
                    min_dist_to_boundary = float('inf')
                    for dim in range(len(bounds)):
                        dist_to_lower = abs(point[dim] - bounds[dim, 0])
                        dist_to_upper = abs(point[dim] - bounds[dim, 1])
                        min_dist_to_boundary = min(min_dist_to_boundary, dist_to_lower, dist_to_upper)
                    boundary_distances.append(min_dist_to_boundary)
                
                avg_boundary_distance = np.mean(boundary_distances)
                if avg_boundary_distance < 0.1:  # Close to boundary
                    boundary_penalty = 0.15
            
            # Final confidence score
            confidence_score = base_confidence - volume_penalty + density_bonus - boundary_penalty
            confidence_score = max(0.0, min(1.0, confidence_score))  # Clamp to [0, 1]
        else:
            confidence_score = 0.0
        
        # Penalty for blocks with very few points
        if total_points < DEFAULT_MIN_POINTS_FOR_PENALTY:
            confidence_score *= 0.5
        
        # Containment confidence (simplified)
        containment_confidence = (own_class_points / (own_class_points + other_class_points) 
                                if (own_class_points + other_class_points) > 0 else 0.0)
        
        # Store scores and statistics
        block['confidence_score'] = confidence_score
        block['containment_confidence'] = containment_confidence
        block['total_points'] = total_points
        block['own_class_points'] = own_class_points
        block['other_class_points'] = other_class_points
        block['volume'] = np.prod(bounds[:, 1] - bounds[:, 0])
        block['density'] = own_class_points / block['volume'] if block['volume'] > 0 else 0
    
    return blocks


def shrink_dominant_blocks(blocks, X, y, epsilon=DEFAULT_SHRINK_EPSILON):
    """
    Shrink margins on dominant blocks to reduce overlap with other classes.
    Performs dimension-wise shrinking: only shrinks along dimensions that contain other-class points.
    
    Args:
        blocks (list): List of hyperblock dictionaries
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target labels
        epsilon (float): Shrink factor
    
    Returns:
        list: Blocks with shrunk bounds
    """
    if not blocks:
        return blocks
    
    block_bounds = [np.array(b['bounds']) for b in blocks]
    block_classes = [b['class'] for b in blocks]
    
    # Calculate class dominance
    class_block_counts = {}
    for class_label in block_classes:
        class_block_counts[class_label] = class_block_counts.get(class_label, 0) + 1
    
    avg_blocks_per_class = len(blocks) / len(set(block_classes))
    dominant_classes = {class_label for class_label, count in class_block_counts.items() 
                       if count > avg_blocks_per_class}
    
    # Shrink margins for dominant blocks
    for i, block in enumerate(blocks):
        if block_classes[i] in dominant_classes:
            bounds = block_bounds[i]
            block_class = block_classes[i]
            
            # Find points within this block
            block_mask = np.ones(len(X), dtype=bool)
            for dim in range(len(bounds)):
                block_mask &= (X[:, dim] >= bounds[dim, 0]) & (X[:, dim] <= bounds[dim, 1])
            
            block_points = X[block_mask]
            block_point_classes = y[block_mask]
            
            if len(block_points) == 0:
                continue
            
            # Check misclassification ratio
            other_class_mask = block_point_classes != block_class
            other_class_points = block_points[other_class_mask]
            misclassification_ratio = len(other_class_points) / len(block_points)
            
            # Shrink if misclassification is high
            if misclassification_ratio > DEFAULT_MISCLASSIFICATION_THRESHOLD:
                # Analyze each dimension to see if it contains other-class points
                dimensions_to_shrink = []
                
                for dim in range(len(bounds)):
                    # Check if this dimension contains other-class points
                    dim_values = block_points[:, dim]
                    other_class_dim_values = other_class_points[:, dim]
                    
                    # Check if other-class points span a significant range in this dimension
                    if len(other_class_dim_values) > 0:
                        other_class_range = np.max(other_class_dim_values) - np.min(other_class_dim_values)
                        total_range = bounds[dim, 1] - bounds[dim, 0]
                        
                        # Shrink if other-class points occupy a significant portion of this dimension
                        if other_class_range > 0.1 * total_range:  # 10% threshold
                            dimensions_to_shrink.append(dim)
                
                # Shrink only the problematic dimensions
                if dimensions_to_shrink:
                    for dim in dimensions_to_shrink:
                        range_size = bounds[dim, 1] - bounds[dim, 0]
                        shrink_amount = range_size * epsilon
                        bounds[dim, 0] += shrink_amount
                        bounds[dim, 1] -= shrink_amount
                    
                    block['bounds'] = bounds.tolist()
                    block['shrunk'] = True
                    block['original_misclassification_ratio'] = misclassification_ratio
                    block['shrunk_dimensions'] = dimensions_to_shrink
                    block['total_dimensions'] = len(bounds)
    
    return blocks


def flag_misclassifying_blocks(blocks, X, y, threshold=DEFAULT_MISCLASSIFICATION_THRESHOLD):
    """
    Flag blocks that misclassify other classes above a threshold.
    
    Args:
        blocks (list): List of hyperblock dictionaries
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target labels
        threshold (float): Misclassification threshold
    
    Returns:
        list: Blocks with misclassification flags
    """
    if not blocks:
        return blocks
    
    for block in blocks:
        bounds = np.array(block['bounds'])
        block_class = block['class']
        
        # Find points within this block
        block_mask = np.ones(len(X), dtype=bool)
        for dim in range(len(bounds)):
            block_mask &= (X[:, dim] >= bounds[dim, 0]) & (X[:, dim] <= bounds[dim, 1])
        
        block_points = X[block_mask]
        block_point_classes = y[block_mask]
        
        if len(block_points) == 0:
            continue
        
        # Calculate misclassification ratio
        other_class_points = block_point_classes[block_point_classes != block_class]
        misclassification_ratio = len(other_class_points) / len(block_points)
        
        # Flag blocks with high misclassification
        if misclassification_ratio > threshold:
            block['flagged'] = True
            block['misclassification_ratio'] = misclassification_ratio
            block['misclassified_points'] = len(other_class_points)
        else:
            block['flagged'] = False
            block['misclassification_ratio'] = misclassification_ratio
    
    return blocks


# =============================================================================
# TIE BREAKER FUNCTIONS
# =============================================================================

def calculate_hyperblock_density(block):
    """
    Calculate the density of a hyperblock as point count / volume.
    
    Args:
        block (dict): Hyperblock dictionary with bounds and total_points
    
    Returns:
        float: Density (points per unit volume), or 0 if volume is 0
    """
    bounds = np.array(block['bounds'])
    
    # Calculate volume
    volume = np.prod(bounds[:, 1] - bounds[:, 0])
    
    # Get point count
    point_count = block.get('total_points', 0)
    
    # Return density (avoid division by zero)
    if volume > 0:
        return point_count / volume
    else:
        return 0.0


def density_based_tie_breaker(blocks, block_indices, block_labels):
    """
    Break ties by selecting the most dense hyperblock.
    
    Args:
        blocks (list): List of hyperblock dictionaries
        block_indices (list): Indices of blocks involved in the tie
        block_labels (np.ndarray): Block class labels
    
    Returns:
        int: Class label of the most dense hyperblock
    """
    if not block_indices:
        return None
    
    # Calculate density for each block in the tie
    block_densities = []
    for idx in block_indices:
        if idx < len(blocks):
            density = calculate_hyperblock_density(blocks[idx])
            block_densities.append((density, idx, block_labels[idx]))
    
    if not block_densities:
        return None
    
    # Sort by density (highest first)
    block_densities.sort(key=lambda x: x[0], reverse=True)
    
    # Return the class of the most dense block
    return block_densities[0][2]


# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================

def analyze_boundary_case(point, block_indices, block_bounds, block_labels, blocks, norm):
    """
    Analyze boundary cases where a point is near multiple blocks or on boundaries.
    
    Args:
        point (np.ndarray): Point to classify
        block_indices (np.ndarray or list): Indices of relevant blocks
        block_bounds (list): List of block bounds
        block_labels (np.ndarray): Block class labels
        blocks (list): List of block dictionaries
        norm (int): Distance norm
    
    Returns:
        int: Predicted class label
    """
    if len(block_indices) == 0:
        return None
    
    # Calculate distances to all relevant blocks
    distances = []
    for idx in block_indices:
        dist = distance_to_hyperblock(point, block_bounds[idx], norm)
        distances.append((dist, idx, block_labels[idx]))
    
    # Sort by distance
    distances.sort()
    
    # Check if point is very close to multiple blocks (boundary case)
    min_dist = distances[0][0]
    close_blocks = [(dist, idx, label) for dist, idx, label in distances if dist <= min_dist * 1.1]
    
    if len(close_blocks) > 1:
        # Boundary case - use weighted voting based on distance and confidence
        class_scores = {}
        
        for dist, idx, label in close_blocks:
            weight = 1.0 / (1.0 + dist)  # Inverse distance weighting
            
            # Add confidence score if available
            if blocks is not None and idx < len(blocks):
                confidence = blocks[idx].get('confidence_score', 0.5)
                weight *= (1.0 + confidence)
            
            if label not in class_scores:
                class_scores[label] = 0.0
            class_scores[label] += weight
        
        # Return class with highest weighted score
        if class_scores:
            return max(class_scores, key=class_scores.get)
    
    # Fallback to closest block
    return distances[0][2]


def classify_batch(points, block_bounds, block_labels, k, norm, blocks=None):
    """
    Classify a batch of points using hyperblocks and k-NN logic.
    
    This function implements a hierarchical tie-breaking system:
    1. First tries confidence scores if available
    2. Falls back to Copeland's method for voting ties
    3. Uses density-based tie breaker (point count / volume) for perfect ties
    4. Enhanced boundary detection for points near multiple blocks
    
    Args:
        points (np.ndarray): Points to classify
        block_bounds (list): List of block bounds
        block_labels (np.ndarray): Block class labels
        k (int): Number of nearest neighbors
        norm (int): Distance norm
        blocks (list): Optional list of block dictionaries for confidence scores
    
    Returns:
        tuple: (predictions, contained_count, knn_count)
    """
    predictions = np.zeros(len(points), dtype=np.int32)
    contained_count = 0
    knn_count = 0
    
    # Precompute all distances
    all_dists = np.zeros((len(points), len(block_bounds)))
    for i, point in enumerate(points):
        for j, bounds in enumerate(block_bounds):
            all_dists[i, j] = distance_to_hyperblock(point, bounds, norm)
    
    for i in range(len(points)):
        dists = all_dists[i]
        min_dist = np.min(dists)
        
        if min_dist == 0:
            # Point is contained in one or more blocks
            contained_indices = np.where(dists == 0)[0]
            contained_classes = [block_labels[idx] for idx in contained_indices]
            unique_classes, counts = np.unique(contained_classes, return_counts=True)
            
            if len(unique_classes) == 1:
                predictions[i] = unique_classes[0]
            else:
                # Tie between different classes - use confidence scores if available
                if blocks is not None:
                    class_confidence = {}
                    for class_label in unique_classes:
                        max_confidence = 0.0
                        for idx in contained_indices:
                            if block_labels[idx] == class_label:
                                if 'containment_confidence' in blocks[idx]:
                                    max_confidence = max(max_confidence, blocks[idx]['containment_confidence'])
                                elif 'confidence_score' in blocks[idx]:
                                    max_confidence = max(max_confidence, blocks[idx]['confidence_score'])
                        class_confidence[class_label] = max_confidence
                    
                    if not class_confidence or all(v == 0 for v in class_confidence.values()):
                        fallback_class = unique_classes[np.argmax(counts)]
                        predictions[i] = fallback_class
                    else:
                        best_class = max(class_confidence, key=class_confidence.get)
                        predictions[i] = best_class
                else:
                    # Fallback to Copeland's method
                    copeland_scores = {}
                    for i_a, class_a in enumerate(unique_classes):
                        score = 0
                        count_a = counts[i_a]
                        for i_b, class_b in enumerate(unique_classes):
                            if i_a != i_b:
                                count_b = counts[i_b]
                                if count_a > count_b:
                                    score += 1
                                elif count_a == count_b:
                                    score += 0.5
                        copeland_scores[class_a] = score
                    
                    best_class = max(copeland_scores, key=copeland_scores.get)
                    
                    # Check if there's still a tie after Copeland's method
                    max_score = copeland_scores[best_class]
                    tied_classes = [cls for cls, score in copeland_scores.items() if score == max_score]
                    
                    if len(tied_classes) > 1 and blocks is not None:
                        # Use density-based tie breaker
                        tied_indices = [idx for idx in contained_indices if block_labels[idx] in tied_classes]
                        density_winner = density_based_tie_breaker(blocks, tied_indices, block_labels)
                        if density_winner is not None:
                            best_class = density_winner
                            # Log when density tie breaker is used
                            diagnostic_print(f"Density tie breaker used: {tied_classes} -> {best_class}")
                    
                    predictions[i] = best_class
            
            contained_count += 1
        else:
            # Point is not contained - use boundary analysis for better accuracy
            min_dist_indices = np.where(dists == min_dist)[0]
            
            # Use boundary analysis for better handling of edge cases
            boundary_prediction = analyze_boundary_case(
                points[i], min_dist_indices, block_bounds, block_labels, blocks, norm
            )
            
            if boundary_prediction is not None:
                predictions[i] = boundary_prediction
            else:
                # Fallback to original method
                classes_k = [block_labels[idx] for idx in min_dist_indices]
                unique_classes, counts = np.unique(classes_k, return_counts=True)
                
                if len(unique_classes) == 1:
                    predictions[i] = unique_classes[0]
                else:
                    # Use Copeland's method for tie-breaking
                    copeland_scores = {}
                    for i_a, class_a in enumerate(unique_classes):
                        score = 0
                        count_a = counts[i_a]
                        for i_b, class_b in enumerate(unique_classes):
                            if i_a != i_b:
                                count_b = counts[i_b]
                                if count_a > count_b:
                                    score += 1
                                elif count_a == count_b:
                                    score += 0.5
                        copeland_scores[class_a] = score
                    
                    best_class = max(copeland_scores, key=copeland_scores.get)
                    
                    # Check if there's still a tie after Copeland's method
                    max_score = copeland_scores[best_class]
                    tied_classes = [cls for cls, score in copeland_scores.items() if score == max_score]
                    
                    if len(tied_classes) > 1 and blocks is not None:
                        # Use density-based tie breaker
                        tied_indices = [idx for idx in min_dist_indices if block_labels[idx] in tied_classes]
                        density_winner = density_based_tie_breaker(blocks, tied_indices, block_labels)
                        if density_winner is not None:
                            best_class = density_winner
                            # Log when density tie breaker is used
                            diagnostic_print(f"Density tie breaker used (k-NN): {tied_classes} -> {best_class}")
                    
                    predictions[i] = best_class
            
            knn_count += 1
    
    return predictions, contained_count, knn_count


def classify_with_hyperblocks(X, y, blocks, k_values=None):
    """
    Classify data using hyperblocks with multiple k values and norms.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target labels
        blocks (list): List of hyperblock dictionaries
        k_values (list): List of k values to try
    
    Returns:
        pd.DataFrame: Classification results
    """
    if not blocks:
        return pd.DataFrame(columns=['norm', 'k', 'accuracy', 'preds', 'contained_count', 'knn_count'])
    
    if k_values is None:
        k_values = range(1, len(blocks) + 1)
    
    block_bounds = [np.array(b['bounds']) for b in blocks]
    block_labels = np.array([b['class'] for b in blocks])
    results = []
    
    for norm in [1, 2, 'inf']:
        for k in k_values:
            if k > len(blocks):
                continue
            
            predictions, contained_count, knn_count = classify_batch(
                X, block_bounds, block_labels, k, norm, blocks
            )
            acc = accuracy_score(y, predictions)
            
            results.append({
                'norm': f'L{norm}' if norm != 'inf' else 'Linf',
                'k': k,
                'accuracy': acc,
                'preds': predictions.tolist(),
                'contained_count': contained_count,
                'knn_count': knn_count
            })
    
    return pd.DataFrame(results)


def learn_optimal_hyperparameters(X_train, y_train, blocks):
    """
    Find the best norm and k for the given training data and blocks.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        blocks (list): List of hyperblock dictionaries
    
    Returns:
        tuple: (best_norm, best_k)
    """
    if not blocks:
        return 'L2', 1
    
    best_accuracy = 0
    best_norm = 'L2'
    best_k = 1
    
    block_bounds = [np.array(b['bounds']) for b in blocks]
    block_labels = np.array([b['class'] for b in blocks])
    
    for norm in [1, 2, 'inf']:
        max_k = len(blocks)
        for k in range(1, max_k + 1):
            predictions, _, _ = classify_batch(X_train, block_bounds, block_labels, k, norm, blocks)
            accuracy = np.mean(predictions == y_train)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_norm = f'L{norm}' if norm != 'inf' else 'Linf'
                best_k = k
    
    return best_norm, best_k


# =============================================================================
# HYPERBLOCK PIPELINE
# =============================================================================

def find_all_envelope_blocks(X, y, feature_indices, classes, pbar=None):
    """
    Find all envelope blocks for all classes.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target labels
        feature_indices (list): Feature indices
        classes (list): Class names
        pbar (tqdm): Progress bar
    
    Returns:
        list: List of hyperblock dictionaries
    """
    encoded_classes = np.unique(y)
    args_list = [(c, X, y, feature_indices) for c in encoded_classes]
    all_blocks = []
    
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for result in executor.map(compute_envelope_blocks_for_class, args_list):
            all_blocks.extend(result)
            if pbar:
                pbar.update(1)
    
    # Post-process blocks
    refined_blocks = check_block_purity_and_refine(all_blocks, X, y, 
                                                  max_splits=DEFAULT_MAX_SPLITS, 
                                                  min_points_per_split=DEFAULT_MIN_POINTS_PER_SPLIT)
    pruned_blocks = prune_and_merge_blocks(refined_blocks)
    deduplicated_blocks = deduplicate_blocks(pruned_blocks)
    merged_blocks = merge_overlapping_blocks(deduplicated_blocks, overlap_threshold=0.3)
    contained_removed_blocks = remove_contained_blocks(merged_blocks)
    scored_blocks = compute_block_confidence_scores(contained_removed_blocks, X, y)
    flagged_blocks = flag_misclassifying_blocks(scored_blocks, X, y)
    shrunk_blocks = shrink_dominant_blocks(flagged_blocks, X, y)
    
    return shrunk_blocks


# =============================================================================
# CROSS-VALIDATION FUNCTIONS
# =============================================================================

def fold_worker(args):
    """
    Worker function for processing a single fold.
    
    Args:
        args (tuple): (train_index, test_index, X, y, feature_indices, classes, fold_num, pbar)
    
    Returns:
        tuple: (df_results, num_blocks, best_norm, best_k, predictions, y_test, X_test, blocks)
    """
    train_index, test_index, X, y, feature_indices, classes, fold_num, pbar = args
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    blocks = find_all_envelope_blocks(X_train, y_train, feature_indices, classes, pbar)
    best_norm, best_k = learn_optimal_hyperparameters(X_train, y_train, blocks)
    df_results = classify_with_hyperblocks(X_test, y_test, blocks, k_values=[best_k])
    
    predictions = []
    if not df_results.empty:
        predictions = df_results.iloc[0]['preds']
        df_results['fold_num'] = fold_num
        df_results['test_size'] = len(test_index)
        df_results['num_blocks'] = len(blocks)
        df_results['learned_norm'] = best_norm
        df_results['learned_k'] = best_k
    
    return df_results, len(blocks), best_norm, best_k, predictions, y_test, X_test, blocks


def cross_validate_blocks(X, y, feature_indices, classes, features, k_folds=DEFAULT_K_FOLDS):
    """
    Perform cross-validation using hyperblocks.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target labels
        feature_indices (list): Feature indices
        classes (list): Class names
        features (list): Feature names
        k_folds (int): Number of folds
    
    Returns:
        pd.DataFrame: Cross-validation results
    """
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    args_list = []
    encoded_classes = np.unique(y)
    total_tasks = k_folds * len(encoded_classes)
    
    # Generate all splits first
    splits = list(kf.split(X, y))
    
    # Print case count per fold
    diagnostic_print(f"\n{k_folds}-fold Cross-Validation Split:")
    total_dataset_size = len(X)
    diagnostic_print(f"Total dataset: {total_dataset_size} cases")
    
    for fold_num, (train_index, test_index) in enumerate(splits):
        train_count = len(train_index)
        test_count = len(test_index)
        diagnostic_print(f"Fold {fold_num + 1}: {train_count} train, {test_count} test")
    
    # Summary
    avg_test_size = total_dataset_size / k_folds
    diagnostic_print(f"Expected test size per fold: ~{avg_test_size:.0f} ({100/k_folds:.1f}% of data)")
    diagnostic_print()
    
    with tqdm(total=total_tasks, desc="Processing hyperblocks", unit="class") as pbar:
        for fold_num, (train_index, test_index) in enumerate(splits):
            args_list.append((train_index, test_index, X, y, feature_indices, classes, fold_num, pbar))
            
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            all_results = list(executor.map(fold_worker, args_list))
    
    # Process results
    valid_results = []
    block_counts = []
    learned_norms = []
    learned_ks = []
    all_predictions = []
    all_true_labels = []
    all_test_data = []
    all_blocks_per_fold = []
    
    for df, num_blocks, norm, k, predictions, y_test, X_test, blocks in all_results:
        if not df.empty:
            valid_results.append(df)
            block_counts.append(num_blocks)
            learned_norms.append(norm)
            learned_ks.append(k)
            all_predictions.append(predictions)
            all_true_labels.append(y_test)
            all_test_data.append(X_test)
            all_blocks_per_fold.append(blocks)
    
    if not valid_results:
        print("No valid results found across all folds")
        return pd.DataFrame()
    
    # Print results
    diagnostic_print("\nClass mapping:")
    for class_num in np.unique(y):
        diagnostic_print(f"  {class_num}: {classes[class_num]}")
    diagnostic_print()
    
    fold_accuracies = []
    fold_contained = []
    fold_knn = []
    
    for i, (df, num_blocks, norm, k, predictions, y_test, X_test, blocks) in enumerate(
        zip(valid_results, block_counts, learned_norms, learned_ks, 
            all_predictions, all_true_labels, all_test_data, all_blocks_per_fold)):
        
        if not df.empty:
            acc = df.iloc[0]['accuracy']
            contained = df.iloc[0]['contained_count']
            knn = df.iloc[0]['knn_count']
            fold_accuracies.append(acc)
            fold_contained.append(contained)
            fold_knn.append(knn)
            
            diagnostic_print(f"Fold {i+1}: accuracy = {acc:.4f}, blocks = {num_blocks}, "
                  f"contained = {contained}, k-NN = {knn}, norm = {norm}, k = {k}")
            diagnostic_print(f"Confusion Matrix (Fold {i+1}):")
            diagnostic_print(confusion_matrix(y_test, predictions))
            diagnostic_print()
            
            # Analyze misclassifications for this fold
            analyze_misclassifications(X_test, y_test, predictions, blocks, classes, features, i, learned_norm=norm)
    
    # Summary statistics
    avg_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    avg_blocks = np.mean(block_counts)
    std_blocks = np.std(block_counts)
    
    print("Cross-validation summary:")
    print(f"Average accuracy across all folds: {avg_accuracy:.4f} +/- {std_accuracy:.4f}")
    print(f"Average blocks per fold: {avg_blocks:.1f} +/- {std_blocks:.1f}")
    
    if fold_contained:
        avg_contained = np.mean(fold_contained)
        std_contained = np.std(fold_contained)
        avg_knn = np.mean(fold_knn)
        std_knn = np.std(fold_knn)
        diagnostic_print(f"Average contained cases per fold: {avg_contained:.1f} +/- {std_contained:.1f}")
        diagnostic_print(f"Average k-NN cases per fold: {avg_knn:.1f} +/- {std_knn:.1f}")
    
    # Save hyperblock bounds
    save_hyperblock_bounds_to_csv(all_blocks_per_fold, classes, features, k_folds)
    
    return pd.DataFrame({'fold_accuracies': fold_accuracies})


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_hyperblock_bounds_to_csv(all_blocks_per_fold, classes, feature_names, k_folds):
    """
    Save hyperblock bounds to CSV with class names and fold information.
    
    Args:
        all_blocks_per_fold (list): List of blocks for each fold
        classes (list): Class names
        feature_names (list): Feature names
        k_folds (int): Number of folds
    """
    # Create output directory
    output_dir = "hyperblock_bounds"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/hyperblock_bounds_{timestamp}.csv"
    
    # Prepare data for CSV
    csv_data = []
    
    for fold_num, blocks in enumerate(all_blocks_per_fold):
        for block_idx, block in enumerate(blocks):
            bounds = np.array(block['bounds'])
            class_label = block['class']
            class_name = classes[class_label]
            
            # Create one row per hyperblock
            row = {
                'fold': fold_num + 1,
                'class_label': class_label,
                'class_name': class_name
            }
            
            # Add feature bounds as columns
            for feature_idx, feature_name in enumerate(feature_names):
                row[f"{feature_name}_lower"] = bounds[feature_idx, 0]
                row[f"{feature_name}_upper"] = bounds[feature_idx, 1]
            
            csv_data.append(row)
    
    # Create DataFrame and save to CSV
    df_bounds = pd.DataFrame(csv_data)
    df_bounds.to_csv(filename, index=False)
    
    diagnostic_print(f"\nHyperblock bounds saved to: {filename}")
    diagnostic_print(f"Total hyperblocks recorded: {len(csv_data)}")
    diagnostic_print(f"Folds processed: {len(all_blocks_per_fold)}")


def analyze_misclassifications(X_test, y_test, predictions, blocks, classes, features, fold_num, learned_norm='L2'):
    """
    Analyze and print detailed information about misclassifications.
    
    Args:
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        predictions (np.ndarray): Predicted labels
        blocks (list): List of hyperblock dictionaries
        classes (list): Class names
        features (list): Feature names
        fold_num (int): Fold number
    """
    def _to_norm(n):
        return 1 if n=='L1' else (2 if n=='L2' else 'inf')
    use_norm = _to_norm(learned_norm)

    misclassified_indices = np.where(y_test != predictions)[0]
    
    if len(misclassified_indices) == 0:
        diagnostic_print(f"Fold {fold_num + 1}: No misclassifications!")
        return
    
    diagnostic_print(f"\n=== DETAILED MISCLASSIFICATION ANALYSIS (Fold {fold_num + 1}) ===")
    diagnostic_print(f"Total misclassifications: {len(misclassified_indices)} out of {len(y_test)} "
          f"({len(misclassified_indices)/len(y_test)*100:.1f}%)")
    
    # Get block information for analysis
    block_bounds = [np.array(b['bounds']) for b in blocks]
    block_labels = np.array([b['class'] for b in blocks])
    
    for idx in misclassified_indices:
        true_class = y_test[idx]
        pred_class = predictions[idx]
        point = X_test[idx]
        
        diagnostic_print(f"\n--- Misclassified Point {idx} ---")
        diagnostic_print(f"True class: {true_class} ({classes[true_class]})")
        diagnostic_print(f"Predicted class: {pred_class} ({classes[pred_class]})")
        diagnostic_print(f"Point features: {dict(zip(features, point))}")
        
        # Find distances to all blocks
        distances = []
        for i, bounds in enumerate(block_bounds):
            dist = distance_to_hyperblock(point, bounds, norm=use_norm)
            distances.append((dist, i, block_labels[i]))
        
        # Sort by distance
        distances.sort()
        
        diagnostic_print(f"Distance to nearest blocks:")
        for i, (dist, block_idx, block_class) in enumerate(distances[:5]):
            class_name = classes[block_class]
            diagnostic_print(f"  {i+1}. Block {block_idx} (class {block_class}: {class_name}): distance = {dist:.6f}")
        
        # Check if point is contained in any block
        contained_in = []
        for i, bounds in enumerate(block_bounds):
            if distance_to_hyperblock(point, bounds, norm=use_norm) == 0:
                contained_in.append((i, block_labels[i]))
        
        if contained_in:
            diagnostic_print(f"Point is contained in {len(contained_in)} block(s):")
            for block_idx, block_class in contained_in:
                diagnostic_print(f"  - Block {block_idx} (class {block_class}: {classes[block_class]})")
        else:
            diagnostic_print("Point is not contained in any block (using k-NN)")
    
    # Summary statistics
    diagnostic_print(f"\n--- Misclassification Summary (Fold {fold_num + 1}) ---")
    misclass_matrix = {}
    for true_class in np.unique(y_test):
        for pred_class in np.unique(y_test):
            if true_class != pred_class:
                count = np.sum((y_test == true_class) & (predictions == pred_class))
                if count > 0:
                    key = (true_class, pred_class)
                    misclass_matrix[key] = count
    
    if misclass_matrix:
        diagnostic_print("Misclassification patterns:")
        for (true_class, pred_class), count in sorted(misclass_matrix.items()):
            true_name = classes[true_class]
            pred_name = classes[pred_class]
            diagnostic_print(f"  {true_name} -> {pred_name}: {count} cases")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function for running the hyperblock classification algorithm."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperblock-based classification')
    parser.add_argument('--train', type=str, help='Path to training dataset CSV')
    parser.add_argument('--test', type=str, help='Path to test dataset CSV')
    parser.add_argument('--dataset', type=str, default='datasets/fisher_iris.csv', 
                       help='Path to single dataset CSV (default)')
    parser.add_argument('--k-folds', type=int, default=DEFAULT_K_FOLDS,
                       help='Number of cross-validation folds')
    parser.add_argument('--test-percentage', type=float, default=DEFAULT_TEST_PERCENTAGE,
                       help='Percentage of data to use for test split (default: 0.3)')
    parser.add_argument('--diagnostic', action='store_true', help='Enable diagnostic mode (more logging)')
    args = parser.parse_args()
    
    # Diagnostic mode if either argument or default is set
    global DIAGNOSTIC_MODE
    DIAGNOSTIC_MODE = args.diagnostic or DIAGNOSTIC_MODE

    if args.train and args.test:
        # Separate train/test datasets
        diagnostic_print("Loading separate train and test datasets...")
        df_train = pd.read_csv(args.train)
        df_test = pd.read_csv(args.test)
        
        features = df_train.columns[:-1]
        X_train_raw = df_train[features].values
        y_train_raw = df_train['class'].values
        X_test_raw = df_test[features].values
        y_test_raw = df_test['class'].values

        # Print the number of cases in the train and test datasets
        diagnostic_print(f"Training set: {len(X_train_raw)} cases")
        diagnostic_print(f"Test set: {len(X_test_raw)} cases")
        
        diagnostic_print("Preprocessing data...")
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        y_encoder = LabelEncoder()
        y_train = y_encoder.fit_transform(y_train_raw)
        y_test = y_encoder.transform(y_test_raw)
        
        feature_indices = list(range(X_train.shape[1]))
        
        diagnostic_print(f"Training set shape: {X_train.shape}")
        diagnostic_print(f"Test set shape: {X_test.shape}")
        diagnostic_print(f"Classes: {y_encoder.classes_}")
        diagnostic_print(f"Features: {len(feature_indices)}")
        
        # Run cross-validation
        diagnostic_print("\nRunning cross-validation...")
        scores = cross_validate_blocks(X_train, y_train, feature_indices, 
                                     y_encoder.classes_, features, args.k_folds)
        
        print("\nFinal results:")
        print(scores)
        
    else:
        # Single dataset with train/test split and cross-validation
        diagnostic_print("Loading dataset...")
        df = pd.read_csv(args.dataset)
        
        features = df.columns[:-1]
        X_raw = df[features].values
        y_raw = df['class'].values

        # Print the number of cases in the dataset
        diagnostic_print(f"Dataset: {len(X_raw)} cases")
        diagnostic_print(f"Features: {len(features)}")
        diagnostic_print(f"Classes: {len(np.unique(y_raw))}")
        
        # Split dataset into train and test sets
        diagnostic_print(f"\nSplitting dataset: {args.test_percentage*100:.1f}% test, {(1-args.test_percentage)*100:.1f}% train")
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X_raw, y_raw, test_size=args.test_percentage, random_state=DEFAULT_RANDOM_STATE, stratify=y_raw
        )
        
        diagnostic_print(f"Training set: {len(X_train_raw)} cases")
        diagnostic_print(f"Test set: {len(X_test_raw)} cases")
        
        diagnostic_print("Preprocessing data...")
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        y_encoder = LabelEncoder()
        y_train = y_encoder.fit_transform(y_train_raw)
        y_test = y_encoder.transform(y_test_raw)
        
        feature_indices = list(range(X_train.shape[1]))
        
        diagnostic_print(f"Training set shape: {X_train.shape}")
        diagnostic_print(f"Test set shape: {X_test.shape}")
        diagnostic_print(f"Classes: {y_encoder.classes_}")
        diagnostic_print(f"Features: {len(feature_indices)}")
        
        # Run cross-validation on training data only
        diagnostic_print("\nRunning cross-validation on training data...")
        scores = cross_validate_blocks(X_train, y_train, feature_indices, 
                                     y_encoder.classes_, features, args.k_folds)
        
        print("\nFinal results:")
        print(scores)


if __name__ == "__main__":
    main()
