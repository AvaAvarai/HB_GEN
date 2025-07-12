import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def distance_to_hyperblock(point, bounds, norm=2):
    """Calculate the distance from a point to a hyperblock (box) using the given norm."""
    projected = np.clip(point, bounds[:, 0], bounds[:, 1])
    diff = point - projected
    if norm == 1:
        return np.sum(np.abs(diff))
    else:
        return np.sqrt(np.sum(diff * diff))

def bounds_calculation(points, num_features):
    """Calculate min/max bounds for a set of points."""
    if len(points) == 0:
        return np.full((num_features, 2), [np.inf, -np.inf])
    bounds = np.column_stack([
        np.min(points, axis=0),
        np.max(points, axis=0)
    ])
    return bounds

def intersection_refinement(class_points, other_points, sorted_features, num_features):
    """Refine bounds by checking for intersection patterns with other classes."""
    if len(other_points) == 0:
        return bounds_calculation(class_points, num_features), False
    refined_bounds = np.full((num_features, 2), [np.inf, -np.inf])
    has_intersection = False
    for i in range(len(sorted_features)):
        left_idx = sorted_features[i - 1] if i > 0 else sorted_features[-1]
        right_idx = sorted_features[i + 1] if i < len(sorted_features) - 1 else sorted_features[0]
        curr_idx = sorted_features[i]
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

def compute_envelope_blocks_for_class(args):
    c, X, y, feature_indices = args
    class_mask = y == c
    other_mask = y != c
    class_points = X[class_mask]
    other_points = X[other_mask]
    if class_points.shape[0] == 0:
        return []
    num_features = X.shape[1]
    blocks = []
    try:
        y_binary = (y == c).astype(int)
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y_binary)
        lda_projection = lda.transform(X).flatten()
        feature_covariances = []
        for i in range(num_features):
            axis_values = X[:, i]
            covariance = np.cov(lda_projection, axis_values)[0, 1]
            feature_covariances.append(abs(covariance))
        feature_importance = np.array(feature_covariances)
    except:
        feature_importance = np.ones(num_features)
    sorted_features = np.argsort(feature_importance)[::-1]
    max_clusters = min(5, class_points.shape[0] // 2)
    best_n_clusters = 1
    best_silhouette = -1
    for n_clusters in range(1, max_clusters + 1):
        if n_clusters == 1:
            silhouette = 0
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(class_points)
            if len(np.unique(cluster_labels)) > 1:
                silhouette = silhouette_score(class_points, cluster_labels)
            else:
                silhouette = 0
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_n_clusters = n_clusters
    if best_n_clusters == 1:
        bounds = bounds_calculation(class_points, num_features)
        if other_points.shape[0] > 0:
            refined_bounds, has_intersection = intersection_refinement(
                class_points, other_points, sorted_features, num_features
            )
            if has_intersection:
                bounds = refined_bounds
        blocks.append({
            'class': c,
            'bounds': bounds.tolist(),
            'feature_order': sorted_features.tolist()
        })
    else:
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(class_points)
        for cluster_id in range(best_n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_points = class_points[cluster_mask]
            if cluster_points.shape[0] == 0:
                continue
            bounds = bounds_calculation(cluster_points, num_features)
            if other_points.shape[0] > 0:
                refined_bounds, has_intersection = intersection_refinement(
                    cluster_points, other_points, sorted_features, num_features
                )
                if has_intersection:
                    bounds = refined_bounds
            blocks.append({
                'class': c,
                'bounds': bounds.tolist(),
                'feature_order': sorted_features.tolist(),
                'cluster_id': cluster_id
            })
    return blocks

def prune_blocks(blocks_bounds, blocks_classes, volume_threshold=1e-10):
    """Prune blocks with non-finite or tiny volume bounds."""
    pruned_bounds = []
    pruned_classes = []
    for i, bounds in enumerate(blocks_bounds):
        if np.all(np.isfinite(bounds)):
            volume = np.prod(bounds[:, 1] - bounds[:, 0])
            if volume > volume_threshold:
                pruned_bounds.append(bounds)
                pruned_classes.append(blocks_classes[i])
            else:
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

def deduplicate_blocks(blocks, tolerance=1e-6):
    """Remove duplicate or nearly identical blocks based on bounds."""
    if not blocks:
        return []
    unique = []
    seen = set()
    for b in blocks:
        bounds = np.round(np.array(b['bounds']), decimals=6)
        key = tuple(bounds.flatten())
        if key not in seen:
            seen.add(key)
            unique.append(b)
    return unique

def compute_block_confidence_scores(blocks, X, y):
    """Compute confidence scores for blocks based on overlap with other classes."""
    if not blocks:
        return blocks
    
    # Convert to numpy arrays for easier manipulation
    block_bounds = [np.array(b['bounds']) for b in blocks]
    block_classes = [b['class'] for b in blocks]
    
    # Calculate overlap scores for each block
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
            # Empty block - low confidence
            block['confidence_score'] = 0.0
            continue
        
        # Count points of different classes within this block
        unique_classes, class_counts = np.unique(block_point_classes, return_counts=True)
        class_point_dict = dict(zip(unique_classes, class_counts))
        
        # Calculate overlap score: points from other classes / total points
        total_points = len(block_points)
        own_class_points = class_point_dict.get(block_class, 0)
        other_class_points = sum(count for class_label, count in class_point_dict.items() 
                               if class_label != block_class)
        
        overlap_score = other_class_points / total_points if total_points > 0 else 1.0
        
        # Confidence score is inverse of overlap (lower overlap = higher confidence)
        confidence_score = 1.0 - overlap_score
        
        # Additional penalty for blocks with very few points
        if total_points < 3:  # Penalize blocks with very few points
            confidence_score *= 0.5
        
        # Containment confidence: intersection size with own-class vs other-class points
        containment_confidence = own_class_points / (own_class_points + other_class_points) if (own_class_points + other_class_points) > 0 else 0.0
        
        block['confidence_score'] = confidence_score
        block['containment_confidence'] = containment_confidence
        block['total_points'] = total_points
        block['own_class_points'] = own_class_points
        block['other_class_points'] = other_class_points
    
    return blocks

def shrink_dominant_blocks(blocks, X, y, epsilon=0.1):
    """Shrink margins on dominant blocks to reduce overlap with other classes."""
    if not blocks:
        return blocks
    
    # Convert to numpy arrays for easier manipulation
    block_bounds = [np.array(b['bounds']) for b in blocks]
    block_classes = [b['class'] for b in blocks]
    
    # Calculate class dominance (number of blocks per class)
    class_block_counts = {}
    for class_label in block_classes:
        class_block_counts[class_label] = class_block_counts.get(class_label, 0) + 1
    
    # Find dominant classes (classes with more blocks than average)
    avg_blocks_per_class = len(blocks) / len(set(block_classes))
    dominant_classes = {class_label for class_label, count in class_block_counts.items() 
                       if count > avg_blocks_per_class}
    
    # Shrink margins for dominant blocks
    for i, block in enumerate(blocks):
        if block_classes[i] in dominant_classes:
            bounds = block_bounds[i]
            block_class = block_classes[i]
            
            # Find points that fall within this block
            block_mask = np.ones(len(X), dtype=bool)
            for dim in range(len(bounds)):
                block_mask &= (X[:, dim] >= bounds[dim, 0]) & (X[:, dim] <= bounds[dim, 1])
            
            block_points = X[block_mask]
            block_point_classes = y[block_mask]
            
            if len(block_points) == 0:
                continue
            
            # Check if this block misclassifies other classes
            other_class_points = block_point_classes[block_point_classes != block_class]
            misclassification_ratio = len(other_class_points) / len(block_points)
            
            # If block misclassifies significantly, shrink its margins
            if misclassification_ratio > 0.1:  # More than 10% misclassification
                # Shrink bounds by epsilon in each dimension
                for dim in range(len(bounds)):
                    range_size = bounds[dim, 1] - bounds[dim, 0]
                    shrink_amount = range_size * epsilon
                    bounds[dim, 0] += shrink_amount
                    bounds[dim, 1] -= shrink_amount
                
                # Update the block bounds
                block['bounds'] = bounds.tolist()
                block['shrunk'] = True
                block['original_misclassification_ratio'] = misclassification_ratio
    
    return blocks

def flag_misclassifying_blocks(blocks, X, y, threshold=0.1):
    """Flag blocks that misclassify other classes above a threshold."""
    if not blocks:
        return blocks
    
    flagged_blocks = []
    
    for i, block in enumerate(blocks):
        bounds = np.array(block['bounds'])
        block_class = block['class']
        
        # Find points that fall within this block
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
            flagged_blocks.append(i)
        else:
            block['flagged'] = False
            block['misclassification_ratio'] = misclassification_ratio
    
    return blocks

def analyze_misclassifications(X_test, y_test, predictions, blocks, classes, features, fold_num):
    """Analyze and print detailed information about misclassifications."""
    misclassified_indices = np.where(y_test != predictions)[0]
    
    if len(misclassified_indices) == 0:
        print(f"Fold {fold_num + 1}: No misclassifications!")
        return
    
    print(f"\n=== DETAILED MISCLASSIFICATION ANALYSIS (Fold {fold_num + 1}) ===")
    print(f"Total misclassifications: {len(misclassified_indices)} out of {len(y_test)} ({len(misclassified_indices)/len(y_test)*100:.1f}%)")
    
    # Get block information for analysis
    block_bounds = [np.array(b['bounds']) for b in blocks]
    block_labels = np.array([b['class'] for b in blocks])
    
    for idx in misclassified_indices:
        true_class = y_test[idx]
        pred_class = predictions[idx]
        point = X_test[idx]
        
        print(f"\n--- Misclassified Point {idx} ---")
        print(f"True class: {true_class} ({classes[true_class]})")
        print(f"Predicted class: {pred_class} ({classes[pred_class]})")
        print(f"Point features: {dict(zip(features, point))}")
        
        # Find distances to all blocks
        distances = []
        for i, bounds in enumerate(block_bounds):
            dist = distance_to_hyperblock(point, bounds, norm=2)
            distances.append((dist, i, block_labels[i]))
        
        # Sort by distance
        distances.sort()
        
        print(f"Distance to nearest blocks:")
        for i, (dist, block_idx, block_class) in enumerate(distances[:5]):  # Show top 5
            class_name = classes[block_class]
            print(f"  {i+1}. Block {block_idx} (class {block_class}: {class_name}): distance = {dist:.6f}")
        
        # Check if point is contained in any block
        contained_in = []
        for i, bounds in enumerate(block_bounds):
            if distance_to_hyperblock(point, bounds, norm=2) == 0:
                contained_in.append((i, block_labels[i]))
        
        if contained_in:
            print(f"Point is contained in {len(contained_in)} block(s):")
            for block_idx, block_class in contained_in:
                print(f"  - Block {block_idx} (class {block_class}: {classes[block_class]})")
        else:
            print("Point is not contained in any block (using k-NN)")
    
    # Summary statistics
    print(f"\n--- Misclassification Summary (Fold {fold_num + 1}) ---")
    misclass_matrix = {}
    for true_class in np.unique(y_test):
        for pred_class in np.unique(y_test):
            if true_class != pred_class:
                count = np.sum((y_test == true_class) & (predictions == pred_class))
                if count > 0:
                    key = (true_class, pred_class)
                    misclass_matrix[key] = count
    
    if misclass_matrix:
        print("Misclassification patterns:")
        for (true_class, pred_class), count in sorted(misclass_matrix.items()):
            true_name = classes[true_class]
            pred_name = classes[pred_class]
            print(f"  {true_name} → {pred_name}: {count} cases")

def save_hyperblock_bounds_to_csv(all_blocks_per_fold, classes, feature_names, k_folds):
    """Save hyperblock bounds to CSV with class names and fold information."""
    import os
    from datetime import datetime
    
    # Create output directory if it doesn't exist
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
            
            # Add feature bounds as columns using actual feature names
            for feature_idx, feature_name in enumerate(feature_names):
                row[f"{feature_name}_lower"] = bounds[feature_idx, 0]
                row[f"{feature_name}_upper"] = bounds[feature_idx, 1]
            
            csv_data.append(row)
    
    # Create DataFrame and save to CSV
    df_bounds = pd.DataFrame(csv_data)
    df_bounds.to_csv(filename, index=False)
    
    print(f"\nHyperblock bounds saved to: {filename}")
    print(f"Total hyperblocks recorded: {len(csv_data)}")
    print(f"Folds processed: {len(all_blocks_per_fold)}")

def classify_batch(points, block_bounds, block_labels, k, norm, blocks=None):
    """Classify a batch of points using the hyperblocks and k-NN logic."""
    predictions = np.zeros(len(points), dtype=np.int32)
    contained_count = 0
    knn_count = 0
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
                    # Use containment confidence scores to break ties
                    class_confidence = {}
                    for class_label in unique_classes:
                        # Find the highest containment confidence block for this class
                        max_confidence = 0.0
                        for idx in contained_indices:
                            if block_labels[idx] == class_label:
                                # Use containment confidence if available, otherwise fall back to confidence_score
                                if 'containment_confidence' in blocks[idx]:
                                    max_confidence = max(max_confidence, blocks[idx]['containment_confidence'])
                                elif 'confidence_score' in blocks[idx]:
                                    max_confidence = max(max_confidence, blocks[idx]['confidence_score'])
                        class_confidence[class_label] = max_confidence
                    
                    if not class_confidence or all(v == 0 for v in class_confidence.values()):
                        # Fallback: choose class with most blocks containing point
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
                    predictions[i] = best_class
            contained_count += 1
        else:
            # Point is not contained - use all blocks at minimum distance
            # Find all blocks at minimum distance (handle ties)
            min_dist_indices = np.where(dists == min_dist)[0]
            
            # Use all blocks at minimum distance, regardless of k
            classes_k = [block_labels[idx] for idx in min_dist_indices]
            
            # Use Copeland's method for tie-breaking
            unique_classes, counts = np.unique(classes_k, return_counts=True)
            if len(unique_classes) == 1:
                predictions[i] = unique_classes[0]
            else:
                # Optimized Copeland calculation
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
                predictions[i] = best_class
            
            knn_count += 1
    return predictions, contained_count, knn_count

def classify_with_hyperblocks(X, y, blocks, k_values=None):
    if not blocks:
        return pd.DataFrame(columns=['norm', 'k', 'accuracy', 'preds', 'contained_count', 'knn_count'])
    if k_values is None:
        k_values = range(1, len(blocks) + 1)
    block_bounds = [np.array(b['bounds']) for b in blocks]
    block_labels = np.array([b['class'] for b in blocks])
    results = []
    for norm in [1, 2]:
        for k in k_values:
            if k > len(blocks):
                continue
            predictions, contained_count, knn_count = classify_batch(
                X, block_bounds, block_labels, k, norm, blocks
            )
            acc = accuracy_score(y, predictions)
            results.append({
                'norm': f'L{norm}',
                'k': k,
                'accuracy': acc,
                'preds': predictions.tolist(),
                'contained_count': contained_count,
                'knn_count': knn_count
            })
    return pd.DataFrame(results)

def learn_optimal_hyperparameters(X_train, y_train, blocks):
    """Find the best norm and k for the given training data and blocks."""
    if not blocks:
        return 'L2', 1
    best_accuracy = 0
    best_norm = 'L2'
    best_k = 1
    block_bounds = [np.array(b['bounds']) for b in blocks]
    block_labels = np.array([b['class'] for b in blocks])
    for norm in [1, 2]:
        max_k = len(blocks)
        for k in range(1, max_k + 1):
            predictions, _, _ = classify_batch(X_train, block_bounds, block_labels, k, norm, blocks)
            accuracy = np.mean(predictions == y_train)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_norm = f'L{norm}'
                best_k = k
    return best_norm, best_k

def find_all_envelope_blocks(X, y, feature_indices, classes, pbar=None):
    encoded_classes = np.unique(y)
    args_list = [(c, X, y, feature_indices) for c in encoded_classes]
    all_blocks = []
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for result in executor.map(compute_envelope_blocks_for_class, args_list):
            all_blocks.extend(result)
            if pbar:
                pbar.update(1)
    pruned_blocks = prune_and_merge_blocks(all_blocks)
    deduplicated_blocks = deduplicate_blocks(pruned_blocks)
    scored_blocks = compute_block_confidence_scores(deduplicated_blocks, X, y)
    flagged_blocks = flag_misclassifying_blocks(scored_blocks, X, y, threshold=0.1)
    shrunk_blocks = shrink_dominant_blocks(flagged_blocks, X, y, epsilon=0.1)
    return shrunk_blocks

def fold_worker(args):
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

def cross_validate_blocks(X, y, feature_indices, classes, features, k_folds=10):
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    args_list = []
    encoded_classes = np.unique(y)
    total_tasks = k_folds * len(encoded_classes)
    with tqdm(total=total_tasks, desc="Processing hyperblocks", unit="class") as pbar:
        for fold_num, (train_index, test_index) in enumerate(kf.split(X, y)):
            args_list.append((train_index, test_index, X, y, feature_indices, classes, fold_num, pbar))
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            all_results = list(executor.map(fold_worker, args_list))
    valid_results = []
    block_counts = []
    learned_norms = []
    learned_ks = []
    all_predictions = []
    all_true_labels = []
    all_test_data = []  # Store test data for each fold
    all_blocks_per_fold = []  # Store blocks for each fold
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
    print("\nClass mapping:")
    for class_num in np.unique(y):
        print(f"  {class_num}: {classes[class_num]}")
    print()
    fold_accuracies = []
    for df in valid_results:
        if not df.empty:
            fold_accuracies.append(df.iloc[0]['accuracy'])
    if fold_accuracies:
        print("Per-fold results:")
        fold_contained = []
        fold_knn = []
        for i, (df, num_blocks, norm, k, predictions, y_test, X_test, blocks) in enumerate(zip(valid_results, block_counts, learned_norms, learned_ks, all_predictions, all_true_labels, all_test_data, all_blocks_per_fold)):
            if not df.empty:
                acc = df.iloc[0]['accuracy']
                contained = df.iloc[0]['contained_count']
                knn = df.iloc[0]['knn_count']
                fold_contained.append(contained)
                fold_knn.append(knn)
                print(f"Fold {i+1}: accuracy = {acc:.4f}, blocks = {num_blocks}, contained = {contained}, k-NN = {knn}, norm = {norm}, k = {k}")
                print(f"Confusion Matrix (Fold {i+1}):")
                cm = confusion_matrix(y_test, predictions)
                print(cm)
                
                # Add detailed misclassification analysis
                analyze_misclassifications(X_test, y_test, predictions, blocks, classes, features, i)
                print()
        avg_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        avg_blocks = np.mean(block_counts)
        std_blocks = np.std(block_counts)
        print("Cross-validation summary:")
        print(f"Average accuracy across all folds: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"Average blocks per fold: {avg_blocks:.1f} ± {std_blocks:.1f}")
        if fold_contained:
            avg_contained = np.mean(fold_contained)
            std_contained = np.std(fold_contained)
            avg_knn = np.mean(fold_knn)
            std_knn = np.std(fold_knn)
            print(f"Average contained cases per fold: {avg_contained:.1f} ± {std_contained:.1f}")
            print(f"Average k-NN cases per fold: {avg_knn:.1f} ± {std_knn:.1f}")
    # Save hyperblock bounds to CSV
    save_hyperblock_bounds_to_csv(all_blocks_per_fold, classes, features, k_folds)
    
    return pd.DataFrame({'fold_accuracies': fold_accuracies})

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Hyperblock-based classification')
    parser.add_argument('--train', type=str, help='Path to training dataset CSV')
    parser.add_argument('--test', type=str, help='Path to test dataset CSV')
    parser.add_argument('--dataset', type=str, default='datasets/fisher_iris.csv', help='Path to single dataset CSV (default)')
    args = parser.parse_args()
    if args.train and args.test:
        print("Loading separate train and test datasets...")
        df_train = pd.read_csv(args.train)
        df_test = pd.read_csv(args.test)
        features = df_train.columns[:-1]
        X_train_raw = df_train[features].values
        y_train_raw = df_train['class'].values
        X_test_raw = df_test[features].values
        y_test_raw = df_test['class'].values
        print("Preprocessing data...")
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        y_encoder = LabelEncoder()
        y_train = y_encoder.fit_transform(y_train_raw)
        y_test = y_encoder.transform(y_test_raw)
        feature_indices = list(range(X_train.shape[1]))
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Classes: {y_encoder.classes_}")
        print(f"Features: {len(feature_indices)}")
        print("\nRunning cross-validation on training data...")
        scores = cross_validate_blocks(X_train, y_train, feature_indices, y_encoder.classes_, features)
        print("\nEvaluating on held-out test set...")
        encoded_classes = np.unique(y_train)
        with tqdm(total=len(encoded_classes), desc="Processing test set hyperblocks", unit="class") as pbar:
            blocks = find_all_envelope_blocks(X_train, y_train, feature_indices, y_encoder.classes_, pbar)
        best_norm, best_k = learn_optimal_hyperparameters(X_train, y_train, blocks)
        df_test_results = classify_with_hyperblocks(X_test, y_test, blocks, k_values=[best_k])
        if not df_test_results.empty:
            test_acc = df_test_results.iloc[0]['accuracy']
            print(f"Test set accuracy: {test_acc:.4f}")
            print(f"Test set predictions: {df_test_results.iloc[0]['preds']}")
            test_predictions = df_test_results.iloc[0]['preds']
            print("Test Set Confusion Matrix:")
            cm = confusion_matrix(y_test, test_predictions)
            print(cm)
            
            # Add detailed misclassification analysis for test set
            print("\n=== TEST SET MISCLASSIFICATION ANALYSIS ===")
            analyze_misclassifications(X_test, y_test, test_predictions, blocks, y_encoder.classes_, features, -1)
        print("\nFinal results:")
        print(scores)
    else:
        print("Loading dataset...")
        df = pd.read_csv(args.dataset)
        features = df.columns[:-1]
        X_raw = df[features].values
        y_raw = df['class'].values
        print("Preprocessing data...")
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X_raw)
        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(y_raw)
        feature_indices = list(range(X.shape[1]))
        print(f"Dataset shape: {X.shape}")
        print(f"Classes: {y_encoder.classes_}")
        print(f"Features: {len(feature_indices)}")
        print("\nRunning cross-validation...")
        scores = cross_validate_blocks(X, y, feature_indices, y_encoder.classes_, features)
        print("\nFinal results:")
        print(scores)

if __name__ == "__main__":
    main()
