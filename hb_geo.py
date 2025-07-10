import pandas as pd
import numpy as np
from itertools import permutations
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def compute_envelope_blocks_for_class(args):
    c, X, y, feature_indices = args
    class_mask = y == c
    other_mask = y != c
    class_points = X[class_mask]
    other_points = X[other_mask]

    if class_points.shape[0] == 0 or other_points.shape[0] == 0:
        return []

    num_features = X.shape[1]
    blocks = []

    for perm in permutations(feature_indices):
        bounds = np.full((num_features, 2), [np.inf, -np.inf])
        has_intersection = False

        # Wrap-around scanning: each feature sees its left and right neighbors
        for i in range(len(perm)):
            # Left neighbor with wrap-around
            left_idx = perm[i - 1] if i > 0 else perm[-1]
            # Right neighbor with wrap-around  
            right_idx = perm[i + 1] if i < len(perm) - 1 else perm[0]
            
            # Current feature
            curr_idx = perm[i]
            
            for p in class_points:
                p_curr = p[curr_idx]
                p_left = p[left_idx]
                p_right = p[right_idx]
                
                for q in other_points:
                    q_curr = q[curr_idx]
                    q_left = q[left_idx]
                    q_right = q[right_idx]
                    
                    # Check if there's an intersection pattern
                    if ((p_curr - q_curr) * (p_left - q_left) < 0 or 
                        (p_curr - q_curr) * (p_right - q_right) < 0):
                        bounds[curr_idx, 0] = min(bounds[curr_idx, 0], p_curr)
                        bounds[curr_idx, 1] = max(bounds[curr_idx, 1], p_curr)
                        bounds[left_idx, 0] = min(bounds[left_idx, 0], p_left)
                        bounds[left_idx, 1] = max(bounds[left_idx, 1], p_left)
                        bounds[right_idx, 0] = min(bounds[right_idx, 0], p_right)
                        bounds[right_idx, 1] = max(bounds[right_idx, 1], p_right)
                        has_intersection = True

        if has_intersection:
            blocks.append({
                'class': c,
                'bounds': bounds.tolist(),
                'perm': perm
            })

    return blocks

def prune_and_merge_blocks(blocks):
    pruned = []
    for b in blocks:
        bounds = np.array(b['bounds'])
        volume = np.prod(bounds[:, 1] - bounds[:, 0])
        if np.all(np.isfinite(bounds)) and volume > 1e-6:
            pruned.append(b)
    return pruned

def deduplicate_blocks(blocks, tolerance=1e-6):
    """Remove duplicate or nearly identical blocks based on bounds."""
    unique = []
    seen = set()

    for b in blocks:
        bounds = np.round(np.array(b['bounds']), decimals=6)
        key = tuple(bounds.flatten())
        if key not in seen:
            seen.add(key)
            unique.append(b)

    return unique

def distance_to_hyperblock(point, bounds, norm=2):
    projected = np.clip(point, bounds[:, 0], bounds[:, 1])
    return np.linalg.norm(point - projected, ord=norm)

def find_all_envelope_blocks(X, y, feature_indices, classes):
    # Use encoded class numbers for block generation
    encoded_classes = np.unique(y)
    args_list = [(c, X, y, feature_indices) for c in encoded_classes]
    all_blocks = []
    # Use ThreadPoolExecutor to avoid multiprocessing issues
    with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 8)) as executor:
        for result in executor.map(compute_envelope_blocks_for_class, args_list):
            all_blocks.extend(result)
    
    # Apply pruning and merging, then deduplication
    pruned_blocks = prune_and_merge_blocks(all_blocks)
    deduplicated_blocks = deduplicate_blocks(pruned_blocks)
    
    return deduplicated_blocks

def classify_with_hyperblocks(X, y, blocks, k_values=None):
    if not blocks:
        # Return empty DataFrame with correct structure if no blocks
        return pd.DataFrame(columns=['norm', 'k', 'accuracy', 'preds', 'contained_count', 'knn_count'])
    
    # Set default k_values if not provided
    if k_values is None:
        k_values = range(1, len(blocks) + 1)
    
    block_bounds = [np.array(b['bounds']) for b in blocks]
    block_labels = [b['class'] for b in blocks]

    results = []
    for norm in [1, 2]:
        dists = np.array([[distance_to_hyperblock(x, bb, norm=norm) for bb in block_bounds] for x in X])
        for k in k_values:
            if k > len(blocks):
                continue  # Skip k values larger than number of blocks
                
            knn_preds = []
            contained_count = 0
            knn_count = 0
            
            for row in dists:
                # Check if point is contained in any block (distance = 0)
                min_dist_idx = np.argmin(row)
                min_dist = row[min_dist_idx]
                
                if min_dist == 0:
                    # Point is contained in a block
                    pred = block_labels[min_dist_idx]
                    contained_count += 1
                else:
                    # Point needs k-NN classification
                    top_k = np.argsort(row)[:k]
                    classes_k = [block_labels[i] for i in top_k]
                    pred = max(set(classes_k), key=classes_k.count)
                    knn_count += 1
                
                knn_preds.append(pred)
            
            acc = accuracy_score(y, knn_preds)
            results.append({
                'norm': f'L{norm}',
                'k': k,
                'accuracy': acc,
                'preds': knn_preds,
                'contained_count': contained_count,
                'knn_count': knn_count
            })
    return pd.DataFrame(results)

def learn_optimal_hyperparameters(X_train, y_train, blocks):
    """Learn optimal k and norm values for a given fold using training data."""
    if not blocks:
        return 'L2', 1  # Default values if no blocks
    
    # Test different norm and k combinations on training data
    best_accuracy = 0
    best_norm = 'L2'
    best_k = 1
    
    # Test both L1 and L2 norms
    for norm in [1, 2]:
        norm_name = f'L{norm}'
        
        # Test k values from 1 to min(10, number of blocks)
        max_k = min(10, len(blocks))
        for k in range(1, max_k + 1):
            # Use training data to evaluate this combination
            predictions = []
            for sample in X_train:
                # Calculate distances to all blocks
                dists = [distance_to_hyperblock(sample, np.array(b['bounds']), norm=norm) for b in blocks]
                
                if min(dists) == 0:
                    # Point is contained in a block
                    min_dist_idx = np.argmin(dists)
                    pred = blocks[min_dist_idx]['class']
                else:
                    # Point needs k-NN classification
                    top_k_indices = np.argsort(dists)[:k]
                    classes_k = [blocks[i]['class'] for i in top_k_indices]
                    pred = max(set(classes_k), key=classes_k.count)
                
                predictions.append(pred)
            
            # Calculate accuracy on training data
            accuracy = np.mean(np.array(predictions) == y_train)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_norm = norm_name
                best_k = k
    
    return best_norm, best_k

def fold_worker(args):
    train_index, test_index, X, y, feature_indices, classes, fold_num = args
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    blocks = find_all_envelope_blocks(X_train, y_train, feature_indices, classes)
    
    # Learn optimal hyperparameters for this fold
    best_norm, best_k = learn_optimal_hyperparameters(X_train, y_train, blocks)
    
    # Use only the optimal hyperparameters for testing
    df_results = classify_with_hyperblocks(X_test, y_test, blocks, k_values=[best_k])
    
    # Get predictions for confusion matrix
    predictions = []
    if not df_results.empty:
        predictions = df_results.iloc[0]['preds']
    
    # Add fold information properly
    if not df_results.empty:
        df_results['fold_num'] = fold_num
        df_results['test_size'] = len(test_index)
        df_results['num_blocks'] = len(blocks)
        df_results['learned_norm'] = best_norm
        df_results['learned_k'] = best_k
    
    return df_results, len(blocks), best_norm, best_k, predictions, y_test

def cross_validate_blocks(X, y, feature_indices, classes, k_folds=10):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    args_list = []
    
    for fold_num, (train_index, test_index) in enumerate(kf.split(X)):
        args_list.append((train_index, test_index, X, y, feature_indices, classes, fold_num))

    # Use ThreadPoolExecutor to avoid multiprocessing issues
    with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 8)) as executor:
        all_results = list(executor.map(fold_worker, args_list))

    # Separate results and metadata
    valid_results = []
    block_counts = []
    learned_norms = []
    learned_ks = []
    all_predictions = []
    all_true_labels = []
    
    for df, num_blocks, norm, k, predictions, y_test in all_results:
        if not df.empty:
            valid_results.append(df)
            block_counts.append(num_blocks)
            learned_norms.append(norm)
            learned_ks.append(k)
            all_predictions.append(predictions)
            all_true_labels.append(y_test)
    
    if not valid_results:
        print("No valid results found across all folds")
        return pd.DataFrame()
    
    # Print class mapping once at the beginning
    print("\nClass mapping:")
    for class_num in np.unique(y):
        print(f"  {class_num}: {classes[class_num]}")
    print()
    
    # Calculate average accuracy across folds
    fold_accuracies = []
    for df in valid_results:
        if not df.empty:
            fold_accuracies.append(df.iloc[0]['accuracy'])
    
    if fold_accuracies:
        print("Per-fold results:")
        fold_contained = []
        fold_knn = []
        for i, (df, num_blocks, norm, k, predictions, y_test) in enumerate(zip(valid_results, block_counts, learned_norms, learned_ks, all_predictions, all_true_labels)):
            if not df.empty:
                acc = df.iloc[0]['accuracy']
                contained = df.iloc[0]['contained_count']
                knn = df.iloc[0]['knn_count']
                fold_contained.append(contained)
                fold_knn.append(knn)
                print(f"Fold {i+1}: accuracy = {acc:.4f}, blocks = {num_blocks}, contained = {contained}, k-NN = {knn}, norm = {norm}, k = {k}")
                
                # Print confusion matrix for this fold
                print(f"Confusion Matrix (Fold {i+1}):")
                cm = confusion_matrix(y_test, predictions)
                print(cm)
                print()
        
        # Print summary statistics at the end
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

    return pd.DataFrame({'fold_accuracies': fold_accuracies})

def main():
    print("Loading Fisher Iris dataset...")
    df = pd.read_csv("datasets/fisher_iris.csv")
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
    # Pass encoded class numbers for block generation, original names for display
    scores = cross_validate_blocks(X, y, feature_indices, y_encoder.classes_)
    print("\nFinal results:")
    print(scores)

if __name__ == "__main__":
    main()
