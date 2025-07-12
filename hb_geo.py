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

    # Use LDA to find discriminative feature order, but only for this class vs rest
    lda = LinearDiscriminantAnalysis()
    try:
        # Prepare binary labels: 1 for class c, 0 for others
        y_binary = (y == c).astype(int)
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

    # Cluster the class points to find natural groupings
    
    # Determine optimal number of clusters using silhouette analysis
    max_clusters = min(5, class_points.shape[0] // 2)  # Limit to reasonable number
    best_n_clusters = 1
    best_silhouette = -1
    
    for n_clusters in range(1, max_clusters + 1):
        if n_clusters == 1:
            silhouette = 0  # Single cluster has no silhouette score
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
    
    # Generate hyperblocks for each cluster
    if best_n_clusters == 1:
        # Single cluster - create one hyperblock for the entire class
        bounds = np.full((num_features, 2), [np.inf, -np.inf])
        
        # Create basic envelope for all class points
        for p in class_points:
            for i in range(num_features):
                bounds[i, 0] = min(bounds[i, 0], p[i])
                bounds[i, 1] = max(bounds[i, 1], p[i])
        
        # Try to refine with intersection patterns if other points exist
        if other_points.shape[0] > 0:
            refined_bounds = np.full((num_features, 2), [np.inf, -np.inf])
            has_intersection = False
            
            # Wrap-around scanning: each feature sees its left and right neighbors
            for i in range(len(sorted_features)):
                left_idx = sorted_features[i - 1] if i > 0 else sorted_features[-1]
                right_idx = sorted_features[i + 1] if i < len(sorted_features) - 1 else sorted_features[0]
                curr_idx = sorted_features[i]

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
                            refined_bounds[curr_idx, 0] = min(refined_bounds[curr_idx, 0], p_curr)
                            refined_bounds[curr_idx, 1] = max(refined_bounds[curr_idx, 1], p_curr)
                            refined_bounds[left_idx, 0] = min(refined_bounds[left_idx, 0], p_left)
                            refined_bounds[left_idx, 1] = max(refined_bounds[left_idx, 1], p_left)
                            refined_bounds[right_idx, 0] = min(refined_bounds[right_idx, 0], p_right)
                            refined_bounds[right_idx, 1] = max(refined_bounds[right_idx, 1], p_right)
                            has_intersection = True

            # If we found intersection patterns, use the refined bounds
            if has_intersection:
                bounds = refined_bounds
        
        blocks.append({
            'class': c,
            'bounds': bounds.tolist(),
            'feature_order': sorted_features.tolist()
        })
        
    else:
        # Multiple clusters - create hyperblocks for each cluster
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(class_points)
        
        for cluster_id in range(best_n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_points = class_points[cluster_mask]
            
            if cluster_points.shape[0] == 0:
                continue
                
            # Create bounds for this cluster
            bounds = np.full((num_features, 2), [np.inf, -np.inf])
            
            # Create envelope for cluster points
            for p in cluster_points:
                for i in range(num_features):
                    bounds[i, 0] = min(bounds[i, 0], p[i])
                    bounds[i, 1] = max(bounds[i, 1], p[i])
            
            # Try to refine with intersection patterns if other points exist
            if other_points.shape[0] > 0:
                refined_bounds = np.full((num_features, 2), [np.inf, -np.inf])
                has_intersection = False
                
                # Wrap-around scanning for this cluster
                for i in range(len(sorted_features)):
                    left_idx = sorted_features[i - 1] if i > 0 else sorted_features[-1]
                    right_idx = sorted_features[i + 1] if i < len(sorted_features) - 1 else sorted_features[0]
                    curr_idx = sorted_features[i]

                    for p in cluster_points:
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
                                refined_bounds[curr_idx, 0] = min(refined_bounds[curr_idx, 0], p_curr)
                                refined_bounds[curr_idx, 1] = max(refined_bounds[curr_idx, 1], p_curr)
                                refined_bounds[left_idx, 0] = min(refined_bounds[left_idx, 0], p_left)
                                refined_bounds[left_idx, 1] = max(refined_bounds[left_idx, 1], p_left)
                                refined_bounds[right_idx, 0] = min(refined_bounds[right_idx, 0], p_right)
                                refined_bounds[right_idx, 1] = max(refined_bounds[right_idx, 1], p_right)
                                has_intersection = True

                # If we found intersection patterns, use the refined bounds
                if has_intersection:
                    bounds = refined_bounds
            
            blocks.append({
                'class': c,
                'bounds': bounds.tolist(),
                'feature_order': sorted_features.tolist(),
                'cluster_id': cluster_id
            })

    return blocks

def prune_and_merge_blocks(blocks):
    pruned = []
    for b in blocks:
        bounds = np.array(b['bounds'])
        # Ensure bounds are finite and have reasonable volume
        if np.all(np.isfinite(bounds)):
            volume = np.prod(bounds[:, 1] - bounds[:, 0])
            # Be more lenient with volume threshold to preserve basic envelopes
            if volume > 1e-10:
                pruned.append(b)
            else:
                # If volume is too small, expand bounds slightly to ensure minimum volume
                expanded_bounds = bounds.copy()
                for i in range(bounds.shape[0]):
                    if bounds[i, 1] - bounds[i, 0] < 1e-6:
                        mid = (bounds[i, 0] + bounds[i, 1]) / 2
                        expanded_bounds[i, 0] = mid - 1e-6
                        expanded_bounds[i, 1] = mid + 1e-6
                b['bounds'] = expanded_bounds.tolist()
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

def find_all_envelope_blocks(X, y, feature_indices, classes, pbar=None):
    # Use encoded class numbers for block generation
    encoded_classes = np.unique(y)
    args_list = [(c, X, y, feature_indices) for c in encoded_classes]
    all_blocks = []
    
    # Use ThreadPoolExecutor to avoid multiprocessing issues
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for result in executor.map(compute_envelope_blocks_for_class, args_list):
            all_blocks.extend(result)
            if pbar:
                pbar.update(1)
    
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
        
        # Test k values from 1 to number of blocks
        max_k = len(blocks)
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
    train_index, test_index, X, y, feature_indices, classes, fold_num, pbar = args
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    blocks = find_all_envelope_blocks(X_train, y_train, feature_indices, classes, pbar)
    
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
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    args_list = []
    
    # Calculate total number of tasks (folds * classes per fold)
    encoded_classes = np.unique(y)
    total_tasks = k_folds * len(encoded_classes)
    
    # Create single progress bar for all processing
    with tqdm(total=total_tasks, desc="Processing hyperblocks", unit="class") as pbar:
        for fold_num, (train_index, test_index) in enumerate(kf.split(X, y)):
            args_list.append((train_index, test_index, X, y, feature_indices, classes, fold_num, pbar))

        # Use ThreadPoolExecutor to avoid multiprocessing issues
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperblock-based classification')
    parser.add_argument('--train', type=str, help='Path to training dataset CSV')
    parser.add_argument('--test', type=str, help='Path to test dataset CSV')
    parser.add_argument('--dataset', type=str, default='datasets/fisher_iris.csv', help='Path to single dataset CSV (default)')
    
    args = parser.parse_args()
    
    if args.train and args.test:
        # Load separate train and test datasets
        print("Loading separate train and test datasets...")
        df_train = pd.read_csv(args.train)
        df_test = pd.read_csv(args.test)
        
        features = df_train.columns[:-1]  # Assume last column is class
        X_train_raw = df_train[features].values
        y_train_raw = df_train['class'].values
        X_test_raw = df_test[features].values
        y_test_raw = df_test['class'].values
        
        print("Preprocessing data...")
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)  # Use same scaler
        
        y_encoder = LabelEncoder()
        y_train = y_encoder.fit_transform(y_train_raw)
        y_test = y_encoder.transform(y_test_raw)  # Use same encoder
        
        feature_indices = list(range(X_train.shape[1]))
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Classes: {y_encoder.classes_}")
        print(f"Features: {len(feature_indices)}")
        
        print("\nRunning cross-validation on training data...")
        # Pass encoded class numbers for block generation, original names for display
        scores = cross_validate_blocks(X_train, y_train, feature_indices, y_encoder.classes_)
        
        print("\nEvaluating on held-out test set...")
        # Train on full training set and evaluate on test set
        # Create progress bar for single evaluation
        encoded_classes = np.unique(y_train)
        with tqdm(total=len(encoded_classes), desc="Processing test set hyperblocks", unit="class") as pbar:
            blocks = find_all_envelope_blocks(X_train, y_train, feature_indices, y_encoder.classes_, pbar)
        best_norm, best_k = learn_optimal_hyperparameters(X_train, y_train, blocks)
        
        df_test_results = classify_with_hyperblocks(X_test, y_test, blocks, k_values=[best_k])
        
        if not df_test_results.empty:
            test_acc = df_test_results.iloc[0]['accuracy']
            print(f"Test set accuracy: {test_acc:.4f}")
            print(f"Test set predictions: {df_test_results.iloc[0]['preds']}")
            
            # Print confusion matrix for test set
            test_predictions = df_test_results.iloc[0]['preds']
            print("Test Set Confusion Matrix:")
            cm = confusion_matrix(y_test, test_predictions)
            print(cm)
        
        print("\nFinal results:")
        print(scores)
        
    else:
        # Load single dataset (original behavior)
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
        # Pass encoded class numbers for block generation, original names for display
        scores = cross_validate_blocks(X, y, feature_indices, y_encoder.classes_)
        print("\nFinal results:")
        print(scores)

if __name__ == "__main__":
    main()
