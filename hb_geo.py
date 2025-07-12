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

def classify_batch(points, block_bounds, block_labels, k, norm):
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
        min_dist_idx = np.argmin(dists)
        min_dist = dists[min_dist_idx]
        if min_dist == 0:
            predictions[i] = block_labels[min_dist_idx]
            contained_count += 1
        else:
            top_k_indices = np.argsort(dists)[:k]
            classes_k = [block_labels[idx] for idx in top_k_indices]
            unique_classes, counts = np.unique(classes_k, return_counts=True)
            predictions[i] = unique_classes[np.argmax(counts)]
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
                X, block_bounds, block_labels, k, norm
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
            predictions, _, _ = classify_batch(X_train, block_bounds, block_labels, k, norm)
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
    return deduplicated_blocks

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
    return df_results, len(blocks), best_norm, best_k, predictions, y_test

def cross_validate_blocks(X, y, feature_indices, classes, k_folds=10):
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
        for i, (df, num_blocks, norm, k, predictions, y_test) in enumerate(zip(valid_results, block_counts, learned_norms, learned_ks, all_predictions, all_true_labels)):
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
        scores = cross_validate_blocks(X_train, y_train, feature_indices, y_encoder.classes_)
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
        scores = cross_validate_blocks(X, y, feature_indices, y_encoder.classes_)
        print("\nFinal results:")
        print(scores)

if __name__ == "__main__":
    main()
