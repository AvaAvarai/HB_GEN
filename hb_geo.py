import pandas as pd
import numpy as np
from itertools import permutations
from sklearn.metrics import accuracy_score
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

def distance_to_hyperblock(point, bounds, norm=2):
    projected = np.clip(point, bounds[:, 0], bounds[:, 1])
    return np.linalg.norm(point - projected, ord=norm)

def find_all_envelope_blocks(X, y, feature_indices, classes):
    args_list = [(c, X, y, feature_indices) for c in classes]
    all_blocks = []
    # Use ThreadPoolExecutor to avoid multiprocessing issues
    with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 8)) as executor:
        for result in executor.map(compute_envelope_blocks_for_class, args_list):
            all_blocks.extend(result)
    return prune_and_merge_blocks(all_blocks)

def classify_with_hyperblocks(X, y, blocks, k_values=range(1, 11)):
    if not blocks:
        # Return empty DataFrame with correct structure if no blocks
        return pd.DataFrame(columns=['norm', 'k', 'accuracy', 'preds', 'contained_count', 'knn_count'])
    
    block_bounds = [np.array(b['bounds']) for b in blocks]
    block_labels = [b['class'] for b in blocks]

    results = []
    for norm in [1, 2]:
        dists = np.array([[distance_to_hyperblock(x, bb, norm=norm) for bb in block_bounds] for x in X])
        for k in k_values:
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

def fold_worker(args):
    train_index, test_index, X, y, feature_indices, classes, fold_num = args
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    blocks = find_all_envelope_blocks(X_train, y_train, feature_indices, classes)
    df_results = classify_with_hyperblocks(X_test, y_test, blocks)
    
    # Add fold information properly
    if not df_results.empty:
        df_results['fold_num'] = fold_num
        df_results['test_size'] = len(test_index)
        df_results['num_blocks'] = len(blocks)
    
    return df_results, len(blocks)

def cross_validate_blocks(X, y, feature_indices, classes, k_folds=10):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    args_list = []
    
    for fold_num, (train_index, test_index) in enumerate(kf.split(X)):
        args_list.append((train_index, test_index, X, y, feature_indices, classes, fold_num))

    # Use ThreadPoolExecutor to avoid multiprocessing issues
    with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 8)) as executor:
        all_results = list(executor.map(fold_worker, args_list))

    # Separate results and block counts
    valid_results = []
    block_counts = []
    for df, num_blocks in all_results:
        if not df.empty:
            valid_results.append(df)
            block_counts.append(num_blocks)
    
    if not valid_results:
        print("No valid results found across all folds")
        return pd.DataFrame()
    
    combined = pd.concat(valid_results, ignore_index=True)
    avg_scores = combined.groupby(['norm', 'k'])['accuracy'].mean().reset_index()
    
    if not avg_scores.empty:
        best_row = avg_scores.loc[avg_scores['accuracy'].idxmax()]
        best_norm, best_k = best_row['norm'], best_row['k']

        print("\nBest hyperparameter configuration across folds:")
        print(best_row)

        print("\nPer-fold results using best configuration:")
        fold_accuracies = []
        fold_contained = []
        fold_knn = []
        for i, (df, num_blocks) in enumerate(zip(valid_results, block_counts)):
            match = df[(df['norm'] == best_norm) & (df['k'] == best_k)]
            if not match.empty:
                acc = match.iloc[0]['accuracy']
                contained = match.iloc[0]['contained_count']
                knn = match.iloc[0]['knn_count']
                fold_accuracies.append(acc)
                fold_contained.append(contained)
                fold_knn.append(knn)
                print(f"Fold {i+1}: accuracy = {acc:.4f}, blocks = {num_blocks}, contained = {contained}, k-NN = {knn}")
        
        # Calculate and display average accuracy
        if fold_accuracies:
            avg_accuracy = np.mean(fold_accuracies)
            std_accuracy = np.std(fold_accuracies)
            avg_blocks = np.mean(block_counts)
            std_blocks = np.std(block_counts)
            avg_contained = np.mean(fold_contained)
            std_contained = np.std(fold_contained)
            avg_knn = np.mean(fold_knn)
            std_knn = np.std(fold_knn)
            print(f"\nAverage accuracy across all folds: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
            print(f"Average blocks per fold: {avg_blocks:.1f} ± {std_blocks:.1f}")
            print(f"Average contained cases per fold: {avg_contained:.1f} ± {std_contained:.1f}")
            print(f"Average k-NN cases per fold: {avg_knn:.1f} ± {std_knn:.1f}")

    return avg_scores

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
    classes = np.unique(y)
    feature_indices = list(range(X.shape[1]))

    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {classes}")
    print(f"Features: {len(feature_indices)}")

    print("\nRunning cross-validation...")
    scores = cross_validate_blocks(X, y, feature_indices, classes)
    # Removed the final results table output
    # print("\nFinal results:")
    # print(scores)

if __name__ == "__main__":
    main()
