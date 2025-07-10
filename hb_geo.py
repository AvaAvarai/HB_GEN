import pandas as pd
import numpy as np
from itertools import permutations
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.spatial.distance import cdist
from concurrent.futures import ProcessPoolExecutor


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

        for i in range(len(perm) - 1):
            a, b = perm[i], perm[i + 1]
            for p in class_points:
                pi, pj = p[a], p[b]
                for q in other_points:
                    qi, qj = q[a], q[b]
                    if (pi - qi) * (pj - qj) < 0:
                        bounds[a, 0] = min(bounds[a, 0], pi)
                        bounds[a, 1] = max(bounds[a, 1], pi)
                        bounds[b, 0] = min(bounds[b, 0], pj)
                        bounds[b, 1] = max(bounds[b, 1], pj)
                        has_intersection = True

        if has_intersection:
            blocks.append({
                'class': c,
                'bounds': bounds.tolist(),
                'perm': perm
            })

    return blocks


def find_all_envelope_blocks(X, y, feature_indices, classes):
    args_list = [(c, X, y, feature_indices) for c in classes]
    all_blocks = []
    with ProcessPoolExecutor() as executor:
        for result in executor.map(compute_envelope_blocks_for_class, args_list):
            all_blocks.extend(result)
    return all_blocks


def classify_with_hyperblocks(X, y, blocks, norm_type='l2', k_values=range(1, 11)):
    block_centers = []
    block_labels = []
    for b in blocks:
        bounds = np.array(b['bounds'])
        center = np.mean(bounds, axis=1)
        block_centers.append(center)
        block_labels.append(b['class'])
    block_centers = np.array(block_centers)

    results = []
    for norm in [1, 2]:
        dists = cdist(X, block_centers, metric='minkowski', p=norm)
        for k in k_values:
            knn_preds = []
            for row in dists:
                top_k = np.argsort(row)[:k]
                classes_k = [block_labels[i] for i in top_k]
                pred = max(set(classes_k), key=classes_k.count)
                knn_preds.append(pred)
            acc = accuracy_score(y, knn_preds)
            results.append({
                'norm': f'L{norm}',
                'k': k,
                'accuracy': acc
            })
    return pd.DataFrame(results)


if __name__ == "__main__":
    df = pd.read_csv("datasets/fisher_iris.csv")
    features = df.columns[:-1]
    X_raw = df[features].values
    y_raw = df['class'].values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw)

    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(y_raw)
    classes = np.unique(y)
    feature_indices = list(range(X.shape[1]))

    blocks = find_all_envelope_blocks(X, y, feature_indices, classes)
    scores = classify_with_hyperblocks(X, y, blocks)
    print(scores)
