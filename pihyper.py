#!/usr/bin/env python3
"""
PIHyper: Parallel Interval Hyperblock classifier

- Loads a CSV with a label column.
- Min–max normalizes features (fit on training split only).
- For each class, iteratively finds the largest high-purity interval on ANY attribute
  using a parallel sliding-window sweep (two pointers), envelopes those points into
  an n-D hyperblock (per-dimension min/max), and removes the covered IN-CLASS points.
- Stores each HB with (bounds, label, purity, size).
- Prediction: if one or more HBs contain x, choose the HB with highest (purity, size).
  If none contain x, fall back to k-NN (Euclidean) on the TRAIN split.

Usage:
  python pihyper.py --csv your.csv --label class --purity 0.95 --min-hb-size 5 \
                    --folds 10 --knn 5

Notes:
- Designed to be deterministic and readable; not micro-optimized.
- Purity T applies while growing attribute intervals; final HB bounds are the envelope
  over the selected window's points across ALL features.
- For interpretability, consider exporting HB rules post-fit.
"""

import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from collections import defaultdict


# ----------------------------- Utilities -----------------------------

def minmax_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    # avoid divide-by-zero: set zero-span cols to [0.5] after transform
    span = np.where(mx > mn, mx - mn, 1.0)
    return mn, span

def minmax_transform(X: np.ndarray, mn: np.ndarray, span: np.ndarray) -> np.ndarray:
    Z = (X - mn) / span
    # columns with zero span become 0 (we’ll later recenter to 0.5 if desired)
    return np.clip(Z, 0.0, 1.0)

def knn_predict(train_X, train_y, test_X, k):
    # simple Euclidean kNN
    preds = []
    for x in test_X:
        d = np.linalg.norm(train_X - x, axis=1)
        idx = np.argpartition(d, k)[:k]
        # majority vote
        vals, counts = np.unique(train_y[idx], return_counts=True)
        preds.append(vals[np.argmax(counts)])
    return np.array(preds)


@dataclass
class Hyperblock:
    """Axis-aligned n-D box: bounds[i] = (low_i, high_i)"""
    bounds: List[Tuple[float, float]]
    label: int
    purity: float
    size: int

    def contains(self, x: np.ndarray) -> bool:
        for (lo, hi), xi in zip(self.bounds, x):
            if xi < lo or xi > hi:
                return False
        return True


# ----------------------------- PIHyper core -----------------------------

def _best_interval_for_attribute(
    X_attr: np.ndarray,
    y: np.ndarray,
    target_label: int,
    purity_T: float,
    min_hb_size: int
) -> Optional[Tuple[int, int, float, int]]:
    """
    Return the best (l, r) window indices (inclusive-exclusive) on sorted X_attr
    maximizing window size while window purity >= T. Ties broken by #target first.

    Returns: (l_idx, r_idx, purity, size) or None if no valid window >= min_hb_size.
    """
    # sort by attribute
    order = np.argsort(X_attr)
    vals = X_attr[order]
    cls = (y[order] == target_label).astype(int)

    best = None  # (l, r, purity, size, target_count)
    l = 0
    target_in = 0
    for r in range(len(vals)):
        target_in += cls[r]
        # shrink until purity ok or window empty
        while l <= r:
            size = r - l + 1
            purity = target_in / size
            if purity >= purity_T:
                # candidate
                # prefer larger size; break ties by more target_in
                if size >= min_hb_size:
                    if best is None:
                        best = (l, r + 1, purity, size, target_in)
                    else:
                        _, _, bp, bs, bt = best
                        if (size > bs) or (size == bs and target_in > bt) or (size == bs and target_in == bt and purity > bp):
                            best = (l, r + 1, purity, size, target_in)
                break
            else:
                # move l forward
                target_in -= cls[l]
                l += 1

    if best is None:
        return None
    l_idx, r_idx, purity, size, _ = best
    # map back to original indices
    window_idx = order[l_idx:r_idx]
    return (l_idx, r_idx, float(purity), int(size)), window_idx, order


def _envelope_bounds(points: np.ndarray) -> List[Tuple[float, float]]:
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    return list(zip(mn.tolist(), mx.tolist()))


def build_pihyper(
    X: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    purity_T: float = 0.95,
    min_hb_size: int = 5,
    max_workers: int = None
) -> List[Hyperblock]:
    """
    Build hyperblocks class-by-class using the PIHyper algorithm.

    For each class:
      while uncovered in-class points remain:
        - For each attribute (in parallel), find the largest interval with purity >= T
          among UNASSIGNED points of ANY label (but purity wrt target class).
        - Pick the best attribute interval overall. Envelope its points across all dims
          to form a HB; record purity & size w.r.t. that window.
        - Mark IN-CLASS points in that window as covered for this class.
    """
    n, d = X.shape
    HBs: List[Hyperblock] = []

    # Track uncovered indices PER CLASS (we allow the same not-target points to be reused by other classes)
    per_class_uncovered: Dict[int, np.ndarray] = {}
    for c in labels:
        per_class_uncovered[c] = np.where(y == c)[0]

    # Precompute arrays we will filter each loop (boolean masks)
    idx_all = np.arange(n)

    for c in labels:
        covered_c = np.zeros(n, dtype=bool)  # marks in-class points already explained by HBs of class c

        while True:
            # candidate search space = all points NOT (in-class & already covered)
            mask_search = np.ones(n, dtype=bool)
            mask_search[per_class_uncovered[c][covered_c[per_class_uncovered[c]]]] = False  # (no-op first pass)
            # The above is a bit awkward; simpler:
            mask_search = np.ones(n, dtype=bool)
            mask_search[np.where((y == c) & covered_c)[0]] = False

            # If all remaining in-class points are covered, stop
            if np.sum((y == c) & (~covered_c)) == 0:
                break

            # If mask_search removes everything (shouldn't), fallback to break
            if mask_search.sum() == 0:
                break

            Xs = X[mask_search]
            ys = y[mask_search]
            # Build mapping to original indices
            orig_idx = idx_all[mask_search]

            # Parallel sweep over attributes
            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for j in range(d):
                    futures.append(ex.submit(
                        _best_interval_for_attribute,
                        Xs[:, j], ys, c, purity_T, min_hb_size
                    ))

                # Gather best across attributes
                best_attr = None
                best_payload = None
                for fut in as_completed(futures):
                    res = fut.result()
                    if res is None:
                        continue
                    (l_idx, r_idx, purity, size), window_idx, order = res
                    # prefer larger size; then larger target count approximated by purity*size
                    if best_attr is None:
                        best_attr = (purity, size)
                        best_payload = (window_idx, purity, size)
                    else:
                        bp, bs = best_attr
                        if (size > bs) or (size == bs and purity > bp):
                            best_attr = (purity, size)
                            best_payload = (window_idx, purity, size)

            if best_payload is None:
                # no more intervals meeting purity & size
                break

            window_local_idx, purity, size = best_payload
            window_global_idx = orig_idx[window_local_idx]
            window_points = X[window_global_idx]

            # Form HB envelope across ALL dimensions for those points
            bounds = _envelope_bounds(window_points)

            # Compute purity in window (w.r.t. class c)
            w_labels = y[window_global_idx]
            target_count = np.sum(w_labels == c)
            window_purity = target_count / len(window_global_idx)

            hb = Hyperblock(bounds=bounds, label=c, purity=float(window_purity), size=int(len(window_global_idx)))
            HBs.append(hb)

            # Mark in-class covered points from this window
            covered_c[window_global_idx[(y[window_global_idx] == c)]] = True

    return HBs


# ----------------------------- Classifier wrapper -----------------------------

class PIHyperClassifier:
    def __init__(self, purity_T=0.95, min_hb_size=5, knn=5, max_workers=None):
        self.purity_T = purity_T
        self.min_hb_size = min_hb_size
        self.knn = knn
        self.max_workers = max_workers
        self.hbs: List[Hyperblock] = []
        self.labels_: Optional[np.ndarray] = None
        self.mn_: Optional[np.ndarray] = None
        self.span_: Optional[np.ndarray] = None
        self.train_X_: Optional[np.ndarray] = None
        self.train_y_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.labels_ = np.unique(y)
        self.mn_, self.span_ = minmax_fit(X)
        Z = minmax_transform(X, self.mn_, self.span_)
        self.hbs = build_pihyper(
            Z, y, labels=self.labels_, purity_T=self.purity_T,
            min_hb_size=self.min_hb_size, max_workers=self.max_workers
        )
        # store train for kNN fallback
        self.train_X_ = Z
        self.train_y_ = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = minmax_transform(X, self.mn_, self.span_)
        preds = []
        for x in Z:
            candidates = []
            for hb in self.hbs:
                if hb.contains(x):
                    # rank by (purity, size)
                    candidates.append((hb.purity, hb.size, hb.label))
            if candidates:
                candidates.sort(reverse=True)  # highest purity/size first
                preds.append(candidates[0][2])
            else:
                # fallback to kNN over training set
                preds.append(knn_predict(self.train_X_, self.train_y_, x[None, :], self.knn)[0])
        return np.array(preds)

    def export_rules(self) -> List[Dict]:
        """Return list of HB rules for inspection/export."""
        rules = []
        for hb in self.hbs:
            rules.append({
                "label": int(hb.label),
                "purity": float(hb.purity),
                "size": int(hb.size),
                "bounds": [(float(lo), float(hi)) for (lo, hi) in hb.bounds]
            })
        return rules


# ----------------------------- CLI / k-fold eval -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV")
    ap.add_argument("--label", required=True, help="Label column name")
    ap.add_argument("--purity", type=float, default=0.95, help="Purity threshold T (0-1)")
    ap.add_argument("--min-hb-size", type=int, default=5, help="Minimum window size to form a HB")
    ap.add_argument("--folds", type=int, default=0, help="If >0, run StratifiedKFold evaluation")
    ap.add_argument("--knn", type=int, default=5, help="k for k-NN fallback")
    ap.add_argument("--workers", type=int, default=None, help="Max worker threads (default: CPU)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.label not in df.columns:
        raise ValueError(f"Label column '{args.label}' not found in CSV.")
    y_raw = df[args.label].values
    X_raw = df.drop(columns=[args.label]).values

    # Map labels to integers if needed
    _, y = np.unique(y_raw, return_inverse=True)

    if args.folds and args.folds > 1:
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        scores = []
        for tr, te in skf.split(X_raw, y):
            clf = PIHyperClassifier(
                purity_T=args.purity, min_hb_size=args.min_hb_size,
                knn=args.knn, max_workers=args.workers
            )
            clf.fit(X_raw[tr], y[tr])
            yp = clf.predict(X_raw[te])
            acc = accuracy_score(y[te], yp)
            scores.append(acc)
        mean = np.mean(scores)
        std = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
        print(f"Accuracy over {args.folds} folds: {mean:.4f} ± {std:.4f}")
    else:
        clf = PIHyperClassifier(
            purity_T=args.purity, min_hb_size=args.min_hb_size,
            knn=args.knn, max_workers=args.workers
        )
        clf.fit(X_raw, y)
        rules = clf.export_rules()
        print(f"Built {len(rules)} hyperblocks.")
        # Preview first few rules
        for i, r in enumerate(rules[:5]):
            print(f"HB #{i+1}: label={r['label']} purity={r['purity']:.3f} size={r['size']}")

if __name__ == "__main__":
    main()
