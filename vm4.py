#!/usr/bin/env python3
# dcp_vm4.py
"""
r'''
Dominance Classifier & Predictor (DCP) — Voting Method VM4
Implements VM4 as described in:
Kovalerchuk & Neuhaus, "Toward Efficient Automation of Interpretable Machine Learning,"
IEEE Big Data 2018. (See paper for VM1–VM4 definitions.)

Features:
- Builds dominance intervals per feature on normalized [0,1] scales.
- VM4 voting: ratio votes (VM3) with per-attribute vote cap (VLP) and minimum interval length (LLI).
- Dominance threshold DTC for including intervals.
- Sequential hyperparameter search (DTC -> VLP -> LLI) on training folds.
- 10-fold stratified cross-validation.
- Works with multi-class labels.

CLI:
  python dcp_vm4.py --csv PATH --label-column class --cv 10
  Optional: --features f1 f2 f3 (defaults to all non-label columns)

Programmatic usage:
  from dcp_vm4 import DCPVM4
  clf = DCPVM4().fit(X_train, y_train)
  y_pred = clf.predict(X_test)
'''
+
r"""
import argparse
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

@dataclass
class Interval:
    start: float
    end: float
    counts: np.ndarray  # counts per class
    dom_class: int      # dominant class index
    dominance_ratio: float  # N_dom / sum_all

@dataclass
class FeatureIntervals:
    feature_index: int
    intervals: List[Interval] = field(default_factory=list)

class DCPVM4:
    def __init__(
        self,
        dtc_grid: Optional[List[float]] = None,  # Dominance threshold candidates
        vlp_grid: Optional[List[float]] = None,  # Vote cap candidates
        lli_grid: Optional[List[float]] = None,  # Min interval length candidates
        random_state: int = 42,
    ):
        # Reasonable defaults (fast-ish). You can widen these for thoroughness.
        self.dtc_grid = dtc_grid if dtc_grid is not None else [0.6, 0.7, 0.8, 0.9]
        self.vlp_grid = vlp_grid if vlp_grid is not None else [1, 2, 3, 5, 10]
        self.lli_grid = lli_grid if lli_grid is not None else [0.0, 0.05, 0.1, 0.2]
        self.random_state = random_state

        # Learned parameters
        self.DTC_: float = 0.7
        self.VLP_: float = 3.0
        self.LLI_: float = 0.05

        self.scaler_: Optional[MinMaxScaler] = None
        self.le_: Optional[LabelEncoder] = None
        self.n_classes_: int = 0
        self.feature_intervals_: List[FeatureIntervals] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DCPVM4":
        # Normalize features to [0,1]
        self.scaler_ = MinMaxScaler()
        Xn = self.scaler_.fit_transform(X)

        # Encode labels to 0..k-1
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.n_classes_ = int(y_enc.max()) + 1

        # Sequential search for DTC, then VLP, then LLI
        # Note: quick/greedy; expand grids for thorough search if desired.
        # 1) DTC
        best_acc = -np.inf
        best_dtc = self.DTC_
        for dtc in self.dtc_grid:
            self.DTC_ = dtc
            # Keep interim defaults for others during this stage:
            self.VLP_ = self.vlp_grid[0]
            self.LLI_ = self.lli_grid[0]
            self._build_all_intervals(Xn, y_enc)
            acc = self._quick_oob_accuracy(Xn, y_enc)
            if acc > best_acc:
                best_acc = acc
                best_dtc = dtc
        self.DTC_ = best_dtc

        # 2) VLP
        best_acc = -np.inf
        best_vlp = self.VLP_
        for vlp in self.vlp_grid:
            self.VLP_ = float(vlp)
            self._build_all_intervals(Xn, y_enc)  # rebuild since thresholds affect pruning
            acc = self._quick_oob_accuracy(Xn, y_enc)
            if acc > best_acc:
                best_acc = acc
                best_vlp = float(vlp)
        self.VLP_ = best_vlp

        # 3) LLI
        best_acc = -np.inf
        best_lli = self.LLI_
        for lli in self.lli_grid:
            self.LLI_ = float(lli)
            self._build_all_intervals(Xn, y_enc)
            acc = self._quick_oob_accuracy(Xn, y_enc)
            if acc > best_acc:
                best_acc = acc
                best_lli = float(lli)
        self.LLI_ = best_lli

        # Final build with learned params
        self._build_all_intervals(Xn, y_enc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_ is None:
            raise RuntimeError("Model is not fitted.")
        Xn = self.scaler_.transform(X)
        y_pred = [self._predict_one(xn) for xn in Xn]
        return self.le_.inverse_transform(np.array(y_pred, dtype=int))

    # ====== Core helpers ======

    def _build_all_intervals(self, Xn: np.ndarray, y: np.ndarray) -> None:
        self.feature_intervals_ = []
        n, d = Xn.shape
        for j in range(d):
            self.feature_intervals_.append(
                FeatureIntervals(feature_index=j, intervals=self._build_intervals_for_feature(Xn[:, j], y))
            )

    def _build_intervals_for_feature(self, col: np.ndarray, y: np.ndarray) -> List[Interval]:
        # Build atomic segments between sorted unique values;
        # Assign each atomic segment a dominant class (by value-level counts),
        # then merge consecutive segments with same dominant class.
        # Finally prune by DTC (dominance ratio) and LLI (min length).
        uniq_vals = np.unique(col)
        # Degenerate case: all values equal -> single "interval" with length 0
        if len(uniq_vals) == 1:
            counts = np.zeros(self.n_classes_, dtype=int)
            for cls in range(self.n_classes_):
                counts[cls] = np.sum(y[(col == uniq_vals[0]) & (y == cls)])
            total = counts.sum()
            if total == 0:
                return []
            dom = int(np.argmax(counts))
            ratio = counts[dom] / total if total > 0 else 0.0
            inter = Interval(start=float(uniq_vals[0]), end=float(uniq_vals[0]), counts=counts, dom_class=dom, dominance_ratio=ratio)
            return [inter] if ratio >= self.DTC_ and (inter.end - inter.start) >= self.LLI_ else []

        # Precompute counts per unique value
        value_counts = []
        for v in uniq_vals:
            counts = np.zeros(self.n_classes_, dtype=int)
            mask_v = (col == v)
            for cls in range(self.n_classes_):
                counts[cls] = np.sum(mask_v & (y == cls))
            value_counts.append(counts)

        # Atomic segments: between midpoints of consecutive unique values.
        # We'll represent an atomic segment i as [a_i, b_i], where
        # a_0 = v0, b_last = v_last, and for inner boundaries we use midpoints.
        mids = (uniq_vals[:-1] + uniq_vals[1:]) / 2.0
        seg_starts = np.concatenate([[uniq_vals[0]], mids])
        seg_ends   = np.concatenate([mids, [uniq_vals[-1]]])

        # For each atomic segment, use the counts from the unique value it "covers".
        # A simple choice: attribute segment i uses counts at uniq_vals[i].
        seg_counts = value_counts  # aligned: len == len(uniq_vals) == number of segments here

        # Assign dominant class per segment
        seg_dom = [int(np.argmax(c)) if c.sum() > 0 else -1 for c in seg_counts]

        # Merge consecutive segments with same dominant class
        merged: List[Interval] = []
        cur_start = seg_starts[0]
        cur_end = seg_ends[0]
        cur_counts = seg_counts[0].astype(int).copy()
        cur_dom = seg_dom[0]

        for i in range(1, len(seg_starts)):
            if seg_dom[i] == cur_dom and cur_dom != -1:
                # extend
                cur_end = seg_ends[i]
                cur_counts += seg_counts[i]
            else:
                # finalize previous interval
                total = cur_counts.sum()
                if total > 0 and cur_dom != -1:
                    ratio = cur_counts[cur_dom] / total
                    inter = Interval(float(cur_start), float(cur_end), cur_counts.copy(), cur_dom, ratio)
                    if ratio >= self.DTC_ and (inter.end - inter.start) >= self.LLI_:
                        merged.append(inter)
                # start new
                cur_start = seg_starts[i]
                cur_end = seg_ends[i]
                cur_counts = seg_counts[i].astype(int).copy()
                cur_dom = seg_dom[i]

        # finalize last
        total = cur_counts.sum()
        if total > 0 and cur_dom != -1:
            ratio = cur_counts[cur_dom] / total
            inter = Interval(float(cur_start), float(cur_end), cur_counts.copy(), cur_dom, ratio)
            if ratio >= self.DTC_ and (inter.end - inter.start) >= self.LLI_:
                merged.append(inter)

        return merged

    def _vote_vm4_for_feature(self, xi: float, finter: FeatureIntervals) -> Optional[Tuple[int, float]]:
        # Find interval containing xi; return (class_idx, vote_value) or None if no interval matches
        for inter in finter.intervals:
            if inter.start <= xi <= inter.end:
                counts = inter.counts
                dom = inter.dom_class
                n_dom = float(counts[dom])
                n_other = float(counts.sum() - counts[dom])
                # VM3 ratio (avoid division by zero): if n_other == 0, use n_dom as strong support
                vote = (n_dom / n_other) if n_other > 0.0 else n_dom
                # VM4 caps the per-attribute vote
                vote_capped = min(vote, float(self.VLP_))
                # Discard vote if interval length is below LLI (already pruned in build, but double-check)
                if (inter.end - inter.start) < self.LLI_:
                    return None
                return (dom, vote_capped)
        return None

    def _predict_one(self, x: np.ndarray) -> int:
        class_scores = np.zeros(self.n_classes_, dtype=float)
        for j, finter in enumerate(self.feature_intervals_):
            xi = float(x[j])
            v = self._vote_vm4_for_feature(xi, finter)
            if v is not None:
                dom, vote = v
                class_scores[dom] += vote
        # If no votes (uncovered), fallback to majority class seen in training
        if np.allclose(class_scores, 0.0):
            return 0  # will be inverse-transformed; by default, label encoder maps the smallest class first
        return int(np.argmax(class_scores))

    def _quick_oob_accuracy(self, Xn: np.ndarray, y: np.ndarray) -> float:
        # Evaluate on the training data (used only for quick hyperparam selection)
        preds = [self._predict_one(xi) for xi in Xn]
        return float(np.mean(np.array(preds, dtype=int) == y))

# ---------- Utilities ----------

def run_cv(X: np.ndarray, y: np.ndarray, cv: int = 10, seed: int = 42,
           dtc_grid=None, vlp_grid=None, lli_grid=None) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    accs = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        clf = DCPVM4(dtc_grid=dtc_grid, vlp_grid=vlp_grid, lli_grid=lli_grid, random_state=seed)
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], y_pred)
        accs.append(acc)
        print(f"Fold {fold:2d}: accuracy = {acc:.4f} | DTC={clf.DTC_} VLP={clf.VLP_} LLI={clf.LLI_}")
    accs = np.array(accs, dtype=float)
    print(f"\nMean accuracy over {cv}-fold CV: {accs.mean():.4f} ± {accs.std(ddof=1):.4f}")
    return {"mean_acc": float(accs.mean()), "std_acc": float(accs.std(ddof=1))}

def load_csv(path: str, label_column: str, feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c != label_column]
    X = df[feature_columns].to_numpy(dtype=float)
    y = df[label_column].to_numpy()
    return X, y

def main():
    parser = argparse.ArgumentParser(description="DCP VM4 with 10-fold CV")
    parser.add_argument("--csv", type=str, help="Path to CSV with a label column.", required=False)
    parser.add_argument("--label-column", type=str, default="class", help="Name of label column in CSV.")
    parser.add_argument("--features", type=str, nargs="*", help="Optional list of feature columns (defaults to all non-label).")
    parser.add_argument("--cv", type=int, default=10, help="Number of CV folds (default: 10).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--demo-iris", action="store_true", help="Run a quick demo on sklearn Iris.")
    args = parser.parse_args()

    if args.demo_iris:
        from sklearn.datasets import load_iris
        data = load_iris()
        X = data.data
        y = data.target
        print("Running VM4 on Iris dataset (3 classes)...")
        run_cv(X, y, cv=args.cv, seed=args.seed)
        sys.exit(0)

    if not args.csv:
        print("Please provide --csv PATH or use --demo-iris.", file=sys.stderr)
        sys.exit(1)

    X, y = load_csv(args.csv, args.label_column, args.features)
    run_cv(X, y, cv=args.cv, seed=args.seed)

if __name__ == "__main__":
    main()