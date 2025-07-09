# Hyperblock Simplification Algorithms

This document describes three novel simplification techniques for reducing hyperblock model complexity and improving interpretability: Remove Redundant Attributes (R2A), Remove Redundant Blocks (R2B), and Disjunctive-Unit HBs. These methods collectively contribute to a more compact and generalized rule set while mitigating overfitting.

## A. Remove Redundant Attributes (R2A)

The R2A algorithm reduces model complexity by identifying and eliminating individual attribute constraints within a hyperblock that do not contribute to its discriminative power. Attributes are removed only when their elimination does not allow any opposing-class training points into the block, thus preserving classification accuracy while reducing the number of clauses.

### Algorithm Outline of R2A

Given a set of hyperblocks and a normalized dataset (all attributes between [0.0, 1.0] for all data):

1) For each hyperblock:
   a) For each attribute:
      - Temporarily expand the hyperblock's interval on the current attribute to the full [0, 1] range.
      - Check if any points from opposing classes now fall within the updated hyperblock.
         - If no opposing points are within the updated hyperblock, keep the expanded interval.
         - If opposing points are now within the updated hyperblock, revert the hyperblock's interval to its previous range.

### Implementation Notes

- This is a greedy algorithm and does not guarantee optimality of attribute removal.
- A domain expert may choose a better order to remove attributes.
- In testing, attributes are sorted by their coefficients found in a multiclass Linear Discriminant Function (LDF).
- The LDF serves as a heuristic to optimize attribute-wise interval generalization order and is not used for classification.

### Example Benefits of R2A

- In Wisconsin Breast Cancer case studies, the largest hyperblock of the benign class required only four out of nine clauses to classify over 400 points.
- In the Fisher Iris dataset, a hyperblock containing all 50 instances of the Setosa class achieves 100% coverage while using only one out of four clauses after R2A simplification.

## B. Remove Redundant Blocks (R2B)

The R2B algorithm eliminates hyperblocks that do not uniquely contribute to the classification of any training data. It is based on the idea that overly specific blocks often represent redundant or overlapping decision regions, which may lead to model overfitting.

### Algorithm Outline of R2B

1) Take each point in the training data, and flag it as belonging to the first hyperblock which it is inside of. (Note that a point can belong to multiple hyperblocks, but that is handled later.)

2) Take a count of how many points belong to each hyperblock.

3) Take each point again, this time flagging it as belonging to the largest hyperblock it is inside of (by count from step 2).

4) Recount the number of cases in each hyperblock.

5) Delete the hyperblocks which have a size (count of training cases) under the removal threshold.

### Benefits

- Greatly reduces overfitting, especially when paired with R2A.
- Often the case that even the largest hyperblocks, classifying huge portions of the dataset, are classifying largely the same group of points with very slight changes between them.
- Multiple hyperblocks may not help to actually classify any new points, since their points can be classified by other hyperblocks.

## C. Disjunctive-Unit Hyperblocks

The Disjunctive HB Simplification algorithm reduces the number of hyperblocks by merging blocks of the same class when it is safe to do so. It attempts to combine two hyperblocks into a single, more general "disjunctive" block that represents multiple allowed intervals per attribute (i.e., logical OR conditions).

### Algorithm Concept

Consider two hyperblocks B1 and B2, both belonging to the same class. Let each block define bounds over attributes x1, x2, ..., xn. Suppose that for all i ≠ j, the bounds on attribute xi are identical in both blocks:

[min_B1^(i), max_B1^(i)] = [min_B2^(i), max_B2^(i)]

For attribute xj, assume the bounds are non-overlapping and distinct:

[min_B1^(j), max_B1^(j)] ∩ [min_B2^(j), max_B2^(j)] = ∅

If merging the bounds on xj into a disjunctive set,

[min_B1^(j), max_B1^(j)] ∪ [min_B2^(j), max_B2^(j)]

results in a hyperblock that does not include any points from other classes, then B1 and B2 may be merged into a single disjunctive hyperblock.

### Implementation Guidelines

- A merge is permitted only if the resulting block does not include any training points from opposing classes.
- In the generalized case, any number of such disjunctions may be applied, provided the resulting hyperblock preserves class purity.
- This strategy enables the model to consolidate overlapping or adjacent decision regions, reducing clause redundancy while preserving class purity and interpretability.

### Example Benefits of R2B

- In higher dimensions, such as with nine attributes, applying a disjunction to just one attribute enables the combination of two hyperblocks using only ten clauses instead of the original eighteen, improving model compactness.
- In 2D examples, two hyperblocks sharing the same width interval while differing in height intervals can be combined by introducing a disjunction that allows a point to fall within either hyperblock's height interval.

## Combined Approach

These three simplification methods can be applied sequentially or in combination:

1. **R2A**: Remove redundant attributes from individual hyperblocks
2. **R2B**: Remove redundant hyperblocks that don't contribute uniquely
3. **Disjunctive Units**: Merge compatible hyperblocks using disjunctive logic

The combined approach typically results in significantly reduced model complexity while maintaining or improving classification performance and interpretability.
