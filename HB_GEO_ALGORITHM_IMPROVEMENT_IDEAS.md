# Ideas to Improve the Hyperblock Algorithm (hb-geo)

- **Learn k value per block from training data**  
  Instead of using a fixed k value across all folds, determine the optimal k for each block based on training data.  
  *Rationale:* Misclassification analysis suggests that some errors could be corrected by allowing k to vary (e.g., k ∈ {3, 5}).

- **Optimize silhouette distance metric on training data**  
  Experiment with different distance metrics (L1, L2) for silhouette score calculation and select the best-performing one during training.

- **Shrink or merge hyperblocks to improve class separation**
  - Iteratively shrink hyperblock edges to exclude training cases from opposing classes.
  - Optionally, merge smaller hyperblocks into disjunctive units (e.g., multi-intervals for a single attribute).
  - Initial experiments are inconclusive and have not shown clear benefits yet.
