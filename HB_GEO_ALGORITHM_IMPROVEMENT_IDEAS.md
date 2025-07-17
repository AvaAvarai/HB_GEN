# Ideas to Improve the Hyperblock Algorithm (hb-geo)

- **Learn k value per block from training data**  
  Instead of using a fixed k value across all folds, determine the optimal k for each block based on training data.  
  *Rationale:* Misclassification analysis suggests that some errors could be corrected by allowing k to vary (e.g., k ∈ {3, 5}).

- **Optimize silhouette distance metric on training data**  
  Experiment with different distance metrics (L1, L2) for silhouette score calculation and select the best-performing one during training.