# Hyperblock Algorithms Rewrite

Algorithms used to independently create HBs

## Interval Hyper (IHyper)

The idea of IHyper is that a hyperblock can be created by repeatedly finding the largest interval of values for some attribute x_i with a purity above a given threshold.

1) For each attribute x_i in a dataset, create an ascending sorted array containing all values for the attribute.
2) For each value a_i in the sorted array of x_i (not just the first), treat it as a possible seed. For each seed:
   a. Initialize b_i = a_i = d_i for a_i.
   b. Create a candidate interval [b_i, d_i] for attribute x_i.
3) For the current interval, attempt to expand it by including the next value e_i in the sorted array:
   a. Only expand if the n-D point e corresponding to e_i is of the same class as a, or if adding e_i keeps the interval's purity above threshold T.
   b. If adding e_i would drop purity below T, do not expand further.
4) If the interval cannot be expanded further:
   a. Save the current interval if it contains at least one n-D point (avoid empty intervals).
   b. If there are more values in the sorted array, repeat step 2 with the next unused value as a new seed (ensure all possible seeds are considered, not just those greater than e_i).
5) For each attribute x_i, among all saved intervals, keep the interval with the largest number of values (if there are ties, clarify how to break them, e.g., by interval width or randomly).
6) Repeat steps 2-5 for all attributes.
7) Among all saved intervals for all attributes, select the interval with the largest number of values (again, clarify tie-breaking if needed).
8) Using the selected interval, create a hyperblock.
9) Remove all n-D points covered by the new hyperblock from the dataset.
10) Repeat steps 1-9 with the remaining n-D points until all points are within a hyperblock or no more intervals above the purity threshold can be found.
11) Note: Ensure that each n-D point is assigned to at most one hyperblock.

## Merger Hyper (MHyper)

The idea for MHyper is that a hyperblock can be created by merging two overlapping hyperblocks.

1) Seed an initial set of pure HBs, each containing a single n-D point (i.e., HBs with length zero).
2) While there are HBs that can be merged:
   a. Select a HB x from the set of all HBs not yet merged.
   b. For each other HB_i:
      i. If HB_i has the same class as x, attempt to combine HB_i with x to get a pure HB:
         - Create a joint HB as the envelope (min/max for each attribute) of HB_i and x.
         - Check if any other n-D point y (not already assigned to another HB) belongs to this envelope; if so, add y to the joint HB.
         - If all points in the joint HB are of the same class, merge HB_i and x into a new HB and remove the originals from the set.
   c. Repeat until no more pure merges are possible.
3) After all pure merges, for any remaining unassigned n-D points, define an impurity threshold that limits the percentage of n-D points from opposite classes allowed in a dominant HB.
4) For each HB x, attempt to merge with other HBs (regardless of class):
   a. Create a joint HB as the envelope of HB_i and x.
   b. Add any n-D point y (not already assigned) that falls within the envelope.
   c. Compute impurity of the joint HB (percentage of n-D points from opposite classes).
   d. Among all possible merges, select the one with the lowest impurity. If this impurity is below the predefined threshold, perform the merge.
5) Repeat step 4 until no more merges below the impurity threshold are possible.
6) Ensure that each n-D point is assigned to at most one HB at any time.

## Interval Merger Hyper (IMHyper)

The idea for IMHyper is to combine the IHyper and MHyper algorithms.

1) Run the IHyper algorithm to create initial hyperblocks.
2) Identify any n-D points not included in the HBs created in step 1.
3) Run the MHyper algorithm on the set of remaining n-D points, but include the HBs from step 1 as the initial set of pure HBs in the MHyper process.
4) Ensure that after both steps, all n-D points are assigned to at most one HB, and that the impurity threshold is respected during the MHyper phase.
