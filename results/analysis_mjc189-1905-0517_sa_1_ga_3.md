# TRANSITION FOR RULE SWITCH (_4): HPC

Can we identify remapping/ changes in the dynamics during the rule switch?

## Multidimensional scaling using cosine distance

**Analysis methods**:
* same trajectories (start arm & goal arm)
* using concatenated trials for transformation and separating them afterwards
* filtered data where speed < 5 cm/s and all zero population vectors
* dimensionality reduction: MDS and cosine


![alt text](../plots/mjc189-1905-0517/man_transition_one_plot_MDS_cos_2D.png)

![alt text](../plots/mjc189-1905-0517/man_transition_MDS_cos_3Dspatial.png)

* separation visible

![alt text](../plots/mjc189-1905-0517/man_transition_MDS_cos_3Dspatial_colored_position.png)

* remapping most obvious in goal arm

## Quantification 

**Overal distance**:
* within rule cos-distance vs. accross rules cos-distance
  * calculate pair-wise cos distance between trials of RULE 1 and RULE 2 for each
spatial bin
  * plot MED/MAD of these pair-wise distances for each spatial bin
  * normalization: divide median of across-rule cos distances by within rule cos median
distance for each spatial bin
  * mann-whitney u test, alpha = 0.01
* filtered data where speed < 5 cm/s and all zero population vectors

![alt text](../plots/mjc189-1905-0517/quant_transition_cos.png)

* can we tell something from different results of the normalization by rule 1 and rule b?

histograms of within vs. across rule cos distance

![alt text](../plots/mjc189-1905-0517/quant_transition_cos_histograms.png)

**Separating trials**:
* for each spatial bin and trial after the switch:
    * pair-wise cos differences between trail and each trial before the switch
    --> get array for all comparisons
    * taking the median

![alt text](../plots/mjc189-1905-0517/quant_transition_cos_trials.png)

* continuous remapping for certain spatial positions. To measure the actual effect the
across-trial variability without rule switch needs to be taken into account.

## Identifying cells that contribute to difference

* average firing rate using both rules
* effect size: diff / pooled std
* contribution to cos diff: leave-one-out analysis

![alt text](../plots/mjc189-1905-0517/quant_transition_cos_cells_char.png)

* removing two most influential cells significantly reduces the cosine difference for the rule switch
for the last couple of spatial bins

![alt text](../plots/mjc189-1905-0517/quant_transition_cos_trials_2_removed_cells.png)

* difference is not statistically significant anymore:

![alt text](../plots/mjc189-1905-0517/quant_transition_cos_2_removed_cells.png)



# RULE LIGHT (_2) VS. RULE WEST (_4)

Do we see significant differences in the dynamics of the system for two different rules?

## Multidimensional scaling using cosine distance
**Analysis methods**:
* same trajectories (start arm & goal arm)
* using concatenated trials for transformation and separating them afterwards
* filtered data where speed < 5 cm/s and all zero population vectors
* dimensionality reduction: MDS and cosine

### Rule: light

![alt text](../plots/mjc189-1905-0517/man_compare_one_plot_light_MDS_cos_2D.png)

* variability is quite different for different bins (greater in the center)

### Rule: west

![alt text](../plots/mjc189-1905-0517/man_compare_one_plot_west_MDS_cos_2D.png)

### Light vs. west

![alt text](../plots/mjc189-1905-0517/man_compare_MDS_cos_3D.png)

* apparent difference between both rules

## Quantification

Do we see significant differences in the dynamics of the system for two different rules?

**Overal distance: RULE 1 (_2/_4) vs. RULE 2 (_6)**:

* calculate pair-wise cos distance between trials of RULE 1 and RULE 2 for each
spatial bin
* plot MED/MAD of these pair-wise distances for each spatial bin
* normalization: divide median of across-rule cos distances by within rule cos median
distance for each spatial bin
* significance: across-rules distance vs. union of within-rule distances

![alt text](../plots/mjc189-1905-0517/quant_compare_cos_2_4_6.png)

distribution of within and across rule distance:

![alt text](../plots/mjc189-1905-0517/quant_compare_cos_histograms.png)

**Separating trials: RULE 1 (_2,_4) vs. RULE 2 (_6)**:
* for each spatial bin and trial of RULE 2:
    * pair-wise cos differences between trial and each trial of RULE 1
    --> get array for all comparisons
    * taking the median

![alt text](../plots/mjc189-1905-0517/quant_compare_cos_trials_2_4_6.png)

* relatively constant population states
* between 60 and 110 cm there seem to be clusters of different dynamics
    * can maybe check with parameters of the experiments what the difference might be
    due to

## Identifying cells that contribute to difference: RULE 1 (_2,_4) vs. RULE 2 (_6):

* average firing rate using both rules
* effect size: diff / pooled std
* contribution to cos diff: leave-one-out analysis

![alt text](../plots/mjc189-1905-0517/quant_compare_cos_cells.png)

* removing two most influential cells:

![alt text](../plots/mjc189-1905-0517/quant_compare_cos_trials_2_removed_cells.png)

* difference seem to depend on other cells after the initial transition as well:

![alt text](../plots/mjc189-1905-0517/quant_compare_cos_2_removed_cells.png)

* significance is still given for the last 30 cm

## Comparison for same rule before/after sleep

* remapping also happens without a rule switch --> seems to be happening
in different spatial locations

![alt text](../plots/mjc189-1905-0517/quant_compare_cos_RULE1.png)

![alt text](../plots/mjc189-1905-0517/quant_compare_cos_trials_RULE1.png)

# GLOBAL VS. LOCAL REMAPPING

* using alternative to leaving-one-out analysis
  
* using shuffling:

  * select subset of n cells randomly and calculate cosine difference
  * repeat for 200 times

![alt text](../plots/mjc189-1905-0517/quant_cell_contrib_2_4_6_shuffling.png)

* Synthetic data:

![alt text](../plots/remapping_synthetic_data_set.png)

* using cohens' d:

![alt text](../plots/mjc189-1905-0517/quant_cell_contrib_cohens.png)

# STATE TRANSITION ANALYSIS

## L1 norm between subsequent population vectors

![alt text](../plots/mjc189-1905-0517/quant_state_transition_L1.png)

## Cosine distance between subsequent population vectors

![alt text](../plots/mjc189-1905-0517/quant_state_transition_cos.png)

## Operations 

* using difference vectors between two subsequent population vectors 
* making difference vectors signed binary
* counting -1 (silencing), 0 (unchanged), +1 (activated)

![alt text](../plots/mjc189-1905-0517/quant_state_transition_operations.png)

# Conclusion

* if rather local remapping for rule A vs. rule B --> results highly depend on cells that
we record from

* is re-consolidation rather global? 

* do we have a local --> global remapping?

* FPC data looks very different --> use other distance measures/methods?