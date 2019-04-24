# COMPARING MANIFOLDS FOR DIFFERENT RULES: HPC

Do we see significant differences in the dynamics of the system for two different rules?

**Analysis methods**:
* same trajectories (start arm & goal arm)
* using concatenated data for transformation and separating them afterwards
* filtered data where speed < 5 cm/s and all zero population vectors
* dimensionality reduction: MDS, PCA etc.
* "confidence intervals" for each bin for each rule to find sections where manifolds differ significantly (comparing population states)
* characterizing/comparing dynamics: 
    * step length
    * step direction (angle between subsequent steps)
    * etc.

## Results using spatial bins (10 cm)

* cannot use jaccard as a difference measure because it doesnt work on continuous data

### 1. Multidimensional scaling

#### Difference measure: cosine

##### Rule: light

![alt text](../plots/mjc189-1905-0517/man_compare_one_plot_light_MDS_cos_2D.png)

* variability is quite different for different bins (greater in the center)

##### Rule: west

![alt text](../plots/mjc189-1905-0517/man_compare_one_plot_west_MDS_cos_2D.png)

##### Light vs. west

![alt text](../plots/mjc189-1905-0517/man_compare_MDS_cos_3D.png)

* apparent difference between both rules

#### Difference measure: euclidean

##### Light vs. west
![alt text](../plots/mjc189-1905-0517/man_compare_MDS_euclidean_3Dspatial.png)

## Results using time bins

### 1. Multidimensional scaling

#### Difference measure: cosine

##### Comparison using one trial for each rule

* time bin: 0.1 seconds

![alt text](../plots/mjc189-1905-0517/man_compare_MDS_cos_one_trial_3D.png)

##### Comparison using all trials for each rule

* time bin: 0.1 seconds

![alt text](../plots/mjc189-1905-0517/man_compare_one_plot_MDS_cos_3Dtemporal.png)

* time bin: 0.5 seconds

![alt text](../plots/mjc189-1905-0517/man_compare_one_plot_MDS_cos_3Dtemporal0_5_time_bin.png)

# MANIFOLD TRANSITION FOR RULE SWITCH: HPC

Can we characterize the transition of the manifold during the rule switch?

**Analysis methods**:
* same trajectories (start arm & goal arm)
* using concatenated trials for transformation and separating them afterwards
* filtered data where speed < 5 cm/s and all zero population vectors
* dimensionality reduction: MDS, PCA etc.
* analysing the transition:
    * rigid rotation?
    * translation?
    * stretching?

## Results using time bins (0.1s)

### 1. Multidimensional scaling

#### Difference measure: cosine

* rule switch after trial 7

![alt text](../plots/mjc189-1905-0517/man_transition_mds_cos_2D.png)


* rule switch after trial 7

![alt text](../plots/mjc189-1905-0517/man_transition_mds_cos_3D.png)


#### Difference measure: jaccard

![alt text](../plots/mjc189-1905-0517/man_transition_MDS_jaccard_2D.png)

### 2. PCA

* title: contribution to variance of first and second principal component

![alt text](../plots/mjc189-1905-0517/man_transition_PCA__2D.png)


### 3. TSNE

![alt text](../plots/mjc189-1905-0517/man_transition_TSNE__2D.png)

## Results using spatial bins (10cm)
* discarding first/last 20 cm

### 1. Multidimensional scaling

#### Difference measure: cosine
* all trials for rule switch in one plot 

![alt text](../plots/mjc189-1905-0517/man_transition_one_plot_MDS_cos_2D.png)

![alt text](../plots/mjc189-1905-0517/man_transition_one_plot_MDS_cos_3D.png)

![alt text](../plots/mjc189-1905-0517/man_transition_MDS_cos_3Dspatial.png)

![alt text](../plots/mjc189-1905-0517/man_transition_MDS_cos_3Dspatial_colored_position.png)

* visible separation for both rules (light --> go west)
* maybe use trials in between (not successful ones) to see how the remapping actually happens

# STATE TRANSITION ANALYSIS: HPC

"Operations" that can change the state of the system. Do we see differences for different rules/spatial positions/rule switching? 

**Analysis methods:**
* filtered data where speed < 5 cm/s and all zero population vectors
* calculate "difference vectors" between two subsequent population vectors
* dimensionality reduction: MDS, PCA etc.


## 1. Multidimensional scaling
### Difference measure: jaccard
* difference vectors are modified (positive and negative integer values)
* using temporal bins
#### rule light

![alt text](../plots/mjc189-1905-0517/trans_analysis_light_MDS_jaccard_2Dtemporal.png)

* the transitions seem to carry information about the location

#### rule west

![alt text](../plots/mjc189-1905-0517/trans_analysis_west_MDS_jaccard_2Dtemporal.png)

#### rule light vs. west

![alt text](../plots/mjc189-1905-0517/trans_analysis_MDS_jaccard_3Dtemporal_no_position_coloring.png)

![alt text](../plots/mjc189-1905-0517/trans_analysis_MDS_jaccard_2Dtemporal.png)

![alt text](../plots/mjc189-1905-0517/trans_analysis_MDS_jaccard_3Dtemporal.png)

### Difference measure: cos

#### rule light

![alt text](../plots/mjc189-1905-0517/trans_analysis_light_MDS_cos_2Dtemporal.png)

* there is no nice separation

### Difference measure: euclidean

#### rule light

![alt text](../plots/mjc189-1905-0517/trans_analysis_light_MDS_euclidean_2Dtemporal.png)

* there is no nice separation