# MANIFOLD TRANSITION FOR RULE SWITCH: HPC

* rule switch after trial 7
* using concatenated trials for transformation and separating them afterwards
* filtered data where speed < 5 cm/s and all zero population vectors

## Results using time bins (0.1s)

### 1. Multidimensional scaling

#### Difference measure: cosine

* number of components: 2

![alt text](plots/man_transition_mds_cos_2D.png)


* number of components: 3

![alt text](plots/man_transition_mds_cos_3D.png)


#### Difference measure: jaccard

![alt text](plots/man_transition_MDS_jaccard_2D.png)

### 2. PCA

* title: contribution to variance of first and second principal component

![alt text](plots/man_transition_PCA__2D.png)


### 3. TSNE

![alt text](plots/man_transition_TSNE__2D.png)

## Results using spatial bins (10cm)
* discarding first/last 20 cm

### 1. Multidimensional scaling

#### Difference measure: cosine
* all trials for rule switch in one plot 

![alt text](plots/man_transition_one_plot_MDS_cos_2D.png)

# COMPARING MANIFOLDS FOR DIFFERENT RULES

### 1. Multidimensional scaling

#### Difference measure: cosine

##### Rule: light

![alt text](plots/man_compare_one_plot_light_MDS_cos_2D.png)


# STATE TRANSITION ANALYSIS

* using "difference vectors" between population states
* filtered data where speed < 5 cm/s and all zero population vectors

## 1. Multidimensional scaling
### Difference measure: jaccard

![alt text](plots/trans_analysis_MDS_jaccard_2D.png)

![alt text](plots/trans_analysis_MDS_jaccard_3D.png)