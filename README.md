# pgm-polarity
Investigating the relative contributions of topologies in heterogeneous social network interactions to the inference of political and ideological sentiment

```
pgm-polarity
|
|--data
|   |--data/icwsm_polarization
|        |--all.edgelist(edge definition)
|        |--all.nodes(nodes id and labels)
|        |--all_igraph.pickle(data in graph format)
|        |--data_formation_specification.docx(detailed documents)
|
|--feature
|   |--objects
|        |--pca200_df.pickle(pca embedding)
|        |--adj_matrix.npz(adjacent matrix for label propagation)
|
|--model
|   |--Baselines
|       |--viz-and-baseline.ipynb
|   |--CRF
|       |--crf.py
|   |--Label propagation
|       |--label-propagation.ipynb
|   |--Combined training
|       |--combine.py
|       |--combine_utils.py
```
## Dependency
numpy

scikit-learn

PyTorch

PyStruct

python-igraph

## Testing Models
The environment is run under Python3

Baselines(KNN and Logistic Regression): please follow the viz-and-baseline.ipynb

CRF: ```python crf.py```

Label Propagtion: please follow the label-propagation.ipynb

Combined Training: ```python combine.py```. The default sampling method is centrality sampling considering influential nodes. Change 'centrality' to 'random' in line 197, if you want to run random sampling.
