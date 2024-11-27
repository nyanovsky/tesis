## Graph Neural Networks for Drug-Target Interaction Prediction
Implementation of Graph Neural Networks in an heterogeneous two layer network consisting Drug and Target (gene products) nodes.

Part of this codebase is due to Ingrid Heuer's work on GNNs for Gene-Disease association predictions. You can check out her work [here](https://github.com/ingridheuer/gcnn_gdas).


## Project Structure

```plaintext
├── data                              <- Data processing (network building and PyG data split), and network analysis
│   ├── ChG_db_merging.ipynb          <- Integration of external DTI data into single bipartite network
│   ├── rdkit_sim.ipynb               <- Drug-Drug network layer building based on rdkit fingerprint similarity
│   ├── pfam_weighted_proy.R          <- Gene-Gene network layer building based on weighted projection from gene-PFAM bipartite graph
│   ├── network_merging.ipynb         <- Merging of three networks into single final network
│   ├── network_analysis.ipynb        <- Full network analysis
│   └── split_graph.py                <- PyG HeteroData generation and splitting
|
|
├── models                            <- Model architecture, training utilities
│   ├── base_model.py                 <- General and flexible GNN architecture similar to described in the Design space for GNNs paper (https://arxiv.org/abs/2011.08843)
│   ├── training_utils.py             <- General utilities for implementing a training pipeline; data loading, negative sampling, train/eval/test
│   └── exp_utils.py                  <- Full training pipeline for quick model implementation
|
|
├── features                          <- Initial features for nodes in graph
│   └── gene                          <- Gene features from Gene Ontology Molecular Function DAG embeddings, similar to described in Go2Vec (https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6272-2)
|
|
├── optimization                      <- Hyperparameter optimization based on random walk with simulated annealing
│   └── results                       <- Walk results
│       ├── v1                        <- Old dataset results. TODO: delete
│       ├── v2                        <- Official dataset walk results
│       └── hp_opt.py                 <- Script for performing random walk with simulated annealing
|
|
├── exploration                       <- Analysis of results; hyperparameter search, initial features, connectivity biases
│   ├── explor_utils.py               <- Utilities for in-depth model exploration and analysis
│   ├── hp_analysis.ipynb             <- Hyperparameter analysis from optimization results
│   ├── feature_analysis.ipynb        <- Analysis of gene initial features' impact on model performance
│   └── connectivity_analysis.ipynb   <- Analysis of predictions based on network connectivity
|
|
└── playground                        <- Toy scripts for testing purposes
|
