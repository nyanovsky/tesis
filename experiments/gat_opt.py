#%%
import sys
sys.path.append("..")

import torch
import numpy as np
import pandas as pd

from models import training_utils
from experiments import hp_opt
from torch_geometric.nn import  GATConv

#%%
version = input("enter dataset version: ")
data_folder = f"/biodata/nyanovsky/datasets/dti/processed/{version}/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
dataset, node_map = training_utils.load_data(data_folder,load_inverted_map=False,load_test=True)

train_set, val_set, test_set = dataset
#%%
param_grid = {'weight_decay': [0.001],
              'lr': [0.01, 0.001, 0.0001],
              'epochs': [300,400, 500],
              'patience': [10],
              'delta': [0.1],
              'feature_dim': [16,32, 64, 128],
              'features': ["random", "go2vec"],
              "pre_process_layers": [0,1,2],
              "post_process_layers":[0,1,2],
              "layer_connectivity":[False,"sum","cat"],
              "hidden_channels":[16,32,64,128],
              "batch_norm": [False, True],
              "dropout":[0, 0.3,0.5],
              "macro_aggregation": ["sum", "mean"],
              "L2_norm": [True, False],
              "msg_passing_layers":[1,2,3,4],
              "normalize_output":[True,False],
              "heads":[1,2,3],
              "add_self_loops":[False],
              "concat": [True, False]}


initial_training_params =  {'weight_decay': 0.001,
 'lr': 0.01,
 'epochs': 500,
 'patience': 10,
 'delta': 0.1,
 'feature_dim': 32,
 'features': "random"}

initial_model_params  = {"pre_process_layers": 1, 
               "post_process_layers":1,
               "layer_connectivity":False,
               "hidden_channels":32,
               "batch_norm": False,
               "dropout":0,
               "macro_aggregation": "sum",
               "L2_norm": True,
               "msg_passing_layers":3,
               "normalize_output":False}

initial_gat_params = {"heads":3, "add_self_loops":False, "concat":True}

training_params_keys = initial_training_params.keys()

model_params_keys = initial_model_params.keys()

initial_keys = [training_params_keys, model_params_keys, initial_gat_params.keys()]

# %%
### RUN ONLY ONCE ###
gat_walk_df = hp_opt.init_df(GATConv, dataset, initial_training_params, initial_model_params, initial_gat_params)
gat_walk_df.to_csv(f"results/{version}/gat_walk_df.csv", index=False)
# %%
# %%
# continue walk here each time:
gat_walk_df = pd.read_csv(f"results/{version}/gat_walk_df.csv") 
gat_walk_df = hp_opt.walk(param_grid, gat_walk_df, 100, 1000, GATConv, initial_keys, dataset)
gat_walk_df.to_csv(f"results/{version}/gat_walk_df.csv", index=False)
# %%
