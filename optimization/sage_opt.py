#%%
import sys
sys.path.append("..")

import torch
import numpy as np
import pandas as pd

from models import training_utils
from experiments import hp_opt
from torch_geometric.nn import  SAGEConv

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
              "normalize_output":[True,False]}


initial_training_params =  {'weight_decay': 0.001,
 'lr': 0.001,
 'epochs': 400,
 'patience': 10,
 'delta': 0.1,
 'feature_dim': 32,
 'features': "go2vec"}

initial_model_params  = {"pre_process_layers": 1, 
               "post_process_layers":1,
               "layer_connectivity":False,
               "hidden_channels":32,
               "batch_norm": False,
               "dropout":0,
               "macro_aggregation": "mean",
               "L2_norm": True,
               "msg_passing_layers":3,
               "normalize_output":False}

training_params_keys = initial_training_params.keys()

model_params_keys = initial_model_params.keys()

#%%
param_grid["aggr"] = ["sum","mean","max"]

initial_conv_params = {"aggr":"sum"}
conv_params_keys = initial_conv_params.keys()

sage_keys = [training_params_keys, model_params_keys, conv_params_keys]

#%%
### RUN ONLY ONCE ###
sage_walk_df = hp_opt.init_df(SAGEConv, dataset, initial_training_params, initial_model_params, initial_conv_params)
sage_walk_df.to_csv(f"results/{version}/sage_walk_full_df.csv", index=False)

#%%
# %%
# continue walk here each time:
sage_walk_df = pd.read_csv(f"results/{version}/sage_walk_full_df.csv")
sage_walk_df = hp_opt.walk(param_grid, sage_walk_df, 199, 1000, SAGEConv, sage_keys, dataset)
sage_walk_df.to_csv(f"results/{version}/sage_walk_full_df.csv", index=False)
# %%
