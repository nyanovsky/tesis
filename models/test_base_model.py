# %%
import torch
import torch_geometric
import base_model
from torch_geometric.nn import SAGEConv, GATConv
import base_model
import training_utils, exp_utils
import pandas as pd

#%%
data_folder = "/biodata/nyanovsky/datasets/dti/processed/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
node_df = pd.read_csv(data_folder+"dti_tensor_df.csv",index_col=0)
#load data
datasets, node_map = training_utils.load_data(data_folder,load_inverted_map=False,load_test=True)

train_set, val_set, test_set = datasets

train_params = {'weight_decay': 0.001,
 'lr': 0.001,
 'epochs': 400,
 'patience': 10,
 'delta': 0.1,
 'feature_dim': 32}

# %%
gral_params = {"pre_process_layers": 0, 
               "post_process_layers":0,
               "hidden_channels":32,
               "batch_norm": False,
               "dropout":0,
               "macro_aggregation": "mean",
               "L2_norm": True,
               "msg_passing_layers":3,
               "normalize_output":False}

conv_params = {"aggr":"sum"}
# %%
model = base_model.base_model(SAGEConv, gral_params, conv_params, datasets[0].metadata(), [("gene", "chg", "chem")])
# %%
train,val, test = exp_utils.init_features(train_set, val_set, test_set, train_params)
# %%
