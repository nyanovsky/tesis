#%% 
import sys 
sys.path.append("..")

import torch
import numpy as np
import pandas as pd

from models import training_utils, exp_utils, base_model
from torch_geometric.nn import  SAGEConv

#%%
version = input("enter dataset version: ")
data_folder = f"/biodata/nyanovsky/datasets/dti/processed/{version}/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

node_df = pd.read_csv(data_folder+"dti_tensor_df.csv",index_col=0)
dataset, node_map = training_utils.load_data(data_folder,load_inverted_map=False,load_test=True)
train_set, val_set, test_set = dataset


gene_features = training_utils.load_feature_dict(data_folder+"prot_features_64.txt", data_folder+"prot_features_ids.txt", 
                                                    node_df, "gene")

#%%
sage_df = pd.read_csv("results/v2/sage_walk_df.csv")
sage_df = sage_df[sage_df["val_auc"]>=0.7].copy()
# %%
new_aucs = []
#%%
from tqdm import tqdm
with tqdm(total=414) as pbar:
    for i in range(414):
        params = sage_df.iloc[i,:-2]
        train_params, model_params, conv_params = params[:7].to_dict(), params[7:-1].to_dict(), {"aggr":params[-1]}

        train_data, val_data = exp_utils.init_features(train_set, val_set, test_set, train_params, gene_features)[:2]

        model = base_model.base_model(SAGEConv, model_params, conv_params, train_set.metadata(), [("gene", "chg", "chem")])

        val_auc = exp_utils.train_model(model, train_params, train_data,val_data)[1]

        new_aucs.append(val_auc)

        pbar.update(1)
# %%
sage_df["val_auc_with_features"] = new_aucs 
sage_df.to_csv("results/v2/sage_df_with_features")
# %%
