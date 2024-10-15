#%% 
import sys 
sys.path.append("..")

import torch
import numpy as np
import pandas as pd

from models import training_utils, exp_utils, base_model
from torch_geometric.nn import  GATConv
# %%
version = input("enter dataset version: ")
data_folder = f"/biodata/nyanovsky/datasets/dti/processed/{version}/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

node_df = pd.read_csv(data_folder+"dti_tensor_df.csv",index_col=0)
dataset, node_map = training_utils.load_data(data_folder,load_inverted_map=False,load_test=True)
train_set, val_set, test_set = dataset


gene_features = training_utils.load_feature_dict(data_folder+"prot_features_64.txt", data_folder+"prot_features_ids.txt", 
                                                    node_df, "gene")
# %%
gat_df = pd.read_csv("results/v2/gat_walk_df.csv")
# %%
gat_top_400_with_features = gat_df.sort_values(by="val_auc", ascending=False).iloc[:400]
gat_top_400_with_features = gat_top_400_with_features[gat_top_400_with_features["features"]=="go2vec"]
# %%
new_aucs = []
#%%
from tqdm import tqdm
with tqdm(total=97) as pbar:
    for i in range(300,397):
        params = gat_top_400_with_features.iloc[i,:-2]
        train_params, model_params, conv_params = params[:7].to_dict(), params[7:-3].to_dict(), params[-3:].to_dict()

        train_data, val_data = exp_utils.init_features(train_set, val_set, test_set, train_params)[:2]

        model = base_model.base_model(GATConv, model_params, conv_params, train_set.metadata(), [("gene", "chg", "chem")])

        val_auc = exp_utils.train_model(model, train_params, train_data,val_data)[1]

        new_aucs.append(val_auc)

        pbar.update(1)
# %%
gat_top_400_with_features["val_auc_random_features"] = new_aucs
gat_top_400_with_features.to_csv("results/v2/gat_features_comp.csv")
# %%
