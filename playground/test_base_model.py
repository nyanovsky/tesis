# %%
import torch
import torch_geometric
import base_model
from torch_geometric.nn import SAGEConv, GATConv
import base_model
import training_utils, exp_utils
import pandas as pd

#%%
version = input("enter dataset version")
data_folder = f"/biodata/nyanovsky/datasets/dti/processed/{version}/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
node_df = pd.read_csv(data_folder+"dti_tensor_df.csv",index_col=0)
#load data
datasets, node_map = training_utils.load_data(data_folder,load_inverted_map=False,load_test=True)

train_set, val_set, test_set = datasets


# %%
train_params = {'weight_decay': 0.001,
 'lr': 0.001,
 'epochs': 400,
 'patience': 10,
 'delta': 0.1,
 'feature_dim': 16}

gral_params = {"pre_process_layers": 0, 
               "post_process_layers":0,
               "hidden_channels":128,
               "batch_norm": False,
               "layer_connectivity":"cat",
               "dropout":0.5,
               "macro_aggregation": "mean",
               "L2_norm": True,
               "msg_passing_layers":4,
               "normalize_output":True}

conv_params = {"aggr":"sum"}
# %%
model = base_model.base_model(SAGEConv, gral_params, conv_params, datasets[0].metadata(), [("gene", "chg", "chem")])
# %%
protein_features = []
protein_ids = []
with open("/biodata/nyanovsky/datasets/dti/processed/v2/prot_features_64.txt","r") as file:
    for line in file:
        protein_features.append([float(x) for x in line.strip().split()])

with open("/biodata/nyanovsky/datasets/dti/processed/v2/prot_features_ids.txt","r") as file:
    for line in file:
        protein_ids.append("G"+line.strip())

protein_feature_tensor = torch.tensor(protein_features, dtype=torch.float32)

#%%
gene_node_df = node_df[node_df["node_type"]=="gene"]
gene_node_df.set_index("node_id", drop=False,inplace=True)

data_gene_node_ids = gene_node_df["node_id"].values
protein_ids = pd.Series(protein_ids)
ids_in_data = protein_ids.isin(data_gene_node_ids)

protein_feature_tensor = protein_feature_tensor[ids_in_data] #type: ignore
protein_ids = protein_ids[ids_in_data].to_list()

protein_tensor_idxs = gene_node_df.loc[protein_ids, "tensor_index"] #type: ignore
protein_feature_dict = {"gene":[protein_feature_tensor, protein_tensor_idxs]}
# %%
train,val, test = exp_utils.init_features(train_set, val_set, test_set, train_params, protein_feature_dict)
# %%
prueba = exp_utils.run_experiment(model, train, val, test, train_params, node_df)
# %%
