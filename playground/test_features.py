#%%
import sys
sys.path.append("..")

import torch, pandas as pd, numpy as np
from models import training_utils, exp_utils, base_model
from torch_geometric.nn import SAGEConv

#%%
version = input("enter dataset version: ")
data_folder = f"/biodata/nyanovsky/datasets/dti/processed/{version}/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
node_df = pd.read_csv(data_folder+"dti_tensor_df.csv",index_col=0)
dataset, node_map = training_utils.load_data(data_folder,load_inverted_map=False,load_test=True)
train_set, val_set, test_set = dataset


gene_features = training_utils.load_feature_dict(data_folder+"prot_features_64.txt", data_folder+"prot_features_ids.txt", 
                                                    node_df, "gene")

#%%
sage_walk_df = pd.read_csv(f"../experiments/results/{version}/sage_walk_df.csv")
# %%
# probamos agregarle features al top 5
top5_params = sage_walk_df.sort_values(by="val_auc", ascending=False).iloc[:5, :-2]

new_val_aucs = []
for i in range(5):
    params = top5_params.iloc[i]
    train_params, model_params = params[:7].to_dict(), params[7:-1].to_dict()
    conv_params = {"aggr": params["aggr"]}
    train_data, val_data = exp_utils.init_features(train_set, val_set, test_set, train_params, 
                                                   gene_features)[:2]
    model = base_model.base_model(SAGEConv, model_params, conv_params, train_set.metadata(), [("gene", "chg", "chem")])
    val_auc = exp_utils.train_model(model, train_params, train_data,val_data)[1]
    new_val_aucs.append(val_auc)

# %%
# lo mismo para el top 5 sin pre-process. 
top5_params = sage_walk_df[sage_walk_df["pre_process_layers"]==0].sort_values(by="val_auc", ascending=False).iloc[:5, :-2]

new_val_aucs = []
for i in range(5):
    params = top5_params.iloc[i]
    train_params, model_params = params[:7].to_dict(), params[7:-1].to_dict()
    conv_params = {"aggr": params["aggr"]}
    train_data, val_data = exp_utils.init_features(train_set, val_set, test_set, train_params, 
                                                   gene_features)[:2]
    model = base_model.base_model(SAGEConv, model_params, conv_params, train_set.metadata(), [("gene", "chg", "chem")])
    val_auc = exp_utils.train_model(model, train_params, train_data,val_data)[1]
    new_val_aucs.append(val_auc)
    print(f'{i} done')
# %%
# lo mismo para 5 al azar con auc entre 0.8 y 0.85
sub_df = sage_walk_df[(sage_walk_df["val_auc"]>=0.80) & (sage_walk_df["val_auc"]<=0.85)]
idxs = sub_df.index.values 
five_random = np.random.choice(idxs, 5, replace=False)
sub_df = sub_df.loc[five_random]
new_val_aucs = []
for idx in five_random:
    params = sub_df.loc[idx][:-2]
    train_params, model_params = params[:7].to_dict(), params[7:-1].to_dict()
    conv_params = {"aggr": params["aggr"]}
    train_data, val_data = exp_utils.init_features(train_set, val_set, test_set, train_params, 
                                                   gene_features)[:2]
    model = base_model.base_model(SAGEConv, model_params, conv_params, train_set.metadata(), [("gene", "chg", "chem")])
    val_auc = exp_utils.train_model(model, train_params, train_data,val_data)[1]
    new_val_aucs.append(val_auc)
    print(f'{i} done')

# para el idx 753 pasamos de 0.83 a 0.948
# %%
