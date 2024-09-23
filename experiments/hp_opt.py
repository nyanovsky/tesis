#%%
from os import sep
import sys 
sys.path.append("..")
#%%
import torch
import numpy as np
import pandas as pd

from models import training_utils, exp_utils, base_model
from param_walker import ParameterWalker, ConvergenceTest
from torch_geometric.nn import SAGEConv, GATConv
# %%
data_folder = "/biodata/nyanovsky/datasets/dti/processed/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
node_df = pd.read_csv(data_folder+"dti_tensor_df.csv",index_col=0)
#load data
datasets, node_map = training_utils.load_data(data_folder,load_inverted_map=False,load_test=True)

train_set, val_set, test_set = datasets


### ---------------- SAGE ---------------###
#%%
initial_training_params =  {'weight_decay': 0.001,
 'lr': 0.001,
 'epochs': 400,
 'patience': 10,
 'delta': 0.1,
 'feature_dim': 32}
training_params_keys = initial_training_params.keys()

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
model_params_keys = initial_model_params.keys()


param_grid = {'weight_decay': [0.001],
              'lr': [0.01, 0.001, 0.0001],
              'epochs': [300,400, 500],
              'patience': [10],
              'delta': [0.1],
              'feature_dim': [16,32, 64, 128],
              "pre_process_layers": [0,1,2] ,
              "post_process_layers":[0,1,2],
              "layer_connectivity":[False,"sum","cat"],
              "hidden_channels":[16,32,64,128],
              "batch_norm": [False, True],
              "dropout":[0, 0.3,0.5],
              "macro_aggregation": ["sum", "mean"],
              "L2_norm": [True, False],
              "msg_passing_layers":[1,2,3,4],
              "normalize_output":[True,False]}

def separate_params(param_dict):
    train_params = {key:val for key,val in param_dict.items() if key in training_params_keys}
    model_params = {key:val for key,val in param_dict.items() if key in model_params_keys}
    return train_params, model_params

#%%
import subprocess

def get_gpu_temp():
    result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                            stdout=subprocess.PIPE, encoding='utf-8')
    
    temp = int(result.stdout.strip())
    return temp
# %%
### RUN THIS ONLY ONCE ###
train_data, val_data = exp_utils.init_features(train_set, val_set, test_set, initial_training_params)[:2]
model = base_model.base_model(SAGEConv, initial_model_params, {"aggr":"sum"}, datasets[0].metadata(), [("gene", "chg", "chem")])
val_auc = exp_utils.train_model(model, initial_training_params, train_data,val_data)[1]

initial_params = initial_training_params|initial_model_params
parameter_walk_df = pd.DataFrame([initial_params|{"val_auc":val_auc}])


#%%
from tqdm import tqdm 
# Intented for repeated using and picking up at the last step of the random walk
def walk(walk_df, max_iters, k_max):

    walker = ParameterWalker(walk_df.iloc[-1,:-1].to_dict(), param_grid)
    test = ConvergenceTest(0.005, 5)
    
    k0 = walk_df.shape[0]
    k= k0
    T = 1-k/k_max
    curr_auc = walk_df.iloc[-1,-1]

    with tqdm(total=max_iters) as pbar:
        while k<k0+max_iters:
            if k%5==0 and get_gpu_temp() > 73:
                print("breaking, too high GPU temp")
                break 

            params = walker.random_step()
            train_params, model_params = separate_params(params)

            train_data, val_data = exp_utils.init_features(train_set, val_set, test_set, train_params)[:2]

            model = base_model.base_model(SAGEConv, model_params, {"aggr":"sum"}, datasets[0].metadata(), [("gene", "chg", "chem")])
            val_auc = exp_utils.train_model(model, train_params, train_data,val_data)[1]

            delta = curr_auc-val_auc 

            accept_step = min(1,np.exp(-delta/T))

            if accept_step > np.random.random():
                walker.accept_step(params)
                curr_auc = val_auc

            walk_df = pd.concat([walk_df, pd.DataFrame([params|{"val_auc":val_auc}])])

            pbar.update(1)
            k+=1
    return walk_df
# %%
walk_df = walk(parameter_walk_df, 100, 1000)
walk_df.to_csv("results/walk_df.csv")
# %%