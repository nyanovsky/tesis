#%%
import sys 
sys.path.append("..")
#%%
import torch
import numpy as np
import pandas as pd

from models import exp_utils, training_utils, base_model
from param_walker import ParameterWalker, ConvergenceTest
# %%
version = input("enter dataset version: ")
data_folder = f"/biodata/nyanovsky/datasets/dti/processed/{version}/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

full_set = torch.load(data_folder+"dti_full_dataset.pt")
negative_sampler = training_utils.NegativeSampler(full_set,("gene","chg","chem"),full_set["gene"]["degree_chg"],full_set["chem"]["degree_chg"])

#%%
node_df = pd.read_csv(data_folder+"dti_tensor_df.csv",index_col=0)
#%%
gene_feature_dict = training_utils.load_feature_dict(data_folder+"prot_features_64.txt", data_folder+"prot_features_ids.txt", 
                                                    node_df, "gene")

#%%
import subprocess

def get_gpu_temp():
    result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                            stdout=subprocess.PIPE, encoding='utf-8')
    
    temp = int(result.stdout.strip())
    return temp

def separate_params(param_dict, keys):
    training_keys, model_keys, conv_keys = keys
    train_params = {key:val for key,val in param_dict.items() if key in training_keys}
    model_params = {key:val for key,val in param_dict.items() if key in model_keys}
    conv_params = {key:val for key,val in param_dict.items() if key in conv_keys}
    return train_params, model_params, conv_params


#%%
from tqdm import tqdm 
# Intented for repeated using and picking up at the last step of the random walk
def walk(param_grid, walk_df, max_iters, k_max, conv, initial_keys, dataset):
    train_set, val_set, test_set = dataset

    k0 = walk_df.shape[0]
    k= k0
    last_accepted = walk_df[walk_df["accepted"]==True].iloc[-1]
    curr_auc, curr_params = last_accepted[-2], last_accepted[:-2]

    if k0==1:
        historial = [walk_df.iloc[0,:-2].values.tolist()]
    else:
        historial = walk_df.iloc[:,:-2].values.tolist()

    walker = ParameterWalker(curr_params.to_dict(), historial, param_grid)
    test = ConvergenceTest(0.005, 10, curr_auc)

    with tqdm(total=max_iters) as pbar:
        while k<k0+max_iters:
            if k%5==0 and get_gpu_temp() > 73:
                print("breaking, too high GPU temp")
                return walk_df 

            T = 1-k/k_max

            params = walker.next()
            train_params, model_params, conv_params = separate_params(params, initial_keys)

            if train_params["features"] != "random":
                train_data, val_data = exp_utils.init_features(train_set, val_set, test_set, train_params, gene_feature_dict)[:2]
            else:
                train_data, val_data = exp_utils.init_features(train_set, val_set, test_set, train_params)[:2]

            model = base_model.base_model(conv, model_params, conv_params, train_set.metadata(), [("gene", "chg", "chem")])
            try:
                val_auc = exp_utils.train_model(model, train_params, train_data,val_data, negative_sampler)[1]
            except:
                print(params)
                k += 1
                continue
            delta = curr_auc-val_auc 

            accept_step = min(1,np.exp(-delta/T))

            accepted = False
            if accept_step > np.random.random():

                walker.accept_step(params)
                curr_auc = val_auc
                accepted = True

                if test.check_convergence(curr_auc):
                    print("walk converged")
                    return walk_df

            walk_df = pd.concat([walk_df, pd.DataFrame([params|{"val_auc":val_auc, "accepted":accepted}])])

            pbar.update(1)
            k+=1
    return walk_df

def init_df(conv, dataset, initial_training_params, initial_model_params, initial_conv_params):
    train_set, val_set, test_set = dataset
    if initial_training_params["features"] == "go2vec":
        train_data, val_data = exp_utils.init_features(train_set, val_set, test_set, initial_training_params, gene_feature_dict)[:2]
    else:
        train_data, val_data = exp_utils.init_features(train_set, val_set, test_set, initial_training_params)[:2]
    model = base_model.base_model(conv, initial_model_params, initial_conv_params, train_set.metadata(), [("gene", "chg", "chem")])
    val_auc = exp_utils.train_model(model, initial_training_params, train_data,val_data, negative_sampler)[1]

    initial_params = initial_training_params|initial_model_params|initial_conv_params
    parameter_walk_df = pd.DataFrame([initial_params|{"val_auc":val_auc, "accepted":True}])

    return parameter_walk_df