#%%
import sys 
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch, torch_geometric
from models import training_utils, base_model, exp_utils, prediction_utils
from torch_geometric.nn import SAGEConv

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score
#%%
def init_train_eval(dataset, params, keys, feature_dict={}):
    train_params, model_params, conv_params = training_utils.separate_params(params, keys)
    train_set, val_set, test_set = dataset
    train_data, val_data, test_data = exp_utils.init_features(train_set, val_set, test_set, train_params, feature_dict)
    
    data = [train_data, val_data, test_data]
    
    model = base_model.base_model(SAGEConv, model_params, conv_params, train_set.metadata(), [("gene", "chg", "chem")])

    model, val_auc, curves = exp_utils.train_model(model, train_params, train_data, val_data)
    
    model = model.to("cpu")
    
    return model, curves, val_auc, data


def get_layers_embeddings(model, data):
    model.eval()
    embs = [data.x_dict]
    for layer_name in model.encoder._modules.keys():
        layer = model.encoder._modules[layer_name]
        if layer_name == "message_passing":
            for conv_layer in layer.layers:
                embs.append(conv_layer(embs[-1], data.edge_index_dict))
        else:
            embs.append(layer(embs[-1]))
    return embs


def layer_tests(model, layer_embeddings, data):
    pos_scores = []
    neg_scores = []
    aucs = []
    
    for emb in layer_embeddings:
        with torch.no_grad():
            preds = model.decoder(emb, data.edge_label_index_dict,[("gene","chg","chem")])[("gene","chg","chem")]
            labels = data.edge_label_dict[("gene","chg","chem")]
            
            first_neg = torch.nonzero(labels==0)[0].item()
            pos_scores.append(preds[:first_neg])
            neg_scores.append(preds[first_neg:])
            
            aucs.append(roc_auc_score(labels, preds))
            
    return aucs, pos_scores, neg_scores


def cos_dist_matrix(embs1, embs2):
    normalized_embs_1 = embs1/embs1.norm(dim=1, keepdim=True)
    normalized_embs_2 = embs2/embs2.norm(dim=1, keepdim=True)
    cos_dist = 1- (normalized_embs_1 @ normalized_embs_2.t())
    return cos_dist


def get_nearest_k(embs1, embs2, k, dist):
    
    if dist=="euc":
        dists = torch.cdist(embs1, embs2)
        dists.fill_diagonal_(float("inf"))
    elif dist=="cos":
        dists = cos_dist_matrix(embs1, embs2)
        dists.fill_diagonal_(float("inf"))
    elif dist=="dec":
        dists = torch.sigmoid(embs1@embs2.T)
        dists.fill_diagonal_(float("-inf"))
        return torch.topk(dists,k)
    dists, idxs = torch.topk(dists, k, largest=False)
    return dists, idxs


def get_neighbors(node_idx, nodetype, edge_index_dict):
    neighbors_dict = {}
    for edgetype, edgetype_edge_index in edge_index_dict.items():
        src, rel, trgt = edgetype
        if nodetype==src:
            neighbors = edgetype_edge_index[1, edgetype_edge_index[0]==node_idx]
            if trgt not in neighbors_dict.keys():
                neighbors_dict[trgt] = neighbors
            else:
                neighbors_dict[trgt] = torch.cat((neighbors_dict[trgt], neighbors)).unique()

    return neighbors_dict 

def get_edgetype_adjdict(edgetype, dataset):
    adjdict = {}
    src, rel, trgt = edgetype
    num_src = dataset[src].num_nodes
    edge_index_dict = dataset.edge_index_dict
    for i in range(num_src):
        adjdict[i] = get_neighbors(i, src, edge_index_dict)[trgt]
    return adjdict   