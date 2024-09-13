#%%
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import copy
import datetime
import pandas as pd
import pickle 
import sys
import tqdm

#%%
def load_data(folder_path,load_inverted_map=True,load_test = False):
    """If load_inverted_map=True, loads map from tensor_index to node_index. 
    if False, loads map from node_index to tensor_index """
    if load_test:
        names = ["train","validation","test"]
    else:
        names = ["train","validation"]
    datasets = []
    for name in names:
        path = folder_path+"dti_"+name+".pt"
        datasets.append(torch.load(path))
    
    with open(folder_path+"dti_"+"node_map.pickle", 'rb') as handle:
        node_map = pickle.load(handle)
    
    if load_inverted_map:
        rev_map = {}
        for node_type, map in node_map.items():
            rev_map[node_type] = {v:k for k,v in map.items()}
        node_map = rev_map
    
    return datasets, node_map


def get_tensor_index_df(node_data, node_map):
    sub_dfs = []
    for node_type in node_map.keys():
        sub_df = node_data[node_data.node_type == node_type]
        node_map_series = pd.Series(node_map[node_type], name="tensor_index") 
        # series with idxs in sub_df idxs and values in their corresponding tensor idx 
        sub_df = sub_df.join(node_map_series) # join combines by idx
        sub_dfs.append(sub_df)
    tensor_df = pd.concat(sub_dfs)
    return tensor_df

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    predictions = model(data.x_dict,data.adj_t_dict,data.edge_label_index_dict)
    edge_label = data.edge_label_dict
    loss = model.loss(predictions, edge_label)
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def get_val_loss(model,val_data):
    model.eval()
    predictions = model(val_data.x_dict,val_data.adj_t_dict,val_data.edge_label_index_dict)
    edge_label = val_data.edge_label_dict
    loss = model.loss(predictions, edge_label)

    return loss.item()

@torch.no_grad()
def test(model,data):
  model.eval()
  predictions = model(data.x_dict,data.adj_t_dict,data.edge_label_index_dict)
  edge_label = data.edge_label_dict
  all_scores = []
  all_true = []
  # por si el modelo predice más de un tipo de enlace, concatenamos todas las labels y preds
  # en un solo tensor y calculamos una métrica global
  for edge_type,y_score in predictions.items():
      y_true = edge_label[edge_type]
      all_scores.append(y_score)
      all_true.append(y_true)
  total_scores = torch.cat(all_scores, dim=0).cpu().numpy()
  total_true = torch.cat(all_true, dim=0).cpu().numpy()
  roc_auc = roc_auc_score(total_true,total_scores)
  return round(roc_auc,3)



def initialize_features(data, dim, feature_dict={}, inplace=False):
    if inplace:
        data_object = data
    else:
        data_object = copy.copy(data)

    for nodetype, store in data_object.node_items():
        if nodetype in feature_dict.keys():
            nodetype_embs, tensor_idxs = feature_dict[nodetype]

            random_init = torch.nn.Parameter(torch.Tensor(store["num_nodes"], nodetype_embs.shape[1]), requires_grad = False)
            torch.nn.init.xavier_uniform_(random_init)
            data_object[nodetype].x = random_init

            data_object[nodetype].x[tensor_idxs] = nodetype_embs 
            # random init and then fill remaining (possibly all) with features
        else:
            random_init = torch.nn.Parameter(torch.Tensor(store["num_nodes"], dim), requires_grad = False)
            torch.nn.init.xavier_uniform_(random_init)
            data_object[nodetype].x = random_init
        
    return data_object

@torch.no_grad()
def get_encodings(model,data):
    model.eval()
    x_dict = data.x_dict
    edge_index = data.edge_index_dict
    encodings = model.encoder(x_dict,edge_index)
    return encodings


# %%
class NegativeSampler:
    def __init__(self,full_dataset,edge_type,src_degrees,dst_degrees) -> None:
        src_type, _ , dst_type = edge_type
        self.num_nodes = (full_dataset[src_type]["num_nodes"],full_dataset[dst_type]["num_nodes"])

        full_positive_index = full_dataset.edge_index_dict[edge_type]
        self.full_positive_hash = self.index_to_hash(full_positive_index)

        self.weights = [src_degrees,dst_degrees]
    
    def index_to_hash(self,edge_index):
        size = self.num_nodes
        row, col = edge_index
        hashed_edges = (row * size[1]).add_(col) 
        # for give edge type (v,r,u), hashes edges as:
        # (v_i, u_j) -> v_i*|u| + u_j (|u| num nodes of type u)
        return hashed_edges

    def hash_to_index(self,hashed_edges):
        size = self.num_nodes
        row = hashed_edges.div(size[1], rounding_mode='floor')
        col = hashed_edges % size[1]
        return torch.stack([row, col], dim=0)
    
    def sample_negatives(self,num_samples,src_or_dst):
        """num_samples: number of samples generated, output will have shape [num_samples]. 
        src_or_dst: use src or dst weights to generate sample. 0:src weights, 1:dst weights
        """
        probs = torch.tensor(self.weights[src_or_dst]**0.75)
        neg_samples = probs.multinomial(num_samples, replacement=True) # returns sampled idxs
        return neg_samples
    
    def generate_negative_edge_index(self,positive_edge_index,method):
        if method == "corrupt_both":
            num_samples = positive_edge_index.shape[1]
            new_src_index = self.sample_negatives(num_samples,0) # samples from src
            new_dst_index = self.sample_negatives(num_samples,1) # samples from trgt
            negative_edge_index = torch.stack([new_src_index,new_dst_index])
            return negative_edge_index
        elif method == "fix_src":
            src_index, _ = positive_edge_index
            new_dst_index = self.sample_negatives(src_index.numel(),1) 
            negative_edge_index = torch.stack([src_index,new_dst_index])
            return negative_edge_index
        elif method == "fix_dst":
            _, dst_index = positive_edge_index
            new_src_index = self.sample_negatives(dst_index.numel(),0)
            negative_edge_index = torch.stack([new_src_index,dst_index])
            return negative_edge_index            
    
    def test_false_negatives(self,negative_edge_index,positive_edge_index):
        full_hash = self.full_positive_hash
        negative_hash = self.index_to_hash(negative_edge_index)
        positive_hash = self.index_to_hash(positive_edge_index) 

        false_negatives_mask = torch.isin(negative_hash,full_hash)
        new_negative_hash = negative_hash[~false_negatives_mask]  # hashed true negative edges
        retry_positive_hash = positive_hash[false_negatives_mask] # hashed false negative edges

        return new_negative_hash, retry_positive_hash
    
    def get_negative_sample(self,positive_edge_index,method):
        true_negatives = []
        retry_positive_hash = torch.tensor([0]) #placeholder
        temp_positive_edge_index = copy.copy(positive_edge_index)

        while retry_positive_hash.numel() > 0:
            negative_edge_index = self.generate_negative_edge_index(temp_positive_edge_index,method)
            true_neg_hash, retry_positive_hash = self.test_false_negatives(negative_edge_index,temp_positive_edge_index)

            true_negatives.append(true_neg_hash)
            temp_positive_edge_index = self.hash_to_index(retry_positive_hash)


        negative_edge_hash = torch.concat(true_negatives)
        negative_edge_index = self.hash_to_index(negative_edge_hash)

        return negative_edge_index
    
    def get_labeled_tensors(self,positive_edge_index,method):
        """positive_edge_index: edge_index with only positive edges. 
        This function will use positive_edge_index as a starting point to generate a negative index
        with the same shape as positive_edge_index.
        
        method: 
        corrupt_both: sample both src and dst nodes with probability deg**0.75
        fix_src: keep original src nodes fixed and sample dst nodes with probability deg**0.75
        fix_dst: like fix_src but keep original dst nodes"""

        sample = self.get_negative_sample(positive_edge_index,method)
        edge_label_index = torch.concat([positive_edge_index,sample],dim=1)
        edge_label = torch.concat([torch.ones(positive_edge_index.shape[1]), torch.zeros(positive_edge_index.shape[1])])
        return edge_label_index, edge_label