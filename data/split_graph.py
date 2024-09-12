#%%
import sys 
import os 
sys.path.append("..")
#%%
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric import seed_everything
import pandas as pd
import os
import pickle
from models.training_utils import get_tensor_index_df, NegativeSampler
#%%
def load_node_csv(path, type_col, **kwargs):
    """Returns node dataframe and a dict of mappings for each node type. 
    Each mapping maps from original df index to "heterodata index" { node_type : { dataframe_index : heterodata_index}}"""
    df = pd.read_csv(path, **kwargs)
    node_types = df[type_col].unique()
    mappings_dict = dict()
    for node_type in node_types:
        mapping = {index: i for i, index in enumerate(
            df[df[type_col] == node_type].index.unique())}
        mappings_dict[node_type] = mapping

    return df, mappings_dict

#%%
def load_edge_csv(path, src_index_col, dst_index_col, mappings, edge_type_col, src_type_col, dst_type_col, **kwargs):
    """Returns edge dataframe and a dict of edge indexes. Nodes are indexed according to the "heterodata index", 
    using the node mappings from load_node_csv. Edge indexes are tensors of shape [2, num_edges]. 
    Dict is indexed by triplets of shape (src_type, edge_type, dst_type)."""
    df = pd.read_csv(path, **kwargs)
    df["edge_triple"] = list(
        zip(df[src_type_col], df[edge_type_col], df[dst_type_col]))
    edge_triplets = df["edge_triple"].unique()

    edge_index_dict = dict()
    for edge_triplet in edge_triplets:

        sub_df = df[df.edge_triple == edge_triplet]
        src_type, edge_type, dst_type = edge_triplet

        src_mapping = mappings[src_type]
        dst_mapping = mappings[dst_type]

        src = [src_mapping[index] for index in sub_df[src_index_col]]
        dst = [dst_mapping[index] for index in sub_df[dst_index_col]]
        edge_index = torch.tensor([src, dst])
        edge_index_dict[edge_triplet] = edge_index

    return df, edge_index_dict


def create_heterodata(node_map, edge_index):
    """Initializes HeteroData object from torch_geometric and creates corresponding nodes and edges, without any features"""
    data = HeteroData()
    for node_type, vals in node_map.items():
        # Initialize all node types without features
        data[node_type].num_nodes = len(vals)

    for edge_triplet, index in edge_index.items():
        src_type, edge_type, dst_type = edge_triplet
        data[src_type, edge_type, dst_type].edge_index = index

    return data


def get_reverse_types(edge_types):
    newlist = []
    for edge in edge_types:
        rev = tuple(reversed(edge))
        if rev != edge:
            if edge not in newlist:
                newlist.append(rev)
        else:
            newlist.append(rev)

    reversed_newlist = [tuple(reversed(edge)) for edge in newlist]

    return newlist, reversed_newlist


# %%
dti_folder = "/biodata/nyanovsky/datasets/dti/processed/"
node_data, node_map = load_node_csv(dti_folder+"node_df.csv", "node_type")
edge_data, edge_index = load_edge_csv(dti_folder+"edge_df.csv","src_node_index", 
                                      "trgt_node_index", node_map, "edge_type",
                                      "src_node_type", "trgt_node_type")
#%%
data = create_heterodata(node_map, edge_index)
edge_types, rev_edge_types = get_reverse_types(data.edge_types) 
# ta bien esto? mas facil hacer [tuple(reversed(x)) for x in data.edge_types]
# %%
p_val = 0.1
p_test = 0.1
p_train = round(1 - p_val - p_test, 1)

split_transform = T.RandomLinkSplit(num_val=p_val, num_test=p_test, is_undirected=True, add_negative_train_samples=False,
                                    disjoint_train_ratio=0.2, edge_types=edge_types, rev_edge_types=rev_edge_types)
transform_dataset = T.Compose([split_transform, T.ToSparseTensor(remove_edge_index=False)])

train_data, val_data, test_data = transform_dataset(data)
# %%
# Generate Deg**0.75 negative sampling distribution for val and test splits
edge_type = ("gene", "chg", "chem")
tensor_df = get_tensor_index_df(node_data,node_map)

src_degrees = tensor_df[tensor_df.node_type == "gene"]["ChG_deg"].values
trgt_degrees = tensor_df[tensor_df.node_type == "chem"]["ChG_deg"].values

negative_sampler = NegativeSampler(data,edge_type,src_degrees,trgt_degrees)
#%%
val_labels = val_data[edge_type]["edge_label"]
val_labeled_edges = val_data[edge_type]["edge_label_index"]
index = torch.nonzero(val_labels == 1).flatten()
val_positive_edges = val_labeled_edges[:,index]

new_val_label_index, _ = negative_sampler.get_labeled_tensors(val_positive_edges,"corrupt_both")
val_data[edge_type]["edge_label_index"] = new_val_label_index

test_labels = test_data[edge_type]["edge_label"]
test_labeled_edges = test_data[edge_type]["edge_label_index"]
index = torch.nonzero(test_labels == 1).flatten()
test_positive_edges = test_labeled_edges[:,index]

new_test_label_index, _ = negative_sampler.get_labeled_tensors(test_positive_edges,"corrupt_both") 

test_data[edge_type]["edge_label_index"] = new_test_label_index
#%%
# add degree data to full_dataset (we use this data in negative sampling)
data["gene"]["degree_chg"] = src_degrees
data["chem"]["degree_chg"] = trgt_degrees
# %%

### test negatives correctness

def test_equal_num_edges(dataset):
    num_chg_r = dataset[("chem", "chg", "gene")
                        ]["edge_index"].shape[1]
    num_chg_l = dataset[("gene", "chg", "chem")
                        ]["edge_index"].shape[1]
    print(
        f"num chg edges in both directions is equal: {num_chg_r == num_chg_l}")


def test_is_correct_p(dataset, p, total_num, prev_edges):
    # num_supervision divided by 2 because the same number of edges is generated as negative samples.
    # These are directed (i.e, a single link in one direction)
    num_supervision = dataset[(
        "gene", "chg", "chem")]["edge_label"].shape[0]/2

    # num_msg divided by 2 because these links are undirected (i.e, two links per edge, one in each direction)
    num_msg = dataset[("gene", "chg", "chem")
                      ]["edge_index"].shape[1]

    num = round(num_supervision + num_msg)
    expected_num = round(p*total_num + prev_edges)
    print(f"Is expected % of edges: {num == expected_num}")
    print(f"Expected {expected_num}, is {num}")
#%%
total_num_chg = data[("gene", "chg", "chem")]["edge_index"].shape[1]

datasets = [train_data, val_data, test_data]
percentage = [p_train, p_val, p_test]
names = ["train", "validation", "test"]

prev_edges = 0
for name, dataset, p in zip(names, datasets, percentage):
    print(name + ":")
    test_equal_num_edges(dataset)
    test_is_correct_p(dataset, p, total_num_chg, prev_edges)
    print("\n")

    prev_edges = round(dataset[("gene", "chg", "chem")]["edge_label"].shape[0] /
                       2 + dataset[("gene", "chg", "chem")]["edge_index"].shape[1])
# %%
# Save splits to cpu

torch.save(data, dti_folder+"dti_full_dataset"+".pt")

for dataset, name in zip(datasets, names):
    path = dti_folder+"dti_"+name+".pt"
    torch.save(dataset, path)


with open(dti_folder+"dti_node_map.pickle", 'wb') as handle:
    pickle.dump(node_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save tensor dataframe 
tensor_df.to_csv(dti_folder+"dti_tensor_df.csv")
# %%
