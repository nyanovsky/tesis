import pandas as pd
import torch
from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T
import networkx as nx
from typing import Union, Optional, Dict, List, Tuple
from torch_geometric.utils import to_networkx, from_networkx
import copy

class GNNData:
    """
    A data class to handle loading, processing, and manipulating graph data for GNNs.
    It can handle both homogeneous and heterogeneous graphs, and provides
    utilities for feature initialization, data splitting, and conversion to NetworkX.
    """
    def __init__(self, node_path: str, edge_path: str,
                 node_index_col: str, 
                 src_edge_col: str,
                 dst_edge_col: str,
                 node_type_col: Optional[str] = None,
                 edge_type_col: Optional[str] = None,
                 src_node_type_col: Optional[str] = None,
                 dst_node_type_col: Optional[str] = None,
                 **kwargs):
        """
        Initializes the GNNData object by loading node and edge data from CSV files.

        Args:
            node_path (str): Path to the node CSV file.
            edge_path (str): Path to the edge CSV file.
            node_index_col (str): The column in the node CSV to be used as index.
            src_edge_col (str): The column in the edge CSV for source nodes.
            dst_edge_col (str): The column in the edge CSV for destination nodes.
            node_type_col (Optional[str]): The column for node types. If None, graph is homogeneous.
            edge_type_col (Optional[str]): The column for edge types.
            src_node_type_col (Optional[str]): The column for source node types.
            dst_node_type_col (Optional[str]): The column for destination node types.
        """
        self.is_hetero = node_type_col is not None

        self._load_data(node_path, edge_path, node_index_col, src_edge_col, dst_edge_col,
                        node_type_col, edge_type_col, src_node_type_col, dst_node_type_col, **kwargs)
        
        self._create_pyg_data()

    def _load_data(self, node_path, edge_path, node_index_col, src_edge_col, dst_edge_col,
                   node_type_col, edge_type_col, src_node_type_col, dst_node_type_col, **kwargs):
        """Loads data from CSVs into pandas DataFrames."""
        self.node_df = pd.read_csv(node_path, index_col=node_index_col, **kwargs)
        self.edge_df = pd.read_csv(edge_path, **kwargs)
        
        self.node_type_col = node_type_col
        self.edge_type_col = edge_type_col
        self.src_node_type_col = src_node_type_col
        self.dst_node_type_col = dst_node_type_col
        self.src_edge_col = src_edge_col
        self.dst_edge_col = dst_edge_col

    def _create_pyg_data(self):
        """Creates the PyG Data or HeteroData object."""
        if self.is_hetero:
            self.data = self._create_hetero_data()
        else:
            self.data = self._create_homo_data()

    def _create_homo_data(self):
        """Creates a homogeneous PyG Data object."""
        node_mapping = {index: i for i, index in enumerate(self.node_df.index.unique())}
        
        src = [node_mapping[index] for index in self.edge_df[self.src_edge_col]]
        dst = [node_mapping[index] for index in self.edge_df[self.dst_edge_col]]
        edge_index = torch.tensor([src, dst])

        data = Data(edge_index=edge_index, num_nodes=len(node_mapping))
        
        # Store mappings for later use
        self.node_mappings = {None: node_mapping}
        self.rev_node_mappings = {None: {v: k for k, v in node_mapping.items()}}
        
        return data

    def _create_hetero_data(self):
        """Creates a heterogeneous PyG HeteroData object."""
        data = HeteroData()
        node_types = self.node_df[self.node_type_col].unique()
        
        node_mappings = {}
        rev_node_mappings = {}
        for node_type in node_types:
            mapping = {index: i for i, index in enumerate(
                self.node_df[self.node_df[self.node_type_col] == node_type].index.unique())}
            node_mappings[node_type] = mapping
            rev_node_mappings[node_type] = {v: k for k, v in mapping.items()}
            data[node_type].num_nodes = len(mapping)

        self.node_mappings = node_mappings
        self.rev_node_mappings = rev_node_mappings

        self.edge_df["edge_triple"] = list(
            zip(self.edge_df[self.src_node_type_col], self.edge_df[self.edge_type_col], self.edge_df[self.dst_node_type_col]))
        edge_triplets = self.edge_df["edge_triple"].unique()

        for edge_triplet in edge_triplets:
            sub_df = self.edge_df[self.edge_df.edge_triple == edge_triplet]
            src_type, _, dst_type = edge_triplet

            src_mapping = self.node_mappings[src_type]
            dst_mapping = self.node_mappings[dst_type]

            src = [src_mapping[index] for index in sub_df[self.src_edge_col]]
            dst = [dst_mapping[index] for index in sub_df[self.dst_edge_col]]
            edge_index = torch.tensor([src, dst])
            data[src_type, edge_triplet[1], dst_type].edge_index = edge_index
        
        return data

    def initialize_features(self, dim: int, feature_dict: Optional[Dict] = None, inplace: bool = False):
        """
        Initializes node features, either randomly or from a provided dictionary.

        Args:
            dim (int): The dimension for random features.
            feature_dict (Optional[Dict]): A dictionary mapping node types to features.
            inplace (bool): If True, modifies the data object in place.
        """
        data_object = self.data if inplace else copy.deepcopy(self.data)

        for nodetype, store in data_object.node_items():
            if feature_dict and nodetype in feature_dict:
                nodetype_embs, tensor_idxs = feature_dict[nodetype]
                random_init = torch.nn.Parameter(torch.Tensor(store["num_nodes"], nodetype_embs.shape[1]), requires_grad=False)
                torch.nn.init.xavier_uniform_(random_init)
                data_object[nodetype].x = random_init
                data_object[nodetype].x[tensor_idxs] = nodetype_embs
            else:
                random_init = torch.nn.Parameter(torch.Tensor(store["num_nodes"], dim), requires_grad=False)
                torch.nn.init.xavier_uniform_(random_init)
                data_object[nodetype].x = random_init
        
        return data_object

    def split_data(self, task: str, **kwargs):
        """
        Splits the data for a specific task like link prediction.

        Args:
            task (str): The task to split for (e.g., 'link_prediction').
            **kwargs: Arguments for the splitter, e.g., num_val, num_test.
        """
        if task == 'link_prediction':
            transform = T.RandomLinkSplit(is_undirected=False, add_negative_train_samples=False, **kwargs)
            train_data, val_data, test_data = transform(self.data)
            return train_data, val_data, test_data
        else:
            raise NotImplementedError(f"Split for task '{task}' is not implemented.")

    @property
    def nx_graph(self) -> nx.DiGraph:
        """
        Returns the graph as a networkx.DiGraph object.
        Node identifiers in NetworkX will be the original IDs from the input files.
        """
        G = nx.DiGraph()
        
        for node_type, mapping in self.node_mappings.items():
            rev_mapping = self.rev_node_mappings[node_type]
            for i in range(len(mapping)):
                node_id = rev_mapping[i]
                attrs = self.node_df.loc[node_id].to_dict()
                if self.is_hetero:
                    attrs['node_type'] = node_type
                G.add_node(node_id, **attrs)

        for edge_type, edge_index in self.data.edge_index_dict.items():
            src_type, rel_type, dst_type = edge_type
            src_nodes = edge_index[0].tolist()
            dst_nodes = edge_index[1].tolist()
            
            rev_src_map = self.rev_node_mappings[src_type]
            rev_dst_map = self.rev_node_mappings[dst_type]

            for i in range(len(src_nodes)):
                src_id = rev_src_map[src_nodes[i]]
                dst_id = rev_dst_map[dst_nodes[i]]
                if self.is_hetero:
                    G.add_edge(src_id, dst_id, edge_type=rel_type)
                else:
                    G.add_edge(src_id, dst_id)
        
        return G

    def update_from_nx(self, G: nx.DiGraph):
        """
        Updates the internal PyG data object from a modified NetworkX graph.
        This is useful for experiments like edge rewiring.
        Currently supports edge modifications, not node additions/deletions.
        """
        # Simple implementation assuming nodes have not changed
        edge_list = []
        for u, v, attrs in G.edges(data=True):
            edge_info = {'source': u, 'target': v}
            if self.is_hetero:
                # This part needs to know how to get the types.
                # A robust implementation might require storing original types on the nx graph
                # or making assumptions. For now, we extract from node attributes.
                u_type = G.nodes[u].get('node_type')
                v_type = G.nodes[v].get('node_type')
                e_type = attrs.get('edge_type')
                edge_info[self.src_node_type_col] = u_type
                edge_info[self.dst_node_type_col] = v_type
                edge_info[self.edge_type_col] = e_type

            edge_list.append(edge_info)
        
        self.edge_df = pd.DataFrame(edge_list)
        self.edge_df.rename(columns={'source': self.src_edge_col, 'target': self.dst_edge_col}, inplace=True)
        
        # Re-create PyG data
        self._create_pyg_data() 