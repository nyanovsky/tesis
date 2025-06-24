import pandas as pd
import torch
from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T
import networkx as nx
from typing import Union, Optional, Dict, List, Tuple
from torch_geometric.utils import to_networkx, from_networkx, degree
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
                 is_undirected: bool = True,
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
        self.is_undirected = is_undirected

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
        
        if self.is_undirected:
            self.data = T.ToUndirected()(self.data)

        self._calculate_and_store_degrees()

    def _calculate_and_store_degrees(self):
        """
        Calculates and stores node degrees for each edge type.
        The degrees are stored in `data[node_type].degrees_by_edge_type`.
        """
        if not self.is_hetero:
            # For homogeneous graphs, there's only one edge type implicitly.
            edge_index = self.data.edge_index
            degrees = degree(edge_index[0], self.data.num_nodes)
            self.data.degree = degrees
            return

        # For heterogeneous graphs
        for node_type in self.data.node_types:
            self.data[node_type].degrees_by_edge_type = {}

        for edge_type in self.data.edge_types:
            if "rev" not in edge_type[1]:
                src_type, _, dst_type = edge_type
                edge_index = self.data[edge_type].edge_index

                # Calculate and store degrees
                src_degree = degree(edge_index[0], self.data[src_type].num_nodes)
                self.data[src_type].degrees_by_edge_type[edge_type] = src_degree

                # Store degrees for the other node type if it's not a looped edge type
                if src_type != dst_type:
                    dst_degree = degree(edge_index[1], self.data[dst_type].num_nodes)
                    self.data[dst_type].degrees_by_edge_type[edge_type] = dst_degree
            
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
            feature_dict (Optional[Dict]): A dictionary mapping node types to a list of [features, tensor_idxs].
            inplace (bool): If True, modifies the data object in place.
        """
        data_object = self if inplace else copy.deepcopy(self)

        for nodetype, store in data_object.data.node_items():
            if feature_dict and nodetype in feature_dict:
                nodetype_embs, tensor_idxs = feature_dict[nodetype]
                random_init = torch.nn.Parameter(torch.Tensor(store["num_nodes"], nodetype_embs.shape[1]), requires_grad=False)
                torch.nn.init.xavier_uniform_(random_init)
                data_object.data[nodetype].x = random_init
                data_object.data[nodetype].x[tensor_idxs] = nodetype_embs
            else:
                random_init = torch.nn.Parameter(torch.Tensor(store["num_nodes"], dim), requires_grad=False)
                torch.nn.init.xavier_uniform_(random_init)
                data_object.data[nodetype].x = random_init
        
        if not inplace:
            return data_object

    def split_data(self,
                   task: str,
                   negative_sampler,
                   supervision_edge_type: tuple,
                   **kwargs):
        """
        Splits the data for a specific task like link prediction.

        For link prediction, this method first uses RandomLinkSplit to partition
        the graph and create positive supervision links. It then replaces the
        default negative edges in the validation and test sets with new ones
        generated by the provided custom NegativeSampler.

        Args:
            task (str): The task to split for (e.g., 'link_prediction').
            negative_sampler: An initialized NegativeSampler instance.
            supervision_edge_type (tuple): The edge type for which to create
                                           supervision links.
            **kwargs: Arguments for the T.RandomLinkSplitter, e.g., num_val, num_test.
        """
        if task == 'link_prediction':
            # Construct the correct reverse edge type, e.g., ('B', 'rev_rel', 'A')
            rev_supervision_edge_type = None
            if self.is_undirected:
                src, rel, dst = supervision_edge_type
                if src != dst:
                    rev_supervision_edge_type = (dst, 'rev_' + rel, src)

            # Step 1: Use RandomLinkSplit to get initial splits with positive labels
            transform = T.RandomLinkSplit(
                is_undirected=self.is_undirected,
                add_negative_train_samples=False,  # Negatives will be sampled on the fly during training
                edge_types=[supervision_edge_type],
                rev_edge_types=[rev_supervision_edge_type] if rev_supervision_edge_type else None,
                **kwargs
            )
            # We don't use ToSparseTensor here as we need to manipulate edge_label_index
            train_data, val_data, test_data = transform(self.data)

            # Step 2: Replace val and test negative edges using the custom sampler
            for split_data in [val_data, test_data]:
                # Isolate the positive supervision edges created by the splitter
                labels = split_data[supervision_edge_type].edge_label
                labeled_edges = split_data[supervision_edge_type].edge_label_index
                positive_mask = (labels == 1)
                positive_edges = labeled_edges[:, positive_mask]

                # Generate new negative edges and create the final supervision data
                new_label_index, new_label = negative_sampler.get_labeled_tensors(
                    positive_edges, "corrupt_both"
                )
                split_data[supervision_edge_type].edge_label_index = new_label_index
                split_data[supervision_edge_type].edge_label = new_label
            
            # Step 3: Apply ToSparseTensor to all splits for efficiency
            sparse_transform = T.ToSparseTensor(remove_edge_index=False)
            train_data = sparse_transform(train_data)
            val_data = sparse_transform(val_data)
            test_data = sparse_transform(test_data)

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