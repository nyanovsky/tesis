import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, to_hetero
from torch_geometric.data import Data, HeteroData
from gnn_interface.data_class import GNNData
from typing import Union, Dict

class LinkPredictor(torch.nn.Module):
    """
    A GNN-based link prediction model. It uses an encoder-decoder framework.
    The encoder generates node embeddings, and the decoder predicts edge existence.
    """
    def __init__(self, gnn_data: GNNData, hidden_channels: int, model_type: str = 'sage', num_layers: int = 2):
        super().__init__()
        self.is_hetero = gnn_data.is_hetero
        self.data = gnn_data.data
        
        self.encoder = self.build_encoder(hidden_channels, model_type, num_layers)

    def build_encoder(self, hidden_channels, model_type, num_layers):
        """Builds the GNN encoder."""
        convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            if model_type == 'sage':
                conv = SAGEConv((-1, -1), hidden_channels)
            elif model_type == 'gat':
                conv = GATConv((-1, -1), hidden_channels)
            else:
                raise ValueError(f"Model type '{model_type}' not supported.")
            
            if self.is_hetero:
                conv = to_hetero(conv, self.data.metadata(), aggr='sum')
            convs.append(conv)
        return torch.nn.Sequential(*convs)

    def encode(self, x, edge_index):
        """Generates node embeddings."""
        for i, conv in enumerate(self.encoder):
            x = conv(x, edge_index)
            if i < len(self.encoder) - 1:
                if self.is_hetero:
                    x = {key: val.relu() for key, val in x.items()}
                else:
                    x = x.relu()
        return x

    def decode(self, z, edge_label_index):
        """
        Decodes node embeddings to predict edge scores.
        Uses a simple dot product for decoding.
        """
        if self.is_hetero:
            # For heterogeneous graphs, we need to handle different node types
            preds = []
            for edge_type in self.data.edge_types:
                src_type, _, dst_type = edge_type
                if edge_type in edge_label_index:
                    src_idx, dst_idx = edge_label_index[edge_type]
                    src_z = z[src_type][src_idx]
                    dst_z = z[dst_type][dst_idx]
                    pred = (src_z * dst_z).sum(dim=-1)
                    preds.append(pred)
            return torch.cat(preds)
        else:
            src_idx, dst_idx = edge_label_index
            return (z[src_idx] * z[dst_idx]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

def train(model: LinkPredictor, data: Union[Data, HeteroData], optimizer):
    model.train()
    optimizer.zero_grad()
    
    if model.is_hetero:
        pred = model(data.x_dict, data.edge_index_dict, data.edge_label_index_dict)
        # Assuming edge_label is stored for a specific edge type or concatenated
        # For simplicity, let's find the first edge type with labels.
        target = torch.cat([data[edge_type].edge_label for edge_type in data.edge_label_index_dict.keys()])
    else:
        pred = model(data.x, data.edge_index, data.edge_label_index)
        target = data.edge_label
    
    loss = F.binary_cross_entropy_with_logits(pred, target.float())
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model: LinkPredictor, data: Union[Data, HeteroData]):
    from sklearn.metrics import roc_auc_score
    model.eval()
    if model.is_hetero:
        pred = model(data.x_dict, data.edge_index_dict, data.edge_label_index_dict)
        target = torch.cat([data[edge_type].edge_label for edge_type in data.edge_label_index_dict.keys()])
    else:
        pred = model(data.x, data.edge_index, data.edge_label_index)
        target = data.edge_label
    
    pred = pred.sigmoid()
    return roc_auc_score(target.cpu().numpy(), pred.cpu().numpy()) 