import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, to_hetero
from torch_geometric.data import Data, HeteroData
from gnn_interface.data_class import GNNData
from typing import Union, Dict, Any
from models.base_model import base_model
from models.training_utils import EarlyStopper, train as train_step, get_val_loss, test as test_step, NegativeSampler
from tqdm.notebook import tqdm
import pandas as pd
from torch_geometric import seed_everything

# Mapping string names to convolution classes for easy selection
CONV_MAP = {
    'GAT': GATConv,
    'SAGE': SAGEConv,
}

class LinkPredictor(torch.nn.Module):
    """
    A configurable GNN Link Predictor with a Scikit-learn-like interface
    for training and evaluation.

    The model is initialized with hyperparameters and remains data-agnostic.
    The training process, specific to link prediction with negative sampling,
    is handled by the `.train()` method.
    """
    def __init__(self,
                 conv_name: str,
                 gral_params: Dict[str, Any],
                 conv_params: Dict[str, Any],
                 optimizer_params: Dict[str, Any]):
        super().__init__()
        if conv_name not in CONV_MAP:
            raise ValueError(f"Unknown convolution '{conv_name}'. Available options: {list(CONV_MAP.keys())}")
        
        # Store configurations
        self.conv_name = conv_name
        self.gral_params = gral_params
        self.conv_params = conv_params
        self.optimizer_params = optimizer_params

        self._model = None  # The actual base_model instance
        self.optimizer = None
        self.history = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def reset(self):
        """
        Resets the model, optimizer, and history.
        This allows re-training the model on new data or with different
        hyperparameters without creating a new LinkPredictor instance.
        """
        self._model = None
        self.optimizer = None
        self.history = {}

    def _build_model_if_needed(self, data: HeteroData, supervision_types: list):
        """Lazily builds the model and optimizer once data is available."""
        if self._model is not None:
            return

        conv_class = CONV_MAP[self.conv_name]
        # This assumes the splits will define the supervision types
        
        self._model = base_model(
            conv=conv_class,
            gral_params=self.gral_params,
            conv_params=self.conv_params,
            metadata=data.metadata(),
            supervision_types=supervision_types
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=self.optimizer_params["lr"], weight_decay=self.optimizer_params["weight_decay"])

    def train(self,
              splits: Dict[str, HeteroData],
              negative_sampler: NegativeSampler,
              supervision_edge_type: tuple,
              epochs: int,
              verbose: bool = True,
              deterministic_sampling=False):
        """
        Trains the model for link prediction using negative sampling.

        Args:
            splits (Dict[str, HeteroData]): A dictionary containing 'train', 'val',
                                            and 'test' data splits.
            negative_sampler: An initialized instance of the NegativeSampler class.
            supervision_edge_type (tuple): The edge type to perform supervision on
                                           (e.g., ('gene', 'chg', 'chem')).
            epochs (int): The number of epochs to train for.
            verbose (bool): If True, shows a progress bar and training metrics.
            deterministic_sampling: If True, fixes seed at each epoch for sampling negative edges at that epoch.

            Only supports one supervision edge type for now.
        """
        train_data, val_data, test_data = splits['train'], splits['val'], splits['test']
        self._build_model_if_needed(train_data, [supervision_edge_type])


        # Prepare negative edges from val/test sets to avoid leakage during training
        val_negs = val_data.edge_label_index_dict[supervision_edge_type][:, val_data.edge_label_dict[supervision_edge_type] == 0]
        test_negs = test_data.edge_label_index_dict[supervision_edge_type][:, test_data.edge_label_dict[supervision_edge_type] == 0]
        avoid_negs = torch.cat((val_negs, test_negs), dim=1)

        train_data.to(self.device)
        val_data.to(self.device)

        # Positive edges from the training set are the basis for sampling
        train_pos_index = train_data.edge_label_index_dict[supervision_edge_type][:, train_data.edge_label_dict[supervision_edge_type] == 1]
        
        early_stopper = EarlyStopper(patience=self.optimizer_params["patience"], min_delta=self.optimizer_params["delta"])
        self.history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}
        
        #iterator = tqdm(range(epochs), desc="Training", disable=not verbose)
        for epoch in range(epochs):
            # Resample negative edges at each epoch
            if deterministic_sampling:
                seed_everything(epoch)
            new_label_index, new_label = negative_sampler.get_labeled_tensors(train_pos_index.cpu(), "corrupt_both", avoid_index=avoid_negs)
            train_data[supervision_edge_type].edge_label_index = new_label_index.to(self.device)
            train_data[supervision_edge_type].edge_label = new_label.to(self.device)

            # Use imported training functions
            train_loss = train_step(self._model, self.optimizer, train_data)
            val_loss = get_val_loss(self._model, val_data)
            
            # Record metrics
            train_auc = test_step(self._model, train_data)
            val_auc = test_step(self._model, val_data)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)
            
            '''
            if verbose:
                iterator.set_postfix({
                    "train_loss": f"{train_loss:.4f}", "val_loss": f"{val_loss:.4f}",
                    "train_auc": f"{train_auc:.4f}", "val_auc": f"{val_auc:.4f}"
                })
            
            if early_stopper.early_stop(val_loss):
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break '''
        
        return self.history

    @torch.no_grad()
    def evaluate(self, data: HeteroData) -> Dict[str, float]:
        """Evaluates the model on a given data split."""
        if self._model is None:
            raise RuntimeError("Model has not been trained. Call .train() first.")
        self._model.eval()
        data.to(self.device)
        
        roc_auc = test_step(self._model, data)
        loss = get_val_loss(self._model, data)

        return {'loss': loss, 'roc_auc': roc_auc}
    
    def get_encodings(self, train_data: HeteroData) -> Dict[str, torch.Tensor]:
        """Returns the node embeddings for the given data."""
        if self._model is None:
            raise RuntimeError("Model has not been trained. Call .train() first.")
        
        self._model.eval()
        train_data.to(self.device)
        return self._model.encoder(train_data.x_dict, train_data.edge_index_dict)
    
    def get_predictions(self, splits:Dict[str, HeteroData], supervision_edge_type:tuple):
        '''
        Returns:
        {train: train_pos_preds, val: {pos:preds, neg:preds}, test:{pos:preds, neg:preds}}
        Negative training predictions correspond to those sampled in the last epoch, since
        since these are re-sampled on each epoch.
        '''
        train, val, test = splits["train"], splits["val"], splits["test"]
        src_type, _, dst_type = supervision_edge_type

        encodings = self.get_encodings(train)
        encodings_src = encodings[src_type]
        encodings_dst = encodings[dst_type]

        pred_dict = {}

        for split, data in splits.items():
            src_nodes = data.edge_label_index_dict[supervision_edge_type][0]
            dst_nodes = data.edge_label_index_dict[supervision_edge_type][1]
            
            preds = torch.sigmoid((encodings_src[src_nodes] * encodings_dst[dst_nodes]).sum(dim=1))
            preds.detach().to("cpu")

            negs = data.edge_label_dict[supervision_edge_type] == 0
            pos = data.edge_label_dict[supervision_edge_type] == 1
                
            pred_dict[split] = {"pos":preds[pos], "negs": preds[negs]}
                
        return pred_dict