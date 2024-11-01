import sys 
sys.path.append("..")
import torch
from models import base_model

class MLP_model(torch.nn.Module):
    def __init__(self, params, metadata, sup_type):
        super().__init__()
        self.encoder = base_model.hetero_MLP(params, metadata, "post_mlp")
        self.decoder = base_model.decoder()
        self.loss_fn = torch.nn.BCELoss()
        self.sup_type = sup_type 

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, x:dict, edge_index:dict, edge_label_index:dict) -> dict:
        x = self.encoder(x)
        pred_dict = self.decoder(x, edge_label_index, self.sup_type)
        return pred_dict
    
    def loss(self, pred_dict, label_dict):
        loss = 0
        for edge_type,pred in pred_dict.items():
            y = label_dict[edge_type]
            loss += self.loss_fn(pred, y.type(pred.dtype))
        return loss