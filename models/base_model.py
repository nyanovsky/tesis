import torch
from torch_geometric.nn import to_hetero

# --------------------- MLP layers
class MLP(torch.nn.Module):
    def __init__(self,num_layers,in_dim,out_dim,activate_last,dropout,bn,hidden_dim=None):
        super().__init__()

        hidden_dim = out_dim if hidden_dim is None else hidden_dim
        self.has_bn = bn
        self.dropout = dropout
        
        modules = []
        if num_layers == 1:
            if in_dim == -1:
                modules.append(torch.nn.LazyLinear(out_dim))
            else:
                modules.append(torch.nn.Linear(in_dim,hidden_dim))
        else:
            for i in range(num_layers):
                final_layer = i == num_layers-1
                first_layer = i == 0
                if first_layer:
                    if in_dim == -1:
                        modules.append(torch.nn.LazyLinear(hidden_dim))
                        self._add_post_modules(modules,hidden_dim)
                    else:
                        modules.append(torch.nn.Linear(in_dim,hidden_dim))
                        self._add_post_modules(modules,hidden_dim)                     
                elif final_layer:
                    modules.append(torch.nn.Linear(hidden_dim,out_dim))
                else:
                    modules.append(torch.nn.Linear(hidden_dim,hidden_dim))
                    self._add_post_modules(modules,hidden_dim)
        
        if activate_last:
            self._add_post_modules(modules,out_dim)
        
        self.model = torch.nn.Sequential(*modules)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def _add_post_modules(self,module_list,dim):
        if self.has_bn:
            module_list.append(torch.nn.BatchNorm1d(dim))
        
        if self.dropout > 0:
            module_list.append(torch.nn.Dropout(p=self.dropout))

        module_list.append(torch.nn.LeakyReLU())

    def forward(self,x):
        x = self.model(x)
        return x
    
   
class hetero_MLP(torch.nn.Module):
    def __init__(self,gral_params,metadata,layer_type):
        super().__init__()
        
        node_types = metadata[0]
        if layer_type == "pre_mlp":
            self.in_dim = -1
            self.num_layers = gral_params["pre_process_layers"]
            self.activate_last = True
            
        elif layer_type == "post_mlp":
            self.in_dim = gral_params["hidden_channels"]
            self.num_layers = gral_params["post_process_layers"]
            self.activate_last = False
        
        self.out_dim = gral_params["hidden_channels"]
        self.mlps = torch.nn.ModuleDict({})
        dropout = gral_params["dropout"]
        bn = gral_params["batch_norm"]

        for node_type in node_types:
            self.mlps[node_type] = MLP(self.num_layers,self.in_dim,self.out_dim,self.activate_last,dropout,bn)
    
    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        for mlp in self.mlps.values():
            mlp.reset_parameters()
        
    def forward(self,x:dict):
        out_dict = {}
        for key, mlp in self.mlps.items():
            assert key in x
            out_dict[key] = mlp(x[key])
        return out_dict    
    
    
    
    
    
class conv_layer(torch.nn.Module):
    def __init__(self, conv, gral_params, conv_params, is_out_layer=False, is_in_layer=False):
        super().__init__()
        self.conv = conv(in_channels=-1, out_channels=gral_params["hidden_channels"], **conv_params)
        
        self.normalize = gral_params["L2_norm"]
        self.is_out_layer = is_out_layer
        self.is_in_layer = is_in_layer

        if not self.is_out_layer:
            post_conv_modules = []
            if gral_params["batch_norm"]:
                bn = torch.nn.BatchNorm1d(gral_params["hidden_channels"])
                post_conv_modules.append(bn)
            
            if gral_params["dropout"] > 0:
                dropout = torch.nn.Dropout(p=gral_params["dropout"])
                post_conv_modules.append(dropout)
            
            post_conv_modules.append(torch.nn.LeakyReLU())
            self.post_conv = torch.nn.Sequential(*post_conv_modules)
    
    def forward(self, x, edge_index):
        identity = x
        out = self.conv(x,edge_index)

        if not self.is_out_layer:
            out = self.post_conv(out)
            
        if self.normalize:
            out = torch.nn.functional.normalize(out,2,-1)
        return out    
    

class msg_passing(torch.nn.Module):
    def __init__(self, conv, gral_params, conv_params, metadata):
        super().__init__()
        
        self.num_layers = gral_params["msg_passing_layers"]
        is_out_layer = gral_params["post_process_layers"] == 0
        is_in_layer = gral_params["pre_process_layers"] == 0
        
        self.layers = torch.nn.ModuleList()
        
        if self.num_layers == 1:
            layer = to_hetero(conv_layer(conv, gral_params, conv_params,is_out_layer=is_out_layer,is_in_layer=is_in_layer),metadata,aggr=gral_params["macro_aggregation"])
            self.layers.append(layer)
        else:
            for i in range(self.num_layers):
                if i == self.num_layers-1:
                    layer = to_hetero(conv_layer(conv, gral_params, conv_params,is_out_layer=is_out_layer),metadata,aggr=gral_params["macro_aggregation"])

                elif i == 0:
                    layer = to_hetero(conv_layer(conv, gral_params, conv_params,is_in_layer=is_in_layer),metadata,aggr=gral_params["macro_aggregation"])

                else:
                    layer = to_hetero(conv_layer(conv, gral_params, conv_params),metadata,aggr=gral_params["macro_aggregation"])
                self.layers.append(layer)
                
    def _reset_child_params(self,module):
        for layer in module.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
            self._reset_child_params(layer)
    
    def reset_parameters(self):
        self._reset_child_params(self)
        
    def forward(self, x:dict, edge_index:dict) -> dict:
        for layer in self.layers:
                x = layer(x,edge_index)
        return x
    
    
class base_encoder(torch.nn.Module):
    def __init__(self,conv, gral_params, conv_params,metadata):
        super().__init__()

        self.has_pre_mlp = gral_params["pre_process_layers"] > 0
        self.has_post_mlp = gral_params["post_process_layers"] > 0

        if self.has_pre_mlp:
            self.pre_mlp = hetero_MLP(gral_params,metadata,"pre_mlp")
        
        self.message_passing = msg_passing(conv, gral_params, conv_params, metadata)

        if self.has_post_mlp:
            self.post_mlp = hetero_MLP(gral_params,metadata,"post_mlp")
        
        self.normalize_output = gral_params["normalize_output"]
    
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.pre_mlp.reset_parameters()
        self.post_mlp.reset_parameters()
        self.message_passing.reset_parameters()
    
    def forward(self,x:dict,edge_index:dict) -> dict :
        if self.has_pre_mlp:
            x = self.pre_mlp(x)

        x = self.message_passing(x,edge_index)
        
        if self.has_post_mlp:
            x = self.post_mlp(x)

        if self.normalize_output:
            for key,val in x.items():
                x[key] = torch.nn.functional.normalize(val,2,-1)
            
        return x
    
class decoder(torch.nn.Module):
    def forward(self,x:dict,edge_label_index:dict,supervision_types,apply_sigmoid=True) -> dict:
        pred_dict = {}
        for edge_type in supervision_types:
            edge_index = edge_label_index[edge_type]
            
            source_type, _ , target_type = edge_type
            
            x_source = x[source_type]
            x_target = x[target_type]

            source_index, target_index = edge_index[0], edge_index[1]
            
            nodes_source = x_source[source_index]
            nodes_target = x_target[target_index]

            pred = (nodes_source * nodes_target).sum(dim=1)

            if apply_sigmoid:
                pred = torch.sigmoid(pred)

            pred_dict[edge_type] = pred
        
        return pred_dict
        
        
class base_model(torch.nn.Module):
    def __init__(self, conv, gral_params, conv_params, metadata, supervision_types):
        super().__init__()
        
        self.encoder = base_encoder(conv, gral_params, conv_params, metadata)
        self.decoder = decoder()
        self.loss_fn = torch.nn.BCELoss()
        self.supervision_types = supervision_types
        
    def reset_parameters(self):
        self.encoder.reset_parameters()
        
    def forward(self, x:dict, edge_index:dict, edge_label_index:dict) -> dict:
        x = self.encoder(x, edge_index)
        pred_dict = self.decoder(x, edge_label_index, self.supervision_types)
        return pred_dict
    
    def loss(self, pred_dict, label_dict):
        loss = 0
        for edge_type,pred in pred_dict.items():
            y = label_dict[edge_type]
            loss += self.loss_fn(pred, y.type(pred.dtype))
        return loss
    