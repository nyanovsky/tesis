import pandas as pd
import torch

class MappedDataset:
    """
    Utilidad para manejar los mapeos de índices entre los dataframes originales y los objetos de torch.

    heterodata: objeto de torch_geometric, alguno de los conjuntos del split (train,val,test)
    node_map: diccionario que contiene la equivalencia entre node_index del dataset y tensor_index de heterodata
    prediction_edge_type: el tipo de enlace que queremos mapear (arma un df para cada tipo, me pareció comodo porque
    para evaluar el modelo solo necesitamos ver un tipo de enlace)

    MappedDataset.dataframe contiene todos los enlaces que están en heterodata indexados por "node_index", osea el índice original
    del dataset. En las columnas está el sub-tipo de enlace: propagación o supervisión y la etiqueta si es de supervisión.
    """

    def __init__(self, heterodata, node_map, prediction_edge_type):
        self.prediction_edge_type = prediction_edge_type
        self.node_map = node_map
        self.edge_dict = self._reverse_map_heterodata(heterodata)
        self.dataframe = self._edge_dict_to_dataframe()

    def _reverse_map_tensor(self, tensor, edge_type):
        """Maps edge dictionary from pyg Heterodata back into the original node indexes from the dataframe"""
        # Tensor to lists [sources], [targets]
        sources = tensor[0, :].tolist()
        targets = tensor[1, :].tolist()

        # Map edge list to node indexes
        src_type, dst_type = edge_type[0], edge_type[2]
        src_map, dst_map = self.node_map[src_type], self.node_map[dst_type]

        mapped_src = [src_map[n] for n in sources]
        mapped_trg = [dst_map[n] for n in targets]

        return {
            f"{src_type}_source": mapped_src,
            f"{dst_type}_target": mapped_trg,
            f"torch_{src_type}_index_source": sources,
            f"torch_{dst_type}_index_target": targets,
        }

    def _reverse_map_heterodata(self, data):
        """Maps full edge data from pyg Heterodata back into the original node indices from the dataframe"""
        edge_dict = {}
        for edge_type in data.edge_types:
            type_dict = {}
            edge_tensor = data[edge_type]["edge_index"]
            mapped_edge_list = self._reverse_map_tensor(edge_tensor, edge_type)

            type_dict["message_passing_edges"] = mapped_edge_list

            if "edge_label_index" in data[edge_type].keys():
                labeled_edges_tensor = data[edge_type]["edge_label_index"]
                # labeled_edges_list = tensor_to_edgelist(labeled_edges_tensor)
                mapped_labeled_edges_list = self._reverse_map_tensor(
                    labeled_edges_tensor, edge_type
                )
                edge_labels = data[edge_type]["edge_label"].tolist()

                type_dict["supervision_edges"] = mapped_labeled_edges_list
                type_dict["supervision_labels"] = edge_labels

            edge_dict[edge_type] = type_dict

        return edge_dict

    def _edge_dict_to_dataframe(self):
        edges_df = []
        e_dict = self.edge_dict[self.prediction_edge_type]
        supervision_edges = pd.DataFrame(e_dict["supervision_edges"])

        labeled_edges = pd.concat(
            [supervision_edges, pd.DataFrame(e_dict["supervision_labels"])], axis=1
        ).rename(columns={0: "label"})
        msg_passing_edges = pd.DataFrame(e_dict["message_passing_edges"])

        msg_passing_edges["edge_type"] = "message_passing"
        labeled_edges["edge_type"] = "supervision"

        edges_df.append(labeled_edges)
        edges_df.append(msg_passing_edges)
        total_df = pd.concat(edges_df, axis=0)
        return total_df
    
    
    
class Predictor:
    """
    Utilidad para hacer predicciones rápidas una vez que ya tenemos los encodings calculados.
    Calcula la probabilidad de enlaces con inner_product_decoder, que es una similaridad producto interno más una logística.
    Se encarga de mapear los indices del grafo ("node_index") a los índices tensoriales que usan los datos de torch.
    Esto es para evitar ambiguedades ya que los indices tensoriales no son únicos (hay una enfermedad 0 y un gen 0),
    mientras que los "node_index" sí son únicos.

    node_df debe contener los indices del grafo "node_index" y su equivalencia a indice de torch "tensor_index" 
    para que pueda hacer el mapeo
    """

    def __init__(self, node_df, encodings_dict):

        self.df = node_df
        self.encodings = encodings_dict
        gene_index = torch.tensor(
            self.df[self.df.node_type ==
                    "gene"]["tensor_index"].index.values
        )
        chem_index = torch.tensor(
            self.df[self.df.node_type ==
                    "chem"]["tensor_index"].index.values
        )
        self.node_index_dict = {
            "gene": gene_index, "chem": chem_index}

    def inner_product_decoder(self, x_source, x_target, apply_sigmoid=True):
        pred = (x_source * x_target).sum(dim=1)

        if apply_sigmoid:
            pred = torch.sigmoid(pred)

        return pred

    def predict_supervision_edges(self, data, edge_type):
        """
        Si queremos calcular la proba de enlace para los datos en el conjunto de
        test en lugar de pasarle nodos elegidos manualmente.
        If return_dataframe_==True, returns dataframe with edges,
        prediction scores and labels. Else, returns predicted scores tensor
        """
        src_type, trg_type = edge_type[0], edge_type[2]
        x_source = self.encodings[src_type]
        x_target = self.encodings[trg_type]

        edge_label_index = data.edge_label_index_dict[edge_type]
        source_index, target_index = edge_label_index[0], edge_label_index[1]

        emb_nodes_source = x_source[source_index]
        emb_nodes_target = x_target[target_index]

        pred = self.inner_product_decoder(emb_nodes_source, emb_nodes_target)
        
        labels = data.edge_label_dict[edge_type].numpy()
        df = pd.DataFrame(
                {
                    "torch_gene_index": source_index,
                    "torch_chem_index": target_index,
                    "score": pred,
                    "label": labels,
                }
            )
        return df

