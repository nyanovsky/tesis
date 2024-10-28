#type: ignore
#%%
import sys
sys.path.append("..")
import torch
import numpy as np
from torch_geometric import seed_everything
from models import prediction_utils, training_utils
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score
seed_everything(0)

#%%
version = input("enter dataset version:")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_folder = f"/biodata/nyanovsky/datasets/dti/processed/{version}/"
#%%
#%%

def init_features(train_set, val_set, test_set, params, feature_dict={}):
    train_set = training_utils.initialize_features(train_set, params["feature_dim"], feature_dict)
    val_set = training_utils.initialize_features(val_set, params["feature_dim"], feature_dict)
    test_set = training_utils.initialize_features(test_set, params["feature_dim"], feature_dict)
    return train_set, val_set, test_set



#full_set = torch.load(data_folder+"dti_full_dataset.pt")
#negative_sampler = training_utils.NegativeSampler(full_set,("gene","chg","chem"),full_set["gene"]["degree_chg"],full_set["chem"]["degree_chg"])

def train_model(model, params, train_set, val_set, negative_sampler):
    
    train_set.to(device)
    val_set.to(device)

    # Initialize model
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    train_losses = []
    val_losses = []
    train_scores = []
    val_scores = []

    train_label_index = train_set["gene","chg","chem"]["edge_label_index"]
    
    early_stopper = training_utils.EarlyStopper(params["patience"], params["delta"])

    for epoch in range(params["epochs"]):
        #Resample negative supervision links every epoch
        new_train_label_index, new_train_label = negative_sampler.get_labeled_tensors(train_label_index.cpu(),"corrupt_both")
        train_set["gene","chg","chem"]["edge_label_index"] = new_train_label_index.to(device)
        train_set["gene","chg","chem"]["edge_label"] = new_train_label.to(device)

        train_loss = training_utils.train(model, optimizer, train_set)
        val_loss = training_utils.get_val_loss(model, val_set)

        train_score = training_utils.test(model, train_set)
        val_score = training_utils.test(model, val_set)

        train_losses.append(train_loss)
        train_scores.append(train_score)

        val_scores.append(val_score)
        val_losses.append(val_loss)
        
        """
        if early_stopper.early_stop(val_loss):
            print("Early stopping")
            break
            """
            

    val_auc = training_utils.test(model, val_set)
    curve_data = [train_losses, val_losses, train_scores, val_scores]


    return model, val_auc , curve_data


def full_eval(data,model,node_df):
    encodings_dict = training_utils.get_encodings(model,data)
    predictor = prediction_utils.Predictor(node_df,encodings_dict)

    preds = predictor.predict_supervision_edges(data,("gene","chg","chem"))
    y_true = preds.label.values
    y_score = preds.score.values
    y_pred_labels = preds.score.values.round()

    auc = roc_auc_score(y_true,y_score)
    acc = accuracy_score(y_true,y_pred_labels) 
    ap = average_precision_score(y_true,y_score) 
    precision = precision_score(y_true,y_pred_labels) 
    recall = recall_score(y_true,y_pred_labels) 

    return preds, {"auc":auc, "acc":acc, "ap":ap, "precision":precision, "recall":recall}


def run_experiment(model, initialized_train_set, initialized_val_set, initialized_test_set, params, negative_sampler, node_df ):

    model, val_auc, curve_data = train_model(model, params, initialized_train_set, initialized_val_set, negative_sampler)

    model = model.to("cpu")
    
    preds, results = full_eval(initialized_test_set, model, node_df)

    return model, results, preds, curve_data
