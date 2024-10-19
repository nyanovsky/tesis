#%%
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric import seed_everything
import copy
import pickle
# %%
class ParameterWalker():
    def __init__(self, start, historial,possible_params) -> None:
        self.possible_params = possible_params
        self.start = start
        self.historial = historial
        self.dimensions = np.array(list(self.possible_params.keys()))
        self.current_params = copy.copy(start)

    def random_step(self):
        chosen_dim = np.random.choice(self.dimensions,1)[0]
        values = self.possible_params[chosen_dim]
        flip_coin = np.random.choice([-1,1],1)[0]
        
        chosen_param = self.current_params[chosen_dim]
        if chosen_dim == "layer_connectivity" and chosen_param == "False":
            chosen_param = False
        current_index = values.index(chosen_param)
        new_index = (current_index + flip_coin)% len(values)

        new_params = copy.copy(self.current_params)
        new_params[chosen_dim] = values[new_index]
      
        return new_params

    def next(self):
        new_params = self.random_step()
        while list(new_params.values()) in self.historial:
            new_params = self.random_step()
        return new_params
    
    def accept_step(self,new_params):
        self.current_params = new_params
        self.historial.append(list(new_params.values()))

class ConvergenceTest:
    def __init__(self, delta,tolerance, last_auc):
        self.tolerance = tolerance
        self.delta = delta
        self.counter = 0
        self.prev_auc = last_auc
    
    def check_convergence(self,current_auc):
        diff = round(abs(current_auc - self.prev_auc),3)
        if diff <= self.delta:
            self.counter += 1
        else:
            self.counter = 0
            
        self.prev_auc = current_auc
        
        if self.counter >= self.tolerance:
            return True
        else:
            return False