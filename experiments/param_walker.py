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
    def __init__(self, start, possible_params) -> None:
        self.possible_params = possible_params
        self.start = start
        self.historial = [list(start.values())]
        self.dimensions = np.array(list(self.possible_params.keys()))
        self.current_params = copy.copy(start)

    def random_step(self):
        chosen_dim = np.random.choice(self.dimensions,1)[0]
        values = self.possible_params[chosen_dim]
        flip_coin = np.random.choice([-1,1],1)[0]
        # izq o der en los posibles valores del parametro elegido 
        # sesgo al orden? quizas mejor elegir al azar dentro de las posibles,
        # impone implicitamente continuidad de la perf en los params (que podrian ni tener orden)
        current_index = values.index(self.current_params[chosen_dim])
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

# %%
# init walker
# k = 0 
# k_max = k_max
# T = 1-(k+1)/k_max 
# while k<k_max and not converged(curr_acu)
#   params = random step 
#   model(params)
#   delta = curr_auc-new_auc
#   accept_step = min(1,np.exp(-delta/T))
#   if accept_step > np.random.random():
#       walker.accept_step(params)
#       record performance (df with steps and auc)
#   k++
# Para correr esto en partes (500 iteraciones un dia y 500 otro, por ejemplo),
# deberia tomar un k_0 y una configuracion inicial (la sacaria del df.iloc[-1,:-1])