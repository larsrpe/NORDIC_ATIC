import torch
import numpy as np
import matplotlib.pyplot as plt

from sys import path
from scipy.integrate import solve_ivp

path.append(".")
from src.densitycontrollers import DensityController

from scipy.integrate._ivp.base import OdeSolver  # this is the class we will monkey patch

from tqdm import tqdm

### monkey patching the ode solvers with a progress bar

# save the old methods - we still need them
old_init = OdeSolver.__init__
old_step = OdeSolver.step

# define our own methods
def new_init(self, fun, t0, y0, t_bound, vectorized, support_complex=False):

    # define the progress bar
    self.pbar = tqdm(total=t_bound - t0, unit='ut', initial=t0, ascii=True, desc='IVP')
    self.last_t = t0
    
    # call the old method - we still want to do the old things too!
    old_init(self, fun, t0, y0, t_bound, vectorized, support_complex)


def new_step(self):
    # call the old method
    old_step(self)
    
    # update the bar
    tst = self.t - self.last_t
    self.pbar.update(tst)
    self.last_t = self.t

    # close the bar if the end is reached
    if self.t >= self.t_bound:
        self.pbar.close()


# overwrite the old methods with our customized ones
OdeSolver.__init__ = new_init
OdeSolver.step = new_step



def dXdt(t: float, X: np.ndarray, controller: DensityController,h) -> np.ndarray:
    
    X = X.reshape(-1,2)
    X = torch.from_numpy(X)
    dXdt = np.empty_like(X)
    for i,x in enumerate(X):
        dXdt[i,:] = gradclip(controller.get_contoll(t,x,X).numpy(),3)
    return dXdt.reshape(-1)


def gradclip(grad: np.ndarray, max: float) -> np.ndarray:
    norm = np.linalg.norm(grad)
    return grad if norm<max else grad/norm*max 

def sim(controller: DensityController, X0: torch.Tensor, t_eval: np.ndarray) -> np.ndarray:
    sol = solve_ivp(dXdt, y0=X0.reshape(-1), t_span=(0,t_eval[-1]), args=(controller,1), method="RK23", t_eval=t_eval)
    return sol.t,sol.y

