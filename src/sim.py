import torch
import numpy as np
import matplotlib.pyplot as plt

from sys import path
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, FFMpegWriter
from functools import partial
from fields import ControlField

path.append(".")

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



def dXdt(t: float, X: np.ndarray, Field: ControlField,h) -> np.ndarray:
    
    X = X.reshape(-1,2)
    X = torch.from_numpy(X)
    dXdt = np.empty_like(X)
    for i,x in enumerate(X):
        dXdt[i,:] = gradclip(Field(t,x,X).numpy(),np.inf)
    return dXdt.reshape(-1)


def gradclip(grad: np.ndarray, max: float) -> np.ndarray:
    norm = np.linalg.norm(grad)
    return grad if norm<max else grad/norm*max 

def sim(Field: ControlField, X0: torch.Tensor, t_eval: np.ndarray) -> np.ndarray:
    sol = solve_ivp(dXdt, y0=X0.reshape(-1), t_span=(0,t_eval[-1]), args=(Field,1), method="RK23", t_eval=t_eval)
    return sol.t,sol.y

def viz(t,y,L,file_name = "test"):
    fig = plt.figure() 
   
    # marking the x-axis and y-axis
    axis = plt.axes(xlim =(0,L), 
                    ylim =(0,L)) 
  
    # initializing a line variable
    plot, = axis.plot([], [], "ko", markersize=5)

    
    # data which the line will 
    # contain (x, y)
    def init(): 
        plot.set_data([], [])
        return plot,
    
    def animate(i,Y):
        X = Y[:,i].reshape(-1,2)
        x = X[:,0]
        y = X[:,1]
        plot.set_data(x, y)
        return plot,

    anim = FuncAnimation(fig, partial(animate,Y=y), init_func = init,
                     frames = len(t), interval = 20, blit = True)
  
    writer = FFMpegWriter(fps = 5)
    anim.save(f'sims/{file_name}.mp4',writer = writer)