import torch
import numpy as np
import matplotlib.pyplot as plt

from sys import path
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, FFMpegWriter
from functools import partial
from fields import VelocityField, LarsField

path.append(".")

def dXdt(t: float, X: np.ndarray, VF: VelocityField | LarsField, h: float) -> np.ndarray:
    print(t)
    X = X.reshape(-1,2)
    X = torch.from_numpy(X)
    dXdt = np.empty_like(X)
    for i,x in enumerate(X):
        dXdt[i,:] = VF(t,x,X).numpy()
    return dXdt.reshape(-1)

def sim(VF: VelocityField | LarsField, X0: torch.Tensor, t_eval: np.ndarray) -> np.ndarray:
    sol = solve_ivp(dXdt, y0=X0.reshape(-1), t_span=(0,t_eval[-1]), args=(VF,1), method="RK23", t_eval=t_eval)
    return sol.t,sol.y

def viz(t,y,L):
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
    anim.save('sims/test.mp4',writer = writer)