import torch
import numpy as np
from typing import Callable
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
import time
import math

from sys import path
path.append(".")
from src.velocity_field import VelocityField



def dXdt(t,X: np.ndarray,f_R: Callable[[float],float], h: float,D: float) -> np.ndarray:
    X = X.reshape(-1,2)
    X = torch.from_numpy(X)
    VF = VelocityField(f_R,h,D)
    dXdt = np.empty_like(X)
    for i,x in enumerate(X):
        dXdt[i,:] = VF(x,X).numpy()
    return dXdt.reshape(-1)


def sim(f_R: Callable[[float],float], h: float,D: float):
    X0 = (4*torch.rand(100,2)-2).reshape(-1)
    sol = solve_ivp(dXdt,y0=X0,t_span=(0,100),args=(f_R,h,D))
    
    return sol.t,sol.y
    

def viz(t,y):
    fig = plt.figure() 
   
    # marking the x-axis and y-axis
    axis = plt.axes(xlim =(-2,2), 
                    ylim =(-2, 2)) 
  
    # initializing a line variable
    plot, = axis.plot([], [], "*") 
    
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
  
   
    anim.save('test.mp4',writer = 'ffmpeg', fps = 1)
    


    



if __name__ == "__main__":
    def f_R(x):
        return 1/torch.pi if torch.norm(x,2) < 1 else 0
    
        
    h = 1/20
    D = 5
    t,y = sim(f_R,h,D)
    print(y.shape)
    viz(t,y)
