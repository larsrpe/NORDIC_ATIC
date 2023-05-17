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
from src.desired_pdf import Desired_PDF



def dXdt(t,X: np.ndarray,VF: VelocityField,h) -> np.ndarray:
    print(t)
    X = X.reshape(-1,2)
    X = torch.from_numpy(X)
    dXdt = np.empty_like(X)
    for i,x in enumerate(X):
        dXdt[i,:] = VF(x,X).numpy()
    return dXdt.reshape(-1)


def sim(VF: VelocityField):
    X0 = (4*torch.rand(100,2)).reshape(-1)
    sol = solve_ivp(dXdt,y0=X0,t_span=(0,100),args=(VF,1))
    
    return sol.t,sol.y
    

def viz(t,y):
    fig = plt.figure() 
   
    # marking the x-axis and y-axis
    axis = plt.axes(xlim =(0,4), 
                    ylim =(0, 4)) 
  
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
  
   
    anim.save('test.mp4',writer = 'ffmpeg', fps = 10)
    


    



if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    X = torch.randn((100,2),device=device)
    r = X[0]
    
    L=4
    f_R = Desired_PDF("images/lena.jpg",L,device)
    h = 4/20
    D = 5

    VF = VelocityField(f_R,h,D).to(device)
    t,y = sim(VF)
    print("sim done")
    viz(t,y)

   
    
