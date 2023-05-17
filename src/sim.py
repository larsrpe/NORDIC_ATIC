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
        u = VF(x,X).numpy()
        #print(np.linalg.norm(u))
        dXdt[i,:] = grad_clip(u,0.1)
    return dXdt.reshape(-1)

def grad_clip(grad,max):
    norm = np.linalg.norm(grad)
    if norm > max:
        return grad/norm*max
    return grad


def sim(VF: VelocityField,X0,t_eval):
    sol = solve_ivp(dXdt,y0=X0.reshape(-1),t_span=(0,100),args=(VF,1),method="RK23",t_eval=t_eval)
    
    return sol.t,sol.y
    

def viz(t,y):
    fig = plt.figure() 
   
    # marking the x-axis and y-axis
    axis = plt.axes(xlim =(0,0.5), 
                    ylim =(0,0.5)) 
  
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
    
    N=100
    X0 = (0.5*torch.rand(N,2))
    L=0.5
    f_R = Desired_PDF("images/lena.jpg",L,device)
    h = L/20
    D = 5

    VF = VelocityField(f_R,h,D).to(device)
    t_eval = np.linspace(0,100,101,endpoint=True)
    t,y = sim(VF,X0,t_eval)
    print("sim done")
    viz(t,y)

   
    
