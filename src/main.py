import torch
import math
import numpy as np


from densities import GridGMM,GaussianPDF,TimevaryingParams
from densitycontrollers import GMMController
from fields import LarsField, VelocityField
from sim import sim
from viz import viz_sim,viz_vectorfield





if __name__ == "__main__":
    
    N=100
    L=10
    T = 15
    t_start =5
    t_end =6
    h = L/20
    D = 5
    t_start = 5


    means = torch.tensor([[5,3.5],
                          [5,6.5]])
    
    w_start = torch.tensor([1,0])
    w_end = torch.tensor([0,1])

    controller = GMMController(means,w_start,w_end,t_start,t_end,h,D,L)

    X0 = L/2*torch.rand(N, 2) + torch.tensor([L/4,1])
    t_eval = np.linspace(0,T,20*T)
    t,y = sim(controller,X0,t_eval)
    viz_sim(t,y,L,'test_with_fast_N=200dt=0.025')
    print("sim done")
    
    


