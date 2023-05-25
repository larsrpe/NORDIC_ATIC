import torch
import numpy as np

from densities import Gaussian_PDF
from fields import LarsField
from sim import sim, viz

if __name__ == "__main__":
    device = "cpu"
    
    N=100
    L=15
    sigma = 1
    T = 10
    h = L/20
    D = 5

    mu_0 = torch.Tensor([3,3])
    mu_T = torch.Tensor([13,13])
    f_R = Gaussian_PDF(sigma, 0, T, mu_0, mu_T)

    LF = LarsField(f_R,h,D)
    
    X0 = L/2*torch.rand(N, 2)
    t_eval = np.linspace(0, T, T*5+1, endpoint=True)
    t,y = sim(LF,X0,t_eval)

    print("sim done")
    viz(t,y,L)