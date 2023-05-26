import torch
import numpy as np

from densities import GridGMM,GaussianPDF
from fields import LarsField, VelocityField
from sim import sim, viz

if __name__ == "__main__":
    
    N=100
    L=6
    T = 2
    h = L/20
    D = 5

    mu_0 = torch.Tensor([3,3])
    mu_T = torch.Tensor([13,13])
    f_d = GridGMM.from_image("eth",L)
    #f_d = GaussianPDF(1,0,T,mu_0,mu_0)
    #f_d.plot(0,500,"eth_plot")

    LF = LarsField(f_d,h,D)
    VF = VelocityField(f_d,h,D)
    
    X0 = L*torch.rand(N, 2)
    t_eval = np.linspace(0, T, T*5+1, endpoint=True)
    t,y = sim(VF,X0,t_eval)

    print("sim done")
    viz(t,y,L,'test_reth')