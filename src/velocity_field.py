import torch
from typing import Callable
from torch.autograd import Variable
from torch.autograd import grad

from sys import path

path.append(".")

from src.kde import gausian_kde

def calculate_VF(X: torch.Tensor, f_R: Callable[[float],float], h: float,D: float) -> torch.Tensor:
    """
    Calculates the velocoty field from the paper.

    X: Tensor of shape N,d with all sampels 
    f_R: The desired PDF
    D: Diffusion parameter
    h: Smoothing parameter for density estimation

    returns: Tensor of shape N,2 where row i is the VF of r_i
    """
    N,d = X.shape
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu" #some bug with float32
    V = torch.zeros((N,d))

    for i in range(N):
        X.type(torch.float32)
        r = Variable(X[i].clone().type(torch.float32),requires_grad=True)
        r = r.to(device)
        X = X.to(device)

        f_hat = gausian_kde(r,X,h) 
        phi = f_hat - f_R(r)
        grad = torch.autograd.grad(phi,r)[0]
        V[i] = -D*grad/f_hat

    return V.detach()
    
   


if __name__ == "__main__":
    X = torch.randn((100,2))
    
    def f_r(x):
        return 1
    
    calculate_VF(X,f_r,0.5,5)






