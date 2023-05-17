import torch
import torch.nn as nn
from typing import Callable
from torch.autograd import Variable
from torch.autograd import grad

from sys import path

path.append(".")

from src.kde import GAUS_KDE

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
    KDE = GAUS_KDE()
    
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
    
   
class VelocityField(nn.Module):
    def __init__(self,f_R: Callable[[torch.Tensor],torch.Tensor], h: float,D: float) -> None:
        super().__init__()
        self.f_R = f_R
        self.KDE = GAUS_KDE(h)
        self.D = D

    def forward(self,r,X):
        """
        r: Tensor of shape
        X: Tensor of shape N,d with all sampels 

        returns the velocity field at r
        """
        r = Variable(r.clone(),requires_grad=True)
        f_hat = self.KDE(r,X)
        f_d = self.f_R(r)
        phi = f_hat - f_d
        grad = torch.autograd.grad(phi,r)[0]
        return -self.D*grad/f_hat.detach()


    

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    X = torch.randn((100,2),device=device)
    
    def f_R(x):
        return 1/torch.pi if torch.norm(x,p=2) < 1 else 0
    
    VF = VelocityField(f_R,1/2,1/20).to(device)(X[0],X)
    print(VF)

    
    







