import torch
import torch.nn as nn
from typing import Callable
from torch.autograd import Variable
from torch.autograd import grad

from sys import path

path.append(".")

from src.kde import GAUS_KDE
from src.desired_pdf import Desired_PDF

class VelocityField(nn.Module):
    def __init__(self,f_R: Desired_PDF, h: float,D: float) -> None:
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
        phi_grad = torch.autograd.grad(f_hat,r)[0] - self.f_R.grad(r)
        return -self.D*phi_grad/f_hat.detach()


    

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    X = torch.randn((100,2),device=device)
    r = X[0]
    
    f_R = Desired_PDF("images/lena.jpg",1,device)
    grad = f_R.grad(r)
    print(grad)
    
    VF = VelocityField(f_R,1/2,1/20).to(device)(X[0],X)
    print(VF)

   

    
    







