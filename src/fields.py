import torch
import torch.nn as nn

from torch.autograd import Variable
from sys import path
from kde import Gaussian_KDE
from densities import Gaussian_PDF

path.append(".")

class VelocityField(nn.Module):
    def __init__(self, f_R: Gaussian_PDF, h: float, D: float) -> None:
        super().__init__()
        self.f_R = f_R
        self.KDE = Gaussian_KDE(h)
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

class LarsField(nn.Module):
    def __init__(self, f_R: Gaussian_PDF, h: float, D: float) -> None:
        super().__init__()
        self.f_R = f_R
        self.KDE = Gaussian_KDE(h)
        self.D = D

    def forward(self, t: float, r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        r: Tensor of shape
        X: Tensor of shape N,d with all sampels 

        returns the velocity field at r
        """
        g = self.f_R.get_g(t,r)
        r = Variable(r.clone(),requires_grad=True)
        f_hat = self.KDE(r,X)
        phi_grad = torch.autograd.grad(f_hat,r)[0] - self.f_R.grad(t,r)
        V = -self.D*phi_grad
        
        return ((V + g)/f_hat).detach()