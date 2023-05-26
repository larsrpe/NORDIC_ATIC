import torch
import torch.nn as nn
from torch.autograd import Variable
from abc import ABC, abstractmethod
from sys import path

path.append(".")
from src.kde import Gaussian_KDE
from src.densities import TimevaryingPDF


class ControlField(nn.Module,ABC):
    def __init__(self, f_d: TimevaryingPDF, h: float, D: float) -> None:
        super().__init__()
        self.f_d = f_d
        self.KDE = Gaussian_KDE(h)
        self.D = D

    @abstractmethod
    def forward(self, t: float, r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        pass


class VelocityField(ControlField):
    
    def forward(self, t: float, r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        r: Tensor of shape
        X: Tensor of shape N,d with all sampels 

        returns the velocity field at r
        """
        r = Variable(r.clone(),requires_grad=True)
        f_hat = self.KDE(r,X)
        phi_grad = torch.autograd.grad(f_hat,r)[0] - self.f_d.grad(t,r)
        return -self.D*phi_grad/f_hat.detach()

class LarsField(ControlField):
    
    def forward(self, t: float, r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        r: Tensor of shape
        X: Tensor of shape N,d with all sampels 

        returns the velocity field at r
        """
        
        g,f_d_grad = self.f_d.get_g_and_grad(t,r)
        r = Variable(r.clone(),requires_grad=True)
        f_hat = self.KDE(r,X)
        phi_grad = torch.autograd.grad(f_hat,r)[0] - f_d_grad
        V = -self.D*phi_grad
        
        return ((V + g)/f_hat).detach()