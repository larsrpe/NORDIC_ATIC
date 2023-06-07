import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from abc import ABC, abstractmethod
from typing import Callable

from sys import path


path.append(".")
from src.kde import Gaussian_KDE
from src.densities import TimevaryingPDF,TimevaryingParams,GMM
from src.pde import ContEQSolver


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
        return (-self.D*phi_grad/f_hat).detach()

class LarsField(ControlField):
     
    
    def forward(self, t: float, r: torch.Tensor, X: torch.Tensor,ff: torch.Tensor) -> torch.Tensor:
        """
        r: Tensor of shape
        X: Tensor of shape N,d with all sampels 
        ff: feedforward gain

        returns the velocity field at r
        """
        
        f_d_grad = self.f_d.grad(t,r)
        f_d = self.f_d.eval(t,r)
        r = Variable(r.clone(),requires_grad=True)
        f_hat = self.KDE(r,X)
        phi_grad = torch.autograd.grad(f_hat,r)[0] - f_d_grad
        V = -self.D*phi_grad
        
        #print(t,ff,f_d)
        return ((V + ff)/f_hat).detach()
    
  
    


