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
    def __init__(self, f_d: TimevaryingPDF, h: float, D: float, feedforward: Callable[[float,torch.Tensor],torch.Tensor]) -> None:
        super().__init__(f_d, h, D)
        self.feedforward = feedforward
    
    def forward(self, t: float, r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        r: Tensor of shape
        X: Tensor of shape N,d with all sampels 

        returns the velocity field at r
        """
        
        f_d_grad = self.f_d.grad(t,r)
        f_d = self.f_d.eval(t,r)
        r = Variable(r.clone(),requires_grad=True)
        f_hat = self.KDE(r,X)
        phi_grad = torch.autograd.grad(f_hat,r)[0] - f_d_grad
        V = -self.D*phi_grad
        
        return ((V + self.feedforward(t,X)*f_d)/f_hat).detach()
    
    @classmethod
    def gmm_controller(cls,means: torch.Tensor, weights_start: torch.Tensor, weights_end: torch.Tensor,  t_start: float,t_end: float, h: float,D: float, L:float) -> "LarsField":
        
        def param_weights(t:float)-> torch.Tensor:
            if t <=t_start:
                return weights_start
            if t > t_end:
                return weights_end
            
            return (1 - (t-t_start)/(t_end-t_start)) * weights_start + (t-t_start)/(t_end-t_start) * weights_end

        def param_weights_dot(t:float)-> torch.Tensor:
            if t <=t_start:
                return torch.zeros_like(weights_start)
            if t > t_end:
                return torch.zeros_like(weights_end)
        
            return -1/(t_end-t_start) * weights_start + (t)/(t_end-t_start) * weights_end
        

        f_d = GMM(means,TimevaryingParams(param_weights,param_weights_dot),sigma=1)

        N = 250
        xs = torch.linspace(0,L,N+1)
        ys = torch.linspace(0,L,N+1)
        
        Rho = np.zeros((N+1,N+1))
        Rho_dot = np.zeros((N+1,N+1))
        Rho_grad = np.zeros((N+1,N+1))


