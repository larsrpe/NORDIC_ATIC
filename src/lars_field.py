import torch
import torch.nn as nn
from typing import Callable
from torch.autograd import Variable
from torch.autograd import grad

from sys import path

path.append(".")

from src.kde import GAUS_KDE
from src.timevarying_pdf import GausianPDF


class LarsField(nn.Module):
    def __init__(self, f_R: GausianPDF, h: float, D: float) -> None:
        super().__init__()
        self.f_R = f_R
        self.KDE = GAUS_KDE(h)
        self.D = D

    def forward(self,t,r,X):
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
        #g = 0
        

        return ((V + g)/f_hat).detach()
