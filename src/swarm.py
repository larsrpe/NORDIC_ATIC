import torch
import numpy as np

from typing import Callable

from sys import path
path.append(".")
from src.velocity_field import VelocityField

class Swarm:
    def __init__(self,N,f_R: Callable[[torch.Tensor],torch.Tensor]) -> None:
        self.f_R = f_R
        self.N = N
        self.d = 2
        self.kde_h = 1/20
        self.vf_D = 5
        self.VF = VelocityField(f_R,self.kde_h,self.vf_D)

    def get_dXdt(self,X:np.ndarray) -> np.ndarray:
        dXdt = np.empty_like(X)
        for i in range(0,len(X),self.d):
            x = torch.from_numpy(X[i:i+self.d])
            dXdt[i:i+self.d] = self.VF(x).numpy()
        return dXdt



