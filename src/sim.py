import torch
from typing import Callable
from scipy.integrate import solve_ivp

from sys import path
path.append(".")
from src.velocity_field import calculate_VF

def dXdt(t,X: torch.Tensor,f_R: Callable[[float],float], h: float,D: float) -> torch.Tensor:
    return calculate_VF(torch.from_numpy(X.reshape(-1,2)),f_R,h,D).reshape(-1)

def sim(f_R: Callable[[float],float], h: float,D: float):
    X0 = torch.randn(100,2).reshape(-1)
    return solve_ivp(dXdt,y0=X0,t_span=(1,10),args=(f_R,h,D))


if __name__ == "__main__":
    def f_R(x):
        return 1/torch.pi if torch.norm(x,2) < 1 else 0
    h = 1/20
    D = 5
    sim(f_R,h,D)