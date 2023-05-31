import numpy as np
import torch
from matplotlib import pyplot as plt
from typing import Callable


from scipy.integrate import dblquad
from kde import Gaussian_KDE
from densities import GaussianPDF 

def eval_exp(Y: np.ndarray, T: np.ndarray, f_R_d: GaussianPDF, h: float) -> np.ndarray:
    KDE = Gaussian_KDE(h)
    Y = torch.from_numpy(Y)

    f_R_hat_t = lambda x, y, t: KDE(torch.Tensor([x, y]).to(torch.double), Y[:, t].reshape(-1, 2)).item()
    f_R_d_t = lambda x, y, t: f_R_d.eval(t, torch.Tensor([x, y]).to(torch.double)).item()
    E_t = lambda t: dblquad(lambda x, y: np.absolute(f_R_hat_t(x, y, t) - f_R_d_t(x, y, t)), 0, 15, 0, 15, epsabs=1.49e-1, epsrel=1.49e-1)[0]

    idt = np.arange(0, len(T))
    return np.vectorize(E_t)(idt)




