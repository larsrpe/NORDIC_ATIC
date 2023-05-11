import torch
from dataclasses import dataclass




def gausian_kde(r: torch.Tensor, X: torch.Tensor,h: float) -> float:
    """
    Implements kde estimator with gausian kernel.

    r: Tensor of shape d 
    X: Tensor of shape N,d with all sampels 
    h: Smoothing parameter

    retuns: estimated density at point r
    """
    N,d = X.shape
    dist_mat =torch.cdist(X,r.reshape(1,-1))
    K_mat = torch.exp(-2/(h**2)*torch.square(dist_mat))*2/torch.pi
    f_hat =  1/(N*h**d)*torch.sum(K_mat,dim=0)

    return f_hat


if __name__ == "__main__":
    X = torch.randn((100,2))
    r = X[0]
    f = gausian_kde(r,X,1/2)
    print(f.shape)



    

