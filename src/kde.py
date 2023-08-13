import torch
import torch.nn as nn

class GaussianKDE(nn.Module):
    """
    Implements kde estimator with gausian kernel.
    h: Smoothing parameter
    """
    def __init__(self, h: float) -> None:
        super().__init__()
        self.h=h
    
    def forward(self, r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        r: Tensor of shape d 
        X: Tensor of shape N,d with all sampels 
        retuns: estimated density at point r
        """
        N,d = X.shape
        dist_vec =torch.cdist(X,r.reshape(1,-1))
        K_vec = torch.exp(-2/(self.h**2)*torch.square(dist_vec))*2/torch.pi
        f_hat =  1/(N*self.h**d)*torch.sum(K_vec,dim=0)
        return f_hat
    
    def estimate_batch(self, R: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        R: Tensor of shape B,d 
        X: Tensor of shape N,d with all sampels 
        retuns: estimated density at point r
        """

        N,d = X.shape
        B = R.shape[0]
        dist_vec =torch.cdist(R,X)
        K_mat =torch.exp(-2/(self.h**2)*torch.square(dist_vec))*2/torch.pi
        f_hat =  1/(N*self.h**d)*torch.sum(K_mat,dim=1)

        return f_hat





if __name__ == "__main__":

    X = torch.randn((100,2),requires_grad=True)
    r = X[0]
    f = GaussianKDE(1/2)(r,X)
    print(f)