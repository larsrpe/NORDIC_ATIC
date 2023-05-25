import torch
from torch import nn
import math
import time
import numpy as np
import matplotlib.pyplot as plt


from scipy.spatial import KDTree

from abc import ABC, abstractmethod
from typing import Callable, Tuple

from functools import cached_property

class TimevaryingParams(nn.Module):
    def __init__(self,param_t: Callable[[float], torch.Tensor], param_dot: Callable[[float], torch.Tensor]) -> None:
        super().__init__()
        self.param_t = param_t
        self.param_dot = param_dot
    def eval(self,t: float) -> torch.Tensor:
        return self.param_t(t)

    def dot(self,t: float) -> torch.Tensor:
        return self.param_dot(t)

class TimevaryingPDF(nn.Module,ABC):
        
    @abstractmethod
    def eval(self,t: float, x: torch.Tensor) -> torch.Tensor:
        pass
    @abstractmethod
    def grad(self,t: float, x: torch.Tensor) -> torch.Tensor:
        pass
    @abstractmethod
    def dot(self,t: float, x: torch.Tensor) -> torch.Tensor:
        pass
    @abstractmethod
    def g(self,t: float, x: torch.Tensor) -> torch.Tensor:
        pass
    @abstractmethod
    def get_g_and_grad(self,t: float, x: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        pass



class GausianPDF:
    def __init__(self,sigma,t_0,T,mu_0,mu_T) -> None:
        self.t_0 = t_0
        self.T = T
        self.mu_0 = mu_0
        self.mu_T = mu_T
        self.sigma = sigma
        self.n_x = mu_0.shape[0]

    @cached_property
    def var(self):
        return self.sigma**2


    def _mu(self,t: float) -> torch.Tensor:
        if t < self.t_0:
            t = self.t_0
        if t > self.T:
            t = self.T
        return self.mu_0 + self._mu_dot(t)*(t-self.t_0)
    
    def _mu_dot(self,t: float):
        if t < self.t_0 or t > self.T:
            return 0
        return (self.mu_T-self.mu_0)/(self.T-self.t_0)
    
    def eval(self,t: float,x: torch.Tensor) -> torch.Tensor:
        mu = self._mu(t)
        return 1/(2*math.pi*self.var)*torch.exp(-1/(2*self.var)*torch.dot(x-mu,x-mu))
    
    def grad(self,t: float, x: torch.Tensor) -> torch.Tensor:
        mu = self._mu(t)
        return -(x-mu)/self.var*self.eval(t,x)
    
    def dt(self,t: float,x: torch.Tensor) -> torch.Tensor:
        return torch.dot(self.grad(t,x),self._mu_dot(t))
    
    def get_g(self,t: float,x: float) -> torch.Tensor:
        g = self.eval(t,x)*self._mu_dot(t)
        return g
    
    @abstractmethod
    def get_g_and_grad(self,t: float, x: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        pass
    


class GridGMM(TimevaryingPDF):
    def __init__(self,means: torch.Tensor,weights: TimevaryingParams,L: float) -> None:
        super().__init__()
        self.L = L
        self.means = means
        self.weights = weights
        self.N,_,self.d = means.shape
       

    def get_nn(self,t: float, x: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        x,y = x[0].item(),x[1].item()
        grid_pos_x = int(self.N/self.L*x)
        grid_pos_y = int(self.N/self.L*y)
        x_min,x_max = max(grid_pos_x-10,0),min(grid_pos_x+10,self.N)
        y_min,y_max = max(grid_pos_y-10,0),min(grid_pos_y+10,self.N)
        weights_t = self.weights.eval(t)
        weights_dot_t = self.weights.dot(t)
        return self.means[x_min:x_max,y_min:y_max:].reshape(-1,2), weights_t[x_min:x_max,y_min:y_max].reshape(-1), weights_dot_t[x_min:x_max,y_min:y_max].reshape(-1)

    def eval(self, t: float, x: torch.Tensor) -> torch.Tensor:
        means,weights_t,_= self.get_nn(t,x)
        n = means.shape[0]
        dist = x - means
        square_dist = torch.bmm(dist.view(n, 1, self.d), dist.view(n, self.d, 1)).flatten()
        return torch.dot(1/(2*math.pi)*torch.exp(-1/2*square_dist),weights_t)
    
    def dot(self, t: float, x: torch.Tensor) -> torch.Tensor:
        means,_,weights_dot_t= self.get_nn(t,x)
        n = means.shape[0]
        dist = x - means
        square_dist = torch.bmm(dist.view(n, 1, self.d), dist.view(n, self.d, 1)).flatten()
        return torch.dot(1/(2*math.pi)*torch.exp(-1/2*square_dist),weights_dot_t)
        

    def grad(self, t: float, x: torch.Tensor) -> torch.Tensor:
        means,weights_t,_= self.get_nn(t,x)
        n = means.shape[0]
        dist = x - means
        square_dist = torch.bmm(dist.view(n, 1, self.d), dist.view(n, self.d, 1)).reshape(-1,1)
        exp = 1/(2*math.pi)*torch.exp(-1/2*square_dist)
        return (exp*dist).T@weights_t
        
    def g(self, t: float, x: torch.Tensor) -> torch.Tensor:
        means,_,weights_dot_t= self.get_nn(t,x)
        n = means.shape[0]
        dist = x - means
        square_dist = torch.bmm(dist.view(n, 1, self.d), dist.view(n, self.d, 1)).reshape(-1,1)
        exp = 1/(2*math.pi)*torch.exp(-1/2*square_dist)
        norm_dist = dist/square_dist.view(n,1)
        g = (norm_dist*exp).T@weights_dot_t
        return g
    
    def get_g_and_grad(self, t: float, x: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        means,weights_t,weights_dot_t= self.get_nn(t,x)
        n = means.shape[0]
        dist = x - means
        square_dist = torch.bmm(dist.view(n, 1, self.d), dist.view(n, self.d, 1)).reshape(-1,1)
        exp = 1/(2*math.pi)*torch.exp(-1/2*square_dist)
        grad = (exp*dist).T@weights_t
        norm_dist = dist/square_dist.view(n,1)
        g = (norm_dist*exp).T@weights_dot_t
        return g,grad

    def plot(self,t:float,points: int,filename: str):
        xs = torch.linspace(0,L,steps=points)
        ys = torch.linspace(0,L,steps=points)
        z = torch.zeros(points,points)
        for j,x in enumerate(xs):
            for i,y in enumerate(ys):
                z[points-1-i,j] = self.eval(t,torch.tensor([x,y]))
        z = z.numpy()
        plt.matshow(z)
        plt.savefig("./images/"+filename)
        

if __name__ == "__main__":
    device = "mps"

    size = 40
    L = 15
    rand =torch.rand(size,size)
    w = rand/rand.sum()
    def weights(t:float) -> torch.Tensor:
        return torch.tensor([0.25,0.25,0.25,0.25]).reshape(2,2)
        
    
    def weights_dot(t:float) -> torch.Tensor:
        return w
    
    params = TimevaryingParams(weights,weights_dot)

    means = torch.tensor([[3,3],[12,12], [12,3],[3,12]]).reshape(2,2,2)

    GMM = GridGMM(means,params,L)

    x = 10*torch.rand(2)
    t = time.time()
    a= GMM.grad(0,x)
    b= GMM.g(0,x)
    c,d = GMM.get_g_and_grad(0,x)
    
    GMM.plot(0,100,"test.jpg")