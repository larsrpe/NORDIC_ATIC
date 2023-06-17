import torch
from torch import nn
import math
from PIL import Image, ImageOps
import cv2 as cv
import time
import numpy as np
import matplotlib.pyplot as plt


from abc import ABC, abstractmethod
from typing import Callable, Tuple

from functools import cached_property


from gmminterpolator import GMMInterpolator

from gmminterpolator import GMMInterpolator

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
    def eval_batch(self,t: float, x: torch.Tensor) -> torch.Tensor:
        pass
    @abstractmethod
    def grad(self,t: float, x: torch.Tensor) -> torch.Tensor:
        pass
    @abstractmethod
    def dot(self,t: float, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def update_params(self,t: float) -> None:
        pass
    

class GMM(TimevaryingPDF):
    def __init__(self,weights: TimevaryingParams, means: TimevaryingParams,sigma: float = 1) -> None:
        super().__init__()
        self.sigma = sigma
        self.var = sigma**2
        self.d = 2
        self.weights = weights
        self.means = means
        self.means_t = None
        self.weights_t = None

    def update_params(self, t: float) -> None:
        self.means_t,self.weights_t = self.get_params(t)
       

    def get_params(self,t: float) -> Tuple[torch.Tensor,torch.Tensor]:
        return self.means.eval(t),self.weights.eval(t)
    

    def eval(self, t: float, x: torch.Tensor) -> torch.Tensor:
        means,weights= self.means_t,self.weights_t
        x = x.view(-1,self.d)
        dmat = torch.cdist(x,means)
        square_dist = dmat**2
        exp = 1/(2*math.pi*self.var)*torch.exp(-1/(2*self.var)*square_dist)
        return torch.dot(exp.flatten(),weights)
    
    def eval_batch(self, t: float, x: torch.Tensor) -> torch.Tensor:
        means,weights= self.means_t,self.weights_t
        x = x.view(-1,self.d)
        dmat = torch.cdist(x,means)
        square_dist = dmat**2
        exp = 1/(2*math.pi*self.var)*torch.exp(-1/(2*self.var)*square_dist)
        return exp@weights
    
    def dot(self, t: float, x: torch.Tensor) -> torch.Tensor:
        pass
        

    def grad(self, t: float, x: torch.Tensor) -> torch.Tensor:
        means,weights= self.means_t,self.weights_t
        n = means.shape[0]
        dist = x - means
        square_dist = torch.bmm(dist.view(n, 1, self.d), dist.view(n, self.d, 1)).reshape(-1,1)
        exp = 1/(2*math.pi*self.var)*torch.exp(-1/(2*self.var)*square_dist)
        return -1/self.var*(exp*dist).T@weights
    
    def get_components(self, t: float, x: torch.Tensor) -> torch.Tensor:
        means,weights= self.means_t,self.weights_t
        x = x.view(-1,self.d)
        dmat = torch.cdist(x,means)
        square_dist = dmat**2
        exp = 1/(2*math.pi*self.var)*torch.exp(-1/(2*self.var)*square_dist)
        
        return exp.flatten()*weights.flatten()





    #def plot(self,t:float,xs: np.ndarray,ys:np.ndarray,filename: str):
    #    points = len(xs)
    #    z = torch.zeros(points,points)
    #    for j,x in enumerate(xs):
    #        for i,y in enumerate(ys):
    #            z[points-1-i,j] = self.eval(t,torch.tensor([x,y]).double())
    #    z = z.numpy()
    #    plt.matshow(z)
    #    plt.savefig("./images/"+filename)

    #def plot(self,t:float,xs: np.ndarray,ys:np.ndarray,filename: str):
    #    points = len(xs)
    #    z = torch.zeros(points,points)
    #    for j,x in enumerate(xs):
    #        for i,y in enumerate(ys):
    #            z[points-1-i,j] = self.eval(t,torch.tensor([x,y]).double())
    #    z = z.numpy()
    #    plt.matshow(z)
    #    plt.savefig("./images/"+filename)




    
        
        
    

      

      



   