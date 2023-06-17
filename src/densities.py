import torch
from torch import nn
import math

from dataclasses import dataclass


from abc import ABC, abstractmethod
from typing import Callable, Tuple

@dataclass
class PDFParams(ABC):
    pass

class PDF(nn.Module,ABC):
    def __init__(self,params: PDFParams) -> None:
        super().__init__()
        self.pdfparams = params

    def set_params(self,p: PDFParams) -> None:
        self.pdfparams = p
    
    @abstractmethod
    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """evaluate"""
        pass
    @abstractmethod
    def eval_batch(self, x: torch.Tensor) -> torch.Tensor:
        pass
    @abstractmethod
    def grad(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    
@dataclass
class GMMParams(PDFParams):
    means: torch.Tensor  
    """(N,d) tensor where N is the number of gaussian components and d is the dimention"""
    weights: torch.Tensor
    """(N) tensor where N is the number of gaussian components"""


class GMM(PDF):
    def __init__(self,params: GMMParams,sigma: float = 1) -> None:
        super().__init__(params)
        self.sigma = sigma
        self.var = sigma**2
        self.pdfparams = params
        self.d = self.pdfparams.means.shape[-1]
       

    def get_params(self) -> Tuple[torch.Tensor,torch.Tensor]:
        return self.pdfparams.means,self.pdfparams.weights
    
    def eval(self, x: torch.Tensor) -> torch.Tensor:
        means,weights= self.get_params()
        x = x.view(-1,self.d)
        dmat = torch.cdist(x,means)
        square_dist = dmat**2
        exp = 1/(2*math.pi*self.var)*torch.exp(-1/(2*self.var)*square_dist)
        return torch.dot(exp.flatten(),weights)
    
    def eval_batch(self, x: torch.Tensor) -> torch.Tensor:
        means,weights= self.get_params()
        x = x.view(-1,self.d)
        dmat = torch.cdist(x,means)
        square_dist = dmat**2
        exp = 1/(2*math.pi*self.var)*torch.exp(-1/(2*self.var)*square_dist)
        return exp@weights
        

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        means,weights=self.get_params()
        n = means.shape[0]
        dist = x - means
        square_dist = torch.bmm(dist.view(n, 1, self.d), dist.view(n, self.d, 1)).reshape(-1,1)
        exp = 1/(2*math.pi*self.var)*torch.exp(-1/(2*self.var)*square_dist)
        return -1/self.var*(exp*dist).T@weights
    
    def get_components(self, x: torch.Tensor) -> torch.Tensor:
        means,weights= self.get_params()
        x = x.view(-1,self.d)
        dmat = torch.cdist(x,means)
        square_dist = dmat**2
        exp = 1/(2*math.pi*self.var)*torch.exp(-1/(2*self.var)*square_dist)
        
        return exp.flatten()*weights.flatten()








    
        
        
    

      

      



   