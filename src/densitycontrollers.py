import torch
import numpy as np
from typing import List

from abc import ABC,abstractmethod,abstractproperty


from sys import path

path.append('.')
from src.fields import ControlField,LarsField,VelocityField
from src.densities import GMM,TimevaryingParams
from src.gmminterpolator import GMMInterpolator

class DensityController(ABC):

    @abstractmethod
    def update_refrence(self,t:float) -> None:
        pass

    @abstractmethod
    def get_contoll(self, t: float, r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        r: Tensor of shape
        X: Tensor of shape N,d with all sampels 
        ff: feedforward gain

        returns the velocity field at r
        """


class GausianController(DensityController):
    "controller for tracking moving gausian"
    def __init__(self,L,h,D,t_start,t_end,use_ff:bool= True) -> None:
        m0 = torch.tensor([L/2,L/4*1.2])
        m1 = torch.tensor([L/2,3*L/4])
        self.mdot = 1/(t_start-t_end)*(m1-m0).double()
        
        def weights(t):
            return torch.tensor([1]).double()
        def means(t):
            if t < t_start:
                return m0.view(1,2).double()
            if t> t_end:
                return m1.view(1,2).double()
            
            h = (t-t_start)/(t_end-t_start)
            return ((1-h)*m0 + h*m1).view(1,2).double()
        
        params_weights =TimevaryingParams(weights,None)
        params_means = TimevaryingParams(means,None)
        self.fd = GMM(params_weights,params_means,sigma=1)
        self.LarsField = LarsField(self.fd,h,D)
        self.t_start=t_start
        self.t_end = t_end
        self.use_ff = use_ff
    
    def update_refrence(self, t: float) -> None:
        self.fd.update_params(t)
        self.LarsField.fd=self.fd

     
    def get_contoll(self, t: float, r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        ff = self.get_feedforward(t,r)
        return self.LarsField(t,r,X,ff)
    
    def get_feedforward(self,t:float,x:torch.Tensor) -> torch.Tensor:
        if not self.use_ff:
            return torch.zeros(2)
        if not (self.t_start < t < self.t_end):
            return torch.zeros(2)
        f =self.fd.eval(t,x)
        return -f*self.mdot
        
        
        
    


        

        



class WalkingManController(DensityController):
    "controller for tracking the walking man"

    def __init__(self,L: float,h:float,D:float,sigma_pixels=1,t_start=1,resolution=(32,32),use_ff: bool = True) -> None:
        self.gmminterpolator = GMMInterpolator.walking_man(t_start,L,resolution=resolution)
        try:
            self.gmminterpolator.load_coeff(f'resolution{resolution[0]}')
        except: 
            self.gmminterpolator.interpolate()
            self.gmminterpolator.save_coeff(f"resolution{resolution[0]}")
        
        #repeat three times
        self.gmminterpolator.extend(5*5)
        
        #make 4 times faster
        self.gmminterpolator.speedup(10)

        params_weights =TimevaryingParams(self.gmminterpolator.get_weights,None)
        params_means = TimevaryingParams(self.gmminterpolator.get_means,None)

        sigma = sigma_pixels*L/resolution[0]
        self.fd = GMM(params_weights,params_means,sigma)

        self.LarsField = LarsField(self.fd,h,D)
        self.means_dot = None
        self.use_ff = use_ff
        self.t_start = t_start

    def update_refrence(self, t: float) -> None:
        self.fd.update_params(t)
        self.LarsField.fd=self.fd
        self.means_dot = self.gmminterpolator.get_means_dot(t)


    def get_contoll(self, t: float, r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        ff = self.get_feedforward(t,r)
        return self.LarsField(t,r,X,ff)

    def get_feedforward(self,t:float,x:torch.Tensor) -> torch.Tensor:
        if not self.use_ff:
            return torch.zeros(2)

        if t < self.gmminterpolator.t_start:
            return torch.zeros(2)
        comps = self.fd.get_components(t,x)
        means_dot = self.means_dot
        return -(means_dot.T)@comps



    
    
        
        



        
        

