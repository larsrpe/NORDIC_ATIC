import torch
import numpy as np
from typing import List

from abc import ABC,abstractmethod


from sys import path

path.append('.')
from src.fields import ControlField,LarsField,VelocityField
from src.densities import GMM,GMMParams
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
    "controller for illustrating tracking of a moving gausian"
    
    def __init__(self,L,h,D,t_start,t_end,use_ff:bool= True) -> None:
        self.m0 = torch.tensor([L/2,L/4*1.2])
        self.m1 = torch.tensor([L/2,3*L/4])
        self.mdot = 1/(t_end-t_start)*(self.m1-self.m0).double()
        self.t_start=t_start
        self.t_end = t_end
        self.use_ff = use_ff

        self.fd = GMM(self.get_gmm_params(0),sigma=1)
        self.LarsField = LarsField(self.fd,h,D)
        

    def get_gmm_params(self,t:float) -> GMMParams:
        
        w = torch.tensor([1]).double()
        
        if t < self.t_start:
            means = self.m0.view(1,2).double()
        elif t> self.t_end:
            means = self.m1.view(1,2).double()
            
        else:
            h = (t-self.t_start)/(self.t_end-self.t_start)
            means = ((1-h)*self.m0 + h*self.m1).view(1,2).double()
        
        return GMMParams(means=means,weights=w)
    
    def update_refrence(self, t: float) -> None:
        """method to update the reference of the swarm at the beginning of each timestep"""
        self.fd.set_params(self.get_gmm_params(t))
        self.LarsField.f_d=self.fd

     
    def get_contoll(self, t: float, r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        ff = self.get_feedforward(t,r)
        return self.LarsField(t,r,X,ff)
    
    def get_feedforward(self,t:float,r:torch.Tensor) -> torch.Tensor:
        if not self.use_ff:
            return torch.zeros(2)
        if not (self.t_start < t < self.t_end):
            return torch.zeros(2)
        f =self.fd.eval(r)
        return f*self.mdot
        
        
        
    


        
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



        sigma = sigma_pixels*L/resolution[0]
        self.fd = GMM(self.get_gmm_params(0),sigma)

        self.LarsField = LarsField(self.fd,h,D)
        self.means_dot = None
        self.use_ff = use_ff
        self.t_start = t_start

    def get_gmm_params(self,t:float) -> GMMParams:
        w = self.gmminterpolator.get_weights(t)
        m = self.gmminterpolator.get_means(t)
        return GMMParams(means=m,weights=w)

    def update_refrence(self, t: float) -> None:
        """method to update the reference of the swarm at the beginning of each timestep"""
        self.fd.set_params(self.get_gmm_params(t))
        self.LarsField.f_d=self.fd
        self.means_dot = self.gmminterpolator.get_means_dot(t)


    def get_contoll(self, t: float, r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        ff = self.get_feedforward(t,r)
        return self.LarsField(t,r,X,ff)

    def get_feedforward(self,t:float,x:torch.Tensor) -> torch.Tensor:
        if not self.use_ff:
            return torch.zeros(2)

        if t < self.gmminterpolator.t_start:
            return torch.zeros(2)
        comps = self.fd.get_components(x)
        means_dot = self.means_dot
        return -(means_dot.T)@comps



    
    
        
        



        
        

