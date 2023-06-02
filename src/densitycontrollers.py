import torch
import numpy as np

from abc import ABC,abstractmethod


from sys import path

path.append('.')
from src.fields import ControlField,LarsField,VelocityField
from src.densities import GMM,TimevaryingParams
from src.pde import ContEQSolver

class DensityController(ABC):
    @abstractmethod
    def get_contoll(self, t: float, r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        r: Tensor of shape
        X: Tensor of shape N,d with all sampels 
        ff: feedforward gain

        returns the velocity field at r
        """


class GMMController(DensityController):
    "Controller for tracking gaussian mixtures with time varying weights"
    def __init__(self,means: torch.Tensor, weights_start: torch.Tensor, weights_end: torch.Tensor,  t_start: float,t_end: float, h: float,D: float, L:float) -> None:
        super().__init__()

        self.L =L
        #the feed forward is calculated solving a pde so wee need to define the spatial resolution of the solver
        self.solver_N = 200
        #Due to expensive computing of the feedforward term we keep it constant over small intervalls of time instead of updating it continously
        self.ff_calculation_interwall = 0.025#sec
        self.last_ff_calculation_time =-1
        self.last_ff_grid = np.zeros((self.solver_N+1,self.solver_N+1,2))

        
        
        def param_weights(t:float)-> torch.Tensor:
            if t <=t_start:
                return weights_start.double()
            if t > t_end:
                return weights_end.double()
            
            return ((1 - (t-t_start)/(t_end-t_start)) * weights_start + (t-t_start)/(t_end-t_start) * weights_end).double()

        def param_weights_dot(t:float)-> torch.Tensor:
            if t <=t_start:
                return torch.zeros_like(weights_start).double()
            if t > t_end:
                return torch.zeros_like(weights_end).double()
        
            return (-1/(t_end-t_start) * weights_start + (t)/(t_end-t_start) * weights_end).double()
        

        self.f_d = GMM(means,TimevaryingParams(param_weights,param_weights_dot),sigma=1)
        self.LF = LarsField(self.f_d,h,D)

    
    def get_contoll(self, t: float, r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        r: Tensor of shape
        X: Tensor of shape N,d with all sampels 
        ff: feedforward gain

        returns the velocity field at r
        """

        ff = self.get_feedforward(t,r)
        return self.LF(t,r,X,ff)

    def get_feedforward(self,t:float,x: torch.Tensor) -> torch.Tensor:
        
        i = int(self.solver_N/self.L*x[0])
        j = int(self.solver_N/self.L*x[1])

        if i > self.solver_N:
            i = self.solver_N
        elif i < 0:  
            i = 0

        if j > self.solver_N:
            j = self.solver_N
        elif j < 0:  
            j = 0
        
        dt = t-self.last_ff_calculation_time
        if dt > self.ff_calculation_interwall and 5 <=t <= 6:
            self.calculate_feedforward(t)
        if dt > 6:
            self.last_ff_grid = np.zeros((self.solver_N+1,self.solver_N+1,2))
        
        return torch.from_numpy(self.last_ff_grid[j,i])


    def calculate_feedforward(self,t:float) -> np.ndarray:
        N = self.solver_N
        xs = torch.linspace(0,self.L,N+1)
        ys = torch.linspace(0,self.L,N+1)
        
        Rho = np.zeros((N+1,N+1))
        Rho_dot = np.zeros((N+1,N+1))
        Rho_grad = np.zeros((N+1,N+1,2))


        for i,x in enumerate(xs):
            for j,y in enumerate(ys):
                r = torch.Tensor([x.item(),y.item()]).double()
                Rho[j,i] = self.f_d.eval(t,r).item()
                Rho_dot[j,i] = self.f_d.dot(t,r).item()
                Rho_grad[j,i,:] = self.f_d.grad(t,r).numpy()

        v1_0=self.last_ff_grid[:,:,0].flatten()
        v2_0=self.last_ff_grid[:,:,1].flatten()
        v0 = np.concatenate((v1_0,v2_0))
        solver = ContEQSolver(self.L,Rho,Rho_dot,Rho_grad)
        #print("solving pde...")
        solver.solve(x0=v0)
        #print("solved pde")
        self.last_ff_grid = np.concatenate((solver.V1.reshape(N+1,N+1,1),solver.V2.reshape(N+1,N+1,1)),axis=2)
        self.last_ff_calculation_time = t


        
        



        
        

