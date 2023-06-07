import torch
import numpy as np
from typing import List

from abc import ABC,abstractmethod,abstractproperty


from sys import path

path.append('.')
from src.fields import ControlField,LarsField,VelocityField
from src.densities import GMM,TimevaryingParams,GridGMM, TimevaryingPDF
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

class PDEFeedforwardController(DensityController):
    def __init__(self,t_start: float,t_end: float, h: float,D: float, L:float) -> None:
        super().__init__()
        self.L =L
        self.t_start = t_start
        self.t_end = t_end
        #the feed forward is calculated solving a pde so wee need to define the spatial resolution of the solver
        self.solver_N = 100
        #Due to expensive computing of the feedforward term we keep it constant over small intervalls of time instead of updating it continously
        self.ff_calculation_interwall = 0.025
        self.ff_t: List[np.ndarray]= []
        self.f_d: TimevaryingPDF = None
        self.LF: LarsField = None


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

        if not self.t_start <= t < self.t_end:
            return torch.zeros(2)
        
        i = int(self.solver_N/self.L*x[0])+1 
        j = int(self.solver_N/self.L*x[1])+1
        t_idx = int((t-self.t_start)/self.ff_calculation_interwall)

        if i > self.solver_N:
            i = self.solver_N
        elif i < 0:  
            i = 0

        if j > self.solver_N:
            j = self.solver_N
        elif j < 0:  
            j = 0
        return torch.from_numpy(self.ff_t[t_idx][j,i])

    def calculate_feedforward(self) -> np.ndarray:
        N = self.solver_N
        xs = torch.linspace(0,self.L,N+1)
        ys = torch.linspace(0,self.L,N+1)
        ts = np.arange(self.t_start,self.t_end,self.ff_calculation_interwall)
        
        Rho = np.zeros((N+1,N+1))
        Rho_dot = np.zeros((N+1,N+1))
        Rho_grad = np.zeros((N+1,N+1,2))

        x0 = np.zeros((N+1)**2*2) #initial guess for the solver

        for t in ts:
            for i,x in enumerate(xs):
                for j,y in enumerate(ys):
                    r = torch.Tensor([x.item(),y.item()]).double()
                    Rho[j,i] = self.f_d.eval(t,r).item()
                    Rho_dot[j,i] = self.f_d.dot(t,r).item()
                    Rho_grad[j,i,:] = self.f_d.grad(t,r).numpy()


            
            solver = ContEQSolver(self.L,Rho,Rho_dot,Rho_grad)
            print("solving pde...")
            solver.solve(x0)
            print(f"solved pde at timestep {t}")
            self.ff_t.append(np.concatenate((solver.G1.reshape(N+1,N+1,1),solver.G2.reshape(N+1,N+1,1)),axis=2))
            X0 = np.concatenate((solver.V1.flatten(),solver.V2.flatten()))

    

            

class GMMController(PDEFeedforwardController):
    "Controller for tracking gaussian mixtures with time varying weights"
    def __init__(self,means: torch.Tensor, weights_start: torch.Tensor, weights_end: torch.Tensor, t_start: float, t_end: float, h: float, D: float, L: float) -> None:
        super().__init__(t_start, t_end, h, D, L)
        
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
        self.calculate_feedforward()


        
        
class ImshowController(PDEFeedforwardController):
    "Controller for tracking interpolation between to images"
    def __init__(self, t_start: float, t_end: float, h: float, D: float, L: float) -> None:
        super().__init__(t_start, t_end, h, D, L)
        self.f_d = GridGMM.showtime('lena','roy_safari',(256,256),t_start,t_end,L,5)
        self.LF = LarsField(self.f_d,h,D)
        self.calculate_feedforward()
        

        

class VideoController(PDEFeedforwardController):
    "Controller for tracking video"
    def __init__(self, t_start: float, h: float, D: float, L: float) -> None:
        super().__init__(t_start, t_start+2, h, D, L)
        self.f_d = GridGMM.from_video(t_start,L)
        self.LF = LarsField(self.f_d,h,D)
        self.calculate_feedforward()
    
    
        
        



        
        

