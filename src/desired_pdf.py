import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import interpolate

import torch

class Desired_PDF:
    def __init__(self, image_path: str, L: float,device) -> None:
        super().__init__()
        self.image = Image.open(image_path)
        self.image = self.image.convert('L')
        
        self.image_points = np.array(self.image)
        self.x_dim = self.image_points.shape[0]
        self.y_dim = self.image_points.shape[1]
        self.dx = L/self.x_dim
        self.dy = L/self.y_dim

        self.f_interp = interpolate.RectBivariateSpline(np.linspace(0,L, self.x_dim), np.linspace(0,L, self.y_dim), self.image_points)
        self.f_interp_int = self.f_interp.integral(0, L, 0, L)
        self.f_nablax = self.f_interp.partial_derivative(1, 0) # 1'st order derivatives
        self.f_nablay = self.f_interp.partial_derivative(0, 1)

        self.device = device

    

    def eval(self,r: torch.Tensor) -> torch.Tensor:
        x = r[0].item()
        y = r[1].item()
        #assert 0 <= x <= self.x_dim
        #assert 0 <= y <= self.y_dim
        print(self.f_interp.ev(x,y))
        return torch.Tensor(self.f_interp(x,y)/self.f_interp_int).reshape(-1).to(self.device)
    
    def grad(self,r: torch.Tensor) -> torch.Tensor:
        x = r[0].item()
        y = r[1].item()
        #assert 0 <= x <= self.x_dim
        #assert 0 <= y <= self.y_dim
        return torch.Tensor([self.f_nablax(x, y)[0][0],self.f_nablay(x,y)[0][0]]).reshape(-1).to(self.device)/self.f_interp_int


class Gaus_Desired_PDF:
    def __init__(self,device) -> None:
        self.device = device
    def eval(self,r: torch.Tensor) -> torch.Tensor:
        r = r.clone().detach()
        return (1/np.sqrt(2*np.pi)*torch.exp(-torch.square(torch.norm(r-torch.Tensor([2,2]),2))/2)).to(self.device)
    def grad(self,r: torch.Tensor) -> torch.Tensor:
        r = r.clone().detach()
        return (-self.eval(r)*(r-torch.Tensor([2,2]))).to(self.device)