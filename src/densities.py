import torch
from torch import nn
import math
from PIL import Image, ImageOps
import time
import numpy as np
import matplotlib.pyplot as plt


from scipy.spatial import KDTree

from abc import ABC, abstractmethod
from typing import Callable, Tuple

from functools import cached_property
from torchvision import transforms

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



class GaussianPDF:
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
    
    def dot(self,t: float,x: torch.Tensor) -> torch.Tensor:
        return torch.dot(self.grad(t,x),self._mu_dot(t))
    
    def g(self,t: float,x: torch.Tensor) -> torch.Tensor:
        g = self.eval(t,x)
        return g
    
    def get_g_and_grad(self,t: float,x: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        f = self.eval(t,x)
        mu = self._mu(t)
        mu_dot= self._mu_dot(t)
        grad = -(x-mu)/self.var*f
        g = f*mu_dot
        return g,grad
        
    

class GridGMM(TimevaryingPDF):
    def __init__(self,means: torch.Tensor,weights: TimevaryingParams,L: float) -> None:
        super().__init__()
        self.L = L
        self.means = means
        self.weights = weights
        self.N,_,self.d = means.shape
        self.sima_pixels = 4
        sigma = self.sima_pixels*(L/self.N)
        self.var = sigma**2
       
    @classmethod
    def from_image(cls,image: str,L: float) -> "GridGMM":
        means,weights =image_to_pdf_args(image,L)
        
        def param_weights(t:float) -> torch.Tensor:
            return weights.double()
        
        def param_weights_dot(t:float) -> torch.Tensor:
            # image has zero temporal gradient
            return torch.zeros_like(weights).double()
        
        params = TimevaryingParams(param_weights,param_weights_dot)
    
        return cls(means,params,L)


    def get_nn(self,t: float, x: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        x,y = x[0].item(),x[1].item()
        grid_pos_x = int(self.N/self.L*x)
        grid_pos_y = self.N-1-int(self.N/self.L*y) # some reversing
        x_min,x_max = max(grid_pos_x-2*self.sima_pixels,0),min(grid_pos_x+2*self.sima_pixels,self.N)
        y_min,y_max = max(grid_pos_y-2*self.sima_pixels,0),min(grid_pos_y+2*self.sima_pixels,self.N)
        weights_t = self.weights.eval(t)
        weights_dot_t = self.weights.dot(t)
        return self.means[y_min:y_max,x_min:x_max].reshape(-1,2), weights_t[y_min:y_max,x_min:x_max].reshape(-1), weights_dot_t[y_min:y_max,x_min:x_max].reshape(-1)

    def eval(self, t: float, x: torch.Tensor) -> torch.Tensor:
        means,weights_t,_= self.get_nn(t,x)
        n = means.shape[0]
        dist = x - means
        square_dist = torch.bmm(dist.view(n, 1, self.d), dist.view(n, self.d, 1)).flatten()
        return torch.dot(1/(2*math.pi*self.var)*torch.exp(-1/(2*self.var)*square_dist),weights_t)
    
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
        exp = 1/(2*math.pi*self.var)*torch.exp(-1/2*square_dist)
        return -1/self.var*(exp*dist).T@weights_t
        
    def g(self, t: float, x: torch.Tensor) -> torch.Tensor:
        means,_,weights_dot_t= self.get_nn(t,x)
        n = means.shape[0]
        dist = x - means
        square_dist = torch.bmm(dist.view(n, 1, self.d), dist.view(n, self.d, 1)).reshape(-1,1)
        exp = 1/(2*math.pi*self.var)*torch.exp(-1/2*square_dist)
        norm_dist = dist/square_dist.view(n,1)
        g = (norm_dist*exp).T@weights_dot_t
        return g
    
    def get_g_and_grad(self, t: float, x: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        means,weights_t,weights_dot_t= self.get_nn(t,x)
        n = means.shape[0]
        dist = x - means
        square_dist = torch.bmm(dist.view(n, 1, self.d), dist.view(n, self.d, 1)).reshape(-1,1)
        exp = 1/(2*math.pi*self.var)*torch.exp(-1/2*square_dist)
        grad = -1/self.var*(exp*dist).T@weights_t
        norm_dist = dist/square_dist.view(n,1)
        g = (norm_dist*exp).T@weights_dot_t
        return g,grad

    def plot(self,t:float,points: int,filename: str):
        xs = torch.linspace(0,self.L,steps=points)
        ys = torch.linspace(0,self.L,steps=points)
        z = torch.zeros(points,points)
        for j,x in enumerate(xs):
            for i,y in enumerate(ys):
                z[points-1-i,j] = self.eval(t,torch.tensor([x,y]).double())
        z = z.numpy()
        plt.matshow(z)
        plt.savefig("./images/"+filename)

    def plot_grid(self):
        zs = np.zeros(self.N*self.N)
        xs = np.zeros(self.N*self.N)
        ys = np.zeros(self.N*self.N)
        weights_t = self.weights.eval(0)

        for i in range(self.N):
            for j in range(self.N):
                xs[i*self.N+j] = self.means[i,j,0].item()
                ys[i*self.N+j] = self.means[i,j,1].item()
                zs[i*self.N+j] = weights_t[i,j].item()
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xs,ys,zs)
        plt.show()


def image_to_pdf_args(image:str, L: float)-> Tuple[torch.Tensor,torch.Tensor]:

    image = Image.open(f'images/{image}.png')
    image = expand2square(image,"white")
    image = image.resize((256,256))
    image = ImageOps.grayscale(image)
    convert_tensor = transforms.ToTensor()
    image = convert_tensor(image)

    #transform gjør at vi kanskje ikke må flippe bildet selv?
    image= torch.flipud(image)
    image = 1 - image
    x_dim = image.shape[1]
    y_dim = image.shape[2]

    weights = image/torch.sum(image)
    weights = weights.reshape((x_dim,y_dim))

    x = torch.linspace(0,L,x_dim)
    y = torch.linspace(0,L,y_dim)
    x_grid,y_grid = torch.meshgrid(x,y,indexing='xy')
    grid = torch.cat((x_grid.reshape(x_dim,y_dim,1),torch.flipud(y_grid).reshape(x_dim,y_dim,1)),dim=2)

    return grid, weights

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result        

if __name__ == "__main__":
    
    
    

    L = 15
    
    GMM = GridGMM.from_image('eth',L)

    x = 10*torch.rand(2)
    t = time.time()
    a= GMM.grad(0,x)
    b= GMM.g(0,x)
    c,d = GMM.get_g_and_grad(0,x)

    #GMM.plot_grid()
    GMM.plot(0,100,"lena_test.jpg")

   