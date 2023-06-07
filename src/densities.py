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
    
class GMM(TimevaryingPDF):
    def __init__(self,means: torch.Tensor,weights: TimevaryingParams,sigma: float = 1) -> None:
        super().__init__()
        self.means = means
        self.weights = weights
        self.sigma = sigma
        self.var = sigma**2
        self.d = means.shape[-1]

    def get_nn(self,t: float, x: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        return self.means,self.weights.eval(t),self.weights.dot(t)

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
        exp = 1/(2*math.pi*self.var)*torch.exp(-1/(2*self.var)*square_dist)
        return -1/self.var*(exp*dist).T@weights_t
        

    def plot(self,t:float,xs: np.ndarray,ys:np.ndarray,filename: str):
        points = len(xs)
        z = torch.zeros(points,points)
        for j,x in enumerate(xs):
            for i,y in enumerate(ys):
                z[points-1-i,j] = self.eval(t,torch.tensor([x,y]).double())
        z = z.numpy()
        plt.matshow(z)
        plt.savefig("./images/"+filename)

    
        
    

class GridGMM(GMM):
    def __init__(self, means: torch.Tensor, weights: TimevaryingParams,L: float, sigma_pixels: int =4) -> None:
        self.L = L
        self.N,_,self.d = means.shape
        self.sigma_pixels = sigma_pixels
        sigma = sigma_pixels*(L/self.N)
        super().__init__(means, weights, sigma)



       
    @classmethod
    def from_image(cls,image: str,L: float, sigma_pixels=4) -> "GridGMM":
        means,weights =image_to_pdf_args(image,L)
        
        def param_weights(t:float) -> torch.Tensor:
            return weights.double()
        
        def param_weights_dot(t:float) -> torch.Tensor:
            # image has zero temporal gradient
            return torch.zeros_like(weights).double()
        
        params = TimevaryingParams(param_weights,param_weights_dot)
    
        return cls(means,params,L,sigma_pixels)

    @classmethod
    def showtime(cls, im1: str, im2: str, size: Tuple, t0:float, T:float, L:float,sigma_pixels: int)-> "GridGMM":
        """
        im1: start image
        im2: end image
        size: Tuple of desired image resolution
        t0: start of transition
        T: end of transition
        L: 
        """
        means,w1 =image_to_pdf_args(im1,L)
        _,w2 =image_to_pdf_args(im2,L)
        w1 = w1.double()
        w2 = w2.double()

        def param_weights(t:float)-> torch.Tensor:
            if t <=t0:
                return w1
            if t > T:
                return w2
            
            return (1 - (t-t0)/(T-t0)) * w1 + (t-t0)/(T-t0) * w2

        def param_weights_dot(t:float)-> torch.Tensor:
            if t <=t0:
                return torch.zeros_like(w1)
            if t > T:
                return torch.zeros_like(w2)
        
            return - w1/(T-t0) + w2/(T-t0)

        params = TimevaryingParams(param_weights,param_weights_dot)
        return cls(means, params, L,sigma_pixels)



    @classmethod
    def from_video(cls,t_start:float,L:int):
        pil_frames = []
        video_T = 2#sec
        cap = cv.VideoCapture('./videos/man_walking.mp4')
        started = False
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                started = True
                pil_frames.append(Image.fromarray(np.uint8(frame)))
            if not ret and started:
                break
        
        pil_frames=pil_frames[::2] #half number of frames
        dt = video_T/(len(pil_frames)-1)
        print(dt)

        W = []
        for frame in pil_frames:
            means,w = image_to_pdf_args(frame,L)
            W.append(w.double())
                
        means = means.double()

        

        def param_weights(t:float)-> torch.Tensor:
            t_index = int((t-t_start)/dt)
            if t_index < 0:
                return W[0]
            if t_index >= len(W)-1:
                return W[-1]
            

            t0 = dt*t_index
            T = t0+dt
            w1 = W[t_index]
            w2 = W[t_index+1]
            return w1
            #return (1 - (t-t0)/(T-t0)) * w1 + (t-t0)/(T-t0) * w2
           

        def param_weights_dot(t:float)-> torch.Tensor:
            t_index = int((t-t_start)/dt)
            if t_index < 0:
                return torch.zeros_like(W[0])
            if t_index >= len(W)-1:
                return torch.zeros_like(W[-1])
            
            t0 = dt*t_index
            T = t0+dt
            w1 = W[t_index]
            w2 = W[t_index+1]
            return (w2-w1)/dt
        
        params = TimevaryingParams(param_weights,param_weights_dot)
        return cls(means, params, L,3)
        



    def get_nn(self,t: float, x: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        x,y = x[0].item(),x[1].item()
        grid_pos_x = int(self.N/self.L*x)
        grid_pos_y = self.N-1-int(self.N/self.L*y) # some reversing
        x_min,x_max = max(grid_pos_x-2*self.sigma_pixels,0),min(grid_pos_x+2*self.sigma_pixels,self.N)
        y_min,y_max = max(grid_pos_y-2*self.sigma_pixels,0),min(grid_pos_y+2*self.sigma_pixels,self.N)
        weights_t = self.weights.eval(t)
        weights_dot_t = self.weights.dot(t)
        return self.means[y_min:y_max,x_min:x_max].reshape(-1,2), weights_t[y_min:y_max,x_min:x_max].reshape(-1), weights_dot_t[y_min:y_max,x_min:x_max].reshape(-1)

    
    def plot(self,t:float,points: int,filename: str):
        xs = torch.linspace(0,self.L,steps=points)
        ys = torch.linspace(0,self.L,steps=points)
        super().plot(t,xs,ys,filename)

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

    if isinstance(image,str):
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
    
    GMM = GridGMM.from_video(t_start=2,L=10)

    xs = np.linspace(0,L,100)
    ys = np.linspace(0,L,100)
    for t in np.linspace(0,4,50):
        GMM.plot(t,100,'man_walking')
        print(t)


    #GMM.plot_grid()
    #GMM.plot(0,100,"lena_test.jpg")

   