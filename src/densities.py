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
    def grad(self,t: float, x: torch.Tensor) -> torch.Tensor:
        pass
    @abstractmethod
    def dot(self,t: float, x: torch.Tensor) -> torch.Tensor:
        pass
    

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
        x = x.view(-1,self.d)
        dmat = torch.cdist(x,means)
        square_dist = dmat**2
        exp = 1/(2*math.pi*self.var)*torch.exp(-1/(2*self.var)*square_dist)
        return exp@weights_t
    
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

class VideoGMM(GMM):
    def __init__(self,t_start:float,L:int,sigma_pixels: int = 4,resolution: Tuple[int,int]= (128,128)) -> None:
        self.t_start = t_start
        self.L = L
       
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
        
        pil_frames=pil_frames[::4] #half number of frames
        dt = video_T/(len(pil_frames)-1)
        print(dt)

        ps = []
        ms = [] 
        for frame in pil_frames:
            means,p = image_to_pdf_args(frame,L,resolution)
            num_pixels = means.shape[0]
            p = p.reshape(-1).double()
            m = means.reshape(-1,2).double()
            ps.append(p)
            ms.append(m)
        

        self.sigma = sigma_pixels*L/num_pixels
        self.d = 2
        self.var = self.sigma**2
        self.interpolation = GMMInterpolator(t_start,dt,ps,ms,self.sigma)
    

    def get_nn(self, t: float, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.interpolation.get_means(t,x[0],x[1]), self.interpolation.get_weights(t,x[0],x[1]),None

    def dot(self, t: float, x: torch.Tensor) -> torch.Tensor:
        pass



    

        

    #def plot(self,t:float,xs: np.ndarray,ys:np.ndarray,filename: str):
    #    points = len(xs)
    #    z = torch.zeros(points,points)
    #    for j,x in enumerate(xs):
    #        for i,y in enumerate(ys):
    #            z[points-1-i,j] = self.eval(t,torch.tensor([x,y]).double())
    #    z = z.numpy()
    #    plt.matshow(z)
    #    plt.savefig("./images/"+filename)


def image_to_pdf_args(image:str, L: float,resolution: Tuple[int,int]= (128,128))-> Tuple[torch.Tensor,torch.Tensor]:

    if isinstance(image,str):
        image = Image.open(f'images/{image}.png')
    image = expand2square(image,"white")
    image = image.resize(resolution)
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

    
        
    

      



   