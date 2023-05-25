import torch
import math
from PIL import Image, ImageOps
from functools import cached_property
from torchvision import transforms

class Gaussian_PDF:
    def __init__(self, sigma: float, t_0: float, T, mu_0: torch.Tensor, mu_T: torch.Tensor) -> None:
        self.t_0 = t_0
        self.T = T
        self.mu_0 = mu_0
        self.mu_T = mu_T
        self.sigma = sigma
        self.n_x = mu_0.shape[0]

    @cached_property
    def var(self):
        return self.sigma**2

    def _mu(self, t: float) -> torch.Tensor:
        if t < self.t_0:
            t = self.t_0
        if t > self.T:
            t = self.T
        return self.mu_0 + self._mu_dot(t)*(t-self.t_0)
    
    def _mu_dot(self, t: float):
        if t < self.t_0 or t > self.T:
            return 0
        return (self.mu_T-self.mu_0)/(self.T-self.t_0)
    
    def eval(self, t: float, x: torch.Tensor) -> torch.Tensor:
        mu = self._mu(t)
        return 1/(2*math.pi*self.var)*torch.exp(-1/(2*self.var)*torch.dot(x-mu,x-mu))
    
    def grad(self, t: float, x: torch.Tensor) -> torch.Tensor:
        mu = self._mu(t)
        return -(x-mu)/self.var*self.eval(t,x)
    
    def dt(self, t: float,x: torch.Tensor) -> torch.Tensor:
        return torch.dot(self.grad(t,x),self._mu_dot(t))
    
    def get_g(self, t: float, x: float) -> torch.Tensor:
        g = self.eval(t,x)*self._mu_dot(t)
        return g
    
def image_to_pdf_args(image, L: float)-> torch.Tensor:

    image = Image.open(f'images/{image}.png')
    image = ImageOps.grayscale(image)
    convert_tensor = transforms.ToTensor()
    image = convert_tensor(image)

    #transform gjør at vi kanskje ikke må flippe bildet selv?
    image= torch.flipud(image/255)
    image = 1 - image
    x_dim = image.shape[1]
    y_dim = image.shape[2]

    weights = image/torch.sum(image)
    weights = weights.reshape((x_dim,y_dim))

    x = torch.linspace(0,L,x_dim)
    y = torch.linspace(0,L,y_dim)
    x_grid,y_grid = torch.meshgrid(x,y,indexing='xy')
    grid = torch.cat((x_grid,torch.flipud(y_grid)))
    grid = grid.reshape((2,x_dim,y_dim))

    return grid, weights
    
