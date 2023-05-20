import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.distributions as dist
import typing

from PIL import Image, ImageOps
from scipy import interpolate

Plot_kind = typing.Literal['3d', 'contour']
Image_title = typing.Literal['lena', 'eth']

Smoothing_params = {
    'lena': 11000,
    'eth': 1
}

def plot_contour_or_3d(xx, yy, f, kind: Plot_kind='contour') -> None:
        if kind == 'contour':
            cs = plt.contourf(xx, yy, f, cmap ="bone")
            plt.colorbar(cs)
        else:
            ax = plt.figure().add_subplot(projection='3d')
            ax.plot_surface(xx, yy, f)
            
        plt.show()


class IMG_PDF:
    def __init__(self, image: Image_title , L: float=1.0) -> None:
        super().__init__()
        assert image in typing.get_args(Image_title)

        self.image = Image.open(f'images/{image}.png')
        self.image = ImageOps.grayscale(self.image)
        
        self.L = L
        self.padding = 20
        
        self.image_points = np.flipud(np.array(self.image)/255)
        self.image_points = 1 - self.image_points
        self.image_points = np.pad(self.image_points, mode='constant', pad_width=(self.padding, self.padding), constant_values=(0,0))

        self.x_dim = self.image_points.shape[0]
        self.y_dim = self.image_points.shape[1]

        self.f_interp = interpolate.RectBivariateSpline(np.linspace(0,L, self.x_dim), np.linspace(0,L, self.y_dim), self.image_points, s=Smoothing_params[image])
        self.f_interp_int = self.f_interp.integral(0, L, 0, L)
        self.f_nablax = self.f_interp.partial_derivative(1, 0)
        self.f_nablay = self.f_interp.partial_derivative(0, 1)

    def eval(self, r: torch.Tensor) -> torch.Tensor:
        x = r[0].item()
        y = r[1].item()
        return torch.Tensor(self.f_interp(x,y)/self.f_interp_int).reshape(-1).to(r.device)
    
    def grad(self, r: torch.Tensor) -> torch.Tensor:
        x = r[0].item()
        y = r[1].item()
        return torch.Tensor([self.f_nablax(x, y)[0][0], self.f_nablay(x,y)[0][0]]).reshape(-1).to(r.device)/self.f_interp_int

    def plot(self, kind: Plot_kind) -> None:
        assert kind in typing.get_args(Plot_kind)

        N = 1000
        x = torch.linspace(0, self.L, N)
        y = torch.linspace(0, self.L, N)
        xx, yy = np.meshgrid(x, y)
        f = self.f_interp(x, y)/self.f_interp_int
        plot_contour_or_3d(xx, yy, f, kind)


class GMM_PDF(nn.Module):
    def __init__(self, means: torch.Tensor, covariances: torch.Tensor, weights: torch.Tensor, L: float=1.0) -> None:
        super().__init__()
        self.components = dist.MultivariateNormal(means, covariances)
        self.weights = weights
        self.covariances = covariances
        self.L = L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape((-1, 2))

        prob = torch.exp(self.components.log_prob(x.unsqueeze(1)))*self.weights
        return prob.sum(dim=1)
    
    def eval(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone().detach()
        return self.forward(x).detach()

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone().detach()
        x.requires_grad = True
        prob = self.forward(x)
        return torch.autograd.grad(prob, x)[0].detach()
    
    def plot(self, kind: Plot_kind) -> None:
        assert kind in typing.get_args(Plot_kind)

        N = 1000
        x = torch.linspace(0, self.L, N)
        y = torch.linspace(0, self.L, N)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        r = torch.stack((xx, yy), axis=2)
        f = self.eval(r).reshape(N, N)
        plot_contour_or_3d(xx, yy, f, kind)


if __name__=="__main__":
    # Gaussian mixture model
    means = torch.tensor([[0.3, 0.2], [0.7, 0.7]])
    covariances = torch.tensor([[[0.01, 0.0], [0.0, 0.05]],
                                [[0.01, 0.0], [0.0, 0.01]]])
    weights = torch.tensor([0.5, 0.5])

    f_R_gmm = GMM_PDF(means, covariances, weights)
    f_R_gmm.plot('contour')

    # PDF from image
    f_R_img = IMG_PDF('eth')
    f_R_img.plot('3d')