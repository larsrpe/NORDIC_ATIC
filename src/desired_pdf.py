import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import interpolate
from typing import string, int, float

class Desired_PDF:
    def __init__(self, image_path: string) -> None:
        self.image = Image.open(image_path)
        self.image = self.image.convert('L')
        
        self.image_points = np.array(self.image)
        self.x_dim = self.image_points.shape[0]
        self.y_dim = self.image_points.shape[1]

        self.f_interp = interpolate.RectBivariateSpline(np.arange(0, self.x_dim, 1), np.arange(0, self.y_dim, 1), self.image_points)
        self.f_interp_int = self.f_interp.integral(0, self.x_dim, 0, self.y_dim)
        self.f_nabla = self.f_interp.partial_derivative(1, 1) # 1'st order derivatives

    def eval(self, x: int, y: int) -> float:
        assert 0 <= x <= self.x_dim
        assert 0 <= y <= self.y_dim
        return self.f(x,y)/self.f_int
    
    def grad(self, x: int, y: int) -> float:
        assert 0 <= x <= self.x_dim
        assert 0 <= y <= self.y_dim
        return self.f_nabla(x, y)
