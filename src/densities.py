import torch
from torch import nn
import math


class GMM(nn.Module):
    def __init__(
        self, means: torch.Tensor, weights: torch.Tensor, sigma: float = 1
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.var = sigma**2
        self.register_buffer("means", means)
        self.register_buffer("weights", weights)
        self.d = self.means.shape[-1]

    def update_means(self, m: torch.Tensor) -> None:
        self.means = m.type_as(self.means)

    def update_weights(self, p: torch.Tensor) -> None:
        self.weights = p.type_as(self.weights)

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.d)
        dmat = torch.cdist(x, self.means)
        square_dist = dmat**2
        exp = (
            1 / (2 * math.pi * self.var) * torch.exp(-1 / (2 * self.var) * square_dist)
        )
        return torch.dot(exp.flatten(), self.weights)

    def eval_batch(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.d)
        dmat = torch.cdist(x, self.means)
        square_dist = dmat**2
        exp = (
            1 / (2 * math.pi * self.var) * torch.exp(-1 / (2 * self.var) * square_dist)
        )
        return exp @ self.weights

    def get_components_batch(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.d)
        dmat = torch.cdist(x, self.means)
        square_dist = dmat**2
        exp = (
            1 / (2 * math.pi * self.var) * torch.exp(-1 / (2 * self.var) * square_dist)
        )
        return exp * self.weights
