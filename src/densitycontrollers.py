from typing import Tuple
from torch import Tensor

import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from src.densities import GMM
from src.gmminterpolator import GMMInterpolator
from src.kde import GaussianKDE as KDE


class DensityController(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("state", torch.tensor([0.0, 0.0]))

    @abstractmethod
    def get_contoll(self, t: float, r: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        r: Tensor of shape
        X: Tensor of shape N,d with all sampels

        returns the velocity field at r
        """
        pass

    def update_measurment(self, X: torch.Tensor) -> None:
        self.state = X.type_as(self.state)


class GausianController(DensityController):
    "controller for illustrating tracking of a moving gausian"

    def __init__(self, L, h, D, t_start, t_end, use_ff: bool = True) -> None:
        super().__init__()
        self.register_buffer("m0", torch.tensor([[L / 2, L / 4 * 1.2]]))
        self.register_buffer("m1", torch.tensor([[L / 2, 3 * L / 4]]))
        self.kde = KDE(h)
        self.t_start = t_start
        self.t_end = t_end
        self.use_ff = use_ff
        self.D = D
        self.gmm = GMM(self.m0, torch.tensor([1.0]))

    def _get_gmm_params(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        if t < self.t_start:
            means = self.m0
        elif t > self.t_end:
            means = self.m1
        else:
            h = (t - self.t_start) / (self.t_end - self.t_start)
            means = (1 - h) * self.m0 + h * self.m1
        return means, torch.tensor([1.0])

    def get_contoll(self, t: float) -> torch.Tensor:
        t_tensor = torch.tensor(t).type_as(self.state).requires_grad_(True)
        m, w = self._get_gmm_params(t_tensor)
        R = self.state.clone().requires_grad_(True)
        # desired density
        self.gmm.update_means(m.detach())
        f_d = self.gmm.eval_batch(R)
        # estimated density
        f_hat = self.kde.estimate_batch(R, self.state)
        # feedback
        phi = f_hat - f_d
        phi_grad = torch.autograd.grad(phi, R, torch.ones_like(phi))[0]
        feed_back = -self.D * phi_grad
        # feedforward
        if (
            self.use_ff
            and m.requires_grad  # use feedforward control  # m is a function of time
        ):
            comps = self.gmm.get_components_batch(self.state)
            feed_foward = torch.zeros_like(self.state)
            for i in range(self.state.size(0)):
                for j in range(2):
                    feed_foward[i, j] = torch.autograd.grad(
                        m[:, j], t_tensor, comps[i], retain_graph=True
                    )[0]
        else:
            feed_foward = 0
        f_hat = f_hat.unsqueeze(1)
        return ((feed_back + feed_foward) / f_hat).detach()


class WalkingManController(DensityController):
    "controller for tracking the walking man"

    def __init__(
        self,
        L: float,
        h: float,
        D: float,
        sigma_pixels=1,
        t_start=1,
        resolution=(64, 64),
        use_ff: bool = True,
    ) -> None:
        super().__init__()
        self.gmminterpolator = GMMInterpolator.walking_man(
            t_start, L, resolution=resolution
        )
        try:
            self.gmminterpolator.load_coeff(
                f"data/walking_man/resolution{resolution[0]}"
            )
        except:
            print(
                "Could not find the interpolation coefficients and have to compute the. This might take a while..."
            )
            print(
                "If this is not the desired behavior please download the precomputed coefficients as described in readme.md."
            )

            self.gmminterpolator.interpolate()
            self.gmminterpolator.save_coeff(
                f"data/walking_man/resolution{resolution[0]}"
            )
        # repeat three times
        self.gmminterpolator.extend(5 * 5)
        # make 4 times faster
        self.gmminterpolator.speedup(10)

        sigma = sigma_pixels * L / resolution[0]
        self.fd = GMM(*self.gmminterpolator.get_params(torch.tensor(0)), sigma)
        self.use_ff = use_ff
        self.t_start = t_start
        self.kde = KDE(h)
        self.D = D

    def get_contoll(self, t: float) -> torch.Tensor:
        t_tensor = torch.tensor(t).type_as(self.state).requires_grad_(True)
        m, w = self.gmminterpolator.get_params(t_tensor)
        R = self.state.clone().requires_grad_(True)
        # desired density
        self.fd.update_means(m.detach())
        self.fd.update_weights(w.detach())
        f_d = self.fd.eval_batch(R)
        # estimated density
        f_hat = self.kde.estimate_batch(R, self.state)
        # feedback
        phi = f_hat - f_d
        # time forloop
        phi_grad = torch.autograd.grad(phi, R, torch.ones_like(phi))[0]
        feed_back = -self.D * phi_grad
        # feedforward
        if (
            self.use_ff
            and m.requires_grad  # use feedforward control  # m is a function of time
        ):
            comps = self.fd.get_components_batch(self.state)
            feed_foward = torch.zeros_like(self.state)
            for i in range(self.state.size(0)):
                for j in range(2):
                    feed_foward[i, j] = torch.autograd.grad(
                        m[:, j], t_tensor, comps[i], retain_graph=True
                    )[
                        0
                    ]  # try with minus here
        else:
            feed_foward = 0

        f_hat = f_hat.unsqueeze(1)
        return ((feed_back + feed_foward) / f_hat).detach()
