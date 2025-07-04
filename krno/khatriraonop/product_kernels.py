import math

import torch
from torch import nn, Tensor
from jaxtyping import Float
from typing import Optional

from .types import CartesianGrid, QuadGrid
from .kronecker_algebra import KhatriRaoMatrix
import numpy as np
import warnings


class SkipLayer(nn.Module):
    def __init__(self, dim: int, nonlinearity: nn.Module, norm: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.nonlinearity = nonlinearity
        if norm:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Float[Tensor, "batch dim"]) -> Float[Tensor, "batch out_dim"]:
        return self.nonlinearity(self.linear(self.norm(x))) + x


silu = nn.SiLU()


class FCNN(nn.Module):
    """A fully connected neural network with ReLU activations."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layers: list[int],
        nonlinearity: nn.Module = silu,
    ) -> None:
        super().__init__()
        if len(layers)!=0:
            if in_dim != layers[0]:
                modules = [nn.Linear(in_dim, layers[0]), nonlinearity]
            else:
                modules = [SkipLayer(in_dim, nonlinearity, norm=False)]
            for l1, l2 in zip(layers[:-1], layers[1:]):
                if l1 != l2:
                    modules += [nn.LayerNorm(l1), nn.Linear(l1, l2), nonlinearity]
                else:
                    modules += [SkipLayer(l1, nonlinearity)]
            modules += [nn.LayerNorm(layers[-1]), nn.Linear(layers[-1], out_dim)]
        else:
            modules = [nn.Linear(in_dim, out_dim)]
        self.net = nn.Sequential(*modules)

    def forward(
        self, x: Float[Tensor, "batch in_dim"]
    ) -> Float[Tensor, "batch out_dim"]:
        return self.net(x)


def block_tensor_to_mat(block: Float[Tensor, "q p N M"]) -> Float[Tensor, "qN pM"]:
    return torch.cat(torch.unbind(block, dim=1), dim=2).flatten(0, 1)


class FourierFeatures(nn.Module):
    """Fourier features for a 2D input

    Args:
        n_fourier (int): number of fourier features
    """

    def __init__(
        self, in_dim: int, n_fourier: int, omega_std: float = 2 * math.pi
    ) -> None:
        super().__init__()
        omega = torch.randn(in_dim, n_fourier) * omega_std
        rotation = torch.rand(n_fourier) * 2 * math.pi
        self.n_fourier = n_fourier
        self.fourier_omega = nn.Parameter(omega)
        self.register_buffer("fourier_rotation", rotation)

    def forward(self, x: Float[Tensor, "batch"]) -> Float[Tensor, "batch dim"]:
        """Time features, placeholder for now."""
        features = x @ self.fourier_omega + self.fourier_rotation
        # dividing by sqrt(dim) is already done in the nn layers to come
        return features.cos() * math.sqrt(2)


class NNKhatriRaoKernel(nn.Module):
    def __init__(self, d: int, q: int, p: int, layers: list[int],
               fourier_features: Optional[int] = None):
        super().__init__()
        self.d = d
        self.q = q
        self.p = p
        # always first transform to Fourier features
        if fourier_features is None:
            fourier_features = layers[0]
        self.fourier = FourierFeatures(2, fourier_features)
        self.prod_kernel = nn.ModuleList(
            [FCNN(fourier_features, q * p, layers=layers[1:]) for _ in range(d)]
        )
        # self.prod_kernel = nn.ModuleList(
        #     [FCNN(2, q * p, layers=layers[:]) for _ in range(d)]
        # )

    def forward(self, cartesian_grid: CartesianGrid) -> KhatriRaoMatrix:
        K_xx = []
        wq_grid, cart_grid = cartesian_grid
        for j in range(self.d):
            n_bar = len(wq_grid[j])
            x_i = cart_grid[j]
            x_i = self.fourier(x_i)
            K_i = self.prod_kernel[j](x_i).t().reshape(self.q, self.p, n_bar, n_bar)
            K_xx.append(K_i / math.sqrt(self.p))
        return KhatriRaoMatrix(K_xx)

    ## old code
    # def super_resolution(
    #     self, out_grid: QuadGrid, in_grid: QuadGrid
    # ) -> Float[Tensor, "qN pM"]:
    #     (_, xq_in), (_, xq_out) = in_grid, out_grid
    #     x_in, x_out = torch.cartesian_prod(*xq_in), torch.cartesian_prod(*xq_out)
    #     if x_in.dim() == 1:
    #         x_in = x_in.unsqueeze(1)
    #     if x_out.dim() == 1:
    #         x_out = x_out.unsqueeze(1)
    #     n_in, n_out = len(x_in), len(x_out)
    #     prod_kern = torch.ones(self.q, self.p, n_out, n_in, device=x_in.device)
    #     for i in range(self.d):
    #         x_i = torch.cartesian_prod(x_out[:, i], x_in[:, i])
    #         x_i = self.fourier(x_i)
    #         K_i = self.prod_kernel[i](x_i).t().reshape(self.q, self.p, n_out, n_in)
    #         prod_kern *= K_i / math.sqrt(self.p)
    #     return block_tensor_to_mat(prod_kern)
    
    def super_resolution(
        self, out_grid: QuadGrid, in_grid: QuadGrid
    ) -> KhatriRaoMatrix:
        (_, xq_in), (_, xq_out) = in_grid, out_grid
        K_xx = []
        for i in range(self.d):
            n_in, n_out = len(xq_in[i]), len(xq_out[i])
            x_i = torch.cartesian_prod(xq_out[i], xq_in[i])
            x_i = self.fourier(x_i)
            K_i = self.prod_kernel[i](x_i).t().reshape(self.q, self.p, n_out, n_in)
            K_xx.append(K_i / math.sqrt(self.p))
        return KhatriRaoMatrix(K_xx)
