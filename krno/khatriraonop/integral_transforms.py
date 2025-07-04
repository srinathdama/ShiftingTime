from jaxtyping import Float
from typing import Optional
import torch
from torch import nn, Tensor

from .kronecker_algebra import KronMatrix
from .types import CartesianGrid, QuadGrid
from .product_kernels import NNKhatriRaoKernel


class KhatriRaoIntTsfm(nn.Module):
    def __init__(self, d: int, p: int, q: int, kernel_layers: list[int],
                 fourier_features: Optional[int] = None):
        super().__init__()
        self.d = d
        self.p = p
        self.q = q
        # grid structured khatri-rao kernel
        self.grid_kernel = NNKhatriRaoKernel(d, q, p, kernel_layers, fourier_features)

    def forward(
        self, cartesian_grid: CartesianGrid, u: Float[Tensor, "batch pN"]
    ) -> Float[Tensor, "batch qN"]:
        """
        Transforms a function v(y) = int K(y,x) u(x) dx using KhatriRao structured NN kernel.

        Args:
            x (Tensor(bs, pN)): input function evaluated at quadrature nodes

        Returns:
            Tensor(bs, qN): output function evaluated at quadrature nodes
        """
        khatri_rao_mat = self.grid_kernel(cartesian_grid)
        assert len(khatri_rao_mat) == self.d
        wq_grid, _ = cartesian_grid
        kron_wq = KronMatrix([wq.diag() for wq in wq_grid])
        return (khatri_rao_mat @ kron_wq.ident_prekron(self.p, u.T)).T

    ## old code 
    # def super_resolution(
    #     self,
    #     out_grid: QuadGrid,
    #     in_grid: QuadGrid,
    #     u: Float[Tensor, "batch pM"],
    # ) -> Float[Tensor, "batch qN"]:
    #     wq_in, _ = in_grid
    #     kron_wq = KronMatrix([wq.diag() for wq in wq_in])
    #     # apply quadrature to input
    #     quad_u = kron_wq.ident_prekron(self.p, u.T)
    #     return (self.grid_kernel.super_resolution(out_grid, in_grid) @ quad_u).T
    
    def super_resolution(
        self,
        out_grid: QuadGrid,
        in_grid: QuadGrid,
        u: Float[Tensor, "batch pM"],
    ) -> Float[Tensor, "batch qN"]:
        khatri_rao_mat = self.grid_kernel.super_resolution(out_grid, in_grid)
        assert len(khatri_rao_mat) == self.d
        wq_in, _ = in_grid
        kron_wq = KronMatrix([wq.diag() for wq in wq_in])
        return (khatri_rao_mat @ kron_wq.ident_prekron(self.p, u.T)).T


class AffineMap(nn.Module):
    """Note: this layer type can't be used with super resolution"""

    def __init__(self, p: int, q: int) -> None:
        super().__init__()
        self.p, self.q = p, q
        self.linear = nn.Linear(p, q)

    def forward(self, u: Float[Tensor, "batch pN"]) -> Float[Tensor, "batch qN"]:
        """Transforms a function using affine mapping v(x) = Au(x) + b

        Args:
            x (Tensor(bs, pN)): input function evaluated at quadrature nodes

        Returns:
            Tensor(bs, qN): output function evaluated at quadrature nodes
        """
        batch, _ = u.shape
        u = u.reshape(batch, self.p, -1).transpose(1, 2)
        return self.linear(u).transpose(1, 2).reshape(batch, -1)

class AffineMapGrid(nn.Module):
    """Note: this layer type can't be used with super resolution"""

    def __init__(self, L: int, T: int) -> None:
        super().__init__()
        self.L, self.T = L, T
        self.linear = nn.Linear(L, T)
        # self.linear.weight = nn.Parameter(
        #         (1 / self.L) * torch.ones([self.T, self.L]))

    def forward(self, u: Float[Tensor, "batch pL"]) -> Float[Tensor, "batch qT"]:
        """Transforms a function using affine mapping v(x) = Au(x) + b

        Args:
            x (Tensor(bs, pL)): input function evaluated at quadrature nodes

        Returns:
            Tensor(bs, qT): output function evaluated at quadrature nodes
        """
        batch, _ = u.shape
        u = u.reshape(batch, -1, self.L).transpose(1, 2) # batch, L, p
        u = self.linear(u.permute(0, 2, 1)).permute(0, 2, 1) # batch, T, p
        return u.transpose(1, 2).reshape(batch, -1)

