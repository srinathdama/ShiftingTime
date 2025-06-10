from jaxtyping import Float
from typing import Optional
import torch
from torch import nn, Tensor

from .kronecker_algebra import KronMatrix
from .types import CartesianGrid, QuadGrid
from .product_kernels import NNKhatriRaoKernel


class KhatriRaoIntTsfm(nn.Module):
    def __init__(self, d: int, p: int, q: int, kernel_layers: list[int],
                 fourier_features: Optional[int] = None,
                 stationary_kernel: bool = False):
        super().__init__()
        self.d = d
        self.p = p
        self.q = q
        # grid structured khatri-rao kernel
        self.grid_kernel = NNKhatriRaoKernel(d, q, p, kernel_layers, fourier_features, stationary_kernel)

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


# TODO: implement SKI based kernel interpolation
# class InterpKhatriRaoIntTsfm(nn.Module):
#     # todo: update n_bar_list_quad method?
#     def __init__(self, d: int, p: int, q: int, kernel_layers: list[int]):
#         super().__init__()
#         self.d = d
#         self.p = p
#         self.q = q
#         # n_bar sets the integration degree
#         wq_grid, xq_grid = self._compute_input_quadrature(
#             n_bar_list_in, lb_list, ub_list, quad_type="trapezoidal"
#         )
#         for j in range(len(wq_grid)):
#             self.register_buffer("wq_{}".format(j), wq_grid[j])
#             self.register_buffer("diag_wq_{}".format(j), torch.diag(wq_grid[j]))
#             self.register_buffer("xq_{}".format(j), xq_grid[j])
#         # getting GL quadrature points
#         _, xq_gl_grid = self._compute_input_quadrature(
#             n_bar_list_quad, lb_list, ub_list, quad_type="gauss-legendre"
#         )
#         # todo: should this be registered?
#         for j in range(len(xq_gl_grid)):
#             self.register_buffer("xq_gl_{}".format(j), xq_gl_grid[j])
#         x_in = grid_to_tensor(xq_gl_grid)
#         # interpolating with SKI
#         interp_indices, interp_values = Interpolation().interpolate(xq_grid, x_in)
#         self.register_buffer("interp_indices", interp_indices)
#         self.register_buffer("interp_values", interp_values)
#         # grid structured khatri-rao kernel
#         self.grid_kernel = NNKhatriRaoKernel(
#             d, q, p, xq_grid, nn_hidden=nn_hidden, nn_layers=nn_layers
#         )

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Transforms x -> y using approximate Khatri-Rao

#         Args:
#             x (Tensor(bs, pN)): input function evaluated at quadrature nodes

#         Returns:
#             Tensor(bs, qN): output function evaluated at quadrature nodes
#         """
#         # check v is the expected shape
#         (bs, pN) = x.shape
#         assert self.p * self.N == pN
#         # K should be a 3D kernel of size d, n_bar, n_bar
#         K = self.grid_kernel()
#         assert len(K) == self.d
#         # multiplying by quadrature weights
#         diag_wq = [getattr(self, "diag_wq_{}".format(j)) for j in range(self.d)]
#         v = ident_kron_mmprod(self.p, diag_wq, x.t()).t()
#         # multiplying by khatri-rao product
#         v = khatri_rao_mmprod(K, v.t()).t()
#         # multiplying by khatri-rao product
#         ind = self.interp_indices
#         val = self.interp_values
#         v = ident_kron_interp(self.q, ind, val, v.t()).t()
#         return v

#     def _compute_input_quadrature(
#         self,
#         n_bar_list: list[int],
#         lb_list: list[float],
#         ub_list: list[float],
#         quad_type: Optional[str] = "trapezoidal",
#     ) -> Tuple[list[Tensor], list[Tensor]]:
#         """
#         Computes 1D quadrature weights along each dimension and quadrature grid

#         Args:
#             n_bar_list (list[int]): number of quadrature points along each dimension
#             lb_list (list[float]): lower bound of domain along each dimension
#             ub_list (list[float]): upper bound of domain along each dimension
#             quad_type (str): quadrature type either ("trapezoidal" or "gauss-legendre")

#         Returns:
#             Tuple[list[Tensor], list[Tensor]]: (quadrature weights, quadrature grid)
#         """
#         # confirm bounds are formatted correctly
#         assert len(n_bar_list) == len(lb_list)
#         assert len(lb_list) == len(ub_list)
#         # 1d quadrature nodes and weights for each dim
#         wq = []
#         xq = []
#         for j in range(len(n_bar_list)):
#             if quad_type == "trapezoidal":
#                 wq_tmp, xq_tmp = trapezoidal_vecs(
#                     n_bar_list[j], a=lb_list[j], b=ub_list[j], dtype=self.dtype
#                 )
#             elif quad_type == "gauss-legendre":
#                 wq_tmp, xq_tmp = gauss_legendre_vecs(
#                     n_bar_list[j], a=lb_list[j], b=ub_list[j], dtype=self.dtype
#                 )
#             wq.append(wq_tmp)
#             xq.append(xq_tmp)
#         return (wq, xq)

#     def update_quadrature(self, n_bar_list: list[int]) -> None:
#         # updating N
#         N = int(np.array(n_bar_list).prod())
#         self.N = N
#         # get device and dtype
#         wq = getattr(self, "wq_{}".format(0))
#         device = wq.device
#         dtype = wq.dtype
#         # get updated quadrature weights and nodes
#         wq_grid, xq_grid = self._compute_input_quadrature(
#             n_bar_list, self.lb_list, self.ub_list, quad_type="trapezoidal"
#         )
#         for j in range(len(n_bar_list)):
#             setattr(self, "wq_{}".format(j), wq_grid[j].to(dtype).to(device))
#             setattr(
#                 self,
#                 "diag_wq_{}".format(j),
#                 torch.diag(wq_grid[j]).to(dtype).to(device),
#             )
#             setattr(self, "xq_{}".format(j), xq_grid[j].to(dtype).to(device))
#         # updating the grid kernel
#         self.grid_kernel.update_quadrature(n_bar_list, xq_grid)
#         # updating interpolation
#         xq_gl_grid = [getattr(self, f"xq_gl_{j}") for j in range(self.d)]
#         x_in = grid_to_tensor(xq_gl_grid).cpu()
#         # interpolating with SKI
#         interp_indices, interp_values = Interpolation().interpolate(xq_grid, x_in)
#         self.register_buffer("interp_indices", interp_indices.to(device))
#         self.register_buffer("interp_values", interp_values.to(device))

#     def update_interpolation(self, n_bar_list_quad: list[int]) -> None:
#         valdtype = self.interp_values.dtype
#         indtype = self.interp_indices.dtype
#         device = self.interp_values.device
#         _, xq_gl_grid = self._compute_input_quadrature(
#             n_bar_list_quad, self.lb_list, self.ub_list, quad_type="gauss-legendre"
#         )
#         x_in = grid_to_tensor(xq_gl_grid)
#         xq_grid = [getattr(self, "xq_{}".format(j)) for j in range(self.d)]
#         interp_indices, interp_values = Interpolation().interpolate(xq_grid, x_in)
#         setattr(self, "interp_indices", interp_indices.to(indtype).to(device))
#         setattr(self, "interp_values", interp_values.to(valdtype).to(device))
