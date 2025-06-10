import math

import torch
from torch import nn, Tensor
from jaxtyping import Float
from typing import Optional

from .types import CartesianGrid, QuadGrid
from .kronecker_algebra import KhatriRaoMatrix
import numpy as np
import warnings

class LogexpTransformation:
    """ apply log transformation to positive parameters for optimization """
    _lim_val = 36.
    _log_lim_val = torch.log(torch.tensor(torch.finfo(torch.float64).max))


    def inverse_transform(self, x):
        x2 = x.clone()
        return torch.where(x2 > self._lim_val, x2, torch.log1p(torch.exp(torch.clip(x2, -self._log_lim_val, self._lim_val))))
        # return torch.log1p(torch.exp(torch.clip(x, -self._log_lim_val, self._lim_val)))  #torch.log1p(torch.exp(x))
        # bool_indx = x < self._lim_val
        # act_indx  = torch.nonzero(bool_indx, as_tuple=True)[0]
        # x2 = x.clone()
        # if len(act_indx) > 0:
        #     x2[act_indx] =  torch.log1p(torch.exp(torch.clip(x2[act_indx], -self._log_lim_val, self._lim_val)))
        # return x2  


    def transform(self, f):
        f2 = f.clone()
        return torch.where(f > self._lim_val, f, torch.log(torch.expm1(f)))
        # return torch.log(torch.expm1(f))
        # bool_indx = f < self._lim_val
        # act_indx  = torch.nonzero(bool_indx, as_tuple=True)[0]
        # f2 = f.clone()
        # if len(act_indx) > 0:
        #     f2[act_indx] =  torch.log(torch.expm1(f2[act_indx]))
        # return f2 


    def transform_grad(self, f, grad_f):
        return grad_f*torch.where(f > self._lim_val, 1.,  - np.expm1(-f))


class LinTransformation:
    """ identity transformation """


    def inverse_transform(self, x):
        return x


    def transform(self, f):
        return f


class RBF_Kernel(torch.nn.Module):
    """RBF kernel class

    """
    def __init__(self, D, sigma_f=None, l=None, transform='logexp'):
        """Initialize RBF kernel
        
        Args:
            D: input dimension 
            sigma_f: signal variance
            l: [D,] numpy array of length scales

        """
        super().__init__()
        self.D = D

        if transform=='logexp':
            self.transformation = LogexpTransformation()
        elif transform=='identity':
            self.transformation = LinTransformation()
        else:
            print('***Invalid transform!***')
            raise 

        if sigma_f is not None:
            assert np.size(sigma_f)==1, 'sigma_f should be scalar'
            sigma_f_raw      = self.transformation.transform(torch.tensor(sigma_f, dtype=torch.float64))
            self.sigma_f_raw = torch.nn.Parameter(sigma_f_raw)
        else:
            # self.sigma_f_raw = torch.nn.Parameter(torch.randn(1, generator=generator)) #initiate sigma with 1
            self.sigma_f_raw = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64 )) #initiate sigma with 0
        # self.sigma_f     = self.transformation.inverse_transform(self.sigma_f_raw)
        if l is not None:
            assert l.size == self.D
            l_raw        = self.transformation.transform(torch.tensor(l, dtype=torch.float64))
            self.l_raw   = torch.nn.Parameter(l_raw)
        else:
            # self.l_raw   = torch.nn.Parameter(torch.randn(self.D, generator=generator))
            self.l_raw   = torch.nn.Parameter(torch.zeros(self.D, dtype=torch.float64))
        # self.l     = self.transformation.inverse_transform(self.l_raw)
        # logger.debug("Initilizing %s kernel", self.__class__.__name__)

    @property
    def l(self):
        """ return actual l from raw l"""
        return self.transformation.inverse_transform(self.l_raw)

    @property
    def sigma_f(self):
        """ return actual sigma_f from raw sigma_f"""
        return self.transformation.inverse_transform(self.sigma_f_raw)

    def cov(self, X, Z=None, diag_shift = None):
        """Compute covariance
        
        Args:
            X: [N, D] numpy array of training input points
            Z(optional): [M, D] numpy array of test input points
 
        Returns:
            covariance matrix
        """
        assert X.ndim == 2, 'X should be a two dimensional array'
        assert X.shape[1] == self.D, 'X.shape[1] should be equal to D'

        if Z is not None:
            assert Z.ndim == 2, 'Z should be a two dimensional array'
            assert Z.shape[1] == self.D, 'Z.shape[1] should be equal to D'
        else:
            Z = X

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                inv_length = 1 / self.l**2
            except Warning as e:
                # logger.warning(" Divide by zero warning! correcting the denominator! ")
                inv_length = 1 / (self.l**2 + 1e-4)
        dist_matrix = torch.sum((torch.unsqueeze(X, 1) - torch.unsqueeze(Z, 0)) * inv_length *
                            (torch.unsqueeze(X, 1) - torch.unsqueeze(Z, 0)), axis = 2)
        
        if diag_shift is not None:
            cov_matrix  = self.sigma_f * torch.exp(-0.5*dist_matrix) + \
                             diag_shift*torch.eye(X.shape[0])
        else:
            cov_matrix  = self.sigma_f * torch.exp(-0.5*dist_matrix)

        # if diag_shift is not None:
        #     cov_matrix  = torch.exp(-0.5*dist_matrix) + \
        #                      diag_shift*torch.eye(X.shape[0])
        # else:
        #     cov_matrix  = torch.exp(-0.5*dist_matrix)

        return cov_matrix


class Matern_Kernel(torch.nn.Module):
    """Matern kernel class

    """

    def __init__(self, D, sigma_f=None, l=None, order = '3/2', transform='logexp'):
        """Initialize Matern kernel
        
        Args:
            D: input dimension 
            sigma_f: signal variance
            l: [D,] numpy array of length scales
            order (string): '3/2' or '5/2' or '1/2'

        """
        super().__init__()
        self.D     = D
        assert order in ['1/2', '3/2', '5/2'], 'Matern order is wrongly specified!'
        self.order = order
        if transform=='logexp':
            self.transformation = LogexpTransformation()
        elif transform=='identity':
            self.transformation = LinTransformation()
        else:
            print('***Invalid transform!***')
            raise 

        if sigma_f is not None:
            assert np.size(sigma_f)==1, 'sigma_f should be scalar'
            sigma_f_raw      = self.transformation.transform(torch.tensor(sigma_f, dtype=torch.float64))
            self.sigma_f_raw = torch.nn.Parameter(sigma_f_raw)
        else:
            # self.sigma_f_raw = torch.nn.Parameter(torch.randn(1, generator=generator)) #initiate sigma with 1
            self.sigma_f_raw = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64 )) #initiate sigma with 0
        # self.sigma_f     = self.transformation.inverse_transform(self.sigma_f_raw)
        if l is not None:
            assert l.size == self.D
            l_raw        = self.transformation.transform(torch.tensor(l, dtype=torch.float64))
            self.l_raw   = torch.nn.Parameter(l_raw)
        else:
            # self.l_raw   = torch.nn.Parameter(torch.randn(self.D, generator=generator))
            self.l_raw   = torch.nn.Parameter(torch.zeros(self.D, dtype=torch.float64))
        # self.l     = self.transformation.inverse_transform(self.l_raw)
        # logger.debug("Initilizing %s kernel", self.__class__.__name__)

    @property
    def l(self):
        """ return actual l from raw l"""
        return self.transformation.inverse_transform(self.l_raw)

    @property
    def sigma_f(self):
        """ return actual sigma_f from raw sigma_f"""
        return self.transformation.inverse_transform(self.sigma_f_raw)

    def cov(self, X, Z=None, diag_shift=None):
        """Compute covariance
        
        Args:
            X: [N, D] numpy array of training input points
            Z(optional): [M, D] numpy array of test input points
 
        Returns:
            covariance matrix
        """
        assert X.ndim == 2, 'X should be a two dimensional array'
        assert X.shape[1] == self.D, 'X.shape[1] should be equal to D'

        if Z is not None:
            assert Z.ndim == 2, 'Z should be a two dimensional array'
            assert Z.shape[1] == self.D, 'Z.shape[1] should be equal to D'
        else:
            # Z = np.copy(X)
            Z = X

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                inv_length = 1 / torch.square(self.l)
            except Warning as e:
                # logger.warning(
                #     " Divide by zero warning! correcting the denominator! ")
                inv_length = 1 / (torch.square(self.l) + 1e-4)
        
        dist_matrix_square = (torch.unsqueeze(X, 1) - torch.unsqueeze(Z, 0)) * inv_length * (torch.unsqueeze(X, 1) - torch.unsqueeze(Z, 0))
        dist_matrix_square = torch.sum(dist_matrix_square, axis=2) + 1e-10
        # print('dist_matrix_square - ', dist_matrix_square)
        # dist_matrix_square.register_hook(lambda t: print(f'hook :\n {t}'))
        # inv_length.register_hook(lambda t: print(f'hook :\n {t}'))

        dist_matrix   = torch.sqrt(dist_matrix_square)
        if self.order == '3/2':
            cov_matrix = self.sigma_f * \
                (1 + np.sqrt(3) * dist_matrix) * torch.exp(- np.sqrt(3)*dist_matrix)
        elif self.order == '5/2':
            cov_matrix = self.sigma_f * \
                (1 + np.sqrt(5) * dist_matrix + (5/3) *
                 dist_matrix_square) * torch.exp(- np.sqrt(5)*dist_matrix)
        elif self.order == '1/2':
            cov_matrix = self.sigma_f * torch.exp(-dist_matrix)
        # if self.order == '3/2':
        #     cov_matrix =  (1 + np.sqrt(3) * dist_matrix) * torch.exp(- np.sqrt(3)*dist_matrix)
        # elif self.order == '5/2':
        #     cov_matrix =  (1 + np.sqrt(5) * dist_matrix + (5/3) *
        #          dist_matrix_square) * torch.exp(- np.sqrt(5)*dist_matrix)
        # elif self.order == '1/2':
        #     cov_matrix =  torch.exp(-dist_matrix)

        if diag_shift is not None:
            cov_matrix = cov_matrix + diag_shift*torch.eye(X.shape[0])

        return cov_matrix


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
               fourier_features: Optional[int] = None, stationary_kernel: bool = False):
        super().__init__()
        self.d = d
        self.q = q
        self.p = p
        self.stationary_kernel_flag = stationary_kernel
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
        if self.stationary_kernel_flag:
            self.rbf_stationary_kernels = nn.ModuleList(
                [RBF_Kernel(1, sigma_f=0.5, l=np.array([0.4])) for _ in range(d)]
            )
            self.matern_stationary_kernels = nn.ModuleList(
                [Matern_Kernel(1, sigma_f=1.0, l=np.array([0.6]), order='5/2') for _ in range(d)]
            )

    def forward(self, cartesian_grid: CartesianGrid) -> KhatriRaoMatrix:
        K_xx = []
        wq_grid, cart_grid = cartesian_grid
        for j in range(self.d):
            n_bar = len(wq_grid[j])
            x_i = cart_grid[j]
            x_i = self.fourier(x_i)
            K_i = self.prod_kernel[j](x_i).t().reshape(self.q, self.p, n_bar, n_bar)
            if self.stationary_kernel_flag:
                raise ('use super_resolution instead with quadrature grid!')
                K_i_stationary = self.rbf_stationary_kernels[j].cov(xq_out[i].unsqueeze(1), xq_in[i].unsqueeze(1))
                K_i_stationary = K_i_stationary.unsqueeze(0).unsqueeze(0)
                K_i += K_i_stationary
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
            if self.stationary_kernel_flag:
                # use different stationary kernel for each output channel 
                K_i_rbf_cov = self.rbf_stationary_kernels[i].cov(xq_out[i].unsqueeze(1), xq_in[i].unsqueeze(1))
                K_i_rbf_cov = K_i_rbf_cov.unsqueeze(0).unsqueeze(0)
                K_i_matern_cov = self.matern_stationary_kernels[i].cov(xq_out[i].unsqueeze(1), xq_in[i].unsqueeze(1))
                K_i_matern_cov = K_i_matern_cov.unsqueeze(0).unsqueeze(0)
                K_i += 0.5*K_i_rbf_cov + 0.5*K_i_matern_cov
            K_xx.append(K_i / math.sqrt(self.p))
        return KhatriRaoMatrix(K_xx)
