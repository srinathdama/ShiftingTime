""" Stacks together a number of kernel integral transforms to create neural operators
"""

from functools import cache

import torch
from torch import nn, Tensor
from jaxtyping import Float
from typing import Optional
from .quadrature import quad_to_cartesian_grid

from .types import CartesianGrid, QuadGrid
from .integral_transforms import AffineMap, KhatriRaoIntTsfm


class KhatriRaoLayer(nn.Module):
    def __init__(
        self,
        d: int,
        p: int,
        q: int,
        kernel_layers: list[int],
        include_affine: bool = False,
        reshape_output: bool = False,
        fourier_features: Optional[int] = None,
        stationary_kernel: bool = False,
    ) -> None:
        super().__init__()
        self.int_tsfm = KhatriRaoIntTsfm(d, p, q, kernel_layers, fourier_features, stationary_kernel)
        if include_affine:
            self.linear = AffineMap(p, q)
        self.include_affine = include_affine
        self.reshape_output = reshape_output
    def forward(
        self, cartesian_grid: CartesianGrid, u: Float[Tensor, "batch pN"]
    ) -> Float[Tensor, "batch qN"]:
        if self.reshape_output:
            # reshape input (batch, p, n_in_1, n_in_2, ..., n_in_d) to (batch pN)
            batch = u.shape[0]
            assert u.shape[1] == self.int_tsfm.p
            # ravel computational grid
            u = u.reshape(batch, -1)
            if self.include_affine:
                u = self.linear(u) + self.int_tsfm(cartesian_grid, u)
            u = self.int_tsfm(cartesian_grid, u)
            # reshape output (batch qN) to (batch, q, n_in_1, n_in_2, ..., n_out_d)
            grid_dims = tuple([len(wq) for wq in cartesian_grid[0]])
            u = u.reshape(batch, self.int_tsfm.q, *grid_dims)
            return u
        else:
            if self.include_affine:
                return self.linear(u) + self.int_tsfm(cartesian_grid, u)
            return self.int_tsfm(cartesian_grid, u)

    def super_resolution(
        self, out_grid: QuadGrid, in_grid: QuadGrid, u: Float[Tensor, "batch pM"]
    ) -> Float[Tensor, "batch qN"]:
        if self.reshape_output:
            # reshape input (batch, p, n_in_1, n_in_2, ..., n_in_d) to (batch pN)
            batch = u.shape[0]
            assert u.shape[1] == self.int_tsfm.p
            # ravel computational grid
            u = u.reshape(batch, -1)
            if self.include_affine:
                raise ValueError("super_resolution not allowed for this model")
            u = self.int_tsfm.super_resolution(out_grid, in_grid, u)
            # reshape output (batch qN) to (batch, q, n_in_1, n_in_2, ..., n_out_d)
            grid_dims = tuple([len(wq) for wq in out_grid[1]])
            u = u.reshape(batch, self.int_tsfm.q, *grid_dims)
            return u
        else:
            if self.include_affine:
                raise ValueError("super_resolution not allowed for this model")
            return self.int_tsfm.super_resolution(out_grid, in_grid, u)


class KhatriRaoNO(nn.Module):
    """
    Args:
        d (int): dimension of domain (i.e. u(x,y) would be 2)
        p (list[int]): sequence of NO function dimensions. p[0] should be the dimension
                       of the input function and p[-1] should be the dimension of the
                       output function
        kernel_layers (list[list[int]]): the number of hidden units in each layer for
                                         example, kernel_layers[i] would correspond
                                         to the sequence of hidden units in layer[i]
    """

    def __init__(
        self,
        d: int,
        lifting_layers: list[int],  # affine maps
        integral_layers: list[int],  # integral transforms
        kernel_layers: list[list[int]],  # nn layers for integral transforms
        projection_layers: list[int],  # affine maps
        nonlinearity: nn.Module,
        include_affine: bool = True,
        affine_in_first_integral_tsfm: bool = False,
    ) -> None:
        super().__init__()
        # checking inputs are provided in the correct format
        assert lifting_layers[-1] == integral_layers[0]
        assert integral_layers[-1] == projection_layers[0]
        if len(integral_layers) != len(kernel_layers) + 1:
            raise ValueError("len(kernel_layers) + 1 should = len(p)")
        self.d = d
        self.nonlinearity = nonlinearity
        self.lifting_layers = lifting_layers
        self.integral_layers = integral_layers
        self.projection_layers = projection_layers
        # affine layers
        self.lifting_map = nn.ModuleList(
            [
                AffineMap(lifting_layers[i], lifting_layers[i + 1])
                for i in range(len(lifting_layers) - 1)
            ]
        )
        # integral layers
        integral_map = []
        for i in range(len(integral_layers) - 1):
            integral_map.append(
                KhatriRaoLayer(  # type: ignore
                    d,
                    integral_layers[i],
                    integral_layers[i + 1],
                    kernel_layers[i],
                    include_affine=(
                        include_affine if i > 0 else affine_in_first_integral_tsfm
                    ),
                )
            )
        self.integral_map = nn.ModuleList(integral_map)
        # projection layers
        self.projection_map = nn.ModuleList(
            [
                AffineMap(projection_layers[i], projection_layers[i + 1])
                for i in range(len(projection_layers) - 1)
            ]
        )

    @cache
    def inputs_are_correct_shape(
        self, grid_dims: tuple[int], u_size: tuple[int]
    ) -> bool:
        correct_shape = len(grid_dims) == self.d
        for j, nbar in enumerate(grid_dims):
            correct_shape = (nbar == u_size[j + 1]) and correct_shape
        return correct_shape

    def apply_lifting(self, u: Float[Tensor, "batch pN"]) -> Float[Tensor, "batch qN"]:
        actns = u
        for layer in self.lifting_map:
            u = layer(actns)
            # actns = u
            actns = self.nonlinearity(u)
        return u

    def apply_int_tsfms(
        self,
        cartesian_grid: CartesianGrid,
        integral_map: nn.ModuleList,
        u: Float[Tensor, "batch pN"],
    ) -> Float[Tensor, "batch qN"]:
        actns = u
        for layer in integral_map:
            u = layer(cartesian_grid, actns)
            actns = self.nonlinearity(u)
        return u

    def apply_projection(
        self, u: Float[Tensor, "batch pN"]
    ) -> Float[Tensor, "batch qN"]:
        actns = u
        for layer in self.projection_map:
            u = layer(actns)
            actns = self.nonlinearity(u)
        return u

    def forward(
        self, cartesian_grid: CartesianGrid, u: Float[Tensor, "batch ... p"]
    ) -> Float[Tensor, "batch ... q"]:
        grid_dims = tuple([len(wq) for wq in cartesian_grid[0]])
        assert self.inputs_are_correct_shape(grid_dims, tuple(u.shape))
        assert u.shape[-1] == self.lifting_layers[0]
        batch = u.shape[0]
        # ravel computational grid
        u = u.reshape(batch, -1, self.lifting_layers[0])
        u = u.transpose(1, 2).reshape(batch, -1)
        u = self.apply_lifting(u)
        u = self.apply_int_tsfms(cartesian_grid, self.integral_map, u)
        u = self.apply_projection(u)
        u = u.reshape(batch, self.projection_layers[-1], -1).transpose(1, 2)
        return u.reshape(batch, *grid_dims, self.projection_layers[-1])

    def super_resolution(
        self, out_grid: QuadGrid, in_grid: QuadGrid, u: Float[Tensor, "batch ... p"]
    ) -> Float[Tensor, "batch ... q"]:
        grid_dims = tuple([len(wq) for wq in in_grid[0]])
        assert self.inputs_are_correct_shape(grid_dims, tuple(u.shape))
        assert u.shape[-1] == self.lifting_layers[0]
        batch = u.shape[0]
        # ravel computational grid
        u = u.reshape(batch, -1, self.lifting_layers[0])
        # flatten for quadrature
        u = u.transpose(1, 2).reshape(batch, -1)
        u = self.apply_lifting(u)
        # first integral_map requires expanding the full krao prod
        u = self.nonlinearity(
            self.integral_map[0].super_resolution(out_grid, in_grid, u)
        )
        # now use kronecker structure in subsequent layers
        cartesian_grid = quad_to_cartesian_grid(out_grid)
        if len(self.integral_map) > 1:
            u = self.apply_int_tsfms(cartesian_grid, self.integral_map[1:], u)  # type: ignore
        u = self.apply_projection(u)
        u = u.reshape(batch, self.projection_layers[-1], -1).transpose(1, 2)
        out_grid_dims = tuple([len(wq) for wq in out_grid[0]])
        return u.reshape(batch, *out_grid_dims, self.projection_layers[-1])

    @classmethod
    def easy_init(
        cls,
        d: int,
        in_channels: int,
        out_channels: int,
        lifting_channels: int,
        integral_channels: int,
        n_integral_layers: int,
        projection_channels: int,
        n_hidden_units: int,
        n_hidden_layers: int,
        nonlinearity: nn.Module,
        include_affine: bool = True,
        affine_in_first_integral_tsfm: bool = False,
    ):
        lifting_layers = [in_channels, lifting_channels, integral_channels]
        # lifting_layers = [in_channels, integral_channels]
        # applies n_integral nonlinear transforms
        integral_layers = [integral_channels] * (n_integral_layers + 1)
        kernel_layers = [[n_hidden_units] * n_hidden_layers] * n_integral_layers
        projection_layers = [integral_channels, projection_channels, out_channels]
        return cls(
            d,
            lifting_layers,
            integral_layers,
            kernel_layers,
            projection_layers,
            nonlinearity,
            include_affine=include_affine,
            affine_in_first_integral_tsfm=affine_in_first_integral_tsfm,
        )


class KhatriRaoNO_v2(nn.Module):
    """
    This is a modified version of the KhatriRaoNO class that allows to have a variable intermediate grid.
    See super_resolution function for more details.

    Args:
        d (int): dimension of domain (i.e. u(x,y) would be 2)
        p (list[int]): sequence of NO function dimensions. p[0] should be the dimension
                       of the input function and p[-1] should be the dimension of the
                       output function
        kernel_layers (list[list[int]]): the number of hidden units in each layer for
                                         example, kernel_layers[i] would correspond
                                         to the sequence of hidden units in layer[i]
    """

    def __init__(
        self,
        d: int,
        lifting_layers: list[int],  # affine maps
        integral_layers: list[int],  # integral transforms
        kernel_layers: list[list[int]],  # nn layers for integral transforms
        projection_layers: list[int],  # affine maps
        nonlinearity: nn.Module,
        include_affine: bool = True,
        affine_in_first_integral_tsfm: bool = False,
    ) -> None:
        super().__init__()
        # checking inputs are provided in the correct format
        assert lifting_layers[-1] == integral_layers[0]
        assert integral_layers[-1] == projection_layers[0]
        if len(integral_layers) != len(kernel_layers) + 1:
            raise ValueError("len(kernel_layers) + 1 should = len(p)")
        self.d = d
        self.nonlinearity = nonlinearity
        self.lifting_layers = lifting_layers
        self.integral_layers = integral_layers
        self.projection_layers = projection_layers
        # affine layers
        self.lifting_map = nn.ModuleList(
            [
                AffineMap(lifting_layers[i], lifting_layers[i + 1])
                for i in range(len(lifting_layers) - 1)
            ]
        )
        # integral layers
        integral_map = []
        for i in range(len(integral_layers) - 1):
            integral_map.append(
                KhatriRaoLayer(  # type: ignore
                    d,
                    integral_layers[i],
                    integral_layers[i + 1],
                    kernel_layers[i],
                    include_affine=(
                        include_affine if ( i > 0 and i < len(kernel_layers) - 1 ) else affine_in_first_integral_tsfm
                    ),
                )
            )
        self.integral_map = nn.ModuleList(integral_map)
        # projection layers
        self.projection_map = nn.ModuleList(
            [
                AffineMap(projection_layers[i], projection_layers[i + 1])
                for i in range(len(projection_layers) - 1)
            ]
        )

    @cache
    def inputs_are_correct_shape(
        self, grid_dims: tuple[int], u_size: tuple[int]
    ) -> bool:
        correct_shape = len(grid_dims) == self.d
        for j, nbar in enumerate(grid_dims):
            correct_shape = (nbar == u_size[j + 1]) and correct_shape
        return correct_shape

    def apply_lifting(self, u: Float[Tensor, "batch pN"]) -> Float[Tensor, "batch qN"]:
        actns = u
        for layer in self.lifting_map:
            u = layer(actns)
            # actns = u
            actns = self.nonlinearity(u)
        return u

    def apply_int_tsfms(
        self,
        cartesian_grid: CartesianGrid,
        integral_map: nn.ModuleList,
        u: Float[Tensor, "batch pN"],
    ) -> Float[Tensor, "batch qN"]:
        actns = u
        for layer in integral_map:
            u = layer(cartesian_grid, actns)
            actns = self.nonlinearity(u)
        return u

    def apply_projection(
        self, u: Float[Tensor, "batch pN"]
    ) -> Float[Tensor, "batch qN"]:
        actns = u
        for layer in self.projection_map:
            u = layer(actns)
            actns = self.nonlinearity(u)
        return u

    def forward(
        self, cartesian_grid: CartesianGrid, u: Float[Tensor, "batch ... p"]
    ) -> Float[Tensor, "batch ... q"]:
        grid_dims = tuple([len(wq) for wq in cartesian_grid[0]])
        assert self.inputs_are_correct_shape(grid_dims, tuple(u.shape))
        assert u.shape[-1] == self.lifting_layers[0]
        batch = u.shape[0]
        # ravel computational grid
        u = u.reshape(batch, -1, self.lifting_layers[0])
        u = u.transpose(1, 2).reshape(batch, -1)
        u = self.apply_lifting(u)
        u = self.apply_int_tsfms(cartesian_grid, self.integral_map, u)
        u = self.apply_projection(u)
        u = u.reshape(batch, self.projection_layers[-1], -1).transpose(1, 2)
        return u.reshape(batch, *grid_dims, self.projection_layers[-1])

    def super_resolution(
        self, out_grid: QuadGrid, in_grid: QuadGrid,
         latent_grid: QuadGrid, u: Float[Tensor, "batch ... p"]
    ) -> Float[Tensor, "batch ... q"]:
        grid_dims = tuple([len(wq) for wq in in_grid[0]])
        assert self.inputs_are_correct_shape(grid_dims, tuple(u.shape))
        assert u.shape[-1] == self.lifting_layers[0]
        batch = u.shape[0]
        # ravel computational grid
        u = u.reshape(batch, -1, self.lifting_layers[0])
        # flatten for quadrature
        u = u.transpose(1, 2).reshape(batch, -1)
        u = self.apply_lifting(u)
        # first integral_map requires expanding the full krao prod
        u = self.nonlinearity(
            self.integral_map[0].super_resolution(latent_grid, in_grid, u)
        )
        # now use kronecker structure in subsequent layers
        cartesian_grid = quad_to_cartesian_grid(latent_grid)
        if len(self.integral_map) > 2:
            u = self.apply_int_tsfms(cartesian_grid, self.integral_map[1:-1], u)  # type: ignore
        # last integral_map requires expanding the full krao prod
        u = self.nonlinearity(
            self.integral_map[-1].super_resolution(out_grid, latent_grid, u)
        )
        u = self.apply_projection(u)
        u = u.reshape(batch, self.projection_layers[-1], -1).transpose(1, 2)
        out_grid_dims = tuple([len(wq) for wq in out_grid[0]])
        return u.reshape(batch, *out_grid_dims, self.projection_layers[-1])

    @classmethod
    def easy_init(
        cls,
        d: int,
        in_channels: int,
        out_channels: int,
        lifting_channels: int,
        integral_channels: int,
        n_integral_layers: int,
        projection_channels: int,
        n_hidden_units: int,
        n_hidden_layers: int,
        nonlinearity: nn.Module,
        include_affine: bool = True,
        affine_in_first_integral_tsfm: bool = False,
    ):
        lifting_layers = [in_channels, lifting_channels, integral_channels]
        # lifting_layers = [in_channels, integral_channels]
        # applies n_integral nonlinear transforms
        integral_layers = [integral_channels] * (n_integral_layers + 1)
        kernel_layers = [[n_hidden_units] * n_hidden_layers] * n_integral_layers
        projection_layers = [integral_channels, projection_channels, out_channels]
        return cls(
            d,
            lifting_layers,
            integral_layers,
            kernel_layers,
            projection_layers,
            nonlinearity,
            include_affine=include_affine,
            affine_in_first_integral_tsfm=affine_in_first_integral_tsfm,
        )
