import torch
from torch import nn

from khatriraonop import models, quadrature


def test_khatri_rao_no():
    # ---------------------------------------------------------------------------------
    # setting up the computational grid

    # get 1D quad rules -- trapz assumes evenly spaced grid here we use a 10x10 grid
    grid = quadrature.get_quad_grid(
        quadrature.trapezoidal_vecs, [10, 10], [-1, -2], [2, 3]
    )
    # convert into a cartesian product grid used by khatri rao integral transforms
    cart_grid = quadrature.quad_to_cartesian_grid(grid)

    # ---------------------------------------------------------------------------------
    # setting up the model grid
    # dimensionality of the input domain
    d = 2
    # number of inputs function
    lifting_layers = [2, 128, 20]
    integral_layers = [20, 20, 20]
    projection_layers = [20, 128, 2]

    # 4 hidden layers, each with 2 hidden layers with 64 hidden units
    # n_hidden_units
    # n_hidden_layers
    kernel_layers = [[64, 64]] * 2
    # make helper method which infers the number of layers
    model1 = models.KhatriRaoNO(
        d, lifting_layers, integral_layers, kernel_layers, projection_layers, nn.ReLU()
    )
    model2 = models.KhatriRaoNO.easy_init(
        d,
        in_channels=2,
        out_channels=2,
        lifting_channels=128,
        integral_channels=20,
        n_integral_layers=2,
        projection_channels=128,
        n_hidden_units=64,
        n_hidden_layers=2,
        nonlinearity=nn.ReLU(),
    )
    assert model1.lifting_layers == model2.lifting_layers
    assert model1.integral_layers == model2.integral_layers
    assert model1.projection_layers == model2.projection_layers

    # ---------------------------------------------------------------------------------
    # generating some dummy data
    batch_size = 8

    u = torch.randn(batch_size, 10, 10, 2)

    # transform u -> v
    v = model1(cart_grid, u)
    assert tuple(v.shape) == (batch_size, 10, 10, 2)

    v = model2(cart_grid, u)
    assert tuple(v.shape) == (batch_size, 10, 10, 2)

    # ---------------------------------------------------------------------------------
    # now do superresolution

    out_grid = quadrature.get_quad_grid(
        quadrature.trapezoidal_vecs, [30, 30], [-1, -2], [2, 3]
    )
    v = model1.super_resolution(out_grid, grid, u)
    assert tuple(v.shape) == (batch_size, 30, 30, 2)
    v = model2.super_resolution(out_grid, grid, u)
    assert tuple(v.shape) == (batch_size, 30, 30, 2)


def test_khatri_rao_no_1D():
    # ---------------------------------------------------------------------------------
    # setting up the computational grid

    # get 1D quad rules -- trapz assumes evenly spaced grid here we use a 10x10 grid
    grid = quadrature.get_quad_grid(quadrature.trapezoidal_vecs, [10], [-1], [2])
    # convert into a cartesian product grid used by khatri rao integral transforms
    cart_grid = quadrature.quad_to_cartesian_grid(grid)

    # ---------------------------------------------------------------------------------
    # setting up the model grid
    # dimensionality of the input domain
    d = 1
    # number of inputs function
    lifting_layers = [1, 128, 20]
    integral_layers = [20, 20, 20]
    projection_layers = [20, 128, 1]

    # 4 hidden layers, each with 2 hidden layers with 64 hidden units
    # n_hidden_units
    # n_hidden_layers
    kernel_layers = [[64, 64]] * 2
    # make helper method which infers the number of layers
    model1 = models.KhatriRaoNO(
        d, lifting_layers, integral_layers, kernel_layers, projection_layers, nn.ReLU()
    )
    model2 = models.KhatriRaoNO.easy_init(
        d,
        in_channels=1,
        out_channels=1,
        lifting_channels=128,
        integral_channels=20,
        n_integral_layers=2,
        projection_channels=128,
        n_hidden_units=64,
        n_hidden_layers=2,
        nonlinearity=nn.ReLU(),
    )
    assert model1.lifting_layers == model2.lifting_layers
    assert model1.integral_layers == model2.integral_layers
    assert model1.projection_layers == model2.projection_layers

    # ---------------------------------------------------------------------------------
    # generating some dummy data
    batch_size = 8

    u = torch.randn(batch_size, 10, 1)

    # transform u -> v
    v = model1(cart_grid, u)
    assert tuple(v.shape) == (batch_size, 10, 1)

    v = model2(cart_grid, u)
    assert tuple(v.shape) == (batch_size, 10, 1)

    # ---------------------------------------------------------------------------------
    # now do superresolution

    out_grid = quadrature.get_quad_grid(quadrature.trapezoidal_vecs, [30], [-1], [2])
    v = model1.super_resolution(out_grid, grid, u)
    assert tuple(v.shape) == (batch_size, 30, 1)
    v = model2.super_resolution(out_grid, grid, u)
    assert tuple(v.shape) == (batch_size, 30, 1)


if __name__ == "__main__":
    test_khatri_rao_no()
