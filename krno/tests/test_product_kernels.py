import math

import torch
from torch.func import vmap  # type: ignore
from khatriraonop import product_kernels, quadrature


def test_fcnn_shape():
    """implementation is pretty standard so just checking that
    things are initialized like we would expect
    """

    in_dim, out_dim, layers = 3, 5, [7, 11, 13]
    model = product_kernels.FCNN(in_dim, out_dim, layers)

    x = torch.randn(32, in_dim)
    assert tuple(model(x).shape) == (32, out_dim)


def test_nn_khatrirao_kernel():
    d = 2
    q = 3
    p = 4
    grid = quadrature.get_quad_grid(
        quadrature.trapezoidal_vecs, [3, 5], [-5, -2], [-1, 3]
    )
    _, xq_grid = grid
    grid_kernel = product_kernels.NNKhatriRaoKernel(d, q, p, layers=[7, 9, 13])
    cart_grid = quadrature.quad_to_cartesian_grid(grid)
    kernel = grid_kernel(cart_grid)
    kmats = kernel._matrices
    for i, xq in enumerate(xq_grid):

        def nn_reshape(xbatch):
            xbatch = grid_kernel.fourier(xbatch.unsqueeze(0))
            return grid_kernel.prod_kernel[i](xbatch).reshape(q, p) / math.sqrt(p)

        n_bar = len(xq_grid[i])
        for j in range(n_bar):
            for k in range(n_bar):
                x_jk = torch.tensor([xq[j], xq[k]])
                # print((nn_reshape(x_jk) - kmats[i][:, :, j, k]))
                assert torch.allclose(nn_reshape(x_jk), kmats[i][:, :, j, k], atol=1e-4)


def block_to_mat(K_block, q, p):
    return torch.cat(
        [torch.cat([K_block[j][k] for k in range(p)], dim=1) for j in range(q)],
        dim=0,
    )


def test_block_tensor_to_mat():
    q, p = 5, 2
    block = torch.randn(q, p, 64, 32)
    assert torch.allclose(
        product_kernels.block_tensor_to_mat(block), block_to_mat(block, q, p)
    )


def test_nn_khatrirao_super_res():
    d, q, p = 2, 3, 4
    grid_kernel = product_kernels.NNKhatriRaoKernel(d, q, p, layers=[7, 9, 13])
    in_quad = quadrature.get_quad_grid(
        quadrature.trapezoidal_vecs, [3, 5], [-5, -2], [-1, 3]
    )
    krao_mat = grid_kernel(quadrature.quad_to_cartesian_grid(in_quad))
    # check things are correct for non super res case
    K_sres = grid_kernel.super_resolution(in_quad, in_quad)
    assert torch.allclose(krao_mat.full_matrix, K_sres.full_matrix, atol=1e-4)

    # now try super res case
    out_quad = quadrature.get_quad_grid(
        quadrature.trapezoidal_vecs, [6, 10], [-5, -2], [-1, 3]
    )
    (_, xq_in), (_, xq_out) = in_quad, out_quad
    x_in, x_out = torch.cartesian_prod(*xq_in), torch.cartesian_prod(*xq_out)
    n_in, n_out = len(x_in), len(x_out)
    block_tensor = [[torch.ones(n_out, n_in) for _ in range(p)] for _ in range(q)]

    def shape_output(K):
        return K.reshape(q, p) / math.sqrt(p)

    for i in range(d):
        x_i = torch.cartesian_prod(x_out[:, i], x_in[:, i])
        x_i = grid_kernel.fourier(x_i)
        K_qp = grid_kernel.prod_kernel[i](x_i)
        K_qp = vmap(shape_output, in_dims=1)(K_qp.t())
        for j in range(q):
            for k in range(p):
                block_tensor[j][k] *= K_qp[:, j, k].reshape(n_out, n_in)

    K_sres = grid_kernel.super_resolution(out_quad, in_quad)
    assert torch.allclose(block_to_mat(block_tensor, q, p), K_sres.full_matrix)


if __name__ == "__main__":
    test_nn_khatrirao_super_res()
    test_nn_khatrirao_kernel()
    test_block_tensor_to_mat()
    test_fcnn_shape()
