from khatriraonop import integral_transforms, quadrature
import torch
import numpy as np


def test_khatrirao_int_tsfm():
    d, p, q = 2, 11, 13
    assert d == 2, "test only works for d == 2"
    proj = integral_transforms.KhatriRaoIntTsfm(d, p, q, [16, 16])
    n_grid = [3, 5]
    grid = quadrature.get_quad_grid(
        quadrature.trapezoidal_vecs, n_grid, [-5, -2], [-1, 3]
    )
    cart_grid = quadrature.quad_to_cartesian_grid(grid)

    batch_size = 3
    x = torch.randn(batch_size, int(np.prod(n_grid) * p))

    proj_out = proj(cart_grid, x)

    khrmat = proj.grid_kernel(cart_grid)
    wq_grid, _ = cart_grid
    # [torch.Size([3, 2, 3, 3]), torch.Size([3, 2, 5, 5])]
    proj_out_man = []
    w_kron = torch.kron(wq_grid[0], wq_grid[1])
    for i in range(q):
        sub = torch.zeros(len(w_kron), batch_size)
        for j in range(p):
            k_kron = torch.kron(khrmat._matrices[0][i, j], khrmat._matrices[1][i, j])
            w_dim = len(w_kron)
            sub += (k_kron * w_kron.view(1, -1)) @ x[:, j * w_dim : (j + 1) * w_dim].T
        proj_out_man.append(sub)

    proj_out_man = torch.cat(proj_out_man, dim=0).T
    assert torch.allclose(proj_out, proj_out_man, atol=1e-3)


def test_khatrirao_int_tsfm_super_res():
    d, p, q = 2, 11, 13
    assert d == 2, "test only works for d == 2"
    proj = integral_transforms.KhatriRaoIntTsfm(d, p, q, [16, 16])
    n_grid = [3, 5]
    in_grid = quadrature.get_quad_grid(
        quadrature.trapezoidal_vecs, n_grid, [-5, -2], [-1, 3]
    )
    out_grid = quadrature.get_quad_grid(
        quadrature.trapezoidal_vecs, [6, 10], [-5, -2], [-1, 3]
    )

    batch_size = 3
    x = torch.randn(batch_size, int(np.prod(n_grid) * p))

    proj_out = proj.super_resolution(out_grid, in_grid, x)

    khrmat = proj.grid_kernel.super_resolution(out_grid, in_grid)
    wq_grid, _ = in_grid
    # apply quadrature
    quad_x = (
        torch.kron(torch.eye(p), torch.kron(wq_grid[0].diag(), wq_grid[1].diag())) @ x.T
    )
    proj_out_man = (khrmat @ quad_x).T
    assert torch.allclose(proj_out, proj_out_man, atol=1e-3)


def test_affine_projection():
    p, q = 11, 13
    affine_proj = integral_transforms.AffineMap(p, q)
    batch = 5
    N = 2
    x = torch.arange(batch * p * N).view(batch, p * N).float()
    proj = affine_proj(x)
    batch_proj = []
    for i in range(batch):
        sub = []
        for j in range(p):
            sub.append(x[i : i + 1, j * N : (j + 1) * N])
        lin = affine_proj.linear(torch.stack(sub, dim=-1)).squeeze().T.ravel()
        batch_proj.append(lin)

    batch_proj = torch.stack(batch_proj)
    assert torch.allclose(proj, batch_proj)


if __name__ == "__main__":
    test_khatrirao_int_tsfm_super_res()
    test_affine_projection()
    test_khatrirao_int_tsfm()
