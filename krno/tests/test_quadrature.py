import math
import torch
from torch.func import vmap  # type: ignore
from khatriraonop import quadrature


def quarter_circle_1d(x):
    return torch.sqrt(2**2 - x**2)


def quadratic_1d(x):
    return x**2


test_fn_1d = {
    "quarter_circle": (quarter_circle_1d, [0.0], [2.0], math.pi * 2**2 / 4),
    "quadratic": (quadratic_1d, [0.0], [2.0], 8 / 3),
}


def test_integrate_on_domain_1d():
    for quad_fn in [
        quadrature.trapezoidal_vecs,
        quadrature.midpoint_vecs,
    ]:
        n = 1000
        u = []
        true_vals = []
        for _, (fn, lb, ub, true_val) in test_fn_1d.items():
            quad_rule = quadrature.get_quad_grid(quad_fn, [n], lb, ub)
            _, xq = quad_rule
            true_vals.append(true_val)
            u.append(fn(xq[0]))
        true_vals = torch.tensor(true_vals)
        true_vals = torch.stack([true_vals, true_vals + 2.0])
        u = torch.stack(u, dim=-1)
        u = torch.stack([u, u + 1.0])
        est = quadrature.integrate_on_domain(u, quad_rule)  # type: ignore
        assert torch.allclose(est, true_vals, atol=1e-5)


def quarter_circle_2d(x, y):
    return torch.sqrt((2**2 - x.pow(2) - y.pow(2)).clamp(0))


def quadratic_2d(x, y):
    return 0.5 * (x.pow(2) + y.pow(2))


test_fn_2d = {
    "quarter_circle": (
        quarter_circle_2d,
        [0.0, 0.0],
        [2.0, 2.0],
        4 / 3 * math.pi * 2**3 / 8,
    ),
    "quadratic": (quadratic_2d, [0.0, 0.0], [2.0, 2.0], 16 / 3),
}


def test_integrate_on_domain_2d():
    for quad_fn in [
        quadrature.trapezoidal_vecs,
        quadrature.midpoint_vecs,
    ]:
        n = 3000
        u = []
        true_vals = []
        for _, (fn, lb, ub, true_val) in test_fn_2d.items():
            quad_rule = quadrature.get_quad_grid(quad_fn, [n, n], lb, ub)
            _, xq = quad_rule
            xs, ys = xq
            true_vals.append(true_val)
            u.append(vmap(vmap(fn, in_dims=(None, 0)), in_dims=(0, None))(xs, ys))
        true_vals = torch.tensor(true_vals)
        true_vals = torch.stack([true_vals, true_vals + 2.0**2])
        u = torch.stack(u, dim=-1)
        u = torch.stack([u, u + 1.0])
        est = quadrature.integrate_on_domain(u, quad_rule)  # type: ignore
        assert torch.allclose(est, true_vals, atol=1e-5)


if __name__ == "__main__":
    test_integrate_on_domain_2d()
