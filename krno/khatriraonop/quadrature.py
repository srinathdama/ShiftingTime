from torch import Tensor
import torch
from torch import Tensor
from jaxtyping import Float

from .types import CartesianGrid, QuadGeneratingFn, QuadGrid
# from .extern import fastgl
from .kronecker_algebra import KronMatrix


# def gauss_legendre_vecs(
#     N: int, a: float, b: float
# ) -> tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
#     """
#     Returns the gauss-legendre quadrature weights and nodes
#     """
#     x = []
#     w = []
#     for j in range(1, N + 1):
#         _, wt, xt = fastgl.glpair(N, j)
#         w.append(wt)
#         x.append(xt)
#     wq = torch.tensor(w) * (b - a) / 2
#     xq = torch.tensor(x) * (b - a) / 2 + (a + b) / 2
#     dtype = torch.get_default_dtype()
#     wq, xq = wq.to(dtype), xq.to(dtype)
#     return (wq, xq)


def trapezoidal_vecs(
    N: int, a: float, b: float
) -> tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
    """
    Returns the trapezoidal qudrature weights and nodes
    """
    x = torch.linspace(a, b, N)
    w = torch.ones(N)
    w[0] = w[0] * 0.5
    w[-1] = w[-1] * 0.5
    w = (b - a) / (N - 1) * w
    return (w, x)

def trapezoidal_vecs_uneven(
    x: Tensor
) -> tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
    """
    Returns the trapezoidal qudrature weights and nodes
    """
    N = x.shape[0]
    w = torch.ones(N)
    w[0] = w[0] * 0.5 * (x[1] - x[0])
    w[-1] = w[-1] * 0.5 * (x[-1] - x[-2])
    for i in range(1, N-1):
        w[i] = (x[i+1] - x[i]) * w[i]
    # w = (x[1] - x[0]) * w
    return (w, x)


def midpoint_vecs(
    N: int, a: float, b: float
) -> tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
    """
    Returns the midpoint quadrature weights and nodes
    """
    x = torch.linspace(a, b, N + 1)
    x = 0.5 * (x[:-1] + x[1:])
    w = (b - a) / N * torch.ones(N)
    return (w, x)

def midpoint_vecs_uneven( x: Tensor
) -> tuple[Float[Tensor, "N"], Float[Tensor, "N"]]:
    """
    Returns the midpoint quadrature weights and nodes
    """
    N = x.shape[0]
    x = 0.5 * (x[:-1] + x[1:])
    w = torch.ones(N-1)
    for i in range(0, N-1):
        w[i] = (x[i+1] - x[i]) * w[i]
    return (w, x)


def get_quad_grid(
    quad_fn: QuadGeneratingFn | list[QuadGeneratingFn],
    n_bar_list: list[int],
    lb_list: list[float],
    ub_list: list[float],
) -> QuadGrid:
    """
    Computes a grid of quad nodes and weights using quad generating function

    Args:
        quad_fn (QuadGeneratingFn | list[QuadGeneratingFn]): quadrature generating function or list of quad fns
        n_bar_list (list[int]): number of quadrature points along each dimension
        lb_list (list[float]): lower bound of domain along each dimension
        ub_list (list[float]): upper bound of domain along each dimension

    Returns:
        Tuple[list[Tensor], list[Tensor]]: (quadrature weights, quadrature grid)
    """
    # allow the user to provide a single quad_fn for backwards compatibility
    if not isinstance(quad_fn, list):
        quad_fn = [quad_fn] * len(n_bar_list)
    # confirm bounds are formatted correctly
    assert len(n_bar_list) == len(lb_list)
    assert len(lb_list) == len(ub_list)
    assert len(quad_fn) == len(n_bar_list)
    # 1d quadrature nodes and weights for each dim
    wq_grid = []
    xq_grid = []
    for n_bar, a, b, qfni in zip(n_bar_list, lb_list, ub_list, quad_fn):
        assert n_bar > 0, "each n_bar must be positive"
        assert a < b, "each lower bound must be strictly < upper bound"
        wq, xq = qfni(n_bar, a, b)
        wq_grid.append(wq)
        xq_grid.append(xq)
    return wq_grid, xq_grid


def quad_grid_to_device(quad_grid: QuadGrid, device: torch.device) -> QuadGrid:
    wq_grid = []
    xq_grid = []
    for wq, xq in zip(*quad_grid):
        wq_grid.append(wq.to(device))
        xq_grid.append(xq.to(device))
    return wq_grid, xq_grid


def quad_to_cartesian_grid(grid: QuadGrid) -> CartesianGrid:
    """Converts a QuadGrid of 1D quad rules into a CartesianGrid"""
    wq_grid, xq_grid = grid
    cart_grid = [torch.cartesian_prod(xq, xq) for xq in xq_grid]
    return wq_grid, cart_grid


def cart_grid_to_device(
    cartesian_grid: CartesianGrid, device: torch.device
) -> CartesianGrid:
    wq_grid, cart_grid = cartesian_grid
    wq_grid = [wq.to(device) for wq in wq_grid]
    cart_grid = [grid.to(device) for grid in cart_grid]
    return wq_grid, cart_grid


def integrate_on_domain(
    u: Float[Tensor, "batch ... p"], grid: QuadGrid | CartesianGrid
) -> Float[Tensor, "batch p"]:
    """Integrates a function on a domain using quadrature weights"""
    wq_grid, _ = grid
    kron_wq = KronMatrix([wq for wq in wq_grid])
    batch, p = u.shape[0], u.shape[-1]
    u = u.reshape(batch, -1, p).transpose(1, 2)
    return (kron_wq @ u.reshape(batch * p, -1).T).T.reshape(batch, p)
