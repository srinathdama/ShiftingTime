from typing import Protocol
from jaxtyping import Float
from torch import Tensor


class QuadGeneratingFn(Protocol):
    """generates a 1D quad rule and weight"""

    def __call__(
        self, N: int, a: float, b: float
    ) -> tuple[Float[Tensor, "N"], Float[Tensor, "N"]]: ...

# TODO: QuadGrid and CartesianGrid really should be refactored into 
# a single class

# quad_weights, quad_nodes
QuadGrid = tuple[list[Float[Tensor, "n_bar"]], list[Float[Tensor, "n_bar"]]]

# quad_weights, cartesian_prod(quad_nodes, quad_nodes)
CartesianGrid = tuple[list[Float[Tensor, "n_bar"]], list[Float[Tensor, "n_bar 2"]]]
