import torch
from torch import Tensor, Size
from jaxtyping import Float
from typing import List, Union
import numpy as np


class KhatriRaoMatrix:
    _matrices: list[Float[Tensor, "q p n n"]]

    def __init__(self, matrices: list[Float[Tensor, "q p n n"]]) -> None:
        self._matrices = matrices

    def __len__(self) -> int:
        return len(self._matrices)

    def __matmul__(self, other: Float[Tensor, "pN bs"]) -> Float[Tensor, "qN batch"]:
        return khatri_rao_mmprod(self._matrices, other)

    @property
    def shape(self) -> list[Size]:
        return [mat.shape for mat in self._matrices]

    @property
    def full_matrix(self) -> Float[Tensor, "q p N N"]:
        d = len(self._matrices)
        q, p, _, _ = self._matrices[0].shape
        block_tensor = [[torch.tensor(1.0) for _ in range(p)] for _ in range(q)]
        for i in range(d):
            K_i = self._matrices[i]
            for j in range(q):
                for k in range(p):
                    block_tensor[j][k] = torch.kron(block_tensor[j][k], K_i[j][k])

        K_full = torch.cat(
            [
                torch.cat([block_tensor[j][k] for k in range(p)], dim=1)
                for j in range(q)
            ],
            dim=0,
        )
        return K_full


class KronMatrix:
    def __init__(self, matrices: list[Float[Tensor, "n m"]]) -> None:
        self._matrices = matrices

    def __matmul__(self, other: Float[Tensor, "p q"]) -> Float[Tensor, "... q"]:
        if other.dim() == 2:
            return kron_mmprod(self._matrices, other)
        elif other.dim() == 1:
            return kron_mvprod(self._matrices, other)
        else:
            raise ValueError("Kron matmul not implemented for > 2D tensor")

    def __len__(self) -> int:
        return len(self._matrices)

    def ident_prekron(
        self, p: int, other: Float[Tensor, "p q"]
    ) -> Float[Tensor, "... q"]:
        pN, bs = other.shape
        X = other.T
        Z = X.reshape(bs, p, -1).transpose(-2, -1)
        X = Z.reshape(bs, -1)
        return self.__matmul__(X.T)

    @property
    def shape(self) -> list[Size]:
        return [mat.shape for mat in self._matrices]

    @property
    def full_matrix(self) -> Float[Tensor, "N M"]:
        K_full = self._matrices[0]
        for K_j in self._matrices[1:]:
            K_full = torch.kron(K_full, K_j)
        return K_full


def khatri_rao_mmprod(
    K: list[Float[Tensor, "q p n1 n2"]], V: Float[Tensor, "pN batch"]
) -> Float[Tensor, "qN batch"]:
    """
    Computes the mm product between the Khatri-Rao structure matrix, K
    and the matrix v

    Args:
        K (List[Tensor]): Khatri-Rao structured matrix K[i].shape == (q,p,n1_bar[i],n2_bar[i]), i = 1,2, ..., d
        V (Tensor): matrix (p*n_bar**d, bs)

    Returns:
        Tensor: product K @ v
    """
    q, p, _, _ = K[0].shape
    pN, bs = V.shape

    X = V.reshape(p, -1, bs).transpose(-2, -1)
    for i in range(len(K)):
        Gd = K[i].shape[-1]
        bs_prod = X.shape[:-1]
        X = X.reshape(*bs_prod, Gd, -1)
        X = K[i].unsqueeze(-3) @ X
        X = X.transpose(-2, -1).reshape(q, p, bs, -1)
    return X.sum(1).transpose(-2, -1).reshape(-1, bs)


# def khatri_rao_mmprod(
#     K: list[Float[Tensor, "q p n1 n2"]], V: Float[Tensor, "pN batch"]
# ) -> Float[Tensor, "qN batch"]:
#     """
#     Computes the mm product between the Khatri-Rao structure matrix, K
#     and the matrix v

#     Args:
#         K (List[Tensor]): Khatri-Rao structured matrix K[i].shape == (q,p,n1_bar[i],n2_bar[i]), i = 1,2, ..., d
#         V (Tensor): matrix (p*n_bar**d, bs)

#     Returns:
#         Tensor: product K @ v
#     """
#     q, p, _, _ = K[0].shape
#     pN, bs = V.shape

#     V = V.reshape(p, -1, bs).transpose(-2, -1)
#     V_out = []
#     for j in range(q):
#         X = V
#         for i in range(len(K)):
#             Gd = K[i].shape[-1]
#             bs_prod = X.shape[:-1]
#             try:
#                 X = X.reshape(*bs_prod, Gd, -1)
#                 X = K[i][j].unsqueeze(-3) @ X
#                 X = X.transpose(-2, -1).reshape(1, p, bs, -1)
#             except torch.cuda.OutOfMemoryError:
#                 torch.cuda.empty_cache()
#                 X = X.reshape(*bs_prod, Gd, -1)
#                 X = K[i][j].unsqueeze(-3) @ X
#                 X = X.transpose(-2, -1).reshape(1, p, bs, -1)
#         X = X.sum(1).transpose(-2, -1).reshape(-1, bs)
#         V_out.append(X)
#     return torch.cat(V_out, dim=0) 


# def khatri_rao_mmprod(
#     K: list[Float[Tensor, "q p n1 n2"]], V: Float[Tensor, "pN batch"]
# ) -> Float[Tensor, "qN batch"]:
#     """
#     Computes the mm product between the Khatri-Rao structure matrix, K
#     and the matrix v

#     Args:
#         K (List[Tensor]): Khatri-Rao structured matrix K[i].shape == (q,p,n1_bar[i],n2_bar[i]), i = 1,2, ..., d
#         V (Tensor): matrix (p*n_bar**d, bs)

#     Returns:
#         Tensor: product K @ v
#     """
#     q, p, _, _ = K[0].shape
#     pN, bs = V.shape

#     X = V.reshape(p, -1, bs).transpose(-2, -1)
#     for i in range(len(K)):
#         Gd = K[i].shape[-1]
#         bs_prod = X.shape[:-1]
#         X = X.reshape(*bs_prod, Gd, -1)
#         Z = torch.einsum('abcd,bedf->abecf', K[i], X)
#         X = Z.transpose(-2, -1).reshape(q, p, bs, -1)
#     return X.sum(1).transpose(-2, -1).reshape(-1, bs)


# def khatri_rao_mmprod(
#     K: list[Float[Tensor, "q p n n"]], V: Float[Tensor, "pN batch"]
# ) -> Float[Tensor, "qN batch"]:
#     q, p, _, _ = K[0].shape
#     pN, bs = V.shape

#     X = V.reshape(p, -1, bs)  # Shape: [p, N, bs]
#     for i in range(len(K)):
#         Gd = K[i].shape[-1]
#         X = X.view(*bs_prod, Gd, -1)
#         # Efficient tensor contraction using einsum
#         X = torch.einsum('qpab,pcb->qcb', K[i], X.view(p, bs, -1))
#         # Reshape X back to the required dimensions
#         X = X.view(q, bs, -1)
#     return X.transpose(1, 2).view(-1, bs)

# def khatri_rao_mmprod(
#     K: list[Tensor], V: Tensor
# ) -> Tensor:
#     q, p, _, _ = K[0].shape
#     pN, bs = V.shape

#     X = V.view(p, -1, bs).permute(0, 2, 1)  # Shape: (p, bs, N)
#     for i in range(len(K)):
#         # Use einsum to avoid large intermediate tensors
#         X = torch.einsum('qpab,pbnd->qand', K[i], X)
#         bs_prod = X.shape[:-1]
#         X = X.view(q, p, bs, -1)
#     result = X.sum(dim=1).permute(2, 0, 1).reshape(-1, bs)
#     return result





def ident_kron_mmprod(p: int, K: Union[Tensor, List[Tensor]], V: Tensor) -> Tensor:
    """
    Computes the mm product between the identity matrix kronecker matrix product and V

    Args:
        p (int): size of identity matrix
        K (Union[Tensor, List[Tensor]]): Kronecker structured matrices
        V (Tensor): matrix (p*n_bar**d, bs)

    Returns:
        Tensor: product (Ip otimes K ) @ V
    """
    pN, bs = V.shape
    X = V.T
    Z = X.reshape(bs, p, -1).transpose(-2, -1)
    X = Z.reshape(bs, -1)
    return kron_mmprod(K, X.T)


def kron_mvprod(K: list[Tensor], v: Float[Tensor, "N"]) -> Float[Tensor, "M"]:
    """
    Returns the mv product between a list of matrices and a vector

    Ref: Saatchi, Yunis. "Scalable Inference for Structured Gaussian Process Models". 2011.

    ie: (kron_{i=1}^d K[i] ) v

    Args:
        K (list of torch.tensors): list of matrices of kronecker products
        v (torch.tensor): input vector
    """
    x = v
    N = len(v)
    for i in range(len(K)):
        Gd = K[i].shape[-1]
        X = x.reshape(Gd, -1)
        Z = K[i] @ X
        x = Z.T.flatten()
    return x


def kron_mmprod(K: Union[Tensor, List[Tensor]], V: Tensor) -> Tensor:
    """
    Returns the mm product between a list of matrices and a vector

    Ref: Saatchi, Yunis. "Scalable Inference for Structured Gaussian Process Models". 2011.

    ie: (kron_{i=1}^d K[i] ) V

    Args:
        K (list of torch.tensors): list of matrices of kronecker products
        v (torch.tensor): input vector
    """
    N, M = V.shape
    X = V.T
    for i in range(len(K)):
        Gd = K[i].shape[-1]
        # Z = X.reshape(M, Gd, N // Gd).transpose(-2, -1) @ K[i].T
        Z = X.reshape(M, Gd, -1).transpose(-2, -1) @ K[i].t()
        X = Z.reshape(M, -1)
    return X.T


def kron_struc_quad_mmprod(
    K_xx: Union[Tensor, List[Tensor]], K_qp: Tensor, v: Tensor
) -> Tensor:
    """
    Computes the quadrature structured matrix matrix product (see paper)

    Args:
        K_xx (Tensor(d,n_bar,n_bar)): grid structured kernel for each dimension
        K_qp (Tensor(q, p)): task kernel
        v (p*n_bar**2, bs): batch last vector of inputs
    """
    K = [K_xx[i] for i in range(len(K_xx))]
    K.append(K_qp)
    return kron_mmprod(K, v)

