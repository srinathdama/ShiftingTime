from khatriraonop import kronecker_algebra
import numpy as np
import torch
import time


def test_khatrirao_mmprod(equal_grids=False):
    d = 3
    bs = 7
    n_bar_list_in = [5, 3, 7]
    if equal_grids:
        n_bar_list_out = n_bar_list_in
    else:
        n_bar_list_out = [4, 2, 6]
        
    p = 6
    q = 8
    n_bar_prod = np.array(n_bar_list_in).prod()
    v = torch.randn(n_bar_prod * p * bs, dtype=torch.float64).reshape(-1, bs)
    K = [
        torch.randn(q, p, n_bar_list_out[i], n_bar_list_in[i], dtype=torch.float64)
        for i in range(d)
    ]
    K_tot = [[K[0][i, j] for j in range(p)] for i in range(q)]
    # naive computation
    naive_time = time.time()
    for i in range(1, d):
        for j in range(q):
            for k in range(p):
                K_tot[j][k] = torch.kron(K_tot[j][k], K[i][j, k])
    K_tot = torch.cat(
        [torch.cat([K_tot[j][k] for k in range(p)], dim=-1) for j in range(q)], dim=0
    )
    true_Kv = (K_tot @ v).squeeze()
    naive_time = time.time() - naive_time
    fast_time = time.time()
    krao_mat = kronecker_algebra.KhatriRaoMatrix(K)
    fast_Kv = (krao_mat @ v).squeeze()
    fast_time = time.time() - fast_time
    assert naive_time > fast_time, "Fast khatri-rao was slower than naive khatri-rao."
    print(f'naive_time: {naive_time}, fast_time: {fast_time}')
    assert torch.allclose(true_Kv, fast_Kv)
    assert torch.allclose(K_tot, krao_mat.full_matrix)


def test_kron_mvprod():
    n = 10
    d = 4
    K = [torch.rand(n, n, dtype=torch.float64) for i in range(d)]
    v = torch.rand(n**d, dtype=torch.float64)
    Ktot = K[0]
    for j in range(1, d):
        Ktot = torch.kron(Ktot, K[j])
    true_Kv = Ktot @ v
    kron_mat = kronecker_algebra.KronMatrix(K)
    fast_Kv = kron_mat @ v
    assert torch.allclose(true_Kv, fast_Kv)
    assert torch.allclose(Ktot, kron_mat.full_matrix)


def test_kron_mmprod():
    n = 10
    m = 10
    d = 4
    K = [torch.rand(n, n, dtype=torch.float64) for i in range(d)]
    V = torch.rand(n**d, m, dtype=torch.float64)
    Ktot = K[0]
    for j in range(1, d):
        Ktot = torch.kron(Ktot, K[j])
    true_Kv = Ktot @ V
    fast_Kv = kronecker_algebra.KronMatrix(K) @ V
    assert torch.allclose(true_Kv, fast_Kv)


def test_ident_kron_mmprod():
    n = 10
    d = 4
    p = 5
    K = [torch.rand(n, n, dtype=torch.float64) for i in range(d)]
    v = torch.rand(p * n**d, 3, dtype=torch.float64)
    fast_Kv = kronecker_algebra.KronMatrix(K).ident_prekron(p, v)
    K.insert(0, torch.eye(p, dtype=torch.float64))
    true_Kv = kronecker_algebra.KronMatrix(K) @ v
    assert torch.allclose(true_Kv, fast_Kv)


if __name__ == "__main__":
    test_khatrirao_mmprod()
    test_kron_mvprod()
    test_kron_mmprod()
    test_ident_kron_mmprod()
