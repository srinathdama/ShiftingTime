import torch
from torch import Tensor
from jaxtyping import Float
from khatriraonop import types
# torch.set_default_dtype(torch.float64)
torch.set_float32_matmul_precision("highest")

@torch.no_grad
def forecast(
    cart_grid: types.CartesianGrid,
    y0: Float[Tensor, "bs lag nx ny 3"],
    n_steps: int,
    model,
    FNO_FLAG=False
):
    lag = y0.shape[1]
    y_out = [y0]
    steps_so_far = lag
    while steps_so_far < n_steps:
        y_prev = y_out[-1]
        if FNO_FLAG:
            xx = y_prev
            T, S, C = xx.shape[1], xx.shape[2], xx.shape[-1]
            batch_size = xx.shape[0]
            xx = xx.permute(0,2,3,4,1) ## [B,lag,Nx,Ny,C] -> [B,Nx,Ny,C,lag]
            xx = xx.reshape((batch_size, S, S, 1, C*T)) #[B,Nx,Ny,C,lag] -> [B,Nx,Ny,1, C*lag]
            xx = xx.repeat([1,1,1,T,1])
            out = model(xx)  ## [B,Nx,Ny,T,C]
            y_pred = out.permute(0,3,1,2,4) ## [B,Nx,Ny,T,C] -> [B,T,Nx,Ny,C]
        else:
            y_pred = model(cart_grid, y_prev)
        y_out.append(y_pred)
        steps_so_far += lag
    return torch.cat(y_out, dim=1)[:, :n_steps]


def rel_l2_vec(lag, y_true, y_pred):
    batch = y_true.shape[0]
    # y_true.shape == (batch, n_steps, nx, ny, 1)
    y_true = y_true[:, lag:].reshape(batch, -1)
    y_pred = y_pred[:, lag:].reshape(batch, -1)
    return torch.norm(y_pred - y_true, p=2, dim=1) / torch.norm(y_true, p=2, dim=1)


def forecast_metrics(lag, cart_grid, y, model, metrics, FNO_FLAG=False):
    y0 = y[:, :lag]
    n_steps = y.shape[1]
    y_pred = forecast(cart_grid, y0, n_steps, model, FNO_FLAG)
    rho_true, v1_true, v2_true = y.chunk(3, dim=-1)
    rho_pred, v1_pred, v2_pred = y_pred.chunk(3, dim=-1)
    rho_err = rel_l2_vec(lag, rho_true, rho_pred)
    v1_err = rel_l2_vec(lag, v1_true, v1_pred)
    v2_err = rel_l2_vec(lag, v2_true, v2_pred)
    batch = len(rho_err)
    if "N" not in metrics:
        metrics["N"] = 0
        metrics["rho_l2"] = 0.0
        metrics["v1_l2"] = 0.0
        metrics["v2_l2"] = 0.0
    metrics["N"] += batch
    scale = batch / metrics["N"]
    metrics["rho_l2"] += scale * (rho_err.mean().item() - metrics["rho_l2"])
    metrics["v1_l2"] += scale * (v1_err.mean().item() - metrics["v1_l2"])
    metrics["v2_l2"] += scale * (v2_err.mean().item() - metrics["v2_l2"])
    return y_pred
