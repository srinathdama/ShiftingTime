import torch
from torch import Tensor
from jaxtyping import Float
from khatriraonop import types


@torch.no_grad
def forecast(
    cart_grid: types.CartesianGrid,
    y0: Float[Tensor, "bs lag nx ny 3"],
    n_steps: int,
    model,
):
    lag = y0.shape[1]
    y_out = [y0]
    steps_so_far = lag
    while steps_so_far < n_steps:
        y_prev = y_out[-1]
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


def forecast_metrics(lag, cart_grid, y, model, norm_fn, metrics):
    y0 = y[:, :lag]
    n_steps = y.shape[1]
    y_pred = forecast(cart_grid, y0, n_steps, model)
    k_true, p_true = norm_fn(y=y).chunk(2, dim=-1)
    k_pred, p_pred = norm_fn(y=y_pred).chunk(2, dim=-1)
    k_err = rel_l2_vec(lag, k_true, k_pred)
    p_err = rel_l2_vec(lag, p_true, p_pred)
    batch = len(k_err)
    if "N" not in metrics:
        metrics["N"] = 0
        metrics["k_l2"] = 0.0
        metrics["p_l2"] = 0.0
    metrics["N"] += batch
    scale = batch / metrics["N"]
    metrics["k_l2"] += scale * (k_err.mean().item() - metrics["k_l2"])
    metrics["p_l2"] += scale * (p_err.mean().item() - metrics["p_l2"])
    return y_pred
