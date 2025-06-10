import argparse
from functools import partial
import os
import sys
import pathlib
from jaxtyping import Float

import torch
import numpy as np
import pandas as pd
from torch import Tensor
from torch.func import vmap  # type: ignore
from torch.utils.data import TensorDataset, DataLoader
from khatriraonop import models, quadrature
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

torch.set_float32_matmul_precision("high")

CURR_DIR = pathlib.Path(__file__).parent.absolute()

sys.path.append(str(CURR_DIR / ".."))

from plotting_config import PlotConfig

from train_script import get_raw_data, TrainLog, seed_everything
from metrics import forecast_metrics

DATA_DIR = CURR_DIR / "loca-data"
CKPT_DIR = CURR_DIR / "ckpts"
FIG_DIR = CURR_DIR / "figs"

os.makedirs(FIG_DIR, exist_ok=True)


@torch.no_grad()
def plot_pred(
    y_true: Float[Tensor, "nt nx ny 3"],
    y_pred: Float[Tensor, "nt nx ny 3"],
    nplots: int,
    name: str,
):

    cmap = sns.color_palette("Spectral", as_cmap=True)
    vars = ["\\rho", "u", "v"]
    PlotConfig.setup()
    figsize = PlotConfig.convert_width((2, 1), page_scale=1.0)
    fig, axs = plt.subplots(6, nplots, figsize=figsize)
    n_time = len(y_true) // nplots
    p_true, u_true, v_true = y_true.cpu().chunk(3, dim=-1)
    p_pred, u_pred, v_pred = y_pred.cpu().chunk(3, dim=-1)

    for j in range(nplots):
        idx = j * n_time
        p_min, p_max = p_true[idx].min(), p_true[idx].max()
        u_min, u_max = u_true[idx].min(), u_true[idx].max()
        v_min, v_max = v_true[idx].min(), v_true[idx].max()
        images = [
            axs[0, j].imshow(p_true[idx], cmap=cmap, vmin=p_min, vmax=p_max, interpolation='bilinear'),
            axs[1, j].imshow(u_true[idx], cmap=cmap, vmin=u_min, vmax=u_max, interpolation='bilinear'),
            axs[2, j].imshow(v_true[idx], cmap=cmap, vmin=v_min, vmax=v_max, interpolation='bilinear'),
            axs[3, j].imshow(p_pred[idx], cmap=cmap, vmin=p_min, vmax=p_max, interpolation='bilinear'),
            axs[4, j].imshow(u_pred[idx], cmap=cmap, vmin=u_min, vmax=u_max, interpolation='bilinear'),
            axs[5, j].imshow(v_pred[idx], cmap=cmap, vmin=v_min, vmax=v_max, interpolation='bilinear'),
        ]
    for ax in axs.flatten():
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
    for i, var in enumerate(vars):
        axs[i, 0].set_ylabel(f"${var}$")
        axs[i + len(vars), 0].set_ylabel("$\\hat{%s}$" % var)

    fig.subplots_adjust(bottom=0.25)
    common_ax = fig.add_subplot(111, frameon=False)  # Add a common subplot
    common_ax.axhline(0.5, alpha=0.5, color="k", linestyle="--", linewidth=2)
    common_ax.set_xlabel("$t$ (sec)")
    common_ax.grid(False)
    common_ax.set_xticks(np.linspace(0, 1.0, 5))
    common_ax.set_xticklabels([f"{v:.2f}" for v in np.linspace(0, 0.6, 5)])
    common_ax.set_yticks([])
    common_ax.set_yticklabels([])
    PlotConfig.save_fig(fig, str(FIG_DIR / name))
    plt.close(fig)


def l2_norm_2D(quad_rule, y_pred, y_true):
    diffs = y_pred - y_true
    numer = quadrature.integrate_on_domain(diffs.pow(2), quad_rule).sqrt()
    denom = quadrature.integrate_on_domain(y_pred.pow(2), quad_rule).sqrt()
    return numer / denom


@torch.no_grad()
def err_over_time(
    y_true: Float[Tensor, "batch nt nx ny 3"],
    y_pred: Float[Tensor, "batch nt nx ny 3"],
    metrics: dict,
    quad_rule: quadrature.QuadGrid,
):
    batch_size, num_t = y_true.shape[0], y_true.shape[1]
    t = torch.linspace(0.0, 0.6, num_t).view(1, -1).repeat(batch_size, 1)
    if "t" not in metrics:
        metrics["t"] = []
        metrics["rho"] = []
        metrics["v1"] = []
        metrics["v2"] = []

    rel_err = vmap(partial(l2_norm_2D, quad_rule))(y_pred, y_true)
    metrics["t"].append(t.ravel().cpu())
    metrics["rho"].append(rel_err[..., 0].ravel().cpu())
    metrics["v1"].append(rel_err[..., 1].ravel().cpu())
    metrics["v2"].append(rel_err[..., 2].ravel().cpu())


def plot_err_over_time():
    err_df = pd.read_pickle(CURR_DIR / "err_time_dict.pkl")
    PlotConfig.setup()
    figsize = PlotConfig.convert_width((10, 1), page_scale=1.0)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # get into percent
    err_df = err_df.melt(id_vars="t", value_vars=["rho", "v1", "v2"])
    err_df = err_df.assign(
        Output=err_df["variable"].map({"rho": "$\\rho$", "v1": "$v_1$", "v2": "$v_2$"})  # type: ignore
    )
    g = sns.lineplot(data=err_df, x="t", y="value", ax=ax, hue="Output", errorbar=("sd", 1.0))  # type: ignore
    g.legend_.set_title(None)  # type: ignore
    # can plot same idx's where loca evaluates
    # loca_test_times = [10, 15, 20, 25, 30]
    # t_vlines = torch.linspace(0, 1, 101)[1:]
    # for ttime in loca_test_times:
    #     ax.axvline(t_vlines[ttime].item(), color="r")
    ax.set_xlabel("$t$ (sec)")
    ax.set_ylabel("rel. $L^2$ error")
    PlotConfig.save_fig(fig, str(FIG_DIR / "err-over-time"))


def main(file: str):
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get index of best model based on validation loss
    with open(file, "rb") as f:
        train_state = torch.load(f)

    log = TrainLog.from_dict(train_state["log"])
    lag = log.hparams.pop("lag")  # type: ignore
    model = models.KhatriRaoNO.easy_init(**log.hparams)  # type: ignore
    model.load_state_dict(log.best_ckpt())

    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Calculate the total number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print the results
    print(f"Total number of parameters: {total_params}")
    print(f"Total number of trainable parameters: {trainable_params}")
    
    model.eval()
    model.to(device)

    quad_fns = [
        quadrature.midpoint_vecs,
        quadrature.trapezoidal_vecs,
        quadrature.trapezoidal_vecs,
    ]
    quad_grid = quadrature.get_quad_grid(
        quad_fns, [lag, 32, 32], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]
    )
    cart_grid = quadrature.quad_to_cartesian_grid(quad_grid)
    cart_grid = quadrature.cart_grid_to_device(cart_grid, device)

    quad_grid_2d = quadrature.get_quad_grid(
        quadrature.trapezoidal_vecs, [32, 32], [-1.0, -1.0], [1.0, 1.0]
    )
    quad_grid_2d = quadrature.quad_grid_to_device(quad_grid_2d, device)

    _, s_test = get_raw_data()

    batch_size = 32
    test_loader = DataLoader(TensorDataset(s_test), batch_size=batch_size)

    metrics = {}
    err_df = {}
    for j, (y,) in enumerate(tqdm(test_loader)):
        y = y.to(device)
        y_pred = forecast_metrics(lag, cart_grid, y, model, metrics)
        if j == 0:
            plot_pred(y[0], y_pred[0], 10, "shallow-water-prediction")
        # relative errors over time
        err_over_time(y, y_pred, err_df, quad_grid_2d)
    err_df = {k: torch.cat(v) for k, v in err_df.items()}
    err_df = pd.DataFrame(err_df)
    err_df.to_pickle(CURR_DIR /"err_time_dict.pkl")
    print(file, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", "-f", type=str, default=CKPT_DIR / "krno_it115000_42.pt"
    )
    # parser.add_argument(
    #     "--file", "-f", type=str, default=CKPT_DIR / "krno_it230000_42.pt"
    # )
    args = parser.parse_args()
    main(args.file)
    plot_err_over_time()
