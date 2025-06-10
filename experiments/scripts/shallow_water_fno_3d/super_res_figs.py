import argparse
import os
import sys
import pathlib

import torch
from jaxtyping import Float
from torch import Tensor
from torch.func import vmap  # type: ignore
import pandas as pd
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from khatriraonop import models, quadrature, types
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

torch.set_float32_matmul_precision("high")

CURR_DIR = pathlib.Path(__file__).parent.absolute()

sys.path.append(str(CURR_DIR / ".."))

from plotting_config import PlotConfig

from train_script import get_raw_data, TrainLog, seed_everything

DATA_DIR = CURR_DIR / "loca-data"
CKPT_DIR = CURR_DIR / "ckpts"
FIG_DIR = CURR_DIR / "figs"

os.makedirs(FIG_DIR, exist_ok=True)


@torch.no_grad()
def super_res_forecast(
    in_quad_grid: types.CartesianGrid,
    out_quad_grid: types.CartesianGrid,
    y0: Float[Tensor, "bs lag nx ny 3"],
    n_steps: int,
    model,
    device,
):
    lag = y0.shape[1]
    model.to("cpu")
    # too big to fit on gpus
    print("applying super resolution...")
    y_out = [model.super_resolution(out_quad_grid, in_quad_grid, y0.cpu()).to(device)]
    model.to(device)
    steps_so_far = lag
    cart_grid = quadrature.quad_to_cartesian_grid(out_quad_grid)
    cart_grid = quadrature.cart_grid_to_device(cart_grid, device)
    print("back on a grid!")
    while steps_so_far < n_steps:
        y_prev = y_out[-1]
        y_pred = model(cart_grid, y_prev)
        y_out.append(y_pred)
        steps_so_far += lag
    y_pred = torch.cat(y_out, dim=1)
    n_ts = y_pred.shape[1]
    t_pred = torch.tensor([0.01 / 2 * j for j in range(n_ts)])
    return (t_pred, y_pred)


def plot_super_res():
    with open(CURR_DIR / "super_res_df.pt", "rb") as fp:
        df = torch.load(fp)
    n_data_plots = 3
    t_data, t_pred = df["t_data"], df["t_pred"]
    y_data, y_pred = df["y_data"][..., 0], df["y_pred"][..., 0]
    skip = len(t_data) // n_data_plots

    PlotConfig.setup()
    figsize = PlotConfig.convert_width((2, 1), page_scale=0.5)
    fig, axs = plt.subplots(2, n_data_plots * 2, figsize=figsize)
    cmap = sns.color_palette("Spectral", as_cmap=True)
    batch_idx = 0
    for j in range(n_data_plots):
        lr_idx = j * skip
        hr_idx1 = 2 * j * skip
        hr_idx2 = (2 * j + 1) * skip
        axs[0, 2 * j].imshow(y_data[batch_idx, lr_idx], cmap=cmap)
        axs[1, 2 * j].imshow(y_pred[batch_idx, hr_idx1], cmap=cmap)
        axs[1, 2 * j + 1].imshow(y_pred[batch_idx, hr_idx2], cmap=cmap)
    for ax in axs.flatten():
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
    axs[0, 0].set_ylabel("data")
    axs[1, 0].set_ylabel("forecast")
    fig.subplots_adjust(bottom=0.25)
    common_ax = fig.add_subplot(111, frameon=False)  # Add a common subplot
    common_ax.grid(True)
    common_ax.set_xticks([])
    common_ax.set_xticklabels([])
    common_ax.set_yticks([])
    common_ax.set_yticklabels([])
    common_ax.set_xlabel("time")

    # common_ax.axhline(y=0.0, color="k", xmin=0.00, xmax=1.0)
    # shrink=0, width=1, headwidth=8, headlength=10,
    common_ax.annotate(
        "",
        xy=(1.0, 0.0),
        xytext=(0.0, 0.0),
        arrowprops=dict(facecolor="k", edgecolor="k", arrowstyle="->"),
        annotation_clip=False,
    )
    PlotConfig.save_fig(fig, str(FIG_DIR / "super-res-ex"))
    plt.close(fig)

def plot_super_res_vert():
    with open(CURR_DIR / "super_res_df.pt", "rb") as fp:
        df = torch.load(fp)
    n_data_plots = 3
    t_data, t_pred = df["t_data"], df["t_pred"]
    y_data, y_pred = df["y_data"][..., 0], df["y_pred"][..., 0]
    skip = len(t_data) // n_data_plots

    PlotConfig.setup()
    figsize = PlotConfig.convert_width((1, 2), page_scale=0.5)
    fig, axs = plt.subplots(n_data_plots * 2, 2, figsize=figsize)
    cmap = sns.color_palette("Spectral", as_cmap=True)
    batch_idx = 0
    for j in range(n_data_plots):
        lr_idx = j * skip
        hr_idx1 = 2 * j * skip
        hr_idx2 = (2 * j + 1) * skip
        axs[2 * j, 0].imshow(y_data[batch_idx, lr_idx], cmap=cmap,) # interpolation='bilinear')
        axs[2 * j, 1].imshow(y_pred[batch_idx, hr_idx1], cmap=cmap,) # interpolation='bilinear')
        axs[2 * j + 1, 1].imshow(y_pred[batch_idx, hr_idx2], cmap=cmap,) # interpolation='bilinear')
    for ax in axs.flatten():
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
    axs[0, 0].set_title("data")
    axs[0, 1].set_title("forecast")
    fig.subplots_adjust(left=0.25)
    common_ax = fig.add_subplot(111, frameon=False)  # Add a common subplot
    common_ax.grid(True)
    common_ax.set_xticks([])
    common_ax.set_xticklabels([])
    common_ax.set_yticks([])
    common_ax.set_yticklabels([])
    common_ax.set_ylabel("time")

    # common_ax.axhline(y=0.0, color="k", xmin=0.00, xmax=1.0)
    # shrink=0, width=1, headwidth=8, headlength=10,
    common_ax.annotate(
        "",
        xy=(0.0, 0.0),
        xytext=(0.0, 1.0),
        arrowprops=dict(facecolor="k", edgecolor="k", arrowstyle="->"),
        annotation_clip=False,
    )
    PlotConfig.save_fig(fig, str(FIG_DIR / "super-res-ex-vert"))
    plt.close(fig)


def main(file: str):
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # get index of best model based on validation loss
    with open(file, "rb") as f:
        train_state = torch.load(f)

    log = TrainLog.from_dict(train_state["log"])
    lag = log.hparams.pop("lag")  # type: ignore
    model = models.KhatriRaoNO.easy_init(**log.hparams)  # type: ignore
    model.load_state_dict(log.best_ckpt())
    model.eval()
    model.to(device)

    quad_fns = [
        quadrature.midpoint_vecs,
        quadrature.trapezoidal_vecs,
        quadrature.trapezoidal_vecs,
    ]
    # in quad grid
    low_res = [8, 8]
    in_quad_grid = quadrature.get_quad_grid(
        quad_fns, [lag, *low_res], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]
    )
    # in_quad_grid = quadrature.quad_grid_to_device(in_quad_grid, device)
    out_quad_grid = quadrature.get_quad_grid(
        quad_fns, [lag * 2, 32, 32], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]
    )
    # out_quad_grid = quadrature.quad_grid_to_device(out_quad_grid, device)
    _, s_test = get_raw_data()
    batch_size = 1

    def resize_tsfm(y):
        return vmap(transforms.Resize(low_res), in_dims=-1, out_dims=-1)(y)

    test_loader = DataLoader(TensorDataset(s_test), batch_size=batch_size)
    sres_df = {}
    for j, (y,) in enumerate(test_loader):
        y = resize_tsfm(y.to(device))
        n_steps = y.shape[1]
        y0 = y[:, :lag]
        t_pred, y_pred = super_res_forecast(
            in_quad_grid, out_quad_grid, y0, n_steps, model, device
        )
        t_data = torch.tensor([0.01 * j for j in range(n_steps)])
        t_pred = t_data[lag] + t_pred
        sres_df = {
            "t_data": t_data[lag:],
            "y_data": y[:, lag:].cpu(),
            "t_pred": t_pred,
            "y_pred": y_pred.cpu(),
        }
        break
    with open(CURR_DIR / "super_res_df.pt", "wb") as fp:
        torch.save(sres_df, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", "-f", type=str, default=CKPT_DIR / "krno_it115000_42.pt"
    )
    args = parser.parse_args()
    # main(args.file)
    plot_super_res()
    plot_super_res_vert()
