import pathlib
import os
import random
import sys

import torch
from torch import nn
import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize

from khatriraonop import models, quadrature

torch.set_float32_matmul_precision("high")

CURR_DIR = pathlib.Path(__file__).parent.absolute()

sys.path.append(str(CURR_DIR / ".."))

from plotting_config import PlotConfig

CKPT_DIR = CURR_DIR / "ckpts"
FIGS = CURR_DIR / "figs"

os.makedirs(FIGS, exist_ok=True)


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class DictDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx].unsqueeze(0)
        y = self.y[idx].unsqueeze(0)
        return {"x": x, "y": y}


def load_darcy_flow_20():
    data = sio.loadmat(CURR_DIR / "Data_Darcy.mat")

    def to_tensor(x):
        return torch.tensor(x, dtype=torch.get_default_dtype())

    train_dset = DictDataset(to_tensor(data["k_train"]), to_tensor(data["u_train"]))
    train_dset, val_dset = random_split(train_dset, [0.8, 0.2])
    train_loader = DataLoader(
        train_dset,
        batch_size=16,
        shuffle=True,
    )
    valid_loader = DataLoader(
        val_dset,
        batch_size=16,
        shuffle=True,
    )
    test_loaders = {
        20: DataLoader(
            DictDataset(to_tensor(data["k_test"]), to_tensor(data["u_test"])),
            batch_size=16,
            shuffle=True,
        )
    }
    return train_loader, valid_loader, test_loaders


def unpack_batch(batch):
    x = batch["x"].permute(0, 2, 3, 1)
    y = batch["y"].permute(0, 2, 3, 1)
    return x, y


# @torch.no_grad
# def plot_prediction(x, y, y_pred, idx: int):
#     PlotConfig.setup()
#     figsize = PlotConfig.convert_width((3, 1), page_scale=0.5)
#     fig, axs = plt.subplots(1, 3, figsize=figsize)
#     cmap = sns.color_palette("viridis", as_cmap=True)
#     axs[0].imshow(x[idx].squeeze().cpu(), cmap=cmap)
#     axs[0].set_title("Test input")
#     axs[1].imshow(y[idx].squeeze().cpu(), cmap=cmap)
#     axs[1].set_title("Test output")
#     axs[2].imshow(y_pred[idx].squeeze().cpu(), cmap=cmap)
#     axs[2].set_title("Pred. output")
#     for ax in axs.flatten():
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_xticks([])
#         ax.set_yticks([])
#     PlotConfig.save_fig(fig, str(FIGS / "darcy-flow-pred"))
#     plt.close(fig)

@torch.no_grad
def plot_prediction(x, y, y_pred, idx: int):

    vmin = np.min([y[idx].squeeze().cpu().min(),y_pred[idx].squeeze().cpu().min()])
    vmax = np.max([y[idx].squeeze().cpu().max(),y_pred[idx].squeeze().cpu().max()])
    # cmap = sns.color_palette("viridis", as_cmap=True)
    cmap = cm.bwr
    normalizer = Normalize(vmin, vmax)
    im         = cm.ScalarMappable(norm=normalizer, cmap=cmap)

    PlotConfig.setup()
    figsize = PlotConfig.convert_width((3, 1), page_scale=0.5)
    fig, axs = plt.subplots(1, 4, figsize=figsize,
                        gridspec_kw={"width_ratios":[1,1,1, 0.05]})
    # cmap = sns.color_palette("viridis", as_cmap=True)
    axs[0].imshow(x[idx].squeeze().cpu(), cmap=cmap, interpolation='bilinear')
    axs[0].set_title("Test input")
    axs[1].imshow(y[idx].squeeze().cpu(), cmap=cmap, norm=normalizer, interpolation='bilinear')
    axs[1].set_title("Test output")
    axs[2].imshow(y_pred[idx].squeeze().cpu(), cmap=cmap, norm=normalizer, interpolation='bilinear')
    axs[2].set_title("Pred. output")
    cbar = fig.colorbar(im, cax=axs[3])
    for i, ax in enumerate(axs.flatten()):
        if i < 3:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
    cbar.outline.set_visible(False)
    PlotConfig.save_fig(fig, str(FIGS / "darcy-flow-pred"))
    plt.close(fig)

@torch.no_grad
def plot_prediction_with_err(x, y, y_pred, idx: int):

    vmin = np.min([y[idx].squeeze().cpu().min(),y_pred[idx].squeeze().cpu().min()])
    vmax = np.max([y[idx].squeeze().cpu().max(),y_pred[idx].squeeze().cpu().max()])
    # cmap = sns.color_palette("viridis", as_cmap=True)
    cmap = cm.bwr
    normalizer = Normalize(vmin, vmax)
    im         = cm.ScalarMappable(norm=normalizer, cmap=cmap)

    PlotConfig.setup()
    figsize = PlotConfig.convert_width((4, 1), page_scale=1)
    fig, axs = plt.subplots(1, 6, figsize=figsize,
                            gridspec_kw={"width_ratios":[1,1,1, 0.05, 1, 0.05]})
    # cmap = sns.color_palette("viridis", as_cmap=True)
    axs[0].imshow(x[idx].squeeze().cpu(), cmap=cmap, interpolation='bilinear')
    axs[0].set_title("Input")
    axs[1].imshow(y[idx].squeeze().cpu(), cmap=cmap, norm=normalizer, interpolation='bilinear')
    axs[1].set_title("Output ($y$)")
    axs[2].imshow(y_pred[idx].squeeze().cpu(), cmap=cmap, norm=normalizer, interpolation='bilinear')
    axs[2].set_title("Pred ($\hat{y}$)")
    cbar = fig.colorbar(im, cax=axs[3])
    cbar.outline.set_visible(False)

    pred = y_pred[idx].squeeze().cpu().numpy()
    true = y[idx].squeeze().cpu().numpy()
    err   = np.abs(pred - true)
    cmap = cm.bwr
    normalizer = Normalize(err.min(), err.max())
    im1        = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    # err   = np.abs(err/ (true+1e-6) )
    axs[4].imshow(err, cmap=cmap, norm=normalizer, interpolation='bilinear')
    axs[4].set_title("$|y - \hat{y}|$")
    cbar1 = fig.colorbar(im1, cax=axs[5])
    cbar1.outline.set_visible(False)
    # Adjust the position of axs[4] and axs[5] to add space
    axs[4].set_position([axs[4].get_position().x0 + 0.05,  # Move left slightly
                        axs[4].get_position().y0,
                        axs[4].get_position().width,
                        axs[4].get_position().height])
    axs[5].set_position([axs[5].get_position().x0 + 0.05,  # Move left slightly
                        axs[5].get_position().y0,
                        axs[5].get_position().width,
                        axs[5].get_position().height])
    for i, ax in enumerate([axs[0], axs[1], axs[2], axs[4]]):
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
    PlotConfig.save_fig(fig, str(FIGS / "darcy-flow-pred-err"))
    plt.close(fig)


def main():
    seed_everything(24)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_models = [
        "krno_20_ep300_40.pt",
        "krno_20_ep300_41.pt",
        "krno_20_ep300_42.pt",
        "krno_20_ep300_43.pt",
        "krno_20_ep300_44.pt",
    ]
    test_l2_rel_vec = []
    for model in trained_models:
        data = torch.load(CKPT_DIR / model)
        test_l2_rel_vec.append(data["test"][20]["l2_rel_vec"])
    test_l2_rel_vec = torch.tensor(test_l2_rel_vec)
    print(f"{test_l2_rel_vec.mean() * 100:.2f} +/- {test_l2_rel_vec.std() * 100:.2f}")

    # setting up computational grid for training
    data_res = 20
    quad_grid = quadrature.get_quad_grid(
        quadrature.trapezoidal_vecs, [data_res, data_res], [-1, -1], [1, 1]
    )
    cart_grid = quadrature.quad_to_cartesian_grid(quad_grid)
    cart_grid = quadrature.cart_grid_to_device(cart_grid, device)

    model = models.KhatriRaoNO.easy_init(
        d=2,
        in_channels=1,
        out_channels=1,
        lifting_channels=20,
        integral_channels=20,
        n_integral_layers=3,
        projection_channels=128,
        n_hidden_units=128,
        n_hidden_layers=3,
        nonlinearity=nn.SiLU(),
    )

    _, _, test_loaders = load_darcy_flow_20()

    _, state_dict = data["log"]["top_k"][0]  # type: ignore
    model.load_state_dict(state_dict)
    model.to(device)
    for i, batch in enumerate(test_loaders[20]):
        x, y = unpack_batch(batch)
        x, y = x.to(device), y.to(device)
        y_pred = model(cart_grid, x)
        # 14 looks more interesting than some of the other outputs
        # plot_prediction(x, y, y_pred, 14)
        # plot_prediction(x, y, y_pred, 14)
        plot_prediction_with_err(x, y, y_pred, 14)
        break


if __name__ == "__main__":
    main()
