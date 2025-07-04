""" Training script for Neural Operators on Darcy Flow dataset
20x20 - 0.96 +/- 0.03 %
"""

import random
import argparse
import os
import pathlib
import time
from typing import Any, OrderedDict

import torch
from torch import nn, Tensor
from jaxtyping import Float
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import scipy.io as sio
from neuralop.models import FNO
from neuralop.datasets import load_darcy_flow_small
import matplotlib.pyplot as plt
import seaborn as sns

from khatriraonop import models, quadrature

torch.set_float32_matmul_precision("high")

CURR_DIR = pathlib.Path(__file__).parent.absolute()
FIG_DIR = CURR_DIR / "figs"
CKPT_DIR = CURR_DIR / "ckpts"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", type=str, required=False, default="krno", choices=["fno", "krno"]
)
parser.add_argument("--dataset", type=int, required=False, default=20, choices=[16, 20])
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if args.model == "fno":
    FNO_FLAG = True
elif args.model == "krno":
    FNO_FLAG = False
else:
    raise ValueError(f"Model type {args.model} not recognized.")


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


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_darcy_flow_16():
    train_loader, test_loaders, _ = load_darcy_flow_small(
        n_train=1000,
        batch_size=16,
        positional_encoding=True,
        test_resolutions=[16, 32],
        n_tests=[100, 50],
        test_batch_sizes=[16, 16],
        encode_input=True,
        encode_output=False,
    )
    train_dset, val_dset = random_split(train_loader.dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(val_dset, batch_size=16, shuffle=True)
    return train_loader, valid_loader, test_loaders


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


load_data_fns = {
    16: load_darcy_flow_16,
    20: load_darcy_flow_20,
}


def load_data(dataset: int):
    return load_data_fns[dataset]()


@torch.no_grad
def get_rel_error(mod_outputs_0, test_y_gt) -> float:
    """
    calculate the relative error between ref and truth outputs
    """
    error_pca = mod_outputs_0 - test_y_gt
    error_pca_norm = np.linalg.norm(error_pca.reshape(test_y_gt.shape[0], -1), axis=1)
    divisor = np.linalg.norm(test_y_gt.reshape(test_y_gt.shape[0], -1), axis=1)
    error_pca_norm_rel = error_pca_norm / divisor
    mean_error = np.mean(error_pca_norm_rel)

    return mean_error


def unpack_batch(batch):
    if FNO_FLAG:
        x = batch["x"]
    else:
        x = batch["x"].permute(0, 2, 3, 1)
    y = batch["y"].permute(0, 2, 3, 1)
    return x, y


def make_prediction(model, x, cart_grid):
    if FNO_FLAG:
        y_pred = model(x).permute(0, 2, 3, 1)
    else:
        y_pred = model(cart_grid, x)
    return y_pred


@torch.no_grad
def compute_test_loss(model, test_loader, cart_grid, device) -> dict[str, float]:
    # mse, vector l2 norm, l2 norm integrated over domain
    errors: dict[str, list[float]] = {"mse": [], "l2_rel_vec": [], "l2_rel_int": []}
    for batch in test_loader:
        x, y = unpack_batch(batch)
        x, y = x.to(device), y.to(device)
        y_pred = make_prediction(model, x, cart_grid)
        errors["mse"].append((y - y_pred).pow(2).mean().item())
        errors["l2_rel_vec"].append(
            get_rel_error(y_pred.cpu().numpy(), y.cpu().numpy())
        )
        errors["l2_rel_int"].append(l2_norm_rel(y_pred, y, cart_grid).item())

    return {k: sum(v) / len(v) for k, v in errors.items()}


@torch.no_grad
def make_test_plot(model, test_loader, cart_grid, device, fname):
    for batch in test_loader:
        x, y = unpack_batch(batch)
        x, y = x.to(device), y.to(device)
        y_pred = make_prediction(model, x, cart_grid)
        batch_size = x.shape[0]
        if FNO_FLAG:
            x = x.permute(0, 2, 3, 1)
        fig, axs = plt.subplots(3, batch_size, figsize=(12, 6))
        for j in range(batch_size):
            axs[0, j].imshow(x[j, :, :, 0].cpu().numpy())
            axs[1, j].imshow(y[j, :, :, 0].cpu().numpy())
            axs[2, j].imshow(y_pred[j, :, :, 0].cpu().numpy())
            for ax in axs[:, j]:
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(FIG_DIR / fname)
        plt.close(fig)
        break


def loss_fn(y_pred, y, quad_rule):
    return quadrature.integrate_on_domain((y - y_pred).pow(2), quad_rule).mean()


def l2_norm_rel(y_pred, y, quad_rule):
    l2_norm = loss_fn(y_pred, y, quad_rule).sqrt()
    divisor = loss_fn(y, torch.zeros_like(y), quad_rule).sqrt()
    return l2_norm / divisor


def get_valid_loss(model, cart_grid, valid_loader, device):
    loss = []
    for batch in valid_loader:
        x, y = unpack_batch(batch)
        x, y = x.to(device), y.to(device)
        y_pred = make_prediction(model, x, cart_grid)
        loss.append(loss_fn(y_pred, y, cart_grid).item())
    return sum(loss) / len(loss)


class TrainLog:
    def __init__(self, n_ckpts: int):
        self.all_metrics: dict[str, list[float]] = {}
        self.val_list = []
        self.n_ckpts = n_ckpts

    def log(self, name: str, iter_count: int, value: float):
        if name not in self.all_metrics:
            self.all_metrics[name + "_iter"] = []
            self.all_metrics[name + "_time"] = []
            self.all_metrics[name] = []
        self.all_metrics[name + "_iter"].append(iter_count)
        self.all_metrics[name + "_time"].append(time.time())
        self.all_metrics[name].append(value)

    def log_val(self, name: str, iter_count: int, val_loss: float, model):
        self.log(name, iter_count, val_loss)
        cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        self.val_list.append((val_loss, cpu_state_dict))
        self.val_list = sorted(self.val_list)
        if len(self.val_list) > self.n_ckpts:
            self.val_list = self.val_list[: self.n_ckpts]

    def log_dict(self, metrics: dict[str, float], iter_count: int, prefix: str):
        for metric, value in metrics.items():
            self.log(f"{prefix}_{metric}", iter_count, value)

    def best_ckpt(self) -> OrderedDict:
        _, state_dict = self.val_list[0]
        return state_dict

    def plot_metrics(self, names: list[str], save_path: str):
        fig, ax = plt.subplots(1, 1)
        sns.set_style("whitegrid")
        for name in names:
            ax.plot(
                self.all_metrics[name + "_iter"], self.all_metrics[name], label=name
            )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Metric")
        ax.legend()
        fig.savefig(save_path, bbox_inches="tight")

    def to_dict(self) -> dict[str, Any]:
        return {"metrics": self.all_metrics, "top_k": self.val_list}


def main():
    seed_everything(args.seed)
    if FNO_FLAG:
        print("Training FNO model")
    else:
        print("Training KhatriRao model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loading data
    data_res = args.dataset
    train_loader, valid_loader, test_loaders = load_data(data_res)

    # setting up computational grid for training
    quad_grid = quadrature.get_quad_grid(
        quadrature.trapezoidal_vecs, [data_res, data_res], [-1, -1], [1, 1]
    )
    cart_grid = quadrature.quad_to_cartesian_grid(quad_grid)
    cart_grid = quadrature.cart_grid_to_device(cart_grid, device)

    # computational grid for testing
    test_grids = {}
    for res, _ in test_loaders.items():
        test_quad_grid = quadrature.get_quad_grid(
            quadrature.trapezoidal_vecs, [res, res], [-1, -1], [1, 1]
        )
        test_grids[res] = quadrature.quad_to_cartesian_grid(test_quad_grid)
        test_grids[res] = quadrature.cart_grid_to_device(test_grids[res], device)

    # initializing the model
    if FNO_FLAG:
        model = FNO(
            n_modes=(8, 8),
            hidden_channels=20,
            in_channels=1,
            out_channels=1,
            lifting_channels=20,
            projection_channels=128,
        )
    else:
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

    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # number of ckpoints to save
    log = TrainLog(n_ckpts=10)
    n_epochs = 300
    epoch, loss = 0, torch.tensor(0.0)
    avg_epoch_time = 0.0
    iter_count = 0
    for epoch in range(1, n_epochs + 1):
        for batch in train_loader:
            # getting into format expected by our package
            x, y = unpack_batch(batch)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = make_prediction(model, x, cart_grid)
            loss = loss_fn(y_pred, y, cart_grid)
            loss.backward()
            optimizer.step()
            iter_count += 1
        log.log("train_loss", iter_count, loss.item())
        model.eval()
        val_loss = get_valid_loss(model, cart_grid, valid_loader, device)
        model.train()
        log.log_val("val_loss", iter_count, val_loss, model)
    # one more
    model.load_state_dict(log.best_ckpt())
    model.to(device)
    model.eval()
    test_loss = {}  # computing test loss
    loss_str = "TEST ERRS: "
    for res, test_loader in test_loaders.items():
        test_loss[res] = compute_test_loss(model, test_loader, test_grids[res], device)
        loss_str += f"{res} - "
        for metric, val in test_loss[res].items():
            loss_str += f"{metric.upper()}: {val:.3f}, "
    print(
        f"[{avg_epoch_time:.2f}secs/epoch] EPOCH {epoch:02d} | "
        f"LOSS: {loss.item():.3f} | "
        f"{loss_str}"
    )
    # making plots
    name = "fno" if FNO_FLAG else "krno"
    for res, test_loader in test_loaders.items():
        make_test_plot(
            model, test_loader, test_grids[res], device, f"{name}_{res}_preds.png"
        )
    save_state = {
        "epoch": epoch,
        "seed": args.seed,
        "log": log.to_dict(),
        "test": test_loss,
    }
    with open(CKPT_DIR / f"{name}_{data_res}_ep{epoch}_{args.seed}.pt", "wb") as f:
        torch.save(save_state, f)


if __name__ == "__main__":
    main()
