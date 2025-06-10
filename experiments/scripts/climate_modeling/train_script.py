from __future__ import annotations

import argparse
import time
import os
import pathlib
import random
from typing import Any, Optional

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader, random_split
from khatriraonop import models, quadrature
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

CURR_DIR = pathlib.Path(__file__).parent.absolute()
DATA_DIR = CURR_DIR / "data"
CKPT_DIR = CURR_DIR / "ckpts"

os.makedirs(CKPT_DIR, exist_ok=True)

torch.set_float32_matmul_precision("high")


class ARDataset(Dataset):
    def __init__(self, x: Float[Tensor, "batch nt ..."], lag: int, forward_steps: int):
        self.x = x
        self.lag = lag
        self.forward_steps = forward_steps

    def __len__(self):
        return self.x.shape[0] * (self.x.shape[1] - self.lag - self.forward_steps + 1)

    def __getitem__(self, idx):
        seq_idx = idx // (self.x.shape[1] - self.lag - self.forward_steps + 1)
        start_idx = idx % (self.x.shape[1] - self.lag - self.forward_steps + 1)
        x = self.x[seq_idx, start_idx : start_idx + self.lag]
        y = self.x[
            seq_idx, start_idx + self.lag : start_idx + self.lag + self.forward_steps
        ]
        return x, y


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def normalize(s_data, mu=None, std=None):
    dim = s_data.shape[-1]
    if (mu is None) or (std is None):
        mu, std = s_data.reshape(-1, dim).mean(0), s_data.reshape(-1, dim).std(0)
    norm_params = {"mu": mu, "std": std}
    return (s_data - mu).div(std + 1e-6), norm_params


def unnormalize(y, mu, std):
    return y * std + mu


def npz_to_autoregressive_data(data_path: str, subscript: str):
    data = np.load(data_path)
    s = torch.tensor(data[f"S_{subscript}"], dtype=torch.get_default_dtype())
    u = torch.tensor(data[f"U_{subscript}"], dtype=torch.get_default_dtype())
    s_total = torch.stack([u, s], dim=-1).reshape(-1, 72, 72, 2)
    s_train = s_total[:1825]
    n_train = int(0.8 * 1825)
    s_train, s_val = s_train[:n_train], s_train[n_train:]
    s_train, norm_params = normalize(s_train)
    s_val, _ = normalize(s_val, **norm_params)
    s_test, _ = normalize(s_total[1825:], **norm_params)
    return s_train.unsqueeze(0), s_val.unsqueeze(0), s_test.unsqueeze(0), norm_params


def get_raw_data():
    s_train, s_val, s_test, norm_params = npz_to_autoregressive_data(
        str(DATA_DIR / "weather_dataset.npz"), "train"
    )
    return s_train, s_val, s_test, norm_params


def get_data_loaders(lag: int, forward_steps: int, batch_size: int = 32):
    s_train, s_val, s_test, _ = get_raw_data()
    s_train = ARDataset(s_train, lag, forward_steps)
    s_val = ARDataset(s_val, lag, forward_steps)
    # s_train, s_val = random_split(s_train, [0.8, 0.2])
    train_loader = DataLoader(s_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(s_val, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        ARDataset(s_test, lag, forward_steps), batch_size=batch_size, shuffle=True
    )
    return train_loader, valid_loader, test_loader


# error metrics


def loss_fn(y_pred, y_true, quad_rule):
    return quadrature.integrate_on_domain((y_true - y_pred).pow(2), quad_rule).mean()


def l2_norm_rel(y_pred, y_true, quad_rule):
    l2_norm = loss_fn(y_pred, y_true, quad_rule).sqrt()
    divisor = loss_fn(y_true, torch.zeros_like(y_true), quad_rule).sqrt()
    return l2_norm / divisor


@torch.no_grad()
def get_rel_error(mod_outputs_0, test_y_gt) -> float:
    """
    calculate the relative error between ref and truth outputs
    """
    mod_outputs_0, test_y_gt = mod_outputs_0.cpu().numpy(), test_y_gt.cpu().numpy()
    error_pca = mod_outputs_0 - test_y_gt
    error_pca_norm = np.linalg.norm(error_pca.reshape(test_y_gt.shape[0], -1), axis=1)
    divisor = np.linalg.norm(test_y_gt.reshape(test_y_gt.shape[0], -1), axis=1)
    error_pca_norm_rel = error_pca_norm / divisor
    mean_error = np.mean(error_pca_norm_rel)

    return mean_error


@torch.no_grad()
def compute_test_metrics(model, test_loader, cart_grid, device) -> dict[str, float]:
    # mse, vector l2 norm, l2 norm integrated over domain
    errors: dict[str, list[float]] = {
        "loss": [],
        "l2_rel_vec": [],
        "l2_rel_int": [],
    }
    for x, y in tqdm(test_loader):
        x, y = x.to(device), y.to(device)
        y_pred = model(cart_grid, x)
        errors["loss"].append(loss_fn(y_pred, y, cart_grid).item())
        errors["l2_rel_vec"].append(get_rel_error(y_pred, y))
        errors["l2_rel_int"].append(l2_norm_rel(y_pred, y, cart_grid).item())

    return {k: sum(v) / len(v) for k, v in errors.items()}


class TrainLog:
    all_metrics: dict[str, list[float]]
    val_list: list[tuple[float, dict]]
    n_ckpts: int
    hparams: dict[str, Any]
    last_ckpt: tuple[int, dict]

    def __init__(self, n_ckpts: int):
        # a dictionary for storing misc. validation / training metrics
        self.all_metrics: dict[str, list[float]] = {}
        # a sorted list containing the n_ckpts best ckpts
        self.val_list = []
        self.n_ckpts = n_ckpts
        # contains the most recent training_iter and most recent ckpt
        self.last_ckpt = (-1, {})
        #  a dictionary containing some hyperparameters (optional)
        self.hparams = {}

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

    def log_hparams(self, hparams: dict[str, Any]):
        self.hparams = hparams

    def log_ckpt(self, iter_count: int, model):
        cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        self.last_ckpt = (iter_count, cpu_state_dict)

    def best_ckpt(self) -> dict:
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
        return {
            "metrics": self.all_metrics,
            "top_k": self.val_list,
            "n_ckpts": self.n_ckpts,
            "hparams": self.hparams,
            "last_ckpt": self.last_ckpt,
        }

    @classmethod
    def from_dict(cls, log_dict) -> TrainLog:
        log = cls(log_dict["n_ckpts"])
        log.all_metrics = log_dict["metrics"]
        log.val_list = log_dict["top_k"]
        log.hparams = log_dict["hparams"]
        log.last_ckpt = log_dict["last_ckpt"]
        return log

    def __repr__(self):
        metric_list = [
            metric
            for metric in self.all_metrics.keys()
            if ("iter" not in metric) and ("time" not in metric)
        ]
        return f"TrainLog({metric_list})"


def init_model_and_log(lag: int, ckpt: Optional[str]):
    if ckpt is None:
        hparams = dict(
            d=3,
            in_channels=2,
            out_channels=2,
            lifting_channels=20,
            integral_channels=20,
            n_integral_layers=3,
            projection_channels=256,
            n_hidden_units=64,
            n_hidden_layers=2,
            nonlinearity=nn.SiLU(),
        )
        # initialize model
        model = models.KhatriRaoNO.easy_init(**hparams)  # type: ignore
        log = TrainLog(n_ckpts=10)
        hparams["lag"] = lag
        log.log_hparams(hparams)
        n_iter_start = 0
    else:
        print("Loading from ckpt:", ckpt)
        with open(ckpt, "rb") as fp:
            train_state = torch.load(fp)
        log = TrainLog.from_dict(train_state["log"])
        lag = log.hparams.pop("lag")
        model = models.KhatriRaoNO.easy_init(**log.hparams)
        log.hparams["lag"] = lag
        n_iter_start, state_dict = log.last_ckpt
        model.load_state_dict(state_dict)
        model.train()
    return model, log, n_iter_start


def main(lag: int, ckpt: str):
    seed = 42
    seed_everything(seed)
    # to leverage quad structure lag must == forward_steps
    print("Loading data...", flush=True)
    forward_steps = lag
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, test_loader = get_data_loaders(
        lag, forward_steps, batch_size=16
    )
    # get computational grid
    quad_fns = [
        quadrature.midpoint_vecs,
        quadrature.trapezoidal_vecs,
        quadrature.trapezoidal_vecs,
    ]
    quad_grid = quadrature.get_quad_grid(
        quad_fns, [lag, 72, 72], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]
    )
    cart_grid = quadrature.quad_to_cartesian_grid(quad_grid)
    cart_grid = quadrature.cart_grid_to_device(cart_grid, device)
    valid_freq = 1000
    model, log, n_iter_start = init_model_and_log(lag, ckpt)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    n_epochs = 1_000
    n_iters, epoch = 0, 0
    epoch = 0

    loss = torch.tensor(0.0)
    print("Training KRNO on weather dataset...", flush=True)
    for epoch in range(1, n_epochs + 1):
        for x, y in tqdm(train_loader):
            if n_iters > n_iter_start:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model(cart_grid, x)
                loss = loss_fn(y_pred, y, cart_grid)
                loss.backward()
                optimizer.step()
            n_iters += 1
            if (n_iters % valid_freq == 0) and (n_iters > n_iter_start):
                log.log("train_loss", n_iters, loss.item())
                model.eval()
                # validation
                valid_metrics = compute_test_metrics(
                    model, valid_loader, cart_grid, device
                )
                log.log_val("valid_loss", n_iters, valid_metrics.pop("loss"), model)
                log.log_dict(valid_metrics, n_iters, "valid")
                log.log_ckpt(n_iters, model)
                model.train()
                # printing
                print_str = f"EPOCH: {epoch:02d} | TRAIN: {loss.item():.3f}"
                print_str += " | VALID: " + " , ".join(
                    [f"{k}: {v:.3f}" for k, v in valid_metrics.items()]
                )
                print(print_str, flush=True)
                with open(CKPT_DIR / f"krno_it{n_iters}_{seed}.pt", "wb") as f:
                    torch.save({"log": log.to_dict()}, f)

    model.load_state_dict(log.best_ckpt())
    model.to(device)
    model.eval()
    print("Computing test metrics...", flush=True)
    # logging valid and test metrics
    valid_metrics = compute_test_metrics(model, valid_loader, cart_grid, device)
    test_metrics = compute_test_metrics(model, test_loader, cart_grid, device)
    print_str = f"EPOCH: {epoch:02d} | TRAIN: {loss.item():.3f}"
    print_str += " | VALID: " + " , ".join(
        [f"{k}: {v:.3f}" for k, v in valid_metrics.items()]
    )
    print_str += " | TEST: " + " , ".join(
        [f"{k}: {v:.3f}" for k, v in test_metrics.items()]
    )
    print(print_str, flush=True)
    log.plot_metrics(
        ["train_loss", "valid_loss"], str(CKPT_DIR / f"krno_loss_curves_{seed}.png")
    )
    save_state = {
        "epoch": epoch,
        "seed": seed,
        "log": log.to_dict(),
        "test": test_metrics,
    }
    with open(CKPT_DIR / f"krno_it{n_iters}_{seed}.pt", "wb") as f:
        torch.save(save_state, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lag", "-l", type=int, default=7)  # 14)
    parser.add_argument("--ckpt", type=str)
    args = parser.parse_args()
    main(args.lag, args.ckpt)
