from __future__ import annotations

import argparse
import time
import os
import pathlib
import random
from typing import Any

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from khatriraonop import models, quadrature
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


CURR_DIR = pathlib.Path(__file__).parent.absolute()
DATA_DIR = CURR_DIR / "loca-data"
CKPT_DIR = CURR_DIR / "ckpts"

root_base_path = os.path.dirname(os.path.dirname(CURR_DIR))
import sys
sys.path
sys.path.append(root_base_path)
from utils.utilities3 import *

os.makedirs(CKPT_DIR, exist_ok=True)

# torch.set_float32_matmul_precision("high")
# torch.set_default_dtype(torch.float64)
torch.set_float32_matmul_precision("highest")

myloss = LpLoss(size_average=False)

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels, out_channels):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 1 timestep + 2 locations (u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2+in_channels, self.width)
        # input size is 2+in_channels: the solution of the previous 1 timesteps (u(t-1, x, y),  x, y), where u(t-1, x, y) is 3D field

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)



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


def npz_to_autoregressive_data(data_path: str, subscript: str):
    data = np.load(data_path)
    s = torch.tensor(data[f"s_{subscript}"], dtype=torch.get_default_dtype())
    u = torch.tensor(data[f"U_{subscript}"], dtype=torch.get_default_dtype())
    s_total = torch.cat([u.unsqueeze(1), s], dim=1)
    return s_total


def get_raw_data():
    if not os.path.exists(str(DATA_DIR / "train_SW.pt")):
        s_train = npz_to_autoregressive_data(str(DATA_DIR / "train_SW.npz"), "train")
        s_test = npz_to_autoregressive_data(str(DATA_DIR / "test_SW.npz"), "test")
        s_test = s_test[:, :60]
        torch.save(s_train, str(DATA_DIR / "train_SW.pt"))
        torch.save(s_test, str(DATA_DIR / "test_SW.pt"))
    else:
        s_train = torch.load(str(DATA_DIR / "train_SW.pt"))
        s_test = torch.load(str(DATA_DIR / "test_SW.pt"))
    return s_train, s_test


def get_data_loaders(lag: int, forward_steps: int, batch_size: int = 32):
    s_train, s_test = get_raw_data()
    s_train = ARDataset(s_train, lag, forward_steps)
    s_train, s_val = random_split(s_train, [0.8, 0.2])
    train_loader = DataLoader(s_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(s_val, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        ARDataset(s_test, lag, forward_steps), batch_size=batch_size, shuffle=True
    )
    return train_loader, valid_loader, test_loader


def make_prediction(model, x, cart_grid):
    if FNO_FLAG:
        y_pred = model(x.squeeze()).unsqueeze(1)
    else:
        y_pred = model(cart_grid, x)
    return y_pred

# error metrics


def loss_fn(y_pred, y_true, quad_rule):
    return quadrature.integrate_on_domain((y_true - y_pred).pow(2), quad_rule).mean()


def l2_norm_rel(y_pred, y_true, quad_rule):
    l2_norm = loss_fn(y_pred, y_true, quad_rule).sqrt()
    divisor = loss_fn(y_true, torch.zeros_like(y_true), quad_rule).sqrt()
    return l2_norm / divisor


@torch.no_grad
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


@torch.no_grad
def compute_test_metrics(model, test_loader, cart_grid, device) -> dict[str, float]:
    # mse, vector l2 norm, l2 norm integrated over domain
    errors: dict[str, list[float]] = {
        "loss": [],
        "l2_rel_vec": [],
        "l2_rel_int": [],
    }
    for xx, yy in tqdm(test_loader):
        loss_ = torch.tensor(0.0).to(device)
        xx, yy = xx.to(device), yy.to(device)
        if FNO_FLAG:
            T = yy.shape[1]
            S = yy.shape[2]
            step = 1
            batch_size = yy.shape[0]
            xx = xx.permute(0,2,3,4,1) ## [B,lag,Nx,Ny,C] -> [B,Nx,Ny,C,lag]
            yy = yy.permute(0,2,3,4,1) ## [B,T,Nx,Ny,C] -> [B,Nx,Ny,C,T]
            ## train autoregressively
            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx.reshape((batch_size, S, S, -1))) #[B,Nx,Ny,C,lag] -> [B,Nx,Ny,C*lag]
                im = im.unsqueeze(-1)
                loss_ += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
                if t == 0:
                    y_pred = im
                else:
                    y_pred = torch.cat((y_pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)
            y_pred = y_pred.permute(0,4,1,2,3) ## [B,Nx,Ny,C,T] -> [B,T,Nx,Ny,C]
            yy     = yy.permute(0,4,1,2,3) ## [B,Nx,Ny,C,T] -> [B,T,Nx,Ny,C]
        else:
            y_pred = model(cart_grid, xx)
            loss_ = loss_fn(y_pred, yy)
        errors["loss"].append(loss_.item())
        errors["l2_rel_vec"].append(get_rel_error(y_pred, yy))
        errors["l2_rel_int"].append(l2_norm_rel(y_pred, yy, cart_grid).item())

    return {k: sum(v) / len(v) for k, v in errors.items()}


class TrainLog:
    all_metrics: dict[str, list[float]]
    val_list: list[tuple[float, dict]]
    n_ckpts: int

    def __init__(self, n_ckpts: int):
        self.all_metrics: dict[str, list[float]] = {}
        self.val_list = []
        self.n_ckpts = n_ckpts
        self.hparams = None

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
            "hparams": self.hparams,
        }

    @classmethod
    def from_dict(cls, log_dict) -> TrainLog:
        n_ckpts = len(log_dict["top_k"])
        log = cls(n_ckpts)
        log.all_metrics = log_dict["metrics"]
        log.val_list = log_dict["top_k"]
        log.hparams = log_dict["hparams"]
        return log

    def __repr__(self):
        metric_list = [
            metric
            for metric in self.all_metrics.keys()
            if ("iter" not in metric) and ("time" not in metric)
        ]
        return f"TrainLog({metric_list})"


def main(lag: int):
    seed = 42
    seed_everything(seed)
    # to leverage quad structure lag must == forward_steps
    print("Loading data...", flush=True)
    forward_steps = lag
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, test_loader = get_data_loaders(
        lag, forward_steps, batch_size=64, #64
    )
    S = 32
    # get computational grid
    quad_fns = [
        quadrature.midpoint_vecs,
        quadrature.trapezoidal_vecs,
        quadrature.trapezoidal_vecs,
    ]
    quad_grid = quadrature.get_quad_grid(
        quad_fns, [lag, S, S], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]
    )
    cart_grid = quadrature.quad_to_cartesian_grid(quad_grid)
    cart_grid = quadrature.cart_grid_to_device(cart_grid, device)

    if FNO_FLAG:
        hparams = dict(
            modes = 12,
            width = 20,
            in_channels=3*lag,
            out_channels=3,
        )
        # initialize model
        model = FNO2d(hparams['modes'], hparams['modes'], hparams['width'],
                    hparams['in_channels'], hparams['out_channels']).to(device)
    else:
        hparams = dict(
            d=3,
            in_channels=3,
            out_channels=3,
            lifting_channels=128,
            integral_channels=20,
            n_integral_layers=3,
            projection_channels=128,
            n_hidden_units=32,
            n_hidden_layers=3,
            nonlinearity=nn.SiLU(),
        )
        # initialize model
        model = models.KhatriRaoNO.easy_init(**hparams)  # type: ignore
    
    hparams["lag"] = lag
    valid_freq = 500
    n_epochs   = 100 # 100
    optim      = 'AdamW'
    learning_rate = 1e-3
    lr_decay_flag = False
    scheduler_step = 100 #100
    scheduler_gamma = 0.5
    model.to(device)
    model.train()
    if optim=='AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif optim=='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        # optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    if lr_decay_flag:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    log = TrainLog(n_ckpts=10)
    log.log_hparams(hparams)
    n_iters, epoch = 0, 0
    print("Training FNO on shallow water autoregressive dataset...", flush=True)
    for epoch in range(1, n_epochs + 1):
        for xx, yy in tqdm(train_loader):
            loss = torch.tensor(0.0).to(device)
            xx, yy = xx.to(device), yy.to(device)
            optimizer.zero_grad()
            if FNO_FLAG:
                T = yy.shape[1]
                S = yy.shape[2]
                step = 1
                batch_size = yy.shape[0]
                xx = xx.permute(0,2,3,4,1) ## [B,lag,Nx,Ny,C] -> [B,Nx,Ny,C,lag]
                yy = yy.permute(0,2,3,4,1) ## [B,T,Nx,Ny,C] -> [B,Nx,Ny,C,T]
                ## train autoregressively
                for t in range(0, T, step):
                    y = yy[..., t:t + step]
                    im = model(xx.reshape((batch_size, S, S, -1))) #[B,Nx,Ny,C,lag] -> [B,Nx,Ny,C*lag]
                    im = im.unsqueeze(-1) #[B,Nx,Ny,C] -> [B,Nx,Ny,C,1]
                    loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
                    if t == 0:
                        y_pred = im
                    else:
                        y_pred = torch.cat((y_pred, im), -1)
                    xx = torch.cat((xx[..., step:], im), dim=-1)
                y_pred = y_pred.permute(0,4,1,2,3) ## [B,Nx,Ny,C,T] -> [B,T,Nx,Ny,C]
            else:
                y_pred = model(cart_grid, xx)
                loss = loss_fn(y_pred, yy)
            loss.backward()
            optimizer.step()
            if lr_decay_flag:
                scheduler.step()
            n_iters += 1
            if n_iters % valid_freq == 0:
                log.log("train_loss", n_iters, loss.item())
                model.eval()
                # validation
                valid_metrics = compute_test_metrics(
                    model, valid_loader, cart_grid, device
                )
                log.log_val("valid_loss", n_iters, valid_metrics.pop("loss"), model)
                log.log_dict(valid_metrics, n_iters, "valid")
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
    with open(CKPT_DIR / f"fno_it{n_iters}_{seed}.pt", "wb") as f:
        torch.save(save_state, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lag", "-l", type=int, default=5)
    parser.add_argument(
        "--model", type=str, required=False, default="fno", choices=["fno", "krno"])
    args = parser.parse_args()
    if args.model == "fno":
        FNO_FLAG = True
    elif args.model == "krno":
        FNO_FLAG = False
    else:
        raise ValueError(f"Model type {args.model} not recognized.")
    main(args.lag)
    
