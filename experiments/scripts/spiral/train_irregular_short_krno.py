from __future__ import annotations

import argparse
import time
import os
import pathlib
import random
from typing import Any
import csv

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader, random_split
from khatriraonop import models, quadrature
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd


CURR_DIR = pathlib.Path(__file__).parent.absolute()
DATA_DIR = CURR_DIR / "loca-data"
CKPT_DIR = CURR_DIR / "ckpts"
FIG_DIR = CURR_DIR / "figs_high_irreg"
OUTPUT_DIR = CURR_DIR / "outputs_high_irreg"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.set_float32_matmul_precision("high")


root_base_path = os.path.dirname(os.path.dirname(CURR_DIR))
import sys
sys.path
sys.path.append(root_base_path)
sys.path.append(str(CURR_DIR / ".."))

from plotting_config import PlotConfig


parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=False)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
args = parser.parse_args()

class CyclicTrajDataset(Dataset):
    def __init__(self, data, timestamps, ntrain_val=100, flag='train', lag=5, forward_steps=5,
                step=1, scale=True):
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.seq_len = lag
        self.step = step
        self.forward_steps = forward_steps

        _N     = len(data)
        border1s = [0, int(0.8*ntrain_val) - self.seq_len, ntrain_val - self.seq_len]
        border2s = [int(0.8*ntrain_val), ntrain_val, _N]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.scaler = StandardScaler()

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = timestamps[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.forward_steps

        seq_x = self.data_x[s_begin:s_end:self.step]
        seq_y = self.data_y[r_begin:r_end:self.step]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.step]
        seq_y_mark = self.data_stamp[r_begin:r_end:self.step]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.forward_steps + 1

    def inverse_transform(self, data, data_time=None):
        data = self.scaler.inverse_transform(data)
        return data

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def generate_sorted_integers(range_min: int, range_max: int, size: int):
    while True:
        numbers = np.sort(np.random.choice(np.arange(range_min, range_max), size=size, replace=False))
        if np.all(np.diff(numbers) <= 4):
            return numbers

def generate_spiral2d(nspiral=10,
                      ntotal=240,
                      nsample=100,
                      sample_freq=2,
                      start=0.,
                      stop=1,  # approximately equal to 6pi
                      noise_std=.1,
                      a=0.,
                      b=1.,
                      noise_flag=False,
                      savefig=True):
    """Parametric formula for 2d spiral is `r = a + b * theta`.

    Args:
      nspiral: number of spirals, i.e. batch dimension
      ntotal: total number of datapoints per spiral
      nsample: number of sampled datapoints for model fitting per spiral
      sample_freq: frequency of sampling for irregularly sampled trajectories
      start: spiral starting theta value
      stop: spiral ending theta value
      noise_std: observation noise standard deviation
      a, b: parameters of the Archimedean spiral
      noise_flag: whether to add noise to the observations
      savefig: plot the ground truth for sanity check

    Returns: 
      Tuple where first element is true trajectory of size (nspiral, ntotal, 2),
      second element is noisy observations of size (nspiral, nsample, 2),
      third element is timestamps of size (ntotal,),
      and fourth element is timestamps of size (nsample,)
    """

    # add 1 all timestamps to avoid division by 0
    orig_ts = np.linspace(start, stop, num=ntotal)

    # generate clock-wise and counter clock-wise spirals in observation space
    # with one set of time-invariant latent dynamics
    def generate_traj(orig_ts):
        zs_cc = orig_ts
        rw_cc = a + b * zs_cc
        xs, ys = rw_cc * np.cos(zs_cc) + 5., rw_cc * np.sin(zs_cc)
        orig_traj_cc = np.stack((xs, ys), axis=1)
        return orig_traj_cc

    orig_traj_cc = generate_traj(orig_ts)

    if savefig:
        plt.figure()
        plt.plot(orig_traj_cc[:, 0], orig_traj_cc[:, 1], 'b*-', label='clock')
        plt.legend()
        plt.savefig(FIG_DIR / 'regular_ground_truth_full.png', dpi=500)
        print('Saved ground truth spiral at {}'.format(FIG_DIR / 'regular_ground_truth_full.png'))
        plt.close()

    ## uniformly sampled trajectory
    samp_uni_traj = orig_traj_cc[::sample_freq, :].copy()
    samp_uni_ts   = orig_ts[::sample_freq].copy()

    if savefig:
        plt.figure()
        plt.plot(samp_uni_traj[:, 0], samp_uni_traj[:, 1], 'b*-', label='clock')
        plt.legend()
        plt.savefig(FIG_DIR / 'regular_ground_truth.png', dpi=500)
        print('Saved ground truth spiral at {}'.format(FIG_DIR / 'regular_ground_truth.png'))
        plt.close()

    # sample irregularly sampled trajectories
    samp_irregular_trajs = []
    samp_irregular_ts = []

    for i, _ in enumerate(range(nspiral)):
        ## sample starting timestamps
        # temp_sample_idx = np.sort(np.random.choice(np.arange(0, int(nsample*sample_freq))-1, size=nsample, replace=False))

        # temp_sample_idx = generate_sorted_integers(0, int(nsample*sample_freq)-1, nsample)
        
        # samp_ts   = orig_ts[temp_sample_idx].copy()
        # samp_traj = orig_traj_cc[temp_sample_idx, :].copy()
        # # samp_traj += npr.randn(*samp_traj.shape) * noise_std
        # print(temp_sample_idx[1:] - temp_sample_idx[:-1])
        
        delta_t_max = (samp_uni_ts[1] - samp_uni_ts[0])
        delta_times = np.random.uniform(0, delta_t_max, nsample)
        samp_ts     = samp_uni_ts[0:nsample] + delta_times
        samp_traj   = generate_traj(samp_ts)
        samp_irregular_trajs.append(samp_traj)
        samp_irregular_ts.append(samp_ts)

        if savefig:
            plt.figure()
            plt.plot(samp_traj[:, 0], samp_traj[:, 1], 'b*-', label='clock')
            plt.legend()
            plt.savefig(FIG_DIR / f'irregular_ground_truth_{i}.png', dpi=500)
            print('Saved ground truth spiral at {}'.format(FIG_DIR / f'irregular_ground_truth_{i}.png'))
            plt.close()

    # batching for sample trajectories is good for RNN; batching for original
    # trajectories only for ease of indexing
    samp_irregular_trajs = np.stack(samp_irregular_trajs, axis=0)
    samp_irregular_ts = np.stack(samp_irregular_ts, axis=0)
    return samp_uni_traj, samp_uni_ts, samp_irregular_trajs, samp_irregular_ts, orig_traj_cc, orig_ts



def generate_data(ntotal = 240, nsample = 100, sample_freq = 2):

    nspiral = 10
    start = 0.
    stop = 6 * np.pi
    noise_std = .3
    a = 0.
    b = .3

    # generate toy spiral data
    samp_uni_traj, samp_uni_ts, samp_irregular_trajs, samp_irregular_ts, orig_traj_cc, orig_ts = generate_spiral2d(
        nspiral=nspiral,
        ntotal=ntotal,
        nsample=nsample,
        sample_freq=sample_freq,
        start=start,
        stop=stop,
        noise_std=noise_std,
        a=a, b=b,
        noise_flag=False,
    )

    return samp_uni_traj, samp_uni_ts, samp_irregular_trajs, samp_irregular_ts, orig_traj_cc, orig_ts


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

def loss_fn(y_pred, y, quad_rule):
    return quadrature.integrate_on_domain((y - y_pred).pow(2), quad_rule).mean()


mse_loss_fn = nn.MSELoss(reduction="mean")

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
def compute_test_metrics(model, seq_data, seq_times, device, past_time, future_time, delta_t, sample_freq, flag='val') -> dict[str, float]:
    # mse, vector l2 norm, l2 norm integrated over domain
    errors: dict[str, list[float]] = {
        "loss": [],
        "l2_rel_vec": [],
        "l2_rel_int": [],
        "MSE": [],
    }

    if flag == 'val':
        min_seq_time = 79*delta_t - past_time 
        max_seq_time =  99*delta_t - past_time - future_time - delta_t/2
    elif flag == 'test':
        min_seq_time = 99*delta_t - past_time 
        max_seq_time = 119*delta_t - past_time - future_time - delta_t/2

    no_steps = int(2*(max_seq_time - min_seq_time)/delta_t)
    start_times  = torch.linspace(min_seq_time, max_seq_time, no_steps, dtype=torch.get_default_dtype())


    for start_time in start_times:

        input_time_range = [start_time, start_time + past_time]
        output_time_range = [start_time + past_time + delta_t/2, start_time + past_time + future_time + delta_t/2]
        x, x_ind = get_seq_data_torch(seq_data, seq_times, input_time_range)
        y, y_ind = get_seq_data_torch(seq_data, seq_times, output_time_range)
        x, y = x.to(device), y.to(device)

        # print(x_ind.shape, y_ind.shape)
        # print(f'input window time ranges, {input_time_range}')
        # print(f'input window times, {x_ind}')
        # print(f'output window time ranges, {output_time_range}')
        # print(f'output window times, {y_ind}')

        x_ind = x_ind - start_time
        y_ind = y_ind - start_time - past_time - delta_t/2
        # print(x_ind, y_ind)

        quad_grid_out, quad_grid_in = get_quad_grid(x_ind, y_ind, device, past_time, future_time, delta_t, sample_freq)
        ##
        y_pred = model.super_resolution(quad_grid_out, quad_grid_in, x)
        
        errors["loss"].append(loss_fn(y_pred, y, quad_grid_out).item())
        errors["l2_rel_vec"].append(get_rel_error(y_pred, y))
        errors["l2_rel_int"].append(l2_norm_rel(y_pred, y, quad_grid_out).item())
        errors["MSE"].append(mse_loss_fn(y_pred, y).item())

    return {k: sum(v) / len(v) for k, v in errors.items()}


@torch.no_grad
def compute_forecast_metrics(model, seq_data, seq_times, device,
                            past_time, future_time, delta_t, sample_freq, scaler, flag='test') -> dict[str, float]:
    # mse, vector l2 norm, l2 norm integrated over domain
    errors: dict[str, list[float]] = {
        "RMSE": [],
        "MSE": [],
    }
    y_true = []
    y_preds = []
    t_preds = []

    ## 
    
    seq_start_time = 99*delta_t - past_time
    x_seq_max_time = 99*delta_t

    if flag == 'val':
        min_seq_time = 79*delta_t - past_time 
        max_seq_time =  99*delta_t - past_time - future_time - delta_t/2
    elif flag == 'test':
        min_seq_time = 99*delta_t - past_time 
        max_seq_time = 119*delta_t - past_time - future_time - delta_t/2
    elif flag == 'train':
        min_seq_time = 0
        max_seq_time = 79*delta_t - past_time - future_time - delta_t/2

    no_steps = int(2*(max_seq_time - min_seq_time)/delta_t)
    start_times  = torch.linspace(min_seq_time, max_seq_time, no_steps, dtype=torch.get_default_dtype())

    skip_steps   =  int(2*future_time/delta_t)
    for i, start_time in enumerate(start_times):
        if i % skip_steps == 0:
            input_time_range = [start_time, start_time + past_time]
            output_time_range = [start_time + past_time + delta_t/2, start_time + past_time + future_time + delta_t/2]
            x, x_ind = get_seq_data_torch(seq_data, seq_times, input_time_range)
            y, y_ind = get_seq_data_torch(seq_data, seq_times, output_time_range)
            x, y = x.to(device), y.to(device)
            t_preds.append(y_ind)

            # print(x_ind.shape, y_ind.shape)
            # print(f'input window time ranges, {input_time_range}')
            # print(f'input window times, {x_ind}')
            # print(f'output window time ranges, {output_time_range}')
            # print(f'output window times, {y_ind}')

            x_ind = x_ind - start_time
            y_ind = y_ind - start_time - past_time - delta_t/2
            # print(x_ind, y_ind)

            quad_grid_out, quad_grid_in = get_quad_grid(x_ind, y_ind, device, past_time, future_time, delta_t, sample_freq)
            ##
            if i == 0:
                y_prev = x.to(device)
            y_pred = model.super_resolution(quad_grid_out, quad_grid_in, y_prev)
            y_prev = y_pred
            y_preds.append(y_pred)
            y_true.append(y)

    y_true = torch.cat(y_true, dim=1)
    y_preds = torch.cat(y_preds, dim=1)
    t_preds = torch.cat(t_preds, dim=0)

    y_true  = y_true.cpu().numpy().squeeze()
    y_preds = y_preds.cpu().numpy().squeeze()

    y_true  = scaler.inverse_transform(y_true)
    y_preds = scaler.inverse_transform(y_preds)

    y_true = torch.tensor(y_true).unsqueeze(0)
    y_preds = torch.tensor(y_preds).unsqueeze(0)

    # print(y_true.shape, y_preds.shape)

    mse = mse_loss_fn(y_preds, y_true)
    rmse = mse.sqrt()
    errors["RMSE"].append(rmse.item())
    errors["MSE"].append(mse.item())

    y_true  = y_true.cpu().numpy().squeeze()
    y_preds = y_preds.cpu().numpy().squeeze()

    return y_true, y_preds, errors

@torch.no_grad
def compute_forecast_metrics_train(model, seq_data, seq_times, device,
                            past_time, future_time, delta_t, sample_freq) -> dict[str, float]:
    # mse, vector l2 norm, l2 norm integrated over domain
    errors: dict[str, list[float]] = {
        "RMSE": [],
        "MSE": [],
    }
    y_true = []
    y_preds = []
    max_seq_time = seq_times[-1] - past_time - future_time + delta_t
    max_iter_idx = torch.argmin(torch.abs(seq_times - max_seq_time)).item()

    iters  = np.arange(0, int(len(seq_times)/(future_time/delta_t)) + 1)

    for iter_i in iters:
        start_idx = iter_i

        lag, forward_steps = get_lag_and_forward_steps(seq_times, start_idx, past_time, future_time)
        forward_steps = 10
        if iter_i> 0:
            lag = 10

        x_ind = seq_times[start_idx : start_idx + lag]  - seq_times[start_idx]
        # y_ind = seq_times[start_idx + lag : start_idx + lag + forward_steps] - seq_times[start_idx + lag ]
        y_ind = np.linspace(0, future_time, forward_steps)
        y_ind = torch.tensor(y_ind, dtype=torch.get_default_dtype())

        x = seq_data[:, start_idx : start_idx + lag]
        y = seq_data[:, start_idx + lag : start_idx + lag + forward_steps]
        if iter_i == 0:
            y_prev = x.to(device)

        y_true.append(y)

        # print(x_ind, y_ind)
        # print(lag, forward_steps)
        # print(seq_times[start_idx: start_idx + lag + forward_steps])

        quad_grid_out, quad_grid_in = get_quad_grid(x_ind, y_ind, device, past_time, future_time, delta_t, sample_freq)
        ##
        y_pred = model.super_resolution(quad_grid_out, quad_grid_in, y_prev)
        y_prev = y_pred
        y_preds.append(y_pred)

    y_true = torch.cat(y_true, dim=1)
    y_preds = torch.cat(y_preds, dim=1).cpu()
    mse = mse_loss_fn(y_preds, y_true)
    rmse = mse.sqrt()
    errors["RMSE"].append(rmse.item())
    errors["MSE"].append(mse.item())

    return y_true, y_preds, errors


def get_seq_data(traj, times, times_range):
    times_idx = np.where(np.logical_and(times >= times_range[0], times <= times_range[1]))[0]
    seq_data = traj[times_idx]
    seq_times = times[times_idx]
    return seq_data, seq_times

def get_seq_data_torch(traj, times, times_range):
    times_idx = torch.where(torch.logical_and(times >= times_range[0], times <= times_range[1]))[0]
    seq_data = traj[:,times_idx, :]
    seq_times = times[times_idx]
    return seq_data, seq_times

def get_lag_and_forward_steps(seq_times, start_idx, start_time, past_time, future_time, delta_t):
    x_seq_max_time = start_time + past_time
    x_seq_max_idx  = torch.argmin(torch.abs(seq_times - x_seq_max_time)).item()
    lag = x_seq_max_idx - start_idx + 1

    y_seq_max_time = start_time + past_time + future_time + delta_t/2
    y_seq_max_idx  = torch.argmin(torch.abs(seq_times - y_seq_max_time)).item()
    forward_steps = y_seq_max_idx - x_seq_max_idx

    return lag, forward_steps


def get_quad_grid(x_ind, y_ind, device, past_time, future_time, delta_t, sample_freq):
    x_ind_max = past_time 
    y_ind_max = future_time 
    x_ind = 2*x_ind/x_ind_max - 1
    y_ind = 2*y_ind/y_ind_max - 1
    # print(x_ind, y_ind)
    quad_grid_in = quadrature.trapezoidal_vecs_uneven(x_ind)
    quad_grid_out = quadrature.trapezoidal_vecs_uneven(y_ind)

    quad_grid_in = ([quad_grid_in[0]], [quad_grid_in[1]])
    quad_grid_out = ([quad_grid_out[0]], [quad_grid_out[1]])

    quad_grid_in = quadrature.quad_grid_to_device(quad_grid_in, device)
    quad_grid_out = quadrature.quad_grid_to_device(quad_grid_out, device)

    return quad_grid_out, quad_grid_in

def find_closest_index(seq_times, start_time, epsilon=1e-6):
    # Calculate the absolute differences
    differences = torch.abs(seq_times - start_time)
    sorted_indices = torch.argsort(differences)
    for i in sorted_indices:
        if differences[i] < epsilon:
            min_diff_index = i.item()
            break
        elif seq_times[i] > start_time:
            min_diff_index = i.item()
            break
    return min_diff_index

def train_one_traj(idx, samp_irregular_traj_i, samp_irregular_ts_i,
                    lag, forward_steps, delta_t, device, sample_freq, seed, n_epochs, traj_flag):

    seed_everything(seed)

    train_shuffle_flag = True
    past_time   = lag*delta_t
    future_time = forward_steps*delta_t

    train_times = [0, 79*delta_t]
    val_times   = [79*delta_t - past_time, 99*delta_t]
    test_times  = [99*delta_t - past_time, 119*delta_t]

    train_seq, train_seq_times = get_seq_data(samp_irregular_traj_i, samp_irregular_ts_i, train_times)
    val_seq, val_seq_times     = get_seq_data(samp_irregular_traj_i, samp_irregular_ts_i, val_times)
    test_seq, test_seq_times   = get_seq_data(samp_irregular_traj_i, samp_irregular_ts_i, test_times)
    if traj_flag == 'uniform':
        np.savez(str(OUTPUT_DIR / f"uniform_traj_test_seq"), test_seq=test_seq, test_seq_times=test_seq_times)
    else:
        test_data = np.load(str(OUTPUT_DIR / f"uniform_traj_test_seq.npz"))
        test_seq = test_data['test_seq']
        test_seq_times = test_data['test_seq_times']

    scaler = StandardScaler()
    train_seq = scaler.fit_transform(train_seq)
    val_seq   = scaler.transform(val_seq)
    test_seq  = scaler.transform(test_seq)

    train_seq = torch.tensor(train_seq, dtype=torch.get_default_dtype()).unsqueeze(0)
    val_seq   = torch.tensor(val_seq, dtype=torch.get_default_dtype()).unsqueeze(0)
    test_seq  = torch.tensor(test_seq, dtype=torch.get_default_dtype()).unsqueeze(0)
    train_seq_times = torch.tensor(train_seq_times, dtype=torch.get_default_dtype())
    val_seq_times   = torch.tensor(val_seq_times, dtype=torch.get_default_dtype())
    test_seq_times  = torch.tensor(test_seq_times, dtype=torch.get_default_dtype())

    batch_size   = 1


    hparams = dict(
        d=1,
        in_channels=2,
        out_channels=2,
        lifting_channels=128,
        integral_channels=4,
        n_integral_layers=3,
        projection_channels=128,
        n_hidden_units=32,
        n_hidden_layers=3,
        nonlinearity=nn.SiLU(),
        include_affine=False,
    )
    # initialize model
    model = models.KhatriRaoNO.easy_init(**hparams)  # type: ignore

    total_params = sum(p.numel() for p in model.parameters())

    # Calculate the total number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print the results
    print(f"Total number of parameters: {total_params}", flush=True)
    print(f"Total number of trainable parameters: {trainable_params}", flush=True)

    hparams["lag"] = lag
    hparams["forward_steps"] = forward_steps
    valid_freq = 10
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # n_epochs = 10 # 200 # 50 #200 #100
    log = TrainLog(n_ckpts=10)
    log.log_hparams(hparams)
    n_iters, epoch = 0, 0
    loss = torch.tensor(0.0)
    print("Training KRNO on cyclic trajectory dataset...", flush=True)
    seq_times = train_seq_times
    seq_data  = train_seq
    print('seq_times: ', seq_times)
    for epoch in range(1, n_epochs + 1):
        max_seq_time = 79*delta_t - past_time - future_time - delta_t/2

        start_times  = torch.linspace(0, max_seq_time, int(2*max_seq_time/delta_t)+1, dtype=torch.get_default_dtype())

        if train_shuffle_flag:
            start_times = np.random.permutation(start_times)
    
        for start_time in start_times:

            input_time_range = [start_time, start_time + past_time]
            output_time_range = [start_time + past_time + delta_t/2, start_time + past_time + future_time + delta_t/2]
            x, x_ind = get_seq_data_torch(seq_data, seq_times, input_time_range)
            y, y_ind = get_seq_data_torch(seq_data, seq_times, output_time_range)
            x, y = x.to(device), y.to(device)

            # print(x_ind.shape, y_ind.shape)
            # print(f'input window time ranges, {input_time_range}')
            # print(f'input window times, {x_ind}')
            # print(f'output window time ranges, {output_time_range}')
            # print(f'output window times, {y_ind}')

            x_ind = x_ind - start_time
            y_ind = y_ind - start_time - past_time - delta_t/2
            # print(x_ind, y_ind)

            quad_grid_out, quad_grid_in = get_quad_grid(x_ind, y_ind, device, past_time, future_time, delta_t, sample_freq)
            ##
            optimizer.zero_grad()
            y_pred = model.super_resolution(quad_grid_out, quad_grid_in, x)
            loss = loss_fn(y_pred, y, quad_grid_out)
        
            loss.backward()
            optimizer.step()
            n_iters += 1
            # print('iter: ', n_iters, 'loss: ', loss.item())
            if n_iters % valid_freq == 0:
                log.log("train_loss", n_iters, loss.item())
                model.eval()
                # validation
                valid_metrics = compute_test_metrics(
                    model, val_seq, val_seq_times, device, past_time, future_time, delta_t, sample_freq, flag='val'
                )
                log.log_val("valid_loss", n_iters, valid_metrics["loss"], model)
                log.log_dict(valid_metrics, n_iters, "valid")
                # printing
                print_str = f"EPOCH: {epoch:02d} | TRAIN: {loss.item():.3f}"
                print_str += " | VALID: " + " , ".join(
                    [f"{k}: {v:.3f}" for k, v in valid_metrics.items()]
                )
                print(print_str, flush=True)
                with open(CKPT_DIR / f"krno_it{n_iters}_{seed}.pt", "wb") as f:
                    torch.save({"log": log.to_dict()}, f)
                model.train()
                # if n_iters == 63:
                #     break

    model.eval()
    model.load_state_dict(log.best_ckpt())
    model.to(device)
    print("Computing test metrics...", flush=True)
    # logging valid and test metrics
    valid_metrics = compute_test_metrics(model, val_seq, val_seq_times, device, past_time,
                                          future_time, delta_t, sample_freq, flag='val')
    print_str = " | VALID: " + " , ".join(
                    [f"{k}: {v:.3f}" for k, v in valid_metrics.items()])
    print(print_str, flush=True)
    test_metrics = compute_test_metrics(model, test_seq, test_seq_times, device, past_time,
                                         future_time, delta_t, sample_freq, flag='test')
    print_str = " | TEST: " + " , ".join(
                    [f"{k}: {v:.3f}" for k, v in test_metrics.items()])
    print(print_str, flush=True)

    # ## forecast on train data
    # y_true_train, y_preds_train, errors_train = compute_forecast_metrics_train(model, train_seq, train_seq_times, device,
    #                                                     past_time, future_time, delta_t, sample_freq)
    # y_true_train  = y_true_train.cpu().numpy().squeeze()
    # y_preds_train = y_preds_train.cpu().numpy().squeeze()

    # y_true_train_inv  = scaler.inverse_transform(y_true_train)
    # y_preds_train_inv = scaler.inverse_transform(y_preds_train)

    ## forecast on test data
    y_true, y_preds, errors = compute_forecast_metrics(model, test_seq, test_seq_times, device,
                                                        past_time, future_time, delta_t, sample_freq, scaler)
    print(errors)
    errors_df = pd.DataFrame(errors)
    errors_df.to_csv(str(OUTPUT_DIR / f"irregular_traj_errors_{idx}.csv"))

    np.savez(str(OUTPUT_DIR / f"irregular_traj_pred_{idx}"), y_true=y_true, y_preds=y_preds)

    PlotConfig.setup()
    figsize  = PlotConfig.convert_width((1, 1), page_scale=0.45)
    fig, ax  = plt.subplots(figsize=figsize)
    # ax.plot(y_preds_train_inv[:, 0], y_preds_train_inv[:, 1], 'r*-', label='pred (train)')
    ax.plot(samp_irregular_traj_i[:80, 0], samp_irregular_traj_i[:80, 1], 'b*--', label='true (train)', markersize=1.5, linewidth=0.5)
    ax.plot(samp_irregular_traj_i[80:100, 0], samp_irregular_traj_i[80:100, 1], 'g*--', label='true (val)', markersize=1.5, linewidth=0.5)
    ax.plot(samp_irregular_traj_i[100:118, 0], samp_irregular_traj_i[100:118, 1], 'k*--', label='true (test)', markersize=1.5, linewidth=0.5)
    ax.plot(y_preds[:, 0], y_preds[:, 1], 'r*-', label='pred (test)', markersize=3.5, linewidth=0.5)
    ax.grid(False)
    ax.legend()
    # ax.set_xticks(np.linspace(min(samp_uni_traj[:, 0]), max(samp_uni_traj[:, 0]), 6))
    # ax.set_yticks(np.linspace(min(samp_uni_traj[:, 1]), max(samp_uni_traj[:, 1]), 6))
    # ax.legend()
    # plt.savefig(str(FIG_DIR / 'uniform_traj_pred.png'), dpi=500)
    PlotConfig.save_fig(fig, str(FIG_DIR / f"irregular_traj_pred_{idx}"))

    return errors["RMSE"][0], errors["MSE"][0]



def main():

    seed = 42
    seed_everything(seed)

    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')
    
    ntotal = 240
    nsample = 100
    sample_freq = 2
    scale_flag = True

    lag = 9
    forward_steps = 9
    n_epochs = 50# 50


    samp_uni_traj, samp_uni_ts, samp_irregular_trajs, samp_irregular_ts, orig_traj_cc, orig_ts = generate_data(ntotal, nsample, sample_freq)

    print(samp_uni_traj.shape, samp_uni_ts.shape)
    print(samp_irregular_trajs.shape, samp_irregular_ts.shape)

    delta_t     = (samp_uni_ts[1] - samp_uni_ts[0])

    rmse, mse = train_one_traj(111, samp_uni_traj, samp_uni_ts,
                                lag, forward_steps, delta_t, device, sample_freq, seed, n_epochs = n_epochs,
                                traj_flag = 'uniform')

    # Write RMSE, MSE, and their means to a CSV file
    metrics_file = OUTPUT_DIR / "metrics_summary_uniform.csv"
    with open(metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Trajectory", "RMSE", "MSE"])
        writer.writerow([0, rmse, mse])
        writer.writerow(["Mean", np.mean(rmse), np.mean(mse)])

    ## train using first irregular trajectory
    ## for loop over all irregular trajectories

    ## get test sequences from the uniform trajectory and use them as test sequences for the irregular trajectories
    # 
    # seed = 0
    rmse_list = []
    mse_list  = []
    # for tarj_i in range(2):
    for tarj_i in range(samp_irregular_trajs.shape[0]):

        print(f"Training on trajectory {tarj_i}...")
        samp_irregular_traj_i  = np.concatenate((samp_irregular_trajs[tarj_i], samp_uni_traj[nsample:]), axis=0)
        samp_irregular_ts_i    = np.concatenate((samp_irregular_ts[tarj_i], samp_uni_ts[nsample:]), axis=0)

        rmse, mse = train_one_traj(tarj_i, samp_irregular_traj_i, samp_irregular_ts_i,
                                    lag, forward_steps, delta_t, device, sample_freq, seed, n_epochs = n_epochs,
                                    traj_flag = 'irregular')
        rmse_list.append(rmse)
        mse_list.append(mse)

    metrics_file = OUTPUT_DIR / "metrics_summary_irregular.csv"
    with open(metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Trajectory", "RMSE", "MSE"])
        for i, (rmse, mse) in enumerate(zip(rmse_list, mse_list)):
            writer.writerow([i, rmse, mse])
        writer.writerow(["Mean", np.mean(rmse_list), np.mean(mse_list)])

        print(f"Metrics summary saved to {metrics_file}")

        print(f"Finished training on trajectory {tarj_i}.")
        


if __name__ == '__main__':
    main()
