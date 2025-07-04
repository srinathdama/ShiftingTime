"""

"""
from __future__ import annotations

import argparse
import time
import os
import pathlib
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader, random_split
from khatriraonop import models, quadrature
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from timeit import default_timer
from datetime import datetime

current_time = datetime.now()
date_time    = current_time.strftime("%d_%m_%Y-%H_%M_%S")

CURR_DIR = pathlib.Path(__file__).parent.absolute()
CKPT_DIR = CURR_DIR / "ckpts"

root_base_path = os.path.dirname(os.path.dirname(CURR_DIR))
import sys
sys.path
sys.path.append(root_base_path)
from utils.utilities3 import *

os.makedirs(CKPT_DIR, exist_ok=True)

DATA_DIR = os.path.join(CURR_DIR, 'dataset/elasticity')

# torch.set_float32_matmul_precision("high")
torch.set_float32_matmul_precision("highest")
# torch.set_default_dtype(torch.float64)

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

# seed_everything(0)
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

seed_everything(args.seed)
print(args.seed, flush=True)
################################################################
# configs
################################################################
INPUT_PATH = os.path.join(DATA_DIR, 'Random_UnitCell_mask_10_interp.npy')
OUTPUT_PATH = os.path.join(DATA_DIR, 'Random_UnitCell_sigma_10_interp.npy')
Ntotal = 2000
ntrain = 1000
ntest = 200

batch_size = 20

modes = 12
width = 32

r = 1
h = int(((41 - 1) / r) + 1)
s = h

################################################################
# load data and data normalization
################################################################
input = np.load(INPUT_PATH)
input = torch.tensor(input, dtype=torch.float).permute(2,0,1)
output = np.load(OUTPUT_PATH)
output = torch.tensor(output, dtype=torch.float).permute(2,0,1)

x_train = input[:Ntotal][:ntrain, ::r, ::r][:, :s, :s]
y_train = output[:Ntotal][:ntrain, ::r, ::r][:, :s, :s]

x_test = input[:Ntotal][-ntest:, ::r, ::r][:, :s, :s]
y_test = output[:Ntotal][-ntest:, ::r, ::r][:, :s, :s]

x_train = x_train.reshape(ntrain, s, s, 1)
x_test = x_test.reshape(ntest, s, s, 1)

print(x_train.shape, y_train.shape, flush = True)
print(x_test.shape, y_test.shape, flush = True)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          shuffle=False)

################################################################
# training and evaluation
################################################################

train_flag = True
model_name = 'KRNO_AdamW_29_09_2024-14_55_42.pt'
epochs = 500 #500
learning_rate = 1e-3
scheduler_step = 100 #100
scheduler_gamma = 0.5
optim = 'AdamW' # 'AdamW' # 'Adam'
lr_decay_flag = True
normalization_flag = True
loss_fn_integral   = True
print(f'using {optim} optimizer! ', flush = True)
print(f'learning rate : {learning_rate} ', flush = True)
print(f'lr_decay_flag : {lr_decay_flag} ', flush = True)
print(f'normalization_flag : {normalization_flag} ', flush = True)
print(f'loss_fn_integral : {loss_fn_integral}', flush = True)

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

if train_flag:
    path_model     = os.path.join(CKPT_DIR, f'KRNO_{optim}_{date_time}.pt')
else:
    path_model     = os.path.join(CKPT_DIR, model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loss_fn(y_pred, y_true, quad_rule):
    return quadrature.integrate_on_domain((y_true - y_pred).pow(2), quad_rule).mean()

################################################################
# training and evaluation
################################################################

quad_fns = [
        quadrature.trapezoidal_vecs,
        quadrature.trapezoidal_vecs
    ]
quad_grid = quadrature.get_quad_grid(
    quad_fns, [41, 41], [-1.0, -1.0], [1.0, 1.0]
)
cart_grid = quadrature.quad_to_cartesian_grid(quad_grid)
cart_grid = quadrature.cart_grid_to_device(cart_grid, device)
hparams = dict(
    d=2,
    in_channels=1,
    out_channels=1,
    lifting_channels= 128, #width,
    integral_channels=width, #20, #32,
    n_integral_layers= 3, #4, #3,
    projection_channels=128,
    n_hidden_units=64, #32, #32, #64
    n_hidden_layers=3,
    nonlinearity=nn.SiLU(),
    affine_in_first_integral_tsfm=True
)

print(hparams, flush = True)
# initialize model
model = models.KhatriRaoNO.easy_init(**hparams)  # type: ignore
valid_freq = 500
model.to(device)
model.train()
if optim=='AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
elif optim=='Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

if lr_decay_flag:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

n_iters, epoch = 0, 0
loss = torch.tensor(0.0)

# Calculate the total number of parameters
total_params = sum(p.numel() for p in model.parameters())

# Calculate the total number of trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print the results
print(f"Total number of parameters: {total_params}", flush = True)
print(f"Total number of trainable parameters: {trainable_params}", flush = True)


# print(model.count_params())


myloss = LpLoss(size_average=False)


if train_flag:
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            mask = x.clone()

            optimizer.zero_grad()
            out = model(cart_grid, x)
            out = out*mask

            loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            loss.backward()

            optimizer.step()
            train_l2 += loss.item()
        
        if lr_decay_flag:
            scheduler.step()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                mask = x.clone()

                out = model(cart_grid, x)
                out2 = out * mask

                test_l2 += myloss(out2.view(batch_size, -1), y.view(batch_size, -1)).item()

        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()
        print(ep, t2 - t1, train_l2, test_l2, flush=True)

    torch.save(model, path_model)

else:
    print('testing', flush = True)
    batch_size  = 1
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          shuffle=False)
    model = torch.load(path_model)

    model.eval()
    test_l2 = 0.0
    pred_y  = []
    true_y  = []
    losses  = []
    masks   = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            mask = x.clone()

            out = model(cart_grid, x)
            out2 = out * mask
            pred_y.append(out2.squeeze().cpu())
            true_y.append(y.cpu())
            masks.append(mask.cpu())

            tmp_loss = myloss(out2.view(batch_size, -1), y.view(batch_size, -1)).item()
            losses.append(tmp_loss)
            test_l2 += tmp_loss

    test_l2 /= ntest
    t2 = default_timer()
    print(test_l2, flush=True)

    # min_index = np.argmin(np.array(losses))
    min_index = 0
    print(min_index, flush = True)
    save_best_res_path  = os.path.join(CKPT_DIR, 'test_results.npz')

    np.savez(save_best_res_path, true_y=true_y[min_index],
              pred_y=pred_y[min_index], mask = masks[min_index])
    


