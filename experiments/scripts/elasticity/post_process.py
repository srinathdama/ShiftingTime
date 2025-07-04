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
import matplotlib.pyplot as plt
import seaborn as sns

torch.set_float32_matmul_precision("high")

CURR_DIR = pathlib.Path(__file__).parent.absolute()

sys.path.append(str(CURR_DIR / ".."))

root_base_path = os.path.dirname(os.path.dirname(CURR_DIR))
import sys
sys.path
sys.path.append(root_base_path)

from plotting_config import PlotConfig

DATA_DIR = CURR_DIR / "loca-data"
CKPT_DIR = CURR_DIR / "ckpts"
FIG_DIR = CURR_DIR / "figs"

os.makedirs(FIG_DIR, exist_ok=True)

save_best_res_path  = os.path.join(CKPT_DIR, 'test_results.npz')

saved_data = np.load(save_best_res_path)
true_y     = saved_data['true_y'].squeeze()
pred_y     = saved_data['pred_y'].squeeze()
mask       = saved_data['mask'].squeeze()

x_grid, y_grid = np.linspace(0, 1, 41), np.linspace(0, 1, 41)
XX, YY = np.meshgrid(x_grid, y_grid, indexing='ij')


PlotConfig.setup()
figsize = PlotConfig.convert_width((3, 1), page_scale=0.5)
fig, axs = plt.subplots(1, 3, figsize=figsize)
cmap = sns.color_palette("viridis", as_cmap=True)
axs[0].imshow(mask, cmap=cmap)
# axs[0].set_title("Test input")
axs[1].imshow(true_y, cmap=cmap)
# axs[1].set_title("Test output")
axs[2].imshow(pred_y, cmap=cmap)
# axs[2].set_title("Pred. output")
for ax in axs.flatten():
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
PlotConfig.save_fig(fig, str(FIG_DIR / "elasticity-pred"))
plt.close(fig)

PlotConfig.setup()
figsize = PlotConfig.convert_width((3, 1), page_scale=0.5)
fig, axs = plt.subplots(1, 3, figsize=figsize)
cmap = sns.color_palette("viridis", as_cmap=True)
axs[0].pcolormesh(XX, YY, mask, cmap=cmap) #, shading='gouraud')
# axs[0].set_title("Test input")
axs[1].pcolormesh(XX, YY, true_y, cmap=cmap) #, shading='gouraud')
# axs[1].set_title("Test output")
axs[2].pcolormesh(XX, YY, pred_y, cmap=cmap) #, shading='gouraud')
# axs[2].set_title("Pred. output")
for ax in axs.flatten():
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
PlotConfig.save_fig(fig, str(FIG_DIR / "elasticity-pred-pcolormesh"))
plt.close(fig)
