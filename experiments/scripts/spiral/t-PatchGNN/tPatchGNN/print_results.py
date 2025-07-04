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
root_base_path = os.path.dirname(root_base_path)
import sys
sys.path
sys.path.append(root_base_path)
sys.path.append(str(CURR_DIR / ".."))
sys.path.append(CURR_DIR)

from plotting_config import PlotConfig

FIG_DIR = CURR_DIR / "figs"
# OUTPUT_DIR = CURR_DIR / "outputs"
# OUTPUT_DIR = CURR_DIR / "outputs_high_irreg"
OUTPUT_DIR = CURR_DIR / "outputs_high_irreg_4x"
# OUTPUT_DIR = CURR_DIR / "outputs_mod_irreg"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 111]


# for idx in idxs:
#     data = np.load(str(OUTPUT_DIR / f"irregular_traj_pred_{idx}.npz"))
#     y_true_inv = data["y_true"]
#     y_preds_inv = data["y_preds"]


# PlotConfig.setup()
# figsize  = PlotConfig.convert_width((1, 1), page_scale=0.5)
# fig, ax  = plt.subplots(figsize=figsize)
# # ax.plot(y_preds_train_inv[:, 0], y_preds_train_inv[:, 1], 'r*-', label='pred (train)')
# ax.plot(samp_irregular_traj_i[:, 0], samp_irregular_traj_i[:, 1], 'b*--', label='true')
# ax.plot(y_preds_inv[:, 0], y_preds_inv[:, 1], 'k*-', label='pred (test)')
# ax.grid(False)
# ax.legend()
# PlotConfig.save_fig(fig, str(FIG_DIR / f"irregular_traj_pred_{idx}"))


df = pd.read_csv(str(OUTPUT_DIR / "metrics_summary_irregular.csv"))
df = df[:-1]
mean_rmse = df['RMSE'].mean()
std_rmse = df['RMSE'].std()
mean_mse = df['MSE'].mean()
std_mse = df['MSE'].std()

# Display results
results = {
    "Mean RMSE": mean_rmse,
    "Std RMSE": std_rmse,
    "Mean MSE": mean_mse,
    "Std MSE": std_mse
}
print(results)
