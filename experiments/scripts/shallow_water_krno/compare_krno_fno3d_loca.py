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

epoch = 200
### load KRNO results

err_df_krno = pd.read_pickle(CURR_DIR / f"err_time_dict_krno_ep_{epoch}.pkl")
time_krno   = err_df_krno['t'].values.reshape(1000, -1)
rho_krno    = err_df_krno['rho'].values.reshape(1000, -1)
v1_krno     = err_df_krno['v1'].values.reshape(1000, -1)
v2_krno     = err_df_krno['v2'].values.reshape(1000, -1)

### load FNO3D results

err_df_fno3d = pd.read_pickle(os.path.join(root_base_path, f"scripts/shallow-water_fno_3d/err_time_dict_ep_{epoch}.pkl"))
time_fno3d   = err_df_fno3d['t'].values.reshape(1000, -1)
rho_fno3d    = err_df_fno3d['rho'].values.reshape(1000, -1)
v1_fno3d     = err_df_fno3d['v1'].values.reshape(1000, -1)
v2_fno3d     = err_df_fno3d['v2'].values.reshape(1000, -1)

### load loca results

err_df_loca = pd.read_pickle(CURR_DIR / f"err_time_dict_loca_ep_{epoch}.pkl")
time_loca   = err_df_loca['t'].values.reshape(1000, -1)
rho_loca    = err_df_loca['rho'].values.reshape(1000, -1)
v1_loca     = err_df_loca['v1'].values.reshape(1000, -1)
v2_loca     = err_df_loca['v2'].values.reshape(1000, -1)

# print(err_df_fno3d)

## 

times_  = time_fno3d[0]

PlotConfig.setup()
figsize  = PlotConfig.convert_width((10, 4), page_scale=1.0)
fig, ax  = plt.subplots(3, 1, figsize=figsize)
ax[0].plot(times_[6:], rho_krno.mean(0)[6:],'k-', label='KRNO')
ax[0].plot(times_[6:], rho_fno3d.mean(0)[6:], 'b--', label='FNO-3D')
ax[0].plot(times_[6:], rho_loca.mean(0)[6:],'r:', label='LOCA')
# ax[0].set_title(f'Rel. $L^2$ error', pad=2)
# ax[0].set_xlabel("$t$ (sec)")
ax[0].legend()
ax[0].set_yscale('log')
ax[0].set_ylabel("$\\rho$")

ax[1].plot(times_[6:], v1_krno.mean(0)[6:],'k-', label='KRNO')
ax[1].plot(times_[6:], v1_fno3d.mean(0)[6:], 'b--', label='FNO-3D')
ax[1].plot(times_[6:], v1_loca.mean(0)[6:],'r:', label='LOCA')
# ax[1].set_xlabel("$t$ (sec)")
# ax[1].set_title(f'Rel. $L^2$ error ', pad=2)
# ax[1].legend()
ax[1].set_yscale('log')
ax[1].set_ylabel("$v_1$")

ax[2].plot(times_[6:], v2_krno.mean(0)[6:],'k-', label='KRNO')
ax[2].plot(times_[6:], v2_fno3d.mean(0)[6:], 'b--', label='FNO-3D')
ax[2].plot(times_[6:], v2_loca.mean(0)[6:],'r:', label='LOCA')
ax[2].set_xlabel("$t$ (sec)")
# ax[2].set_title(f'Rel. $L^2$ error in $v_2$', pad=2)
# ax[2].legend()
ax[2].set_yscale('log')
ax[2].set_ylabel("$v_2$")

# fig.subplots_adjust(bottom=0.23)
# legend_ax = fig.add_axes([0.5, 0.02, 0.4, 0.1])  # Adjust these values as needed
# legend_ax.axis('off')  # Hide the axis
# handles, labels = ax[0].get_legend_handles_labels()  # Get handles and labels for the legend
# legend_ax.legend(handles, labels, loc='center', ncol=2) 

PlotConfig.save_fig(fig, str(FIG_DIR / f"krno_fno3d_loca_ep_{epoch}"))


