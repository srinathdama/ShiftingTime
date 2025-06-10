import argparse
from functools import partial
import math
import os
import re
import sys
import pathlib
from jaxtyping import Float
import matplotlib.ticker as ticker

import torch
import numpy as np
import pandas as pd
from torch import Tensor
from torch.func import vmap  # type: ignore
from torch.utils.data import DataLoader, Dataset
from khatriraonop import models, quadrature
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import AxesGrid

import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

torch.set_float32_matmul_precision("high")

CURR_DIR = pathlib.Path(__file__).parent.absolute()

sys.path.append(str(CURR_DIR / ".."))

from plotting_config import PlotConfig

from train_script import get_raw_data, TrainLog, seed_everything, unnormalize
from metrics import forecast_metrics

DATA_DIR = CURR_DIR / "data"
CKPT_DIR = CURR_DIR / "ckpts"
FIG_DIR = CURR_DIR / "figs"

os.makedirs(FIG_DIR, exist_ok=True)


def natural_sort(some_list):
    """from: https://stackoverflow.com/a/4836734/12184561"""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(some_list, key=alphanum_key)


def get_latest_ckpt() -> str:
    ckpts = [f for f in os.listdir(CKPT_DIR) if f.endswith(".pt")]
    return str(CKPT_DIR / natural_sort(ckpts)[-1])


class NonOverlapDS(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self.n_samples = data.size(0) // window_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start_idx = idx * self.window_size
        end_idx = start_idx + self.window_size
        sample = self.data[start_idx:end_idx]
        return sample


# @torch.no_grad()
# def plot_pred(nplots: int, name: str):
#     with open("prediction.pt", "rb") as fp:
#         data = torch.load(fp)
#     y_true, y_pred = data["y"], data["y_pred"]
#     cmap = sns.color_palette("Spectral_r", as_cmap=True)
#     cmap_err = sns.color_palette("Blues", as_cmap=True)
#     vars = ["T", "p"]
#     PlotConfig.setup()
#     figsize = PlotConfig.convert_width((3 / 2, 1), page_scale=1.0)
#     fig, axs = plt.subplots(2 * 3, nplots, figsize=figsize)
#     t_true, p_true = y_true.cpu().chunk(2, dim=-1)
#     t_pred, p_pred = y_pred.cpu().chunk(2, dim=-1)

#     t_cbar = dict(vmin=t_true.min(), vmax=t_true.max())
#     p_cbar = dict(vmin=p_true.min(), vmax=p_true.max())
#     t_err_cbar = dict(
#         vmin=(t_pred - t_true).abs().min(), vmax=(t_pred - t_true).abs().max()
#     )
#     p_err_cbar = dict(
#         vmin=(p_pred - p_true).abs().min(), vmax=(p_pred - p_true).abs().max()
#     )
#     for j in range(nplots):
#         idx = j
#         images = [
#             axs[0, j].imshow(t_true[idx], cmap=cmap, **t_cbar),
#             axs[1, j].imshow(p_true[idx], cmap=cmap, **p_cbar),
#             axs[2, j].imshow(t_pred[idx], cmap=cmap, **t_cbar),
#             axs[3, j].imshow(p_pred[idx], cmap=cmap, **p_cbar),
#             axs[4, j].imshow(
#                 (t_pred[idx] - t_true[idx]).abs(), cmap=cmap_err, **t_err_cbar
#             ),
#             axs[5, j].imshow(
#                 (p_pred[idx] - p_true[idx]).abs(), cmap=cmap_err, **p_err_cbar
#             ),
#         ]
#         axs[5, j].set_xlabel(f"day {j+1}")
#         if j == nplots - 1:
#             for i in range(len(images)):
#                 box = axs[i, j].get_position()
#                 axs[i, j].set_position(
#                     [box.x0, box.y0, box.width * 0.9, box.height]
#                 )  # Reduce width by 10%
#                 cbar = plt.colorbar(images[i], ax=axs[i, j], fraction=0.046, pad=0.04)
#                 cbar.outline.set_visible(False)  # type: ignore
#     for ax in axs.flatten():
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_xticks([])
#         ax.set_yticks([])
#     for i, var in enumerate(vars):
#         axs[i, 0].set_ylabel(f"${var}$")
#         axs[i + len(vars), 0].set_ylabel("$\\hat{%s}$" % var)
#         axs[i + 2 * len(vars), 0].set_ylabel("$|{%s} - \\hat{%s}|$" % (var, var))

#     fig.subplots_adjust(bottom=0.25)
#     common_ax = fig.add_subplot(111, frameon=False)  # Add a common subplot
#     common_ax.axhline(1.0 / 3, alpha=0.5, color="k", linestyle="--", linewidth=2)
#     common_ax.axhline(2.0 / 3, alpha=0.5, color="k", linestyle="--", linewidth=2)
#     common_ax.grid(False)
#     common_ax.set_xticks([])
#     common_ax.set_yticks([])
#     common_ax.set_yticklabels([])
#     PlotConfig.save_fig(fig, str(FIG_DIR / name))
#     plt.close(fig)


# @torch.no_grad()
# def plot_pred(nplots: int, name: str, days_idx: list):
#     with open("prediction.pt", "rb") as fp:
#         data = torch.load(fp)
#     y_true, y_pred = data["y"], data["y_pred"]
#     # cmap_temp = sns.color_palette("Spectral_r", as_cmap=True)
#     # cmap_pres = sns.color_palette("Spectral_r", as_cmap=True)
#     cmap_err = sns.color_palette("Blues", as_cmap=True)
#     cmap_temp = cm.get_cmap('viridis')
#     cmap_pres = cm.get_cmap('coolwarm')
#     # cmap_temp = cm.get_cmap('Inferno')
#     # cmap_pres = cm.get_cmap('Cividis')

#     vars = ["T", "p"]
#     PlotConfig.setup()
#     figsize = PlotConfig.convert_width((2.5, 1), page_scale=1.0)
#     fig, axs = plt.subplots(nplots, 2 * 3, figsize=figsize)
#     t_true, p_true = y_true.cpu().chunk(2, dim=-1)
#     t_pred, p_pred = y_pred.cpu().chunk(2, dim=-1)

#     t_cbar = dict(vmin=t_true.min(), vmax=t_true.max())
#     p_cbar = dict(vmin=p_true.min(), vmax=p_true.max())
#     t_err_rel = (t_pred - t_true).abs()/t_true.abs()
#     t_err_cbar = dict(
#         vmin=t_err_rel.min(), vmax=t_err_rel.max()
#     )
#     p_err_rel = (p_pred - p_true).abs()/p_true.abs()
#     p_err_cbar = dict(
#         vmin=p_err_rel.min(), vmax=p_err_rel.max()
#     )
#     for j in range(nplots):
#         idx = days_idx[j] - 1
#         images = [
#             axs[j, 0].imshow(t_pred[idx], cmap=cmap_temp, **t_cbar, interpolation='bilinear'),
#             axs[j, 1].imshow(t_true[idx], cmap=cmap_temp, **t_cbar, interpolation='bilinear'),
#             axs[j, 2].imshow(
#                 (t_pred[idx] - t_true[idx]).abs()/t_true[idx].abs(), cmap=cmap_err, **t_err_cbar
#             , interpolation='bilinear'),
#             axs[j, 3].imshow(p_pred[idx], cmap=cmap_pres, **p_cbar, interpolation='bilinear'),
#             axs[j, 4].imshow(p_true[idx], cmap=cmap_pres, **p_cbar, interpolation='bilinear'),
#             axs[j, 5].imshow(
#                 (p_pred[idx] - p_true[idx]).abs()/p_true[idx].abs(), cmap=cmap_err, **p_err_cbar
#             , interpolation='bilinear'),
#         ]
#         axs[j, 0].set_ylabel(f"day {idx+1}")
#         if j == nplots - 1:
#             for i in range(len(images)):
#                 box = axs[j, i].get_position()
#                 axs[j, i].set_position(
#                     [box.x0, box.y0, box.width * 0.9, box.height]
#                 )  # Reduce width by 10%
#                 cbar = plt.colorbar(images[i], ax=axs[j, i], fraction=0.046, pad=0.04, orientation='horizontal')
#                 cbar.outline.set_visible(False)  # type: ignore
#                 # Set scientific notation
#                 cbar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1e}'))
#                 cbar.ax.tick_params(axis='x', which='both', rotation=30)
#                 # cbar.ax.xaxis.get_offset_text().set_fontsize(10)  # Adjust fontsize if needed
#                 # cbar.ax.tick_params(axis='x', which='both', labelsize=10)
#                 # Adjust the colorbar position
#                 cbar_ax = cbar.ax
#                 cbar_box = cbar_ax.get_position()
#                 cbar_ax.set_position([cbar_box.x0, box.y0 - 0.12, cbar_box.width, 0.02])
#     for ax in axs.flatten():
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_xticks([])
#         ax.set_yticks([])
#     for i, var in enumerate(vars):
#         axs[0, i* (len(vars) +1)].set_title("$\\hat{%s}$" % var)
#         axs[0, i* (len(vars) +1) + 1].set_title(f"${var}$")
#         axs[0, i* (len(vars) +1) + 2].set_title("$|{%s} - \\hat{%s}|/{|{%s}|}$" % (var, var, var))

#     # fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.05, wspace=0.12)
#     fig.subplots_adjust(bottom=0.25)
#     common_ax = fig.add_subplot(111, frameon=False)  # Add a common subplot
#     common_ax.axvline(1.0 / 2, alpha=0.5, color="k", linestyle="--", linewidth=2)
#     # common_ax.axvline(2.0 / 2, alpha=0.5, color="k", linestyle="--", linewidth=2)
#     common_ax.grid(False)
#     common_ax.set_xticks([])
#     common_ax.set_yticks([])
#     common_ax.set_yticklabels([])
#     PlotConfig.save_fig(fig, str(FIG_DIR / name))
#     plt.close(fig)

# @torch.no_grad()
# def plot_pred(nplots: int, name: str, days_idx: list):
#     with open("prediction.pt", "rb") as fp:
#         data = torch.load(fp)
#     y_true, y_pred = data["y"], data["y_pred"]

#     # Use predefined colormaps
#     cmap_err = sns.color_palette("Blues", as_cmap=True)
#     cmap_temp = cm.get_cmap('viridis')
#     cmap_pres = cm.get_cmap('coolwarm')

#     # Define a regular 72x72 grid for latitude and longitude
#     # lons = np.concatenate((np.linspace(0, 180, 36, endpoint=False), np.linspace(-180, 0, 36, endpoint=False)))   # Longitude grid from -180 to 180 degrees
#     lons = np.linspace(0, 360, 72)
#     lats = np.linspace(-90, 90, 72)    # Latitude grid from -90 to 90 degrees

#     t_true, p_true = y_true.cpu().chunk(2, dim=-1)
#     t_pred, p_pred = y_pred.cpu().chunk(2, dim=-1)

#     # Temperature-related plots
#     PlotConfig.setup()
#     figsize = PlotConfig.convert_width((2.5, 1), page_scale=1.0)
#     fig_temp = plt.figure(figsize=figsize)
    
#     grid_temp = AxesGrid(fig_temp, 111,  # 1x1 grid for layout
#                      nrows_ncols=(nplots, 3),
#                      axes_pad=0.05,  # Padding between plots
#                      cbar_mode='edge',  
#                      cbar_location='bottom',
#                      cbar_pad=0.05,
#                      share_all=True,
#                      label_mode="L",
#                      axes_class=(GeoAxes, dict(projection=ccrs.PlateCarree(central_longitude=180))))
    

#     t_cbar = dict(vmin=t_true.min(), vmax=t_true.max())
#     t_err_rel = (t_pred - t_true).abs() / t_true.abs()
#     t_err_cbar = dict(vmin=t_err_rel.min(), vmax=t_err_rel.max())

#     for j in range(nplots):
#         idx = days_idx[j] - 1

#         # Plot temperature (predicted, true, and relative error)
#         images = [
#             grid_temp[j * 3].contourf(lons, lats, t_pred[idx].squeeze(), 50, transform=ccrs.PlateCarree(), cmap=cmap_temp, **t_cbar),
#             grid_temp[j * 3 + 1].contourf(lons, lats, t_true[idx].squeeze(), 50, transform=ccrs.PlateCarree(), cmap=cmap_temp, **t_cbar),
#             grid_temp[j * 3 + 2].contourf(lons, lats, t_err_rel[idx].squeeze(), 50, transform=ccrs.PlateCarree(), cmap=cmap_err, **t_err_cbar),
#         ]

#         # Add coastlines and colorbars for each plot
#         for i in range(3):
#             # grid_temp[j * 3 + i].coastlines()
#             # grid_temp[j * 3 + i].add_feature(cfeature.LAND)
#             grid_temp[j * 3 + i].add_feature(cfeature.COASTLINE, facecolor='none', linewidth=0.3)
#             grid_temp[j * 3 + i].add_feature(cfeature.BORDERS, facecolor='none', linewidth=0.3)
#             cbar = fig_temp.colorbar(images[i], cax=grid_temp.cbar_axes[j * 3 + i], orientation='horizontal')
#             cbar.ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
#             # cbar.ax.xaxis.get_offset_text().set_position((0.5, -0.25)) 
#             # cbar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1e}'))
#             cbar.ax.tick_params(axis='x', which='both', rotation=90)
#             cbar.outline.set_visible(False)

#         # grid_temp[j * 3].set_ylabel(f"day {idx + 1}")
#         grid_temp[j * 3].text(-0.1, 0.5, f"day {idx + 1}", va='center', ha='center', rotation='vertical', transform=grid_temp[j * 3].transAxes)

#     # Set titles for each column
#     for i, var in enumerate(["T"]):
#         grid_temp[0].set_title(f"Predicted $\\hat{{{var}}}$ in K")
#         grid_temp[1].set_title(f"True ${var}$ in K")
#         grid_temp[2].set_title(f"$|{{{var}}} - \\hat{{{var}}}|/{{|{var}|}}$")

#     fig_temp.subplots_adjust(bottom=0.25)
#     PlotConfig.save_fig(fig_temp, str(FIG_DIR / name) + '_temp')
#     plt.close(fig_temp)

#     # Pressure-related plots
#     PlotConfig.setup()
#     figsize = PlotConfig.convert_width((2.5, 1), page_scale=1.0)
#     fig_pres = plt.figure(figsize=figsize)

#     grid_pres = AxesGrid(fig_pres, 111,  # 1x1 grid for layout
#                      nrows_ncols=(nplots, 3),
#                      axes_pad=0.05,  # Padding between plots
#                      cbar_mode='edge',  
#                      cbar_location='bottom',
#                      cbar_pad=0.05,
#                      share_all=True,
#                      label_mode="L",
#                      axes_class=(GeoAxes, dict(projection=ccrs.PlateCarree(central_longitude=180))))
    

#     p_pred = p_pred/1000
#     p_true = p_true/1000
#     p_cbar = dict(vmin=p_true.min(), vmax=p_true.max())
#     p_err_rel = (p_pred - p_true).abs() / p_true.abs()
#     p_err_cbar = dict(vmin=p_err_rel.min(), vmax=p_err_rel.max())

#     for j in range(nplots):
#         idx = days_idx[j] - 1

#         # Plot pressure (predicted, true, and relative error)
#         images = [
#             grid_pres[j * 3].contourf(lons, lats, p_pred[idx].squeeze(), 50, transform=ccrs.PlateCarree(), cmap=cmap_pres, **p_cbar),
#             grid_pres[j * 3 + 1].contourf(lons, lats, p_true[idx].squeeze(), 50, transform=ccrs.PlateCarree(), cmap=cmap_pres, **p_cbar),
#             grid_pres[j * 3 + 2].contourf(lons, lats, p_err_rel[idx].squeeze(), 50, transform=ccrs.PlateCarree(), cmap=cmap_err, **p_err_cbar),
#         ]

#         # Add coastlines and colorbars for each plot
#         for i in range(3):
#             # grid_pres[j * 3 + i].coastlines()
#             # grid_pres[j * 3 + i].add_feature(cfeature.LAND)
#             grid_pres[j * 3 + i].add_feature(cfeature.COASTLINE, facecolor='none', linewidth=0.3)
#             grid_pres[j * 3 + i].add_feature(cfeature.BORDERS, facecolor='none', linewidth=0.3)
#             cbar = fig_pres.colorbar(images[i], cax=grid_pres.cbar_axes[j * 3 + i], orientation='horizontal')
#             cbar.ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
#             # cbar.ax.xaxis.get_offset_text().set_position((0.5, -0.25)) 
#             # cbar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1e}'))
#             cbar.ax.tick_params(axis='x', which='both', rotation=90)
#             cbar.outline.set_visible(False)

#         # grid_pres[j * 3].set_ylabel(f"day {idx + 1}")
#         grid_pres[j * 3].text(-0.1, 0.5, f"day {idx + 1}", va='center', ha='center', rotation='vertical', transform=grid_pres[j * 3].transAxes)

#     # Set titles for each column
#     for i, var in enumerate(["p"]):
#         grid_pres[0].set_title(f"Predicted $\\hat{{{var}}}$ in KPa")
#         grid_pres[1].set_title(f"True ${var}$ in KPa")
#         grid_pres[2].set_title(f"$|{{{var}}} - \\hat{{{var}}}|/{{|{var}|}}$")

#     fig_pres.subplots_adjust(bottom=0.25)
#     PlotConfig.save_fig(fig_pres, str(FIG_DIR / name) + '_pres')
#     plt.close(fig_pres)

@torch.no_grad()
def plot_pred(nplots: int, name: str, days_idx: list):
    with open("prediction.pt", "rb") as fp:
        data = torch.load(fp)
    y_true, y_pred = data["y"], data["y_pred"]

    # Use predefined colormaps
    cmap_err = sns.color_palette("Blues", as_cmap=True)
    cmap_temp = cm.get_cmap('viridis')
    cmap_pres = cm.get_cmap('coolwarm')

    # Define a regular 72x72 grid for latitude and longitude
    # lons = np.concatenate((np.linspace(0, 180, 36, endpoint=False), np.linspace(-180, 0, 36, endpoint=False)))   # Longitude grid from -180 to 180 degrees
    lons = np.linspace(0, 360, 72)
    lats = np.linspace(-90, 90, 72)    # Latitude grid from -90 to 90 degrees

    t_true, p_true = y_true.cpu().chunk(2, dim=-1)
    t_pred, p_pred = y_pred.cpu().chunk(2, dim=-1)

    # Temperature-related plots
    PlotConfig.setup()
    figsize = PlotConfig.convert_width((2.5, 1), page_scale=1.0)
    fig_temp = plt.figure(figsize=figsize)
    
    grid_temp = AxesGrid(fig_temp, 111,  # 1x1 grid for layout
                     nrows_ncols=(nplots, 3),
                     axes_pad=0.05,  # Padding between plots
                     cbar_mode='edge',  
                     cbar_location='bottom',
                     cbar_pad=0.05,
                     share_all=True,
                     label_mode="L",
                     axes_class=(GeoAxes, dict(projection=ccrs.PlateCarree(central_longitude=180))))
    

    t_min  = np.floor(np.min([t_true.min(), t_pred.min()]))
    t_max  = np.ceil(np.max([t_true.max(), t_pred.max()]))
    normalizer = Normalize(t_min, t_max)
    temp_im    = cm.ScalarMappable(norm=normalizer, cmap=cmap_temp)
    t_cbar     = dict(vmin=t_min, vmax=t_max)
    num_levels = 100
    levels = np.linspace(t_min, t_max, num_levels, dtype=np.int64)
    t_err_rel  = (t_pred - t_true).abs() / t_true.abs()
    t_err_cbar = dict(vmin=t_err_rel.min(), vmax=t_err_rel.max())

    for j in range(nplots):
        idx = days_idx[j] - 1

        # Plot temperature (predicted, true, and relative error)
        images = [
            grid_temp[j * 3].contourf(lons, lats, t_pred[idx].squeeze(), transform=ccrs.PlateCarree(), cmap=cmap_temp, **t_cbar, 
                                      levels=levels, norm=normalizer),
            grid_temp[j * 3 + 1].contourf(lons, lats, t_true[idx].squeeze(), transform=ccrs.PlateCarree(), cmap=cmap_temp, **t_cbar,
                                      levels=levels, norm=normalizer),
            grid_temp[j * 3 + 2].contourf(lons, lats, t_err_rel[idx].squeeze(), 50, transform=ccrs.PlateCarree(), cmap=cmap_err, **t_err_cbar),
        ]

        # Add coastlines and colorbars for each plot
        for i in range(3):
            # grid_temp[j * 3 + i].coastlines()
            # grid_temp[j * 3 + i].add_feature(cfeature.LAND)
            grid_temp[j * 3 + i].add_feature(cfeature.COASTLINE, facecolor='none', linewidth=0.3)
            grid_temp[j * 3 + i].add_feature(cfeature.BORDERS, facecolor='none', linewidth=0.3)
            if i<2:
                cbar = fig_temp.colorbar(temp_im, cax=grid_temp.cbar_axes[j * 3 + i], orientation='horizontal')
                cbar.set_ticks(levels[::10]) 
            else:
                ## plot colorbar for t_err
                cbar = fig_temp.colorbar(images[i], cax=grid_temp.cbar_axes[j * 3 + i], orientation='horizontal')
            cbar.ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            # cbar.ax.xaxis.get_offset_text().set_position((0.5, -0.25)) 
            # cbar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1e}'))
            cbar.ax.tick_params(axis='x', which='both', rotation=90)
            cbar.outline.set_visible(False)

        # grid_temp[j * 3].set_ylabel(f"day {idx + 1}")
        grid_temp[j * 3].text(-0.1, 0.5, f"day {idx + 1}", va='center', ha='center', rotation='vertical', transform=grid_temp[j * 3].transAxes)

    # Set titles for each column
    for i, var in enumerate(["T"]):
        grid_temp[0].set_title(f"Predicted $\\hat{{{var}}}$ in K")
        grid_temp[1].set_title(f"True ${var}$ in K")
        grid_temp[2].set_title(f"$|{{{var}}} - \\hat{{{var}}}|/{{|{var}|}}$")

    fig_temp.subplots_adjust(bottom=0.25)
    PlotConfig.save_fig(fig_temp, str(FIG_DIR / name) + '_temp')
    plt.close(fig_temp)

    # Pressure-related plots
    PlotConfig.setup()
    figsize = PlotConfig.convert_width((2.5, 1), page_scale=1.0)
    fig_pres = plt.figure(figsize=figsize)

    grid_pres = AxesGrid(fig_pres, 111,  # 1x1 grid for layout
                     nrows_ncols=(nplots, 3),
                     axes_pad=0.05,  # Padding between plots
                     cbar_mode='edge',  
                     cbar_location='bottom',
                     cbar_pad=0.05,
                     share_all=True,
                     label_mode="L",
                     axes_class=(GeoAxes, dict(projection=ccrs.PlateCarree(central_longitude=180))))
    

    p_pred = p_pred/1000
    p_true = p_true/1000
    p_min  = np.floor(np.min([p_pred.min(), p_true.min()]))
    p_max  = np.ceil(np.max([p_pred.max(), p_true.max()]))
    normalizer = Normalize(p_min, p_max)
    pres_im    = cm.ScalarMappable(norm=normalizer, cmap=cmap_pres)
    p_cbar     = dict(vmin=p_min, vmax=p_max)
    num_levels = 50
    levels = np.linspace(p_min, p_max, num_levels, dtype=np.int64)
    p_err_rel = (p_pred - p_true).abs() / p_true.abs()
    p_err_cbar = dict(vmin=p_err_rel.min(), vmax=p_err_rel.max())

    for j in range(nplots):
        idx = days_idx[j] - 1

        # Plot pressure (predicted, true, and relative error)
        images = [
            grid_pres[j * 3].contourf(lons, lats, p_pred[idx].squeeze(), transform=ccrs.PlateCarree(), cmap=cmap_pres, **p_cbar,
                                      levels=levels, norm=normalizer),
            grid_pres[j * 3 + 1].contourf(lons, lats, p_true[idx].squeeze(), transform=ccrs.PlateCarree(), cmap=cmap_pres, **p_cbar,
                                          levels=levels, norm=normalizer),
            grid_pres[j * 3 + 2].contourf(lons, lats, p_err_rel[idx].squeeze(), 50, transform=ccrs.PlateCarree(), cmap=cmap_err, **p_err_cbar),
        ]

        # Add coastlines and colorbars for each plot
        for i in range(3):
            # grid_pres[j * 3 + i].coastlines()
            # grid_pres[j * 3 + i].add_feature(cfeature.LAND)
            grid_pres[j * 3 + i].add_feature(cfeature.COASTLINE, facecolor='none', linewidth=0.3)
            grid_pres[j * 3 + i].add_feature(cfeature.BORDERS, facecolor='none', linewidth=0.3)
            if i<2:
                cbar = fig_pres.colorbar(pres_im, cax=grid_pres.cbar_axes[j * 3 + i], orientation='horizontal')
                cbar.set_ticks(levels[::8]) 
                cbar.ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            else:
                ## plot colorbar for t_err
                cbar = fig_pres.colorbar(images[i], cax=grid_pres.cbar_axes[j * 3 + i], orientation='horizontal')
                cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
            # cbar.ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            # cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            # cbar.ax.xaxis.get_offset_text().set_position((0.5, -0.25)) 
            # cbar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1e}'))
            cbar.ax.tick_params(axis='x', which='both', rotation=90)
            cbar.outline.set_visible(False)

        # grid_pres[j * 3].set_ylabel(f"day {idx + 1}")
        grid_pres[j * 3].text(-0.1, 0.5, f"day {idx + 1}", va='center', ha='center', rotation='vertical', transform=grid_pres[j * 3].transAxes)

    # Set titles for each column
    for i, var in enumerate(["p"]):
        grid_pres[0].set_title(f"Predicted $\\hat{{{var}}}$ in KPa")
        grid_pres[1].set_title(f"True ${var}$ in KPa")
        grid_pres[2].set_title(f"$|{{{var}}} - \\hat{{{var}}}|/{{|{var}|}}$")

    fig_pres.subplots_adjust(bottom=0.25)
    PlotConfig.save_fig(fig_pres, str(FIG_DIR / name) + '_pres')
    plt.close(fig_pres)




def l2_norm_2D(quad_rule, y_mean, y_pred, y_true):
    diffs = y_true - y_pred
    numer = quadrature.integrate_on_domain(diffs.pow(2), quad_rule).sqrt()
    denom = quadrature.integrate_on_domain((y_true).pow(2), quad_rule).sqrt()
    return numer / denom


@torch.no_grad()
def err_over_time(
    lag: int,
    y_mean: Float[Tensor, "batch 1 nx ny 3"],
    y_true: Float[Tensor, "batch nt nx ny 3"],
    y_pred: Float[Tensor, "batch nt nx ny 3"],
    metrics: dict,
    quad_rule: quadrature.QuadGrid,
):
    y_true, y_pred = y_true[:, lag:], y_pred[:, lag:]
    batch_size, num_t = y_true.shape[0], y_true.shape[1]
    t = torch.arange(num_t).view(1, -1).repeat(batch_size, 1)
    if "t" not in metrics:
        metrics["t"] = []
        metrics["T"] = []
        metrics["p"] = []

    rel_err = vmap(partial(l2_norm_2D, quad_rule))(y_mean, y_pred, y_true)
    # check that things are being raveled like they should be
    metrics["t"].append(t.ravel().cpu())
    metrics["T"].append(rel_err[..., 0].ravel().cpu())
    metrics["p"].append(rel_err[..., 1].ravel().cpu())


def plot_err_over_time():
    err_df = pd.read_pickle(CURR_DIR / "err_time_dict.pkl")
    PlotConfig.setup()
    figsize = PlotConfig.convert_width((10, 1), page_scale=1.0)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # get into percent
    err_df = err_df.melt(id_vars="t", value_vars=["p", "T"])
    err_df = err_df.assign(
        Output=err_df["variable"].map({"p": "$p$", "T": "$T$"})  # type: ignore
    )
    g = sns.lineplot(data=err_df, x="t", y="value", ax=ax, hue="Output", errorbar=("sd", 1.0))  # type: ignore
    g.legend_.set_title(None)  # type: ignore
    ax.set_xlabel("$t$ (days)")
    ax.set_ylabel("rel. $L^2$ error")
    PlotConfig.save_fig(fig, str(FIG_DIR / "err-over-time"))


def main(file: str):
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get index of best model based on validation loss
    with open(file, "rb") as fp:
        train_state = torch.load(fp)

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
    quad_grid = quadrature.get_quad_grid(
        quad_fns, [lag, 72, 72], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]
    )
    cart_grid = quadrature.quad_to_cartesian_grid(quad_grid)
    cart_grid = quadrature.cart_grid_to_device(cart_grid, device)

    quad_grid_2d = quadrature.get_quad_grid(
        quadrature.trapezoidal_vecs, [72, 72], [-1.0, -1.0], [1.0, 1.0]
    )
    quad_grid_2d = quadrature.quad_grid_to_device(quad_grid_2d, device)

    _, _, s_test, norm_params = get_raw_data()
    norm_params = {k: v.to(device) for k, v in norm_params.items()}
    s_test.squeeze_(0)

    win_size = lag + 365
    test_loader = DataLoader(NonOverlapDS(s_test, win_size), batch_size=1)

    metrics = {}
    err_df = {}
    norm_fn = partial(unnormalize, **norm_params)
    for j, y in enumerate(tqdm(test_loader)):
        y = y.to(device)
        y_pred = forecast_metrics(lag, cart_grid, y, model, norm_fn, metrics)
        if j == 0:
            with open("prediction.pt", "wb") as fp:
                torch.save(
                    {"y": norm_fn(y[0, lag:]), "y_pred": norm_fn(y_pred[0, lag:])}, fp
                )
        # relative errors over time
        y_mean = y[:, :lag].mean(1, keepdim=True)
        err_over_time(lag, y_mean, norm_fn(y), norm_fn(y_pred), err_df, quad_grid_2d)
    err_df = {k: torch.cat(v) for k, v in err_df.items()}
    err_df = pd.DataFrame(err_df)
    err_df.to_pickle(CURR_DIR /"err_time_dict.pkl")
    print(file, metrics)


if __name__ == "__main__":
    get_latest_ckpt()
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, default=get_latest_ckpt())
    args = parser.parse_args()
    main(args.file)
    plot_err_over_time()
    plot_pred(1, "weather-prediction", days_idx = [120])
