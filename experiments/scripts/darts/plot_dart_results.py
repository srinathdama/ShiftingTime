import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec

import os
CURR_DIR       = os.path.dirname(os.path.abspath(__file__))
root_base_path = os.path.dirname(os.path.dirname(CURR_DIR))
import sys
sys.path
sys.path.append(root_base_path)

from scripts.plotting_config import PlotConfig

output_save_dir  = os.path.join(CURR_DIR, f'figures')

from scipy.stats.mstats import gmean

name_order = ['N-BEATS','SM-GP','TCN','N-HiTS', "LLaMA-2", 'GPT-3', 'ARIMA', 'TS-KRNO']

colors = ['lightgrey' if name != 'TS-KRNO' else 'dimgrey' for name in name_order]

csv_fn = os.path.join(output_save_dir, "darts_results_agg.csv")
df = pd.read_csv(csv_fn)
df['Type'] = df['Type'].apply(lambda x: x.replace(" 70B", ""))

geometric_means = df.groupby('Type')['MAE'].apply(gmean).reset_index()

geometric_means = geometric_means[geometric_means['Type'].isin(name_order)]

geometric_means_list = []
for name_ in name_order:
    geometric_means_list.append(geometric_means[geometric_means['Type']==name_]['MAE'].values[0])

print(geometric_means_list)

### plot the predictions on Darts dataset
def plot_darts_all(datasets, data_dir):

    ncols = int(len(datasets)/2)
    PlotConfig.setup()
    figsize = PlotConfig.convert_width((10, 4), page_scale=1.0)
    # Now create the plot
    fig, axs = plt.subplots(2, ncols, figsize=figsize)

    for i, dataset in enumerate(datasets):
        exp_dir_path = os.path.join(root_base_path, data_dir)
        npz_file     = np.load(os.path.join(exp_dir_path, f'{dataset}/pred_with_out_val.npz'))
        gt_data      = npz_file['gt_data']
        test_idx     = npz_file['test_idx']
        y_test_pred  = npz_file['y_test_pred']

        row_idx  = 0 if i < 4 else 1
        col_idx  = i if i < 4 else i-4
        axs[row_idx, col_idx].plot(gt_data, 'k-', label='GroundTruth', linewidth=0.2)
        axs[row_idx, col_idx].plot(test_idx, y_test_pred, 'b-', label='TS-KRNO', linewidth=0.3)
        axs[row_idx, col_idx].set_title(f'{dataset}', pad=2)
        axs[row_idx, col_idx].grid(False)
        axs[row_idx, col_idx].set_xticks([]) 
        axs[row_idx, col_idx].set_yticks([]) 
        # axs.grid(axis='x', which='major', linestyle='--', linewidth=0.2)
        # axs[row_idx, col_idx].legend()
    fig.subplots_adjust(bottom=0.15)
    legend_ax = fig.add_axes([0.3, 0.05, 0.4, 0.1])  # Adjust these values as needed
    legend_ax.axis('off')  # Hide the axis
    handles, labels = axs[0, 0].get_legend_handles_labels()  # Get handles and labels for the legend
    legend_ax.legend(handles, labels, loc='center', ncol=2) 
    PlotConfig.save_fig(fig, os.path.join(output_save_dir, 'darts_plots_combined'))
    # fig.savefig(os.path.join(output_save_dir, 'darts_plots_combined1.pdf'), format='pdf', dpi=1000, bbox_inches="tight")
    plt.close(fig)


def plot_bar():
    ### plot the NMAE bar plot
    PlotConfig.setup()
    figsize = PlotConfig.convert_width((2, 1), page_scale=.5)
    # Now create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot geometric mean, for example as a scatter plot
    # ax.scatter(name_order, geometric_means_list, color='grey', edgecolor='black')
    ax.bar(name_order, geometric_means_list, color=colors, edgecolor='black')

    # Customizing the plot
    ax.set_ylabel(f'Geometric Mean \n of NMAE')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='y', which='major', pad=0, labelsize=9)
    ax.set(xlabel=None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=50, horizontalalignment='right')
    ax.grid(axis='x')
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_title("Darts")

    PlotConfig.save_fig(fig, os.path.join(output_save_dir, 'darts_geo_mean_NMAE'))
    plt.close(fig)


if __name__ == "__main__":
    
    ### plot the NMAE bar plot
    # plot_bar()
    data_dir      = 'outputs/darts/KRNO'
    datasets      = ['AirPassengers', 'AusBeer', 'GasRateCO2','MonthlyMilk',  
                     'sunspots', 'Wine', 'Wooly', 'HeartRate']
    plot_darts_all(datasets, data_dir)
