# KRNO
Code repository for the ICML 2025 paper 'Shifting Time: Time-series Forecasting with Khatri-Rao Neural Operators'

- The scripts used for training and post-processing all the datasets are provided in the folder `experiments/scripts`.

## Scripts Details

### Spatial and Spatio-temporal problems
- `experiments/scripts/darcy-flow/run_all.sh` contains the commands for training and testing using KRNO on Darcy flow dataset.
- In the case of **Hyper-Elastic**, **Shallow water**, and **Climate modeling** problems, `train_script.py` in the corresponding folders is used to train KRNO. Testing is done using the script `post_process.py`.

### Temporal forecasting problems
-  Scripts `experiments/scripts/mujoco/run_all${i}.sh` are used for training KRNO on all MuJoCo datasets. We used the training pipeline and evaluation metric implemented by Oh, Y. et al. [1]  ([https://github.com/yongkyung-oh/Stable-Neural-SDEs](https://github.com/yongkyung-oh/Stable-Neural-SDEs)) 
- Scripts to train KRNO on MIMIC, USHCN, and Human Activity datasets are given in `experiments/scripts/irregular_time_series/krno`. We used the training pipeline and evaluation merics implemented  by Zhang, Weijia, et al. [2] ([https://github.com/usail-hkust/t-PatchGNN](https://github.com/usail-hkust/t-PatchGNN)). Instructions to download these datasets are also provided in their github page.  
- Training scripts for spiral trajectory (short and long) are provided in `experiments/scripts/spiral`. 
- `experiments/scripts/darts/run_darts_KRNO.sh` contains the commands for hyperparameter tuning and testing using KRNO on all the **Darts** datasets. Final testing results for each dataset is written to the test file 'outputs/KRNO/*<dataset*>/final_test_results.txt' along with intermediate results from hyperparameter studies. 
- `experiments/scripts/m4_crypto_traj/run_crypto_KRNO.sh` contains the commands for hyperparameter tuning and testing using KRNO on all the **Crypto** test cases.
- `experiments/scripts/m4_crypto_traj/run_traj_KRNO.sh` contains the commands for hyperparameter tuning and testing using KRNO on all the **Player Trajectory** datasets.  
- `experiments/scripts/m4_crypto_traj/run_m4_KRNO.sh` contains the commands for hyperparameter tuning and testing using KRNO on all the **M4** test cases.  

[1] Oh, Y., Lim, D., & Kim, S. (2024, May). Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data. In The Twelfth International Conference on Learning Representations

[2] Zhang, Weijia, et al. "Irregular multivariate time series forecasting: A transformable patching graph neural networks approach." ICML 2024.

## Datasets

- The datasets for the Darcy, Hyper-elasticity, and climate modeling are already included in the corresponding scripts folders. 
- Dataset for shallow water problem can be downloaded from [LOCA github page](https://github.com/PredictiveIntelligenceLab/LOCA?tab=readme-ov-file). Copy the downloaded npz files to the folder `experiments/scripts/shallow_water_<model>/loca-data/`.
- Darts datasets are included in the folder `experiments/dataset/darts`
- Instructions to download the datasets for M4, Crypto and Player Trajectory are given in `experiments/dataset/ReadMe.md`. 





## Installing dependencies

`conda create --name KRNO_env python=3.10`

`conda install numpy==1.26.2 scipy matplotlib seaborn h5py`

`conda install -c conda-forge pandas scikit-learn patool tqdm sktime wandb cartopy`

`conda install pytorch==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia`

`pip install neuraloperator==0.3.0 torch-harmonics==0.6.5`


## Installing KRNO

`cd krno/`

`pip install .`


## Usage

``` python

import torch
from torch import nn

from khatriraonop import models, quadrature

# setting up the computational grid

# get 1D quad rules -- trapz assumes evenly spaced grid here we use a 10x10 grid
grid = quadrature.get_quad_grid(
    quadrature.trapezoidal_vecs, [10, 10], [-1, -2], [2, 3]
)
# convert into a cartesian product grid used by khatri rao integral transforms
cart_grid = quadrature.quad_to_cartesian_grid(grid)

# ---------------------------------------------------------------------------------
# setting up the model grid
# dimensionality of the input domain
d = 2
# number of inputs function
lifting_layers = [2, 128, 20]
integral_layers = [20, 20, 20]
projection_layers = [20, 128, 2]

# 4 hidden layers, each with 2 hidden layers with 64 hidden units
# n_hidden_units
# n_hidden_layers
kernel_layers = [[64, 64]] * 2
# make helper method which infers the number of layers
model1 = models.KhatriRaoNO(
    d, lifting_layers, integral_layers, kernel_layers, projection_layers, nn.ReLU()
)
model2 = models.KhatriRaoNO.easy_init(
    d,
    in_channels=2,
    out_channels=2,
    lifting_channels=128,
    integral_channels=20,
    n_integral_layers=2,
    projection_channels=128,
    n_hidden_units=64,
    n_hidden_layers=2,
    nonlinearity=nn.ReLU(),
)
assert model1.lifting_layers == model2.lifting_layers
assert model1.integral_layers == model2.integral_layers
assert model1.projection_layers == model2.projection_layers

# ---------------------------------------------------------------------------------
# generating some dummy data
batch_size = 8

u = torch.randn(batch_size, 10, 10, 2)

# transform u -> v
v = model1(cart_grid, u)
assert tuple(v.shape) == (batch_size, 10, 10, 2)

v = model2(cart_grid, u)
assert tuple(v.shape) == (batch_size, 10, 10, 2)

# ---------------------------------------------------------------------------------
# now do superresolution

out_grid = quadrature.get_quad_grid(
    quadrature.trapezoidal_vecs, [30, 30], [-1, -2], [2, 3]
)
v = model1.super_resolution(out_grid, grid, u)
assert tuple(v.shape) == (batch_size, 30, 30, 2)
v = model2.super_resolution(out_grid, grid, u)
assert tuple(v.shape) == (batch_size, 30, 30, 2)

```




