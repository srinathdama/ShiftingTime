from __future__ import annotations
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from jax.core import as_named_shape

import argparse
import time
import os
import pathlib
import random
from typing import Any

from pathos.pools import ProcessPool
from scipy import linalg, interpolate
from sklearn import gaussian_process as gp
import argparse
from jax.example_libraries.stax import Dense, Gelu, Relu
from jax.example_libraries import stax
import os, sys

import timeit

from jax.example_libraries import optimizers

from absl import app
import jax
from jax import vjp
import jax.numpy as jnp
import numpy as np
from jax.numpy.linalg import norm

from jax import random, grad, vmap, jit, pmap
from functools import partial 
from torch.func import vmap as tvmap

from torch.utils import data
from jaxtyping import Float
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import TensorDataset
from khatriraonop import quadrature

from scipy import interpolate

from tqdm import trange
from math import log, sqrt, sin, cos

import itertools
import torch

from kymatio.sklearn import HarmonicScattering3D

import flax

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import pandas as pd


CURR_DIR = pathlib.Path(__file__).parent.absolute()
DATA_DIR = CURR_DIR / "loca-data"
CKPT_DIR = CURR_DIR / "ckpts"
FIG_DIR = CURR_DIR / "figs"

sys.path.append(str(CURR_DIR / ".."))
root_base_path = os.path.dirname(os.path.dirname(CURR_DIR))
sys.path
sys.path.append(root_base_path)
from utils.utilities3 import *
from plotting_config import PlotConfig

os.makedirs(CKPT_DIR, exist_ok=True)

# torch.set_float32_matmul_precision("high")
# torch.set_default_dtype(torch.float64)
torch.set_float32_matmul_precision("highest")

myloss = LpLoss(size_average=False)

################################################################
# 3d fourier layers
################################################################

def save_params(params, filename):
    beta, gamma, q_params, g_params, v_params = params
    # Serialize each component separately
    with open(filename, 'wb') as f:
        f.write(flax.serialization.to_bytes((beta, gamma, q_params, g_params, v_params)))

def load_params(filename, template_params):
    with open(filename, 'rb') as f:
        serialized_params = f.read()
    beta, gamma, q_params, g_params, v_params = template_params
    beta, gamma, q_params, g_params, v_params = flax.serialization.from_bytes((beta, gamma, q_params, g_params, v_params), serialized_params)
    return [beta, gamma, q_params, g_params, v_params]


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return str(np.argmin(memory_available))

@torch.no_grad()
def plot_pred(
    y_true: Float[Tensor, "nt nx ny 3"],
    y_pred: Float[Tensor, "nt nx ny 3"],
    nplots: int,
    name: str,
):

    cmap = sns.color_palette("Spectral", as_cmap=True)
    vars = ["\\rho", "u", "v"]
    PlotConfig.setup()
    figsize = PlotConfig.convert_width((2, 1), page_scale=1.0)
    fig, axs = plt.subplots(6, nplots, figsize=figsize)
    n_time = len(y_true) // nplots
    p_true, u_true, v_true = y_true.cpu().chunk(3, dim=-1)
    p_pred, u_pred, v_pred = y_pred.cpu().chunk(3, dim=-1)

    for j in range(nplots):
        idx = j * n_time
        p_min, p_max = p_true[idx].min(), p_true[idx].max()
        u_min, u_max = u_true[idx].min(), u_true[idx].max()
        v_min, v_max = v_true[idx].min(), v_true[idx].max()
        images = [
            axs[0, j].imshow(p_true[idx], cmap=cmap, vmin=p_min, vmax=p_max, interpolation='bilinear'),
            axs[1, j].imshow(u_true[idx], cmap=cmap, vmin=u_min, vmax=u_max, interpolation='bilinear'),
            axs[2, j].imshow(v_true[idx], cmap=cmap, vmin=v_min, vmax=v_max, interpolation='bilinear'),
            axs[3, j].imshow(p_pred[idx], cmap=cmap, vmin=p_min, vmax=p_max, interpolation='bilinear'),
            axs[4, j].imshow(u_pred[idx], cmap=cmap, vmin=u_min, vmax=u_max, interpolation='bilinear'),
            axs[5, j].imshow(v_pred[idx], cmap=cmap, vmin=v_min, vmax=v_max, interpolation='bilinear'),
        ]
    for ax in axs.flatten():
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
    for i, var in enumerate(vars):
        axs[i, 0].set_ylabel(f"${var}$")
        axs[i + len(vars), 0].set_ylabel("$\\hat{%s}$" % var)

    fig.subplots_adjust(bottom=0.25)
    common_ax = fig.add_subplot(111, frameon=False)  # Add a common subplot
    common_ax.axhline(0.5, alpha=0.5, color="k", linestyle="--", linewidth=2)
    common_ax.set_xlabel("$t$ (sec)")
    common_ax.grid(False)
    common_ax.set_xticks(np.linspace(0, 1.0, 5))
    common_ax.set_xticklabels([f"{v:.2f}" for v in np.linspace(0, 0.6, 5)])
    common_ax.set_yticks([])
    common_ax.set_yticklabels([])
    PlotConfig.save_fig(fig, str(FIG_DIR / name))
    plt.close(fig)



os.environ['CUDA_VISIBLE_DEVICES']= get_freer_gpu()

def output_construction(Ux,t_his,cx, cy, ng,P=1000, num_train=1000, ds=3, Nx=30, Ny=30, Nt=100):
    U_all = np.zeros((P,ds))
    Y_all = np.zeros((P,ds))
    it = np.random.randint(Nt, size=P)
    x  = np.random.randint(Nx, size=P)
    y  = np.random.randint(Ny, size=P)
    T, X, Y = np.meshgrid(t_his,cx,cy,indexing="ij")
    Y_all[:,:] = np.concatenate((T[it,x][range(P),y][:,None], X[it,x][range(P),y][:,None], Y[it,x][range(P),y][:,None]),axis=-1)
    U_all[:,:] = Ux[it,x][range(P),y]
    return U_all, Y_all

def pairwise_distances(dist,**arg):
    return jit(vmap(vmap(partial(dist,**arg),in_axes=(None,0)),in_axes=(0,None)))

def euclid_distance(x,y,square=True):
    XX=jnp.dot(x,x)
    YY=jnp.dot(y,y)
    XY=jnp.dot(x,y)
    return XX+YY-2*XY

class DataGenerator(data.Dataset):
    def __init__(self, u, y, s,
                 batch_size=100, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u  = u
        self.y = y
        self.s = s
        
        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs,outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        s = self.s[idx,:,:]
        inputsxu  = self.u[idx,:,:]
        y = self.y[idx,:,:]
        inputs = (inputsxu, y)
        return inputs, s

class PositionalEncodingY: 
    def __init__(self, Y, d_model, max_len = 100, H=4): 
        self.d_model = int(np.ceil(d_model/6)*2)
        self.Y = Y 
        self.max_len = max_len 
        self.H = H
 
    def forward(self, x):
        pet = np.zeros((x.shape[0], self.max_len, self.H))
        pex = np.zeros((x.shape[0], self.max_len, self.H))
        pey = np.zeros((x.shape[0], self.max_len, self.H))
        T = jnp.take(self.Y, 0, axis=2)[:,:,None]
        X = jnp.take(self.Y, 1, axis=2)[:,:,None]
        Y = jnp.take(self.Y, 2, axis=2)[:,:,None]
        positionT = jnp.tile(T,(1,1,self.H))
        positionX = jnp.tile(X,(1,1,self.H))
        positionY = jnp.tile(Y,(1,1,self.H))
        div_term = 2**jnp.arange(0,int(self.H/2),1)*jnp.pi
        # pet = jax.ops.index_update(pet, jax.ops.index[:,:,0::2], jnp.cos(positionT[:,:,0::2] * div_term))
        # pet = jax.ops.index_update(pet, jax.ops.index[:,:,1::2], jnp.sin(positionT[:,:,1::2] * div_term))
        # pex = jax.ops.index_update(pex, jax.ops.index[:,:,0::2], jnp.cos(positionX[:,:,0::2] * div_term))
        # pex = jax.ops.index_update(pex, jax.ops.index[:,:,1::2], jnp.sin(positionX[:,:,1::2] * div_term))
        # pey = jax.ops.index_update(pey, jax.ops.index[:,:,0::2], jnp.cos(positionY[:,:,0::2] * div_term))
        # pey = jax.ops.index_update(pey, jax.ops.index[:,:,1::2], jnp.sin(positionY[:,:,1::2] * div_term))
        pet = jnp.array(pet)
        pex = jnp.array(pex)
        pey = jnp.array(pey)
        # Update pet
        pet = pet.at[:,:,0::2].set(jnp.cos(positionT[:,:,0::2] * div_term))
        pet = pet.at[:,:,1::2].set(jnp.sin(positionT[:,:,1::2] * div_term))

        # Update pex
        pex = pex.at[:,:,0::2].set(jnp.cos(positionX[:,:,0::2] * div_term))
        pex = pex.at[:,:,1::2].set(jnp.sin(positionX[:,:,1::2] * div_term))

        # Update pey
        pey = pey.at[:,:,0::2].set(jnp.cos(positionY[:,:,0::2] * div_term))
        pey = pey.at[:,:,1::2].set(jnp.sin(positionY[:,:,1::2] * div_term))

        pos_embedding =  jnp.concatenate((pet,pex,pey),axis=-1)
        x =  jnp.concatenate([x, pos_embedding], -1)
        return x

def scatteringTransform(sig, l=100, m=100, training_batch_size = 100):
    # scattering = Scattering3D(J=1, L=3, max_order=2, shape=(32, 32))
    scattering = HarmonicScattering3D(J=1, shape=(5, 32, 32), L=3)
    cwtmatr = np.zeros((training_batch_size, 1024*5, 1))
    sig = np.array(sig)
    for i in range(0,training_batch_size):
        scatteringCoeffs = scattering(sig[i,:,:].reshape(5,32,32))
        cwtmatr[i,:,:] = scatteringCoeffs[:,:,:].flatten()[:,None]
    return cwtmatr


def process_batch(xx, yy, global_vars, test_flag = False):

    # if test_flag:
    #     T = global_vars.T_test
    # else:
    #     T = global_vars.T_train
    num_train =  xx.shape[0]

    CX = global_vars.CX
    CY = global_vars.CY
    P = global_vars.P
    m = global_vars.m
    L = global_vars.L
    dx = global_vars.dx 
    du = global_vars.du
    dy = global_vars.dy
    ds = global_vars.ds
    n_hat = global_vars.n_hat
    l  = global_vars.l
    Nx = global_vars.Nx
    Ny = global_vars.Ny
    Nt = global_vars.Nt
    Ng = global_vars.Ng
    H  = global_vars.H
    T = global_vars.T_train[0:Nt]

    s_train = np.zeros((num_train,P,ds))
    y_train = np.zeros((num_train,P,dy))
    U_train = np.zeros((num_train,m,du))
    X_train = np.zeros((num_train,m,dx))

    ##  output_construction
    for i in range(0,num_train):
        s_train[i,:,:], y_train[i,:,:] = output_construction(yy[i,:,:,:,:], T, CX, CY, Ng,P=P,Nt=Nt)
        U_train[i,:,:] = xx[i,:,:,:].reshape(Nx*Ny*Nt, du)

    X_train = jnp.asarray(X_train)
    U_train =  np.asarray(U_train)
    y_train = jnp.asarray(y_train)
    s_train = jnp.asarray(s_train)

    ## scatteringTransform on inputs --- not working for 3D inputs
    # inputs_trainxu = np.zeros((num_train,m,3))
    # inputs_trainxu[:,:,0:1] = jnp.asarray(scatteringTransform(U_train[:,:,0:1], l=L, m=m, training_batch_size=num_train))
    # inputs_trainxu[:,:,1:2] = jnp.asarray(scatteringTransform(U_train[:,:,1:2], l=L, m=m, training_batch_size=num_train))
    # inputs_trainxu[:,:,2:3] = jnp.asarray(scatteringTransform(U_train[:,:,2:3], l=L, m=m, training_batch_size=num_train))
    # inputs_trainxu = jnp.array(inputs_trainxu)

    inputs_trainxu = jnp.array(U_train)

    ## PositionalEncodingY on y_train
    pos_encodingy  = PositionalEncodingY(y_train,int(y_train.shape[1]*y_train.shape[2]), max_len = P, H=H) 
    y_train  = pos_encodingy.forward(y_train) 
    del pos_encodingy 

    return [[inputs_trainxu, y_train], s_train]

class LOCA:
    def __init__(self, q_layers, g_layers, v_layers , m=100, P=100, H=100):    
        # Network initialization and evaluation functions

        self.encoder_init2, self.encoder_apply2 = self.init_NN(q_layers, activation=Gelu)
        self.in_shape = (-1, q_layers[0])
        self.out_shape, encoder_params2 = self.encoder_init2(random.PRNGKey(10000), self.in_shape)
        self.encoder_apply2 = self.encoder_apply2

        self.v_init, self.v_apply = self.init_NN(v_layers, activation=Gelu)
        self.in_shape = (-1, v_layers[0])
        self.out_shape, v_params = self.v_init(random.PRNGKey(10000), self.in_shape)
        self.v_apply = self.v_apply

        self.g_init, self.g_apply = self.init_NN(g_layers, activation=Gelu)
        self.in_shape = (-1, g_layers[0])
        self.out_shape, g_params = self.g_init(random.PRNGKey(10000), self.in_shape)
        self.g_apply = self.g_apply

        beta = [1.]
        gamma = [1.]

        params = (beta,gamma,encoder_params2, g_params, v_params)

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init,self.opt_update,self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3, 
                                                                      decay_steps=100, 
                                                                      decay_rate=0.99))
        self.opt_state = self.opt_init(params)
        # Logger
        self.itercount = itertools.count()
        self.loss_log = []

        self.vdistance_function = vmap(pairwise_distances(euclid_distance))

        print("Model initialized")
        
    def init_NN(self, Q, activation=Gelu):
        layers = []
        num_layers = len(Q)
        if num_layers < 2:
            net_init, net_apply = stax.serial()
        else:
            for i in range(0, num_layers-2):
                layers.append(Dense(Q[i+1]))
                layers.append(activation)
            layers.append(Dense(Q[-1]))
            net_init, net_apply = stax.serial(*layers)
        return net_init, net_apply

    def LOCA_net(self, params, inputs, ds=3):
        beta, gamma, q_params, g_params, v_params = params
        inputsxu, inputsy = inputs
        inputsy  = self.encoder_apply2(q_params,inputsy)

        d = self.vdistance_function(inputsy, inputsy)
        K =  beta[0]*jnp.exp(-gamma[0]*d)
        Kzz =  jnp.sqrt((1./K.shape[1])*jnp.sum(K ,axis=1,keepdims=True))
        Kyz =  jnp.sqrt((1./K.shape[1])*jnp.sum(K ,axis=-1,keepdims=True))
        mean_K = jnp.matmul(Kyz, Kzz)
        K = jnp.divide(K,mean_K)

        g  = self.g_apply(g_params, inputsy)
        g = (1./K.shape[1])*jnp.einsum("ijk,ikml->ijml",K,g.reshape(inputsy.shape[0], inputsy.shape[1], ds, int(g.shape[2]/ds)))
        g = jax.nn.softmax(g, axis=-1)

        value_heads = self.v_apply(v_params, inputsxu.reshape(inputsxu.shape[0],1,inputsxu.shape[1]*inputsxu.shape[2]))
        value_heads = value_heads.reshape(value_heads.shape[0],int(value_heads.shape[2]/ds),ds)
        attn_vec = jnp.einsum("ijkl,ilk->ijk", g,value_heads)

        return attn_vec


    @partial(jax.jit, static_argnums=0)
    def loss(self, params, batch):
        inputs, outputs = batch
        y_pred = self.LOCA_net(params,inputs)
        loss = np.mean((outputs- y_pred)**2)
        return loss    

    @partial(jax.jit, static_argnums=0)
    def lossT(self, params, batch):
        inputs, outputs = batch
        y_pred = self.LOCA_net(params,inputs)
        loss = np.mean((outputs - y_pred)**2)
        return loss    
    
    @partial(jax.jit, static_argnums=0)
    def L2errorT(self, params, batch):
        inputs, y = batch
        y_pred = self.LOCA_net(params,inputs)
        return norm(y.flatten() - y_pred.flatten(), 2)/norm(y.flatten(),2)


    @partial(jax.jit, static_argnums=0)
    def L2error(self, params, batch):
        inputs, y = batch
        y_pred = self.LOCA_net(params,inputs)
        return norm(y.flatten() - y_pred.flatten(), 2)/norm(y.flatten(),2)
    
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, batch)
        return self.opt_update(i, g, opt_state)

    @partial(jit, static_argnums=(0,))
    def predict(self, params, inputs):
        s_pred = self.LOCA_net(params,inputs)
        return s_pred

    @partial(jit, static_argnums=(0,))
    def predictT(self, params, inputs):
        s_pred = self.LOCA_net(params,inputs)
        return s_pred

    def ravel_list(self, *lst):
        return jnp.concatenate([jnp.ravel(elt) for elt in lst]) if lst else jnp.array([])

    def ravel_pytree(self, pytree):
        leaves, treedef = jax.tree_util.tree_flatten(pytree)
        flat, unravel_list = vjp(self.ravel_list, *leaves)
        unravel_pytree = lambda flat: jax.tree_util.tree_unflatten(treedef, unravel_list(flat))
        return flat, unravel_pytree

    def count_params(self, params):
        beta, gamma,q_params, g_params, v_params = params
        qlv, _ = self.ravel_pytree(q_params)
        vlv, _ = self.ravel_pytree(v_params)
        glv, _ = self.ravel_pytree(g_params)
        print("The number of model parameters is:",qlv.shape[0]+vlv.shape[0]+glv.shape[0])


def predict_function(inputs_trainxu,y, model=None,params= None, H=None):
    uCNN_super_all = model.predict(params, (inputs_trainxu, y))
    return uCNN_super_all, y[:,:,0:1], y[:,:,1:2], y[:,:,2:3]


def error_full_resolution(uCNN_super_all, s_all,tag='train', num_train=1000,P=128, Nx=30, Ny=30, Nt=10, idx=None, ds=3):
    print(s_all.shape)
    z = uCNN_super_all.reshape(num_train,Nx*Ny*Nt,ds)
    s = s_all.reshape(num_train,Nx*Ny*Nt,ds)
    test_error_rho = []
    for i in range(0,num_train):
        test_error_rho.append(norm(s[i,:,0]- z[i,:,0], 2)/norm(s[i,:,0], 2))
    print("The average "+tag+" rho error for the super resolution is %e, the standard deviation %e, the minimum error is %e and the maximum error is %e"%(np.mean(test_error_rho),np.std(test_error_rho),np.min(test_error_rho),np.max(test_error_rho)))

    test_error_u = []
    for i in range(0,num_train):
        test_error_u.append(norm(s[i,:,1]- z[i,:,1], 2)/norm(s[i,:,1], 2))
    print("The average "+tag+" u error for the super resolution is %e, the standard deviation %e, the minimum error is %e and the maximum error is %e"%(np.mean(test_error_u),np.std(test_error_u),np.min(test_error_u),np.max(test_error_u)))

    test_error_v = []
    for i in range(0,num_train):
        test_error_v.append(norm(s[i,:,2]- z[i,:,2], 2)/norm(s[i,:,2], 2))
    print("The average "+tag+" v error for the super resolution is %e, the standard deviation %e, the minimum error is %e and the maximum error is %e"%(np.mean(test_error_v),np.std(test_error_v),np.min(test_error_v),np.max(test_error_v)))

    absolute_error = np.abs(z-s)
    return absolute_error, np.mean(test_error_rho), np.mean(test_error_u),np.mean(test_error_v), (test_error_rho, test_error_u, test_error_v) 


def minmax(a, n, mean):
    minpos = a.index(min(a))
    maxpos = a.index(max(a)) 
    meanpos = min(range(len(a)), key=lambda i: abs(a[i]-mean))

    print("The maximum is at position", maxpos)  
    print("The minimum is at position", minpos)
    print("The mean is at position", meanpos)
    return minpos,maxpos,meanpos


##########

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
    jax.random.PRNGKey(seed)


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
    train_loader = DataLoader(s_train, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(s_val, batch_size=batch_size, shuffle=False)
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

def l2_norm_2D(quad_rule, y_pred, y_true):
    diffs = y_pred - y_true
    numer = quadrature.integrate_on_domain(diffs.pow(2), quad_rule).sqrt()
    denom = quadrature.integrate_on_domain(y_pred.pow(2), quad_rule).sqrt()
    return numer / denom

@torch.no_grad()
def err_over_time(
    y_true: Float[Tensor, "batch nt nx ny 3"],
    y_pred: Float[Tensor, "batch nt nx ny 3"],
    metrics: dict,
    quad_rule: quadrature.QuadGrid,
):
    batch_size, num_t = y_true.shape[0], y_true.shape[1]
    t = torch.linspace(0.0, 0.6, num_t).view(1, -1).repeat(batch_size, 1)
    if "t" not in metrics:
        metrics["t"] = []
        metrics["rho"] = []
        metrics["v1"] = []
        metrics["v2"] = []

    rel_err = tvmap(partial(l2_norm_2D, quad_rule))(y_pred, y_true)
    metrics["t"].append(t.ravel().cpu())
    metrics["rho"].append(rel_err[..., 0].ravel().cpu())
    metrics["v1"].append(rel_err[..., 1].ravel().cpu())
    metrics["v2"].append(rel_err[..., 2].ravel().cpu())



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
            T, S, C = yy.shape[1], yy.shape[2], yy.shape[-1]
            batch_size = yy.shape[0]
            xx = xx.permute(0,2,3,4,1) ## [B,lag,Nx,Ny,C] -> [B,Nx,Ny,C,lag]
            xx = xx.reshape((batch_size, S, S, 1, C*T)) #[B,Nx,Ny,C,lag] -> [B,Nx,Ny,1, C*lag]
            xx = xx.repeat([1,1,1,T,1])
            yy = yy.permute(0,2,3,1,4) ## [B,T,Nx,Ny,C] -> [B,Nx,Ny,T,C]
            # yy = yy.reshape(batch_size, S, S, C*T) #[B,Nx,Ny,C,lag] -> [B,Nx,Ny,C*lag]
            # out = model(xx).view(batch_size, S, S, C*T)
            out = model(xx)  ## [B,Nx,Ny,T,C]
            loss_ = myloss(out.reshape(batch_size, -1), yy.reshape(batch_size, -1))
            y_pred = out.permute(0,3,1,2,4) ## [B,Nx,Ny,T,C] -> [B,T,Nx,Ny,C]
            yy     = yy.permute(0,3,1,2,4) ## [B,Nx,Ny,T,C] -> [B,T,Nx,Ny,C]
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



def train(model, train_loader, valid_loader, global_vars,
                                    n_epochs = 200, valid_freq = 500):
    n_iters, epoch = 0, 0
    print("Training FNO on shallow water autoregressive dataset...", flush=True)
    
    for epoch in range(1, n_epochs + 1):
        print('epoch :', epoch)
        errors: dict[str, list[float]] = {
                                "loss_train": [],
                                "loss_test": [],
                                "errorTrain": [],
                                "errorTest": [],
                            }
        ## train for epoch
        for xx, yy in train_loader:

            train_batch = process_batch(xx, yy, global_vars)

            model.opt_state = model.step(next(model.itercount), model.opt_state, train_batch)
            
            params = model.get_params(model.opt_state)
            loss_train = model.loss(params, train_batch)
            errorTrain = model.L2error(params, train_batch)
            errors["loss_train"].append(loss_train)
            errors["errorTrain"].append(errorTrain)
            
            n_iters += 1
            # if n_iters % valid_freq == 0:

        ## validation
        params = model.get_params(model.opt_state)
        for xx, yy in valid_loader:
            test_batch = process_batch(xx, yy, global_vars)
            loss_test  = model.lossT(params, test_batch)
            errorTest  = model.L2errorT(params, test_batch)

            errors["loss_test"].append(loss_test)
            errors["errorTest"].append(errorTest)

        ## save model
        model.loss_log.append(loss_train)
        out_file = CKPT_DIR / f"loca_ep_{epoch}.pkl"

        save_params(params, out_file)

        print('Training loss:', np.mean(errors["loss_train"]), 'Train error:',  np.mean(errors["errorTrain"]),
                'val error:',  np.mean(errors["loss_test"]), 'val loss:', np.mean(errors["errorTest"]))
                

def rel_l2_vec(lag, y_true, y_pred):
    batch = y_true.shape[0]
    # y_true.shape == (batch, n_steps, nx, ny, 1)
    y_true = y_true[:, lag:].reshape(batch, -1)
    y_pred = y_pred[:, lag:].reshape(batch, -1)
    return torch.norm(y_pred - y_true, p=2, dim=1) / torch.norm(y_true, p=2, dim=1)

@torch.no_grad
def forecast(
    global_vars,
    y0: Float[Tensor, "bs lag nx ny 3"],
    n_steps: int,
    model,
    params
):
    lag = y0.shape[1]
    y_out = [y0]
    steps_so_far = lag
    while steps_so_far < n_steps:
        y_prev = y_out[-1]
        ## preprocess inputs

        num_test =  y_prev.shape[0]

        CX = global_vars.CX
        CY = global_vars.CY
        P = global_vars.P
        m = global_vars.m
        L = global_vars.L
        dx = global_vars.dx 
        du = global_vars.du
        dy = global_vars.dy
        ds = global_vars.ds
        n_hat = global_vars.n_hat
        l  = global_vars.l
        Nx = global_vars.Nx
        Ny = global_vars.Ny
        Nt = global_vars.Nt
        Ng = global_vars.Ng
        H  = global_vars.H
        T = global_vars.T_train[0:Nt]

        TT, XX, YY = np.meshgrid(T, CX, CY, indexing="ij")

        TT = np.expand_dims(TT,axis=0)
        XX = np.expand_dims(XX,axis=0)
        YY = np.expand_dims(YY,axis=0)

        TT = np.tile(TT,(num_test,1,1)).reshape(num_test,Nx*Ny*Nt,1)
        XX = np.tile(XX,(num_test,1,1)).reshape(num_test,Nx*Ny*Nt,1)
        YY = np.tile(YY,(num_test,1,1)).reshape(num_test,Nx*Ny*Nt,1)

        Y_test_in = np.concatenate((TT, XX, YY),axis=-1)

        pos_encodingy  = PositionalEncodingY(Y_test_in,int(Y_test_in.shape[1]*Y_test_in.shape[2]), max_len = Y_test_in.shape[1], H=H)
        Y_test_in  = pos_encodingy.forward(Y_test_in)
        del pos_encodingy

        U_test = np.zeros((num_test,m,du))

        for i in range(num_test):
            U_test[i,:,:] = y_prev[i,:,:,:].reshape(Nx*Ny*Nt,du)

        U_test =  np.asarray(U_test)
        U_test =  np.reshape(U_test,(num_test,m,du))
        inputs_testxu = jnp.array(U_test)

        print("Predicting the solution for the full resolution")
        y_pred = np.zeros_like(y_prev).reshape(num_test, Nx*Ny*Nt, ds)
        for i in range(0, Nx*Ny*Nt, P):
            idx = i + np.arange(0,P)
            y_pred[:,idx,:], T_out, X, Y  = predict_function(inputs_testxu , Y_test_in[:,idx,:], model=model, params=params, H=H)

        ##
        y_pred = y_pred.reshape(y_prev.shape)
        y_out.append(torch.tensor(y_pred))
        steps_so_far += lag
    return torch.cat(y_out, dim=1)[:, :n_steps]

def forecast_metrics(lag, global_vars, y, model, params, metrics):
    y0 = y[:, :lag]
    n_steps = y.shape[1]
    y_pred = forecast(global_vars, y0, n_steps, model, params)
    rho_true, v1_true, v2_true = y.chunk(3, dim=-1)
    rho_pred, v1_pred, v2_pred = y_pred.chunk(3, dim=-1)
    rho_err = rel_l2_vec(lag, rho_true, rho_pred)
    v1_err = rel_l2_vec(lag, v1_true, v1_pred)
    v2_err = rel_l2_vec(lag, v2_true, v2_pred)
    batch = len(rho_err)
    if "N" not in metrics:
        metrics["N"] = 0
        metrics["rho_l2"] = 0.0
        metrics["v1_l2"] = 0.0
        metrics["v2_l2"] = 0.0
    metrics["N"] += batch
    scale = batch / metrics["N"]
    metrics["rho_l2"] += scale * (rho_err.mean().item() - metrics["rho_l2"])
    metrics["v1_l2"] += scale * (v1_err.mean().item() - metrics["v1_l2"])
    metrics["v2_l2"] += scale * (v2_err.mean().item() - metrics["v2_l2"])
    return y_pred


def main(lag: int, train_flag = True, plot_flag = False):
    seed = 42
    seed_everything(seed)
    # to leverage quad structure lag must == forward_steps
    print("Loading data...", flush=True)
    class GlobalVar():
        pass

    global_vars = GlobalVar()

    global_vars.forward_steps = lag
    global_vars.S = 32
    d = np.load("code/scripts/shallow-water/loca-data/test_SW.npz")
    global_vars.T_train  = d["T_test"]
    global_vars.CX = d["X_test"]
    global_vars.CY = d["Y_test"]
    global_vars.T_test  = d["T_test"][0:60]

    n_epochs = 200
    valid_freq = 500
    global_vars.P = 128
    global_vars.m = 1024*lag
    global_vars.L = 1
    global_vars.T = 1
    num_train = 1000
    num_test  = 1000
    casenum_train = 2
    casenum_test  = 2
    training_batch_size = 100
    global_vars.dx = 3
    global_vars.du = 3
    global_vars.dy = 3
    global_vars.ds = 3
    global_vars.n_hat  = 480
    global_vars.l  = 100
    global_vars.Nx = 32
    global_vars.Ny = 32
    global_vars.Nt = lag
    global_vars.Ng = 0
    global_vars.H = 2

    train_loader, valid_loader, test_loader = get_data_loaders(
        lag, global_vars.forward_steps, batch_size=training_batch_size, #64
    )
    
    q_layers = [global_vars.L*global_vars.dy+global_vars.H*global_vars.dy, global_vars.m, global_vars.l]
    v_layers = [global_vars.m*global_vars.du, global_vars.m, global_vars.ds*global_vars.n_hat]
    g_layers  = [global_vars.l, global_vars.m, global_vars.ds*global_vars.n_hat]

    print("DataGenerator defined")

    model = LOCA(q_layers, g_layers, v_layers, m=global_vars.m, P=global_vars.P, H=global_vars.H) 

    model.count_params(model.get_params(model.opt_state))

    if train_flag:
        start_time = timeit.default_timer()
        train(model, train_loader, valid_loader, global_vars, n_epochs=n_epochs, valid_freq = valid_freq)
        elapsed = timeit.default_timer() - start_time
        print("The training wall-clock time is seconds is equal to %f seconds"%elapsed)
    else:
        print(f'testing using save model loca_ep_{n_epochs}.pkl')
    out_file = CKPT_DIR / f"loca_ep_{n_epochs}.pkl"
    params  = model.get_params(model.opt_state)
    # Assuming you have some template_params already, perhaps loaded initially or defined for model creation
    params = load_params(out_file, params)

    ### testing 
    _, s_test = get_raw_data()

    batch_size = 32
    test_loader = DataLoader(TensorDataset(s_test), batch_size=batch_size)

    quad_fns = [
        quadrature.midpoint_vecs,
        quadrature.trapezoidal_vecs,
        quadrature.trapezoidal_vecs,
    ]
    quad_grid = quadrature.get_quad_grid(
        quad_fns, [lag, 32, 32], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]
    )
    cart_grid = quadrature.quad_to_cartesian_grid(quad_grid)
    cart_grid = quadrature.cart_grid_to_device(cart_grid, device)

    quad_grid_2d = quadrature.get_quad_grid(
        quadrature.trapezoidal_vecs, [32, 32], [-1.0, -1.0], [1.0, 1.0]
    )
    quad_grid_2d = quadrature.quad_grid_to_device(quad_grid_2d, device)

    metrics = {}
    err_df = {}
    for j, (y,) in enumerate(tqdm(test_loader)):
        y_pred = forecast_metrics(lag, global_vars, y, model, params, metrics)
        if j == 0 and plot_flag:
            plot_pred(y[0], y_pred[0], 10, "shallow-water-prediction-loca")
        # relative errors over time
        err_over_time(y, y_pred, err_df, quad_grid_2d)
    err_df = {k: torch.cat(v) for k, v in err_df.items()}
    err_df = pd.DataFrame(err_df)
    err_df.to_pickle(CURR_DIR /"err_time_dict_loca.pkl")
    print(metrics)


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
    # main(args.lag)
    main(args.lag, train_flag = False, plot_flag = True)
    
