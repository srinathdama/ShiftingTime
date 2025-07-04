import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.KNF_normalizer import RevIN
from khatriraonop import models, quadrature


class Model(nn.Module):
    """
    FNO 
    Paper link: 
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        if configs.task_name not in ['long_term_forecast', 'short_term_forecast']:
            raise("task should be either 'long_term_forecast' or 'short_term_forecast'!")
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len  = configs.seq_len
        
        if hasattr(configs, 'use_revin'):
            if configs.use_revin:
                self.normalizer = RevIN(num_features=configs.num_channels,
                                     affine= configs.revin_affine, axis=(1))
                self.use_revin = True
            else:
                self.use_revin = False
        else:
            self.use_revin = False

        ### 
        self._KRNO1d = models.KhatriRaoNO.easy_init(d = 1,
                        in_channels       = configs.num_channels,
                        out_channels      = configs.num_channels,
                        lifting_channels  = configs.lifting_channels,
                        integral_channels = configs.width,
                        n_integral_layers = configs.n_integral_layers , #4, [1,2,3]
                        projection_channels = configs.lifting_channels,
                        n_hidden_units    = configs.hidden_units,
                        n_hidden_layers   = configs.n_hidden_layers, #3, [1,2,3]
                        nonlinearity      = nn.SiLU(),
                        include_affine=configs.include_affine,
                        affine_in_first_integral_tsfm=configs.affine_in_first_layer)
        self.quadrature = quadrature


    def loss_fn(self, y_pred, y, quad_rule):
        return quadrature.integrate_on_domain((y - y_pred).pow(2), quad_rule).mean()


    def forecast(self, x, cart_grid):
        
        # Embedding
        out = self._KRNO1d(cart_grid, x)

        return out


    def forward(self, x, cart_grid):

        if self.use_revin:
            x = self.normalizer.forward(x, mode="norm")
        
        if self.seq_len == self.pred_len:
            out = self._KRNO1d(cart_grid, x)
        else:
            out = self._KRNO1d.super_resolution(cart_grid[0], cart_grid[1],  x)
        
        if self.use_revin:
            out = self.normalizer.forward(out, mode="denorm")

        return out  # [B, L, D]
