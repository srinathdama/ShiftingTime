import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from khatriraonop import models, quadrature

import lib.utils as utils
from lib.evaluation import *

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


	
class Conv1DNetwork(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(Conv1DNetwork, self).__init__()

		self.conv1 = nn.Conv1d(in_channels, 30, kernel_size=9, stride=1, padding=4)  
		self.conv2 = nn.Conv1d(30, 20, kernel_size=9, stride=1, padding=4)  
		self.conv3 = nn.Conv1d(20, out_channels, kernel_size=9, stride=1, padding=4) 

		self.activation = nn.LeakyReLU(negative_slope=0.01)
		# self.activation = nn.LeakyReLU(negative_slope=0.1)

	def forward(self, x):
		x = self.activation(self.conv1(x))
		x = self.activation(self.conv2(x))
		x = self.activation(self.conv3(x))
		return x


class KRNO(nn.Module):
	def __init__(self, hparams, supports = None, dropout = 0):
	
		super(KRNO, self).__init__()

		# encoder to handle 
		self.encoder = Conv1DNetwork(2, 1)

		# initialize model
		self.KRNO_model = models.KhatriRaoNO.easy_init(**hparams)  # type: ignore
		

	def forecasting(self, time_steps_to_predict, X, truth_time_steps, mask = None):
		
		""" 
		# time_steps_to_predict (B, L) [0, 1]
		# truth_time_steps (B, L) [0, 1]
		time_steps_to_predict (L) [0]
		truth_time_steps (L) [0]
		X (B, L, N) 
		mask (B, L, N)

		To ====>

        X (B*N, L, 1)
        mask_X (B*N, L, 1)
        """

		time_steps_to_predict  = time_steps_to_predict.squeeze()
		truth_time_steps       =  truth_time_steps.squeeze()

		## get quad grids
		
		quad_grid_in = quadrature.trapezoidal_vecs_uneven(truth_time_steps)
		quad_grid_out = quadrature.trapezoidal_vecs_uneven(time_steps_to_predict)

		quad_grid_in = ([quad_grid_in[0]], [quad_grid_in[1]])
		quad_grid_out = ([quad_grid_out[0]], [quad_grid_out[1]])

		quad_grid_in = quadrature.quad_grid_to_device(quad_grid_in, X.device)
		quad_grid_out = quadrature.quad_grid_to_device(quad_grid_out, X.device)

		##

		B, L_in, N = X.shape
		self.batch_size = B

		assert B ==1, 'Current KRNO code supports only batch size 1!'

		## use a 1D conv for encoding each scalar val, mask image [B, L, N, 2] -> [B, N, L, 1]
		X = X.permute(0, 2, 1).unsqueeze(-1) # (B, N, L, 1)
		mask = mask.permute(0, 2, 1).unsqueeze(-1)  # (B, N, L, 1)
		# truth_time_steps = truth_time_steps.repeat(B, N, 1).unsqueeze(-1)
		X = torch.cat([X, mask], dim=-1)  # (B, N, L, 2)
		X = X.reshape(-1, L_in, 2).transpose(1,2) # (B*N, 2, L)

		# ### *** use a 1D conv encoder to extract features
		X = self.encoder(X) # (B*N, 1, L)
		X = X.reshape(B, N, L_in).permute(0,2,1) # (B, L, N)
		## pass the conv features to KRNO
		outputs = self.KRNO_model.super_resolution(quad_grid_out, quad_grid_in, X)
		
		return outputs # (B, Lp, N)

