import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from khatriraonop import models, quadrature


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

	
class Conv1DNetwork(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(Conv1DNetwork, self).__init__()

		self.conv1 = nn.Conv1d(in_channels, 30, kernel_size=9, stride=1, padding=4)  
		self.conv2 = nn.Conv1d(30, 20, kernel_size=9, stride=1, padding=4)  
		self.conv3 = nn.Conv1d(20, out_channels, kernel_size=9, stride=1, padding=4) 

		# self.conv1 = nn.Conv1d(in_channels, 20, kernel_size=9, stride=1, padding=4)  
		# self.conv2 = nn.Conv1d(20, 20, kernel_size=9, stride=1, padding=4)  
		# self.conv3 = nn.Conv1d(20, out_channels, kernel_size=9, stride=1, padding=4)

		# self.conv1 = nn.Conv1d(in_channels, 10, kernel_size=9, stride=1, padding=4)  
		# self.conv2 = nn.Conv1d(10, 10, kernel_size=9, stride=1, padding=4)  
		# self.conv3 = nn.Conv1d(10, out_channels, kernel_size=9, stride=1, padding=4) 

		# self.conv1 = nn.Conv1d(in_channels, 10, kernel_size=9, stride=1, padding=4)  
		# self.conv2 = nn.Conv1d(10, out_channels, kernel_size=9, stride=1, padding=4)

		# self.activation = nn.ReLU()
		# self.activation = nn.SiLU()
		self.activation = nn.LeakyReLU(negative_slope=0.01)
		# self.activation = nn.LeakyReLU(negative_slope=0.1)

	def forward(self, x):
		x = self.activation(self.conv1(x))
		x = self.activation(self.conv2(x))
		x = self.activation(self.conv3(x))
		return x

class KRNO(nn.Module):
	def __init__(self, hparams, output_time):
	
		super(KRNO, self).__init__()

		# encoder to handle 
		self.encoder = Conv1DNetwork(2, 1)

		# initialize model
		self.KRNO_model = models.KhatriRaoNO.easy_init(**hparams)  # type: ignore

		self.output_time = output_time
		

	def forward(self, X, mask = None):
		
		""" 
		X (B, L, N) 
		mask (B, L, N)

		To ====>

        X (B*N, L, 1)
        mask_X (B*N, L, 1)
        """



		##

		B, L_in, N = X.shape
		self.batch_size = B

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
		outputs = self.KRNO_model.super_resolution(self.quad_grid_out, self.quad_grid_in, X)


		return outputs # (B, Lp, N)

