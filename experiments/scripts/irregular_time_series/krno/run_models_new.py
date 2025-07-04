import os
import sys
sys.path.append("..")
sys.path.append("./")

import time
import datetime
import argparse
import numpy as np
import pandas as pd
import random
from random import SystemRandom
from sklearn import model_selection

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import lib.utils as utils
from lib.parse_datasets import parse_datasets
from model.KRNO import *

parser = argparse.ArgumentParser('KRNO Forecasting')

parser.add_argument('--state', type=str, default='def')
parser.add_argument('-n',  type=int, default=int(1e8), help="Size of the dataset")
parser.add_argument('--epoch', type=int, default=100, help="training epoches")
parser.add_argument('--patience', type=int, default=10, help="patience for early stop")
# parser.add_argument('--dataset', type=str, default='physionet', help="Dataset to load. Available: physionet, mimic, ushcn")
# parser.add_argument('--history', type=int, default=24, help="number of hours (months for ushcn and ms for activity) as historical window")


parser.add_argument('--dataset', type=str, default='mimic', help="Dataset to load. Available: physionet, mimic, ushcn")
parser.add_argument('--history', type=int, default=24, help="number of hours (months for ushcn and ms for activity) as historical window")


# parser.add_argument('--dataset', type=str, default='activity', help="Dataset to load. Available: physionet, mimic, ushcn")
# parser.add_argument('--history', type=int, default=3000, help="number of hours (months for ushcn and ms for activity) as historical window")


# parser.add_argument('--dataset', type=str, default='ushcn', help="Dataset to load. Available: physionet, mimic, ushcn")
# parser.add_argument('--history', type=int, default=24, help="number of hours (months for ushcn and ms for activity) as historical window")


parser.add_argument('--logmode', type=str, default="a", help='File mode of logging.')

parser.add_argument('--lr',  type=float, default=1e-3, help="Starting learning rate.")
parser.add_argument('--w_decay', type=float, default=0.0, help="weight decay.")
parser.add_argument('-b', '--batch_size', type=int, default=1)
parser.add_argument('--accumulation_steps', type=int, default=16, help="accumulation steps")

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('--seed', type=int, default=5, help="Random seed")

# value 0 means using original time granularity, Value 1 means quantization by 1 hour, 
# value 0.1 means quantization by 0.1 hour = 6 min, value 0.016 means quantization by 0.016 hour = 1 min
parser.add_argument('--quantization', type=float, default=0.0, help="Quantization on the physionet dataset.")
parser.add_argument('--model', type=str, default='KRNO', help="Model name")
parser.add_argument('--int_channels', type=int, default=50, help="Number of channels in the integral layers")
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use.')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
file_name = os.path.basename(__file__)[:-3]
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.PID = os.getpid()
print("PID, device:", args.PID, args.device)

#####################################################################################################

if __name__ == '__main__':
	utils.setup_seed(args.seed)

	experimentID = args.load
	if experimentID is None:
		# Make a new experiment ID
		experimentID = int(SystemRandom().random()*100000)
	ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')
	
	input_command = sys.argv
	ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
	if len(ind) == 1:
		ind = ind[0]
		input_command = input_command[:ind] + input_command[(ind+2):]
	input_command = " ".join(input_command)

	# utils.makedirs("results/")

	##################################################################
	data_obj = parse_datasets(args)
	input_dim = data_obj["input_dim"]

	
	### Model setting ###
	args.ndim = input_dim
	print(f'Dimension of the data: {args.ndim}')
	# input_dim = 1  ## share KRNO params across channels
	hparams = dict(
        d=1,
        in_channels=input_dim,
        out_channels=input_dim,
        lifting_channels=128, #128,
        integral_channels=args.int_channels, #50 #20 #4 
        n_integral_layers=3, #3 try 1 for activity
        projection_channels=128, #128,
        n_hidden_units=32,
        n_hidden_layers=3,
        nonlinearity=nn.SiLU(),
        include_affine=True,
		)
	
	model = KRNO(hparams).to(args.device)
	print(model)

	print(hparams, flush=True)

	# Calculate the total number of parameters
	total_params = sum(p.numel() for p in model.parameters())
	# Calculate the total number of trainable parameters
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

	# Print the results
	print(f"Total number of parameters: {total_params}", flush=True)
	print(f"Total number of trainable parameters: {trainable_params}", flush=True)

	##################################################################
	
	# # Load checkpoint and evaluate the model
	# if args.load is not None:
	# 	utils.get_ckpt_model(ckpt_path, model, args.device)
	# 	exit()

	##################################################################

	if(args.n < 12000):
		args.state = "debug"
		log_path = "logs/{}_{}_{}.log".format(args.dataset, args.model, args.state)
	else:
		log_path = "logs/{}_{}_{}_{}lr.log". \
			format(args.dataset, args.model, args.state, args.lr)
	
	if not os.path.exists("logs/"):
		utils.makedirs("logs/")
	logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode)
	logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
	logger.info(input_command)
	logger.info(args)

	# optimizer = optim.Adam(model.parameters(), lr=args.lr)
	# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
	optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
	# optimizer = optim.AdamW(model.parameters(), lr=args.lr)
	print(optimizer, flush=True)

	scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3, verbose=True)

	num_batches = data_obj["n_train_batches"] # n_sample / batch_size
	print("n_train_batches:", num_batches)

	best_val_mse = np.inf
	test_res = None
	accumulation_steps = args.accumulation_steps  # Simulating batch size of 8
	optimizer.zero_grad()
	for itr in range(args.epoch):
		st = time.time()

		### Training ###
		model.train()
		train_loss = []
		for i in range(num_batches):
			# optimizer.zero_grad()
			batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
			if batch_dict['tp_to_predict'].shape[-1] <2 or batch_dict['observed_tp'].shape[-1] <2 :
				# print('skipping this batch as len of tp_to_predict or observed_tp is < 2! ')
				continue
			train_res, _ = compute_all_losses(model, batch_dict)
			loss = train_res["loss"] / accumulation_steps
			loss.backward()
			train_loss.append(train_res["loss"].item())
			if (i + 1) % accumulation_steps == 0:
				optimizer.step()
				optimizer.zero_grad()
			# optimizer.step()
			# logger.info(" batch: {:d} Train - Loss (one batch): {:.5f}".format(i, loss.item()))
		if (i + 1) % accumulation_steps != 0:
			optimizer.step()
			optimizer.zero_grad()

		### Validation ###
		model.eval()
		with torch.no_grad():
			val_res = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"])
			
			### Testing ###
			if(val_res["mse"] < best_val_mse):
				best_val_mse = val_res["mse"]
				best_iter = itr
				test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"])
			
			logger.info('- Epoch {:03d}, ExpID {}'.format(itr, experimentID))
			logger.info("Train - Loss: {:.5f}".format(np.mean(train_loss)))
			logger.info("Val - Loss, MSE, RMSE, MAE, MAPE: {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%" \
				.format(val_res["loss"], val_res["mse"], val_res["rmse"], val_res["mae"], val_res["mape"]*100))
			if(test_res != None):
				logger.info("Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%" \
					.format(best_iter, test_res["loss"], test_res["mse"],\
			 		 test_res["rmse"], test_res["mae"], test_res["mape"]*100))
			logger.info("Time spent: {:.2f}s".format(time.time()-st))

		scheduler.step(val_res["loss"])

		if(itr - best_iter >= args.patience):
			print("Exp has been early stopped!")
			sys.exit(0)


