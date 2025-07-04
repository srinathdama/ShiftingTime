import argparse
import os
import torch
import random
import numpy as np
from datetime import datetime
import pandas as pd
import copy
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
# Set the start method for multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import time

root_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_base_path = os.path.dirname(root_base_path)
import sys
sys.path
sys.path.append(root_base_path)

from data_provider.data_factory import data_provider
from utils.print_args import print_args
from utils.tools import grid_iter, update_results_dict
from utils.tools import club_axes, plot_train_val_test_together, seed_everything, plot_idx
from utils.KNF_eval_metrics import RMSE
from utils.KNF_eval_metrics import SMAPE
from utils.KNF_eval_metrics import WRMSE
from utils.metrics import process_testing_data
from scripts.m4_crypto_traj.run_exp import run_exp
from scripts.m4_crypto_traj.run_exp_weight_decay_tuning import run_exp as run_exp_weight_decay

metric_dict = {
        "mini": SMAPE,
        "m4": SMAPE,
        "Cryptos": WRMSE,
        "Traj": RMSE
    }

def update_results_dicts(final_model_metrics, best_model_metrics, val_loss_min,
                         output_metrics_final_model, output_metrics_best_model,
                         exp_hyper_params, ii, args):
    output_metrics_final_model['short'].append(final_model_metrics[0])
    output_metrics_final_model['medium'].append(final_model_metrics[1])
    output_metrics_final_model['long'].append(final_model_metrics[2])
    output_metrics_final_model['total'].append(final_model_metrics[3])

    print('results from final model \n')
    print(final_model_metrics)

    output_metrics_best_model['val_loss'].append(val_loss_min)
    output_metrics_best_model['val_loss_norm'].append(val_loss_min/args.pred_len)
    output_metrics_best_model['short'].append(best_model_metrics[0])
    output_metrics_best_model['medium'].append(best_model_metrics[1])
    output_metrics_best_model['long'].append(best_model_metrics[2])
    output_metrics_best_model['total'].append(best_model_metrics[3])

    print('results from best model \n')
    print(best_model_metrics)

    ## 
    exp_hyper_params['exp'].append(ii)
    exp_hyper_params['learning_rate'].append(args.learning_rate)
    exp_hyper_params['seq_len'].append(args.seq_len)
    exp_hyper_params['pred_len'].append(args.pred_len)
    exp_hyper_params['width'].append(args.width)
    exp_hyper_params['loss_pred_len'].append(args.loss_pred_len)
    exp_hyper_params['hidden_units'].append(args.hidden_units)
    exp_hyper_params['include_affine'].append(args.include_affine)
    exp_hyper_params['affine_in_first_layer'].append(args.affine_in_first_layer)
    exp_hyper_params['n_hidden_layers'].append(args.n_hidden_layers)
    exp_hyper_params['n_integral_layers'].append(args.n_integral_layers)
    exp_hyper_params['use_revin'].append(args.use_revin)

    return [output_metrics_final_model, output_metrics_best_model, exp_hyper_params]

def process_hyper_param(base_args, hyper_param_i=None, index=0, train_loss_optimal=None,
                         train_final_model = False):
    args = copy.deepcopy(base_args)

    if hyper_param_i is not None:
        args.learning_rate = hyper_param_i['learning_rate']
        args.seq_len       = hyper_param_i['seq_len']
        args.pred_len      = hyper_param_i['pred_len']
        args.width         = hyper_param_i['width']
        args.hidden_units  = hyper_param_i['hidden_units']
        args.n_hidden_layers  = hyper_param_i['n_hidden_layers']
        args.n_integral_layers  = hyper_param_i['n_integral_layers']
        args.include_affine = hyper_param_i['include_affine']
        args.use_revin      = hyper_param_i['use_revin']
        if args.include_affine:
            args.affine_in_first_layer = hyper_param_i['affine_in_first_layer']
        else:
            args.affine_in_first_layer = False

        if args.pred_len == 'half_len':
            args.pred_len = int(args.test_output_length/2)
        elif args.pred_len == 'full_len':
            args.pred_len = args.test_output_length
        args.loss_pred_len = args.pred_len

    try:
        if train_final_model:
            final_model_metrics, best_model_metrics, val_loss_min = run_exp(args, index,
                                                                            train_loss_optimal=train_loss_optimal)
        elif args.weight_decay_tuning:
            final_model_metrics, best_model_metrics, val_loss_min = run_exp_weight_decay(args, index)
    
        else:
            final_model_metrics, best_model_metrics, val_loss_min = run_exp(args, index,
                                                                            train_loss_optimal=train_loss_optimal)
        torch.cuda.empty_cache()
        return [final_model_metrics, best_model_metrics, val_loss_min, args, index]
        
    except Exception as e:
        print(f"Error processing hyperparameter set {index}: {str(e)}")
        return [None, None, None, args, index]
        
def run_hyper_param_exps(args):
    exp_hyper_params    = dict(
                            exp  = [],
                            learning_rate=[],
                            seq_len = [],
                            pred_len = [],
                            loss_pred_len=[],
                            width   = [],
                            hidden_units = [],
                            include_affine = [],
                            affine_in_first_layer=[],
                            n_hidden_layers=[],
                            n_integral_layers=[],
                            use_revin=[]
                            )
    output_metrics_final_model      = dict(short=[], medium=[], long=[],
                                    total=[])
    output_metrics_best_model       = dict(val_loss=[], val_loss_norm=[], short=[], medium=[], long=[],
                                    total=[])
    
    if args.run_parallel:

        with ProcessPoolExecutor(max_workers=args.max_subprocesses) as executor:
            futures = [executor.submit(process_hyper_param, args, hyper_param_i, idx) for idx, hyper_param_i in enumerate(hypers)]

        # Collect results if needed
        results = [future.result() for future in futures]

        for result_i in results:
            try:
                final_model_metrics, best_model_metrics, val_loss_min, args, index = result_i
                results_out  = update_results_dicts(final_model_metrics, best_model_metrics, val_loss_min,
                                        output_metrics_final_model, output_metrics_best_model,
                                        exp_hyper_params, index, args)
                output_metrics_final_model = results_out[0]
                output_metrics_best_model  = results_out[1]
                exp_hyper_params           = results_out[2]
            except Exception as e:
                print(f"Failed run")
                print(e)
                continue

    else:

        for ii, hyper_param_i in enumerate(hypers):

            try:

                final_model_metrics, best_model_metrics, val_loss_min, args, index = process_hyper_param(args, hyper_param_i, ii)
                results_out = update_results_dicts(final_model_metrics, best_model_metrics, val_loss_min,
                                                   output_metrics_final_model,output_metrics_best_model,
                                                    exp_hyper_params, index, args)
                output_metrics_final_model = results_out[0]
                output_metrics_best_model  = results_out[1]
                exp_hyper_params           = results_out[2]
                torch.cuda.empty_cache()

                ## saving after each run as there is some memory leak with Crypto exp
                output_metrics_final_model = {**exp_hyper_params, **output_metrics_final_model}
                output_metrics = pd.DataFrame(output_metrics_final_model)
                output_metrics.to_csv( os.path.join(args.output_save_dir, 'final_model_metrics.csv'))

                output_metrics_best_model = {**exp_hyper_params, **output_metrics_best_model}
                output_metrics = pd.DataFrame(output_metrics_best_model)
                output_metrics.to_csv( os.path.join(args.output_save_dir, 'best_model_metrics.csv'))

                sorted_output_metrics = output_metrics.sort_values(by='val_loss_norm', ascending=True, ignore_index=True) 
                csv_file_path  = os.path.join(args.output_save_dir, f'sorted_best_model_metrics.csv')
                sorted_output_metrics.to_csv(csv_file_path)

            except Exception as e:
                print(f"Failed run {ii}")
                print(e)
                continue

    ### 
    output_metrics_final_model = {**exp_hyper_params, **output_metrics_final_model}
    output_metrics = pd.DataFrame(output_metrics_final_model)
    output_metrics.to_csv( os.path.join(args.output_save_dir, 'final_model_metrics.csv'))

    output_metrics_best_model = {**exp_hyper_params, **output_metrics_best_model}
    output_metrics = pd.DataFrame(output_metrics_best_model)
    output_metrics.to_csv( os.path.join(args.output_save_dir, 'best_model_metrics.csv'))

    sorted_output_metrics = output_metrics.sort_values(by='val_loss_norm', ascending=True, ignore_index=True) 
    csv_file_path  = os.path.join(args.output_save_dir, f'sorted_best_model_metrics.csv')
    sorted_output_metrics.to_csv(csv_file_path)

    return args, sorted_output_metrics

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FNO')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')

    # data loader
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--data', type=str, required=False, default='m4', help='dataset type')
    parser.add_argument('--root_path', type=str, default='dataset/M4_KNF/', help='root path of the data file')
    # parser.add_argument('--data', type=str, required=False, default='Cryptos', help='dataset type')
    # parser.add_argument('--root_path', type=str, default='dataset/Cryptos/', help='root path of the data file')
    # parser.add_argument('--data', type=str, required=False, default='Traj', help='dataset type')
    # parser.add_argument('--root_path', type=str, default='dataset/PlayerTraj/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='AirPassengers.csv', help='data file')
    parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='#Passengers', help='target feature in S or MS task')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    # parser.add_argument('--shuffle_flag', type=bool, default=False, help='shuffle data')
    parser.add_argument('--val_train_ratio', type=float, default=0.1, help='ratio of val to train data sizes')
    parser.add_argument('--jumps', type=int, default=1, help='input sequence length')
    
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=45, help='input sequence length')
    parser.add_argument('--step', type=int, default=1, help='step in a sequence')
    parser.add_argument('--pred_len', type=int, default=45, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Weekly',
                        help='data freq, options:[Hourly, Daily, Weekly, Monthly, Quarterly, Yearly]')
    parser.add_argument('--inverse', action='store_false', help='inverse output data', default=True)
    parser.add_argument('--loss_pred_len', type=int, default=45, help='prediction sequence length used to compute loss')
    parser.add_argument('--use_revin', type=bool, default=True, help='flag to set ReVIN')
    parser.add_argument('--revin_affine', type=bool, default=True, help='Affine flag in ReVIN')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs') #100
    parser.add_argument('--batch_size', type=int, default=512, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    #slightly better results came with lr 0.0001, epochs = 500  
    parser.add_argument('--learning_rate', type=float, default=0.005, help='optimizer learning rate')
    parser.add_argument('--optim', type=str, default='AdamW', help='optimizer [AdamW, Adam]')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='optimizer weight decay')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--run_parallel', type=bool, default=True, help='use gpu')
    parser.add_argument('--max_subprocesses', type=int, default=3, help='gpu')

    # model selection
    parser.add_argument('--model', type=str, required=False, default='KRNO',
                        help='model name, options: [KRNO, FNO, FNO_residual, FNO_scale, FNO_without_spectconv]')
    
    # KRNO params
    parser.add_argument('--width', type=int, default=20, help='input/output channels in KRNO/FNO blocks')
    parser.add_argument('--lifting_channels', type=int, default=128, help='hidden layers in lifting/projection FC ')
    parser.add_argument('--hidden_units', type=int, default=32, help='hidden units in MLP of KRNO kernel ')
    parser.add_argument('--include_affine', type=bool, default=True, help='use affine in KRNO') 
    parser.add_argument('--n_integral_layers', type=int, default=3, help='no of integral_layers ')
    parser.add_argument('--n_hidden_layers', type=int, default=3, help='no of integral_layers in MLP of KRNO kernel ')
    parser.add_argument('--affine_in_first_layer', type=bool, default=False,
                         help='when seq_len and pred_len are same, this flag will turn on affine mapping in first layer')

    # FNO params
    parser.add_argument('--modes', type=int, default=6, help='Fourier modes')

    # cluster params
    parser.add_argument('--tmp_storage', type=bool, default=False,
                             help='if True, checkpoints are stored and moved from the host tmp folder')
    parser.add_argument('--dest_dir_path', type=str, default='/tmp/TNO/code', help='dest_dir_path')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.root_path = os.path.join(root_base_path, args.root_path)


    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.model == 'KRNO':
        args.learning_rate = 1e-3
        args.weight_decay  = 1e-4

    current_time = datetime.now()
    args.date_time    = current_time.strftime("%d_%m_%Y-%H_%M_%S")

    print('Args in experiment:')
    # print_args(args)

    ## hyper-params
    # fno_hypers = dict(learning_rate=[5e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2],
    #                   seq_len      = [10, 24, 48, 96, 192, 400],
    #                   width        = [10, 20, 32, 48],
    #                   )

    # fno_hypers = dict(learning_rate=[1e-4, 1e-3, 5e-3],
    #                   seq_len      = ['min', 'twice'],
    #                   loss_pred_len = [1, 5, 'half_len', 'full_len'],
    #                   width        = [5, 10, 20, 32, 48, 96],
    #                   )

    
    if args.data=='m4':
        args.num_channels = 1 #input channels
        if args.seasonal_patterns == 'Daily':
            args.test_output_length = 14
            seq_len_list = [18, 30, 45]  #max input length should not be more than 93
            pred_len_list = [6, 10, 14]
            args.jumps   = 5
        elif args.seasonal_patterns == 'Weekly':
            args.test_output_length = 13
            seq_len_list = [30, 45, 60] #max input length should not be more than 80
            pred_len_list = [6, 10, 13]
            # n_hidden_layers_list   = [1,2,3]
            # n_integral_layers_list = [1,2,3,4]
            args.jumps   = 3
        elif args.seasonal_patterns == 'Monthly':
            args.test_output_length = 18
            seq_len_list = [18, 40] #[18, 30, 40] #max input length should not be more than 42
            pred_len_list = [6, 12, 18]
            args.jumps   = 3 # 10 
            args.val_train_ratio = 0.05
        elif args.seasonal_patterns == 'Quarterly':
            args.test_output_length = 8
            seq_len_list = [8, 12, 14] #max input length should not be more than 16
            pred_len_list = [4, 8]
            args.jumps   = 2 # 4
            args.val_train_ratio = 0.05
        elif args.seasonal_patterns == 'Yearly':
            args.test_output_length = 6
            seq_len_list = [3, 6] #max input length should not be more than 13
            pred_len_list = [1, 2]
            args.jumps   = 1 # 2
            args.val_train_ratio = 0.05
        elif args.seasonal_patterns == 'Hourly':
            args.test_output_length = 48
            seq_len_list = [48, 96, 192] #max input length should not be more than 700
            pred_len_list = [5, 10]
            args.jumps   = 1
        else:
            raise ValueError("data freq, options:[Hourly, Daily, Weekly, Monthly, Quarterly, Yearly]")
    elif args.data=='Cryptos':
        args.num_channels = 8
        args.test_output_length = 15
        seq_len_list = [63, 80]  #[48, 63, 80]
        pred_len_list = [5, 10, 15]
        args.jumps   = 100
        args.seasonal_patterns = 'NA'
        args.run_parallel = False
    elif args.data=='Traj':
        args.num_channels = 2
        args.test_output_length = 30
        seq_len_list = [10, 21, 30] #[10, 21, 30]
        pred_len_list = [15, 30]
        args.jumps   = 2
        args.seasonal_patterns = 'NA'
    else:
        raise('args.data should be either m4/Cryptos/Traj!') 
        
    # fno_hypers = dict(learning_rate=[1e-3],
    #                   include_affine = [True],
    #                   affine_in_first_layer=[False],
    #                   seq_len      = seq_len_list,
    #                   pred_len     = pred_len_list, # [1, 5, 'half_len', 'full_len']
    #                   width        = [8, 16, 32, 48],
    #                   hidden_units = [32, 64]
    #                   )

    fno_hypers = dict(learning_rate=[1e-3],
                      include_affine = [True],
                      affine_in_first_layer=[False],
                      seq_len      = seq_len_list,
                      pred_len     = pred_len_list, 
                      width        = [16, 32],  #[8, 16, 32, 48],
                      hidden_units = [32, 64], # [32, 64],
                      n_hidden_layers = [3],
                      n_integral_layers = [3],
                      use_revin    = [True, False]
                      )
    
    # fno_hypers = dict(learning_rate=[1e-3],
    #                   include_affine = [True],
    #                   affine_in_first_layer=[False],
    #                   seq_len      = seq_len_list,
    #                   pred_len     = [1], # [1, 5, 'half_len', 'full_len']
    #                   width        = [8],
    #                   hidden_units = [32],
    #                   n_hidden_layers = [3],
    #                   n_integral_layers = [3],
    #                   use_revin    = [True]
    #                   )

    args.weight_decay_tuning  = False
    hypers = list(grid_iter(fno_hypers, shuffle=True))

    args.output_save_dir  = os.path.join(root_base_path, f'outputs/{args.data}_{args.seasonal_patterns}')
    os.makedirs(args.output_save_dir, exist_ok=True)

    csv_file_path  = os.path.join(args.output_save_dir, f'sorted_best_model_metrics.csv')

    if os.path.exists(csv_file_path):
        ## load the existing sorted hyperparams csv file
        sorted_output_metrics = pd.read_csv(csv_file_path)
    else:
        os.makedirs(args.output_save_dir, exist_ok=True)
        args, sorted_output_metrics = run_hyper_param_exps(args)

    ###
    args.weight_decay_tuning = True
    ## initial weight decay grid
    args.weight_decay_list = [1e0, 5e-1, 1e-1, 7.5e-2, 5e-2, 2.5e-2, 1e-2, 5e-3]
    ## new weight decay grid
    # args.weight_decay_list = [2e0, 1e0, 5e-1, 1e-1, 7.5e-2, 5e-2, 2.5e-2, 1e-2, 5e-3]

    best_exp_idx          = sorted_output_metrics['exp'][0]
    # best_hyperparams      = hypers[best_exp_idx]
    # args.train_epochs     = int(sorted_output_metrics['epoch'][0])
    for column_i in sorted_output_metrics.columns:
        if hasattr(args, column_i):
            setattr(args, column_i, sorted_output_metrics[column_i].values[0])
        
    if args.weight_decay_tuning:
        final_model_metrics, best_model_metrics, val_loss_min, _, _ = process_hyper_param(args, index ='final')
    elif args.val_train_ratio==0:
        # best_model_metrics, val_train_loss_min, _, _ = process_hyper_param(args, best_exp_idx,
        #                                                                              train_loss_optimal=train_loss_optimal)
        final_model_metrics, best_model_metrics, val_loss_min, _, _ = process_hyper_param(args, index ='final',
                                                                                    train_loss_optimal=train_loss_optimal,
                                                                                    train_final_model=True)
    else:
        final_model_metrics, best_model_metrics, val_loss_min, _, _ = process_hyper_param(args, index ='final',
                                                                                    train_final_model=True)
    
    f = open(os.path.join(args.output_save_dir, "result_long_term_forecast.txt"), 'a')
    f.write("Test results  \n")
    f.write("Final model  \n")
    f.write('SMAPE:{}'.format(final_model_metrics))
    f.write('\n')
    f.write("Best model  \n")
    f.write('SMAPE:{}'.format(best_model_metrics))
    f.write('\n')
    f.close()

