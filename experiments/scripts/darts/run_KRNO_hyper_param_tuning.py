import argparse
import os
import torch
import random
import numpy as np
from datetime import datetime
import pandas as pd
from copy import deepcopy
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
# Set the start method for multiprocessing
multiprocessing.set_start_method('spawn', force=True)

CURR_DIR       = os.path.dirname(os.path.abspath(__file__))
root_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_base_path = os.path.dirname(root_base_path)
import sys
sys.path
sys.path.append(root_base_path)

from data_provider.data_factory import data_provider
from utils.print_args import print_args
from utils.tools import grid_iter, update_results_dict
from utils.tools import club_axes, plot_train_val_test_together, seed_everything
from scripts.darts.run_exp import run_exp
from scripts.darts.run_exp_weight_decay import run_exp as run_exp_weight_decay

def average_dictionaries(dict_lists):

    # Initialize a dictionary to store the averages
    average_dict = {}

    # Iterate over the keys of the first dictionary
    for key in ['nmse', 'nmae', 'mse', 'mae']:
        # Calculate the average for each key
        value_ = []
        for dict_i in dict_lists:
            value_.append(dict_i[key])
        average_dict[key] = np.mean(value_)

    return average_dict

def average_lists(list_of_lists):
    # Assuming all lists are of the same length and contain two elements each
    len_list = len(list_of_lists[0])
    if not all(len(lst) == len_list for lst in list_of_lists):
        raise ValueError(f"All lists must contain exactly {len_list} elements")
    
    # Compute the average for each corresponding element in the lists
    average_list = []
    for i in range(len_list):
        value_ = []
        for list_j in list_of_lists:
            value_.append(list_j[i])
        average_list.append(np.mean(value_))
    return average_list

def process_hyper_param(base_args, hyper_param_i=None, index=0, train_loss_optimal=None,
                         train_final_model = False):
    args = deepcopy(base_args)

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
            args.pred_len = int(args.seq_len/2)
        elif args.pred_len == 'full_len':
            args.pred_len = args.seq_len
        args.loss_pred_len = args.pred_len

    try:
        # best_model_metrics, val_train_loss_min = run_exp(args, index, train_loss_optimal=train_loss_optimal)
        # torch.cuda.empty_cache()
        # return [best_model_metrics, val_train_loss_min, args, index]
        if train_final_model:
            best_model_metrics, best_model_metrics_window, val_train_loss_min = run_exp(args, index,
                                                                         train_loss_optimal=train_loss_optimal)
            torch.cuda.empty_cache()
        elif args.weight_decay_tuning:
            best_model_metrics, best_model_metrics_window, val_train_loss_min = run_exp_weight_decay(args, index)
        else:
            val_train_loss_min_list = []
            train_results_best_dicts, val_results_best_dicts, test_results_best_dicts = [], [], []
            train_results_best_window_dicts, val_results_best_window_dicts, test_results_best_window_dicts = [], [], []
            for i_,val_train_ratio in enumerate(args.val_train_ratio_list):
                print(f'{i_}: val_train_ratio:{val_train_ratio}')
                args.val_train_ratio = val_train_ratio
                try:
                    best_model_metrics, best_model_metrics_window, val_train_loss_min = run_exp(args,
                                                                         index, train_loss_optimal=train_loss_optimal)
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(e)
                    continue
                train_results_best_dicts.append(best_model_metrics[0])
                val_results_best_dicts.append(best_model_metrics[1])
                test_results_best_dicts.append(best_model_metrics[2])
                train_results_best_window_dicts.append(best_model_metrics_window[0])
                val_results_best_window_dicts.append(best_model_metrics_window[1])
                test_results_best_window_dicts.append(best_model_metrics_window[2])
                val_train_loss_min_list.append(val_train_loss_min)
            if len(val_train_loss_min_list) != len(args.val_train_ratio_list):
                runs_failed = len(args.val_train_ratio_list) - len(val_train_loss_min_list)
                raise(f'failed cross validation run - {runs_failed}/{len(val_train_loss_min_list)}')
            train_results_best = average_dictionaries(train_results_best_dicts)
            val_results_best   = average_dictionaries(val_results_best_dicts)
            test_results_best  = average_dictionaries(test_results_best_dicts)
            best_model_metrics = [train_results_best, val_results_best, test_results_best]

            train_results_best_window = average_dictionaries(train_results_best_window_dicts)
            val_results_best_window   = average_dictionaries(val_results_best_window_dicts)
            test_results_best_window  = average_dictionaries(test_results_best_dicts)
            best_model_metrics_window = [train_results_best_window, val_results_best_window, test_results_best_window]

            val_train_loss_min = average_lists(val_train_loss_min_list)
        
        return [best_model_metrics, best_model_metrics_window, val_train_loss_min, args, index]
        
    except Exception as e:
        print(f"Error processing hyperparameter set {index}: {str(e)}")
        return [None, None, None, args, index]
        

def run_hyper_param_exps(args):
    
    exp_hyper_params_final_model        = dict(
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
    exp_hyper_params_best_model     = deepcopy(exp_hyper_params_final_model)
    output_metrics_best_model       = dict(val_loss=[], val_loss_norm=[],  train_loss=[], epoch=[],
                                     train_mse=[], val_mse=[], test_mse=[],
                                    train_mae=[], val_mae=[], test_mae=[],
                                   train_nmse=[], val_nmse=[], test_nmse=[],
                                   train_nmae=[], val_nmae=[], test_nmae=[])
    output_metrics_best_model_window       = dict(val_loss=[], val_loss_norm=[],  train_loss=[], epoch=[],
                                     train_mse=[], val_mse=[], test_mse=[],
                                    train_mae=[], val_mae=[], test_mae=[],
                                   train_nmse=[], val_nmse=[], test_nmse=[],
                                   train_nmae=[], val_nmae=[], test_nmae=[])
    

    if args.run_parallel:

        with ProcessPoolExecutor(max_workers=args.max_subprocesses) as executor:
            futures = [executor.submit(process_hyper_param, args, hyper_param_i, idx) for idx, hyper_param_i in enumerate(hypers)]

        # Collect results if needed
        results = [future.result() for future in futures]

        for result_i in results:
            try:
                best_model_metrics, best_model_metrics_window, val_train_loss_min, args, index = result_i

                try:
                    output_metrics_best_model = update_results_dict(*best_model_metrics,
                                                                output_metrics_best_model)
                    output_metrics_best_model['val_loss'].append(val_train_loss_min[0])
                    output_metrics_best_model['val_loss_norm'].append(val_train_loss_min[0]/args.pred_len)
                    output_metrics_best_model['train_loss'].append(val_train_loss_min[1])
                    output_metrics_best_model['epoch'].append(val_train_loss_min[2])

                    output_metrics_best_model_window = update_results_dict(*best_model_metrics_window,
                                                                output_metrics_best_model_window)
                    output_metrics_best_model_window['val_loss'].append(val_train_loss_min[0])
                    output_metrics_best_model_window['val_loss_norm'].append(val_train_loss_min[0]/args.pred_len)
                    output_metrics_best_model_window['train_loss'].append(val_train_loss_min[1])
                    output_metrics_best_model_window['epoch'].append(val_train_loss_min[2])
                    
                    exp_hyper_params_best_model['exp'].append(index)
                    exp_hyper_params_best_model['learning_rate'].append(args.learning_rate)
                    exp_hyper_params_best_model['seq_len'].append(args.seq_len)
                    exp_hyper_params_best_model['pred_len'].append(args.pred_len)
                    exp_hyper_params_best_model['width'].append(args.width)
                    exp_hyper_params_best_model['loss_pred_len'].append(args.loss_pred_len)
                    exp_hyper_params_best_model['hidden_units'].append(args.hidden_units)
                    exp_hyper_params_best_model['include_affine'].append(args.include_affine)
                    exp_hyper_params_best_model['affine_in_first_layer'].append(args.affine_in_first_layer)
                    exp_hyper_params_best_model['n_hidden_layers'].append(args.n_hidden_layers)
                    exp_hyper_params_best_model['n_integral_layers'].append(args.n_integral_layers)
                    exp_hyper_params_best_model['use_revin'].append(args.use_revin)
                except Exception as e:
                    print(f"Failed testing on best model, run - {index}")
                    print(e)
                
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Failed run")
                print(e)
                continue

    else:

        for ii, hyper_param_i in enumerate(hypers):

            try:
                [best_model_metrics, val_train_loss_min, args, index] = process_hyper_param(args, hyper_param_i, ii)
            except Exception as e:
                print(f"Failed run {ii}")
                print(e)
                continue

            try:
                output_metrics_best_model = update_results_dict(*best_model_metrics,
                                                            output_metrics_best_model)
                output_metrics_best_model['val_loss'].append(val_train_loss_min[0])
                output_metrics_best_model['val_loss_norm'].append(val_train_loss_min[0]/args.pred_len)
                output_metrics_best_model['train_loss'].append(val_train_loss_min[1])
                output_metrics_best_model['epoch'].append(val_train_loss_min[2])
                
                exp_hyper_params_best_model['exp'].append(ii)
                exp_hyper_params_best_model['learning_rate'].append(args.learning_rate)
                exp_hyper_params_best_model['seq_len'].append(args.seq_len)
                exp_hyper_params_best_model['pred_len'].append(args.pred_len)
                exp_hyper_params_best_model['width'].append(args.width)
                exp_hyper_params_best_model['loss_pred_len'].append(args.loss_pred_len)
                exp_hyper_params_best_model['hidden_units'].append(args.hidden_units)
                exp_hyper_params_best_model['include_affine'].append(args.include_affine)
                exp_hyper_params_best_model['affine_in_first_layer'].append(args.affine_in_first_layer)
                exp_hyper_params_best_model['n_hidden_layers'].append(args.n_hidden_layers)
                exp_hyper_params_best_model['n_integral_layers'].append(args.n_integral_layers)
                exp_hyper_params_best_model['use_revin'].append(args.use_revin)
            except Exception as e:
                print(f"Failed testing on final model, run - {ii}")
                print(e)

    ### 

    output_metrics_best_model = {**exp_hyper_params_best_model, **output_metrics_best_model}
    output_metrics = pd.DataFrame(output_metrics_best_model)
    output_metrics.to_csv( os.path.join(args.output_save_dir, 'best_model_metrics.csv'))

    sorted_output_metrics = output_metrics.sort_values(by='val_nmae', ascending=True, ignore_index=True) 
    csv_file_path  = os.path.join(args.output_save_dir, f'sorted_best_model_metrics.csv')
    sorted_output_metrics.to_csv(csv_file_path)

    # output_metrics_best_model_window = {**exp_hyper_params_best_model, **output_metrics_best_model_window}
    # output_metrics = pd.DataFrame(output_metrics_best_model_window)
    # output_metrics.to_csv( os.path.join(args.output_save_dir, 'best_model_metrics_window.csv'))

    # sorted_output_metrics_window = output_metrics.sort_values(by='val_nmae', ascending=True, ignore_index=True) 
    # csv_file_path  = os.path.join(args.output_save_dir, f'sorted_best_model_metrics_window.csv')
    # sorted_output_metrics_window.to_csv(csv_file_path)

    return args, sorted_output_metrics

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FNO')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')

    # data loader
    parser.add_argument('--root_path', type=str, default='dataset/darts/', help='root path of the data file')
    # parser.add_argument('--data', type=str, required=False, default='AirPassengers', help='dataset type')
    # parser.add_argument('--data_path', type=str, default='AirPassengers.csv', help='data file')
    # parser.add_argument('--data', type=str, required=False, default='AusBeer', help='dataset type')
    # parser.add_argument('--data_path', type=str, default='ausbeer.csv', help='data file')
    # parser.add_argument('--data', type=str, required=False, default='GasRateCO2', help='dataset type')
    # parser.add_argument('--data_path', type=str, default='gasrate_co2.csv', help='data file')
    # parser.add_argument('--data', type=str, required=False, default='HeartRate', help='dataset type')
    # parser.add_argument('--data_path', type=str, default='heart_rate.csv', help='data file')
    # parser.add_argument('--data', type=str, required=False, default='MonthlyMilk', help='dataset type')
    # parser.add_argument('--data_path', type=str, default='monthly-milk.csv', help='data file')
    # parser.add_argument('--data', type=str, required=False, default='sunspots', help='dataset type')
    # parser.add_argument('--data_path', type=str, default='monthly-sunspots.csv', help='data file')
    # parser.add_argument('--data', type=str, required=False, default='Wine', help='dataset type')
    # parser.add_argument('--data_path', type=str, default='wineind.csv', help='data file')
    parser.add_argument('--data', type=str, required=False, default='Wooly', help='dataset type')
    parser.add_argument('--data_path', type=str, default='woolyrnq.csv', help='data file')
    parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default=None, help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--shuffle_flag', type=bool, default=False, help='shuffle data')
    parser.add_argument('--val_train_ratio', type=float, default=0.33, help='ratio of val to train data sizes')
    
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=24, help='input sequence length')
    parser.add_argument('--step', type=int, default=1, help='step in a sequence')
    parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_false', help='inverse output data', default=True)
    parser.add_argument('--loss_pred_len', type=int, default=12, help='prediction sequence length used to compute loss')
    parser.add_argument('--use_revin', type=bool, default=True, help='flag to set instance normalization ReVIN')
    parser.add_argument('--revin_affine', type=bool, default=True, help='Affine flag in ReVIN')
    parser.add_argument('--detrend_flag', type=bool, default=False, help='Detrend the data using linear fit')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs') #100
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
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
    parser.add_argument('--max_subprocesses', type=int, default=4, help='gpu')

    # model selection
    parser.add_argument('--model', type=str, required=False, default='KRNO',
                        help='model name, options: [KRNO, FNO, FNO_residual, FNO_scale, FNO_without_spectconv]')
    
    # KRNO params
    parser.add_argument('--width', type=int, default=20, help='input/output channels in KRNO/FNO blocks')
    parser.add_argument('--lifting_channels', type=int, default=128, help='hidden layers in lifting/projection FC ')
    parser.add_argument('--hidden_units', type=int, default=128, help='hidden units in MLP of KRNO kernel ')
    parser.add_argument('--n_integral_layers', type=int, default=4, help='no of integral_layers ')
    parser.add_argument('--n_hidden_layers', type=int, default=3, help='no of integral_layers in MLP of KRNO kernel ')
    parser.add_argument('--include_affine', type=bool, default=True, help='use affine in KRNO') 
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
    args.root_path = os.path.join(CURR_DIR, args.root_path)

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

    # KRNO model hyper params 
    
    krno_hypers = dict(
                      learning_rate = [1e-3, 5e-3], #[1e-3, 5e-3],
                      include_affine = [True],
                      affine_in_first_layer=[False],
                      seq_len      = [10, 24, 48, 96],
                      pred_len     = [5, 'half_len', 'full_len'],
                      width        = [5, 10, 32],
                      hidden_units = [32, 64], # [32, 64]
                      n_hidden_layers = [3],
                      n_integral_layers = [3],
                      use_revin    = [True, False]
                      )
    
    # krno_hypers = dict(
    #                   learning_rate = [1e-3],
    #                   include_affine = [True],
    #                   affine_in_first_layer=[False],
    #                   seq_len      = [24],
    #                   pred_len     = [5],
    #                   width        = [32],
    #                   hidden_units = [32], # [32, 64]
    #                   n_hidden_layers = [3],
    #                   n_integral_layers = [4]
    #                   )

    args.val_train_ratio_list = [0.33]
    args.weight_decay_tuning  = False

    hypers = list(grid_iter(krno_hypers, shuffle=True))

    args.output_save_dir  = os.path.join(CURR_DIR, f'outputs/{args.model}/{args.data}')
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
    # args.weight_decay_list = [1e0, 5e-1, 1e-1, 7.5e-2, 5e-2, 2.5e-2, 1e-2, 5e-3]
    ## new weight decay grid
    args.weight_decay_list = [2e0, 1e0, 5e-1, 1e-1, 7.5e-2, 5e-2, 2.5e-2, 1e-2, 5e-3]
    args.val_train_ratio = 0.33 

    ## train final model using the best hyperparams on the train and validation data
    # args.val_train_ratio = 0.33 # 0.33
    # args.patience        = 30
    # best_hyperparam_idx = np.argmin(output_metrics_best_model['val_nmae'])
    # train_loss_optimal  = output_metrics_best_model['train_loss'][best_hyperparam_idx]
    # best_exp_idx        = output_metrics_best_model['exp'][best_hyperparam_idx]
    # import pdb;pdb.set_trace()

    train_loss_optimal    = sorted_output_metrics['train_loss'][0]
    best_exp_idx          = sorted_output_metrics['exp'][0]
    # best_hyperparams      = hypers[best_exp_idx]
    # args.train_epochs     = int(sorted_output_metrics['epoch'][0])
    for column_i in sorted_output_metrics.columns:
        if hasattr(args, column_i):
            setattr(args, column_i, sorted_output_metrics[column_i].values[0])

    if args.weight_decay_tuning:
        best_model_metrics, best_model_metrics_window, val_train_loss_min, _, _ = process_hyper_param(args, index ='final',
                                                                                    train_loss_optimal=train_loss_optimal)
    elif args.val_train_ratio==0:
        # best_model_metrics, val_train_loss_min, _, _ = process_hyper_param(args, best_exp_idx,
        #                                                                              train_loss_optimal=train_loss_optimal)
        best_model_metrics, best_model_metrics_window, val_train_loss_min, _, _ = process_hyper_param(args, index ='final',
                                                                                    train_loss_optimal=train_loss_optimal,
                                                                                    train_final_model=True)
    else:
        best_model_metrics, best_model_metrics_window, val_train_loss_min, _, _ = process_hyper_param(args, index ='final',
                                                                                    train_final_model=True)
    
    # print('KRNO - mse:{}, mae:{}'.format(best_model_metrics[2]['mse'], best_model_metrics[2]['mae']))
    # print('KRNO - nmse:{}, nmae:{}'.format(best_model_metrics[2]['nmse'], best_model_metrics[2]['nmae']))
    # f = open(os.path.join(args.output_save_dir, "final_test_results.txt"), 'a')
    # f.write(f'Best exp idx - {best_exp_idx}' + "  \n")
    # f.write('Using mean of future predictions while sliding window by one step\n')
    # f.write('\n')
    # f.write('KRNO metrics \n')
    # f.write('mse:{}, mae:{}'.format(best_model_metrics[2]['mse'], best_model_metrics[2]['mae']))
    # f.write('\n')
    # f.write('nmse:{}, nmae:{} \n'.format(best_model_metrics[2]['nmse'], best_model_metrics[2]['nmae']))
    # f.write('\n')
    # f.write('\n')
    # f.close()

    print('KRNO - mse:{}, mae:{}'.format(best_model_metrics_window[2]['mse'], best_model_metrics_window[2]['mae']))
    print('KRNO - nmse:{}, nmae:{}'.format(best_model_metrics_window[2]['nmse'], best_model_metrics_window[2]['nmae']))
    f = open(os.path.join(args.output_save_dir, "final_test_results.txt"), 'a')
    f.write(f'Best exp idx - {best_exp_idx}' + "  \n")
    f.write('Using the full predictions while sliding window by loss_pred_length\n')
    f.write('\n')
    f.write('KRNO metrics \n')
    f.write('mse:{}, mae:{}'.format(best_model_metrics_window[2]['mse'], best_model_metrics_window[2]['mae']))
    f.write('\n')
    f.write('nmse:{}, nmae:{} \n'.format(best_model_metrics_window[2]['nmse'], best_model_metrics_window[2]['nmae']))
    f.write('\n')
    f.write('\n')
    f.close()

