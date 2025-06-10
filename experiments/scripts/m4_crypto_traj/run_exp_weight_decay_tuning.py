import argparse
import os
import torch
import random
import numpy as np
from datetime import datetime
import json

root_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_base_path = os.path.dirname(root_base_path)
import sys
sys.path
sys.path.append(root_base_path)

from exp.exp_long_term_forecasting_knf import Exp_Long_Term_Forecast
from data_provider.data_factory import data_provider
from utils.print_args import print_args
from utils.tools import club_axes, plot_train_val_test_together, seed_everything, plot_idx, NumpyEncoder
from utils.KNF_eval_metrics import RMSE
from utils.KNF_eval_metrics import SMAPE
from utils.KNF_eval_metrics import WRMSE
from utils.metrics import process_testing_data

metric_dict = {
        "mini": SMAPE,
        "m4": SMAPE,
        "Cryptos": WRMSE,
        "Traj": RMSE
    }

class Global_args:
    def __init__(self):
        pass 

    def update_attributes(self, data):
        for key, value in data.items():
            setattr(self, key, value)


def run_exp(args, run_i=0, train_loss_optimal=None):
    fix_seed = args.seed #2021
    seed_everything(seed=fix_seed)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    else:
        Exp = Exp_Long_Term_Forecast

    if args.data=='m4':
        args.num_channels = 1 #FNO channels
        if args.seasonal_patterns == 'Hourly':
            args.test_output_length = 48
        elif args.seasonal_patterns == 'Daily':
            args.test_output_length = 14
        elif args.seasonal_patterns == 'Weekly':
            args.test_output_length = 13
        elif args.seasonal_patterns == 'Monthly':
            args.test_output_length = 18
            args.val_train_ratio = 0.05
        elif args.seasonal_patterns == 'Quarterly':
            args.test_output_length = 8
            args.val_train_ratio = 0.05
        elif args.seasonal_patterns == 'Yearly':
            args.test_output_length = 6
            args.val_train_ratio = 0.05
        else:
            raise ValueError("data freq, options:[Hourly, Daily, Weekly, Monthly, Quarterly, Yearly]")
    elif args.data=='Cryptos':
        args.num_channels = 8
        args.test_output_length = 15
    elif args.data=='Traj':
        args.num_channels = 2
        args.test_output_length = 30
    else:
        raise('args.data should be either m4/Cryptos/Traj!')


    eval_metric = metric_dict[args.data]

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            
            setting = '{}_{}_{}_{}_{}_ft{}_sl{}_step{}_pl{}_lpl_{}_dt{}_FNO_nc{}_w{}_m{}_nlc_{}_time_{}_exp_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.seasonal_patterns,
                args.features,
                args.seq_len,
                args.step,
                args.pred_len,
                args.loss_pred_len,
                args.des,
                args.num_channels,
                args.width,
                args.modes,
                args.lifting_channels,
                args.date_time, run_i)
            
            ### store the args
            # root_path       = os.path.split(os.path.abspath(__file__))[0]
            checkpoint_dir  = os.path.join(root_base_path, f'checkpoints/{setting}')
            os.makedirs(checkpoint_dir, exist_ok=True)

            with open(os.path.join(checkpoint_dir, f'exp_parameters.json'), 'w') as fp:
                json.dump(args.__dict__, fp, indent ="", cls=NumpyEncoder)

            train_set, train_loader = exp._get_data(flag='train')
            if train_loss_optimal is None:
                vali_set, vali_loader = exp._get_data(flag='val')
                if len(vali_loader)==0:
                    raise ('len of validation set is 0!')
            else:
                vali_loader = None
            test_set, test_loader = exp._get_data(flag='test')


            vali_loss_prev = -np.Inf
            counter        = 0
            for j, weight_decay in enumerate(args.weight_decay_list):
                print(f'>>>>>>> {j}: current weight decay: {weight_decay}>>>>>>>')
                args.weight_decay = weight_decay
                # setting record of experiments
                exp = Exp(args)  # set experiments
                if j > 0:
                    print('loading model')
                    checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pth'))
                    exp.model.load_state_dict(checkpoint['model_state_dict'])

                    optimizer_state_dict = checkpoint['optimizer_state_dict']
                    for param_group in optimizer_state_dict['param_groups']:
                        param_group['weight_decay'] = args.weight_decay
                    exp.model_optim.load_state_dict(optimizer_state_dict)

                    train_loss_lrun = checkpoint['train_loss']
                    val_loss_lrun   = checkpoint['val_loss']
                else:
                    train_loss_lrun = np.Inf
                    val_loss_lrun   = np.Inf

                # Calculate the total number of parameters
                total_params = sum(p.numel() for p in exp.model.parameters())

                # Calculate the total number of trainable parameters
                trainable_params = sum(p.numel() for p in exp.model.parameters() if p.requires_grad)
                
                # Print the results
                print(f"Total number of parameters: {total_params}")
                print(f"Total number of trainable parameters: {trainable_params}")


                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                _, vali_loss, train_loss_optimal = exp.train(setting, train_loader,test_loader,
                                                    vali_loader, plot_idx = j,
                                                    val_loss_min=val_loss_lrun,
                                                    train_loss_min =train_loss_lrun,
                                                    weight_decay_flag=True)
                
                if j>0:
                    if vali_loss < vali_loss_prev:
                        counter += 1 
                        weight_decay_optimal = weight_decay
                    elif vali_loss > vali_loss_prev:
                        weight_decay_optimal = args.weight_decay_list[j-1]
                        break
                    elif vali_loss == vali_loss_prev:
                        weight_decay_optimal = args.weight_decay_list[j-1]
                        break
                if j==len(args.weight_decay_list)-1 and counter == 0:
                    weight_decay_optimal = weight_decay
                vali_loss_prev  = vali_loss

            ## train model with decay 1e-4 on 0.33 validation set
            print('>>>>>>> training final model : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            args.weight_decay = 1e-4
            exp = Exp(args)

            train_loss_lrun = np.Inf
            val_loss_lrun   = np.Inf

            train_data, train_loader = data_provider(args, flag='train')
            if args.val_train_ratio == 0:
                vali_loader          = None
                train_without_val    = True
            else:
                vali_set, vali_loader = exp._get_data(flag='val')
                train_without_val     =  False
            test_data, test_loader    = data_provider(args, flag='test')

            exp.train(setting, train_loader, test_loader, vali_loader,
                                                    plot_idx = len(args.weight_decay_list),
                                                    val_loss_min=val_loss_lrun,
                                                    train_without_val=train_without_val)

            ## retrain the model using the rest of the validation data until train_loss is close to train_loss_optimal
            print('>>>>>>> training final model : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            args.weight_decay = weight_decay_optimal
            args.val_train_ratio = 0
            exp = Exp(args)

            print('loading model')
            checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pth'))
            exp.model.load_state_dict(checkpoint['model_state_dict'])

            # optimizer_state_dict = checkpoint['optimizer_state_dict']
            # for param_group in optimizer_state_dict['param_groups']:
            #     param_group['weight_decay'] = weight_decay_optimal
            # exp.model_optim.load_state_dict(optimizer_state_dict)
            # train_loss_lrun = checkpoint['train_loss']
            # val_loss_lrun   = checkpoint['val_loss']

            train_loss_lrun = np.Inf
            val_loss_lrun   = np.Inf

            train_data, train_loader = data_provider(args, flag='train')
            if args.val_train_ratio == 0:
                vali_loader          = None
                train_without_val    = True
            else:
                vali_set, vali_loader = exp._get_data(flag='val')
                train_without_val     =  False
            test_data, test_loader    = data_provider(args, flag='test')

            print('train_loss_optimal:{}, \n'.format(train_loss_optimal))
            print('weight_decay_optimal:{}, \n'.format(weight_decay_optimal))

            exp.train(setting, train_loader, test_loader, vali_loader,
                                                    plot_idx = len(args.weight_decay_list),
                                                    val_loss_min=val_loss_lrun,
                                                    train_without_val=train_without_val)
            
            f = open(os.path.join(checkpoint_dir, "weight_decay_optimal_train_error.txt"), 'a')
            f.write(setting + "  \n")
            f.write('\n')
            f.write('\n')
            f.write('LLMtime metrics \n')
            f.write('train_loss_optimal:{}, \n'.format(train_loss_optimal))
            f.write('weight_decay_optimal:{}, \n'.format(weight_decay_optimal))
            f.write('\n')
            f.write('\n')
            f.close()



            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # test results using model trained until the last epoch
            _, test_preds, test_tgts = exp.vali(test_loader, exp._select_criterion(), test=0,
                                                onestep_pred=False, flag='test')
            test_preds, test_tgts    = process_testing_data(args, test_preds, test_tgts, test_set)
            final_model_metrics = eval_metric(test_preds, test_tgts)

            torch.save(
                {
                    "test_preds": test_preds,
                    "test_tgts": test_tgts,
                    "eval_score": final_model_metrics
                }, checkpoint_dir + "/test_results_final_model.pt")

            print(setting)
            print('results from final model \n')
            print(final_model_metrics)

            # test results using model trained using the best
            _, test_preds, test_tgts = exp.vali(test_loader, exp._select_criterion(),
                                                 setting=setting,  test=1, onestep_pred=False, flag='test')
            test_preds, test_tgts    = process_testing_data(args, test_preds, test_tgts, test_set)
            best_model_metrics = eval_metric(test_preds, test_tgts)

            print('results from best model', best_model_metrics)

            torch.save(
                {
                    "test_preds": test_preds,
                    "test_tgts": test_tgts,
                    "eval_score": best_model_metrics
                }, checkpoint_dir + "/test_results_best_model.pt")

            f = open(os.path.join(checkpoint_dir, "result_long_term_forecast.txt"), 'a')
            f.write(setting + "  \n")
            f.write("Test results  \n")
            f.write("Final model  \n")
            f.write('SMAPE:{}'.format(final_model_metrics))
            f.write('\n')
            f.write("Best model  \n")
            f.write('SMAPE:{}'.format(best_model_metrics))
            f.write('\n')
            f.close()

            ## plot m4 predictions
            if args.data=='m4':
                test_inputs = []
                for i_ in range(test_preds.shape[0]):
                    test_inputs.append(test_loader.dataset[i_][0])
                test_inputs = np.stack(test_inputs,0)
                test_inputs, _    = process_testing_data(args, test_inputs, test_inputs, test_set)
                
                # plot predictions
                test_error = test_preds - test_tgts
                test_error = np.linalg.norm(test_error, axis=1)[:, 0]
                test_error_rel = test_error/np.linalg.norm(test_tgts, axis=1)[:, 0]

                sort_idx       = np.argsort(test_error_rel)
                min_error_idx  = sort_idx[0:3]
                max_error_idx  = sort_idx[-3:]


                for idx in min_error_idx:
                    input_    = np.expand_dims(test_loader.dataset[idx][0], 0)
                    file_name = os.path.join(checkpoint_dir, f"pred_min_error_{idx}.pdf")
                    plot_idx(test_inputs[idx][-6:], test_preds[idx], test_tgts[idx], idx, file_name=file_name)
                for idx in max_error_idx:
                    file_name = os.path.join(checkpoint_dir, f"pred_max_error_{idx}.pdf")
                    plot_idx(test_inputs[idx][-6:], test_preds[idx], test_tgts[idx], idx, file_name=file_name)

            torch.cuda.empty_cache()

        return final_model_metrics, best_model_metrics, None
    else:
        ii = 0
        
        setting = 'long_term_forecast_test_FNO_m4_Monthly_ftS_sl48_step1_pl48_lpl_48_dttest_FNO_nc1_w48_m24_nlc_128_time_18_04_2024-00_30_07_exp_0'
        
        exp_folder = os.path.join(root_base_path, f'checkpoints/{setting}')
        with open( os.path.join(exp_folder, 'exp_parameters.json'), 'r') as f:
            config_data = json.load(f)

        checkpoint_dir  = os.path.join(root_base_path, f'checkpoints/{setting}')

        args = Global_args()
        args.update_attributes(config_data)

        # setting = '{}_{}_{}_{}_ft{}_sl{}_step{}_pl{}_dt{}_FNO_nc{}_w{}_m{}_nlc_{}_time_{}_{}'.format(
        #         args.task_name,
        #         args.model_id,
        #         args.model,
        #         args.data,
        #         args.features,
        #         args.seq_len,
        #         args.step,
        #         args.pred_len,
        #         args.des,
        #         args.num_channels,
        #         args.width,
        #         args.modes,
        #         args.lifting_channels,
        #         date_time, ii)

        exp = Exp(args)  # set experiments

        train_set, train_loader = exp._get_data(flag='train')
        vali_set, vali_loader = exp._get_data(flag='val')
        test_set, test_loader = exp._get_data(flag='test')
        

        # test results using model trained using the best
        _, test_preds, test_tgts = exp.vali(test_loader, exp._select_criterion(),
                                                setting=setting,  test=1)
        test_preds, test_tgts    = process_testing_data(args, test_preds, test_tgts)
        best_model_metrics = eval_metric(test_preds, test_tgts)

        print('results from best model', best_model_metrics)

        torch.save(
            {
                "test_preds": test_preds,
                "test_tgts": test_tgts,
                "eval_score": best_model_metrics
            }, checkpoint_dir + "/test_results_best_model.pt")

        f = open(os.path.join(checkpoint_dir, "result_long_term_forecast.txt"), 'a')
        f.write(setting + "  \n")
        f.write("Test results  \n")
        f.write('\n')
        f.write("Best model  \n")
        f.write('SMAPE:{}'.format(best_model_metrics))
        f.write('\n')
        f.close()
        torch.cuda.empty_cache()





if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='FNO')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--resume_training', type=int, required=False, default=0, help='status')
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
    parser.add_argument('--jumps', type=int, default=3, help='input sequence length')
    
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=60, help='input sequence length')
    parser.add_argument('--step', type=int, default=1, help='step in a sequence')
    parser.add_argument('--pred_len', type=int, default=13, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Weekly',
                        help='data freq, options:[Hourly, Daily, Weekly, Monthly, Quarterly, Yearly]')
    parser.add_argument('--inverse', action='store_false', help='inverse output data', default=True)
    parser.add_argument('--loss_pred_len', type=int, default=13, help='prediction sequence length used to compute loss')
    parser.add_argument('--use_revin', type=bool, default=True, help='flag to set ReVIN')
    parser.add_argument('--revin_affine', type=bool, default=True, help='Affine flag in ReVIN')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs') #100
    parser.add_argument('--batch_size', type=int, default=512, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    #slightly better results came with lr 0.0001, epochs = 500  
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
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

    # model selection
    parser.add_argument('--model', type=str, required=False, default='KRNO',
                        help='model name, options: [KRNO, FNO, FNO_residual, FNO_scale, FNO_without_spectconv]')
    
    # KRNO params
    parser.add_argument('--width', type=int, default=32, help='input/output channels in KRNO/FNO blocks')
    parser.add_argument('--lifting_channels', type=int, default=128, help='hidden layers in lifting/projection FC ')
    parser.add_argument('--hidden_units', type=int, default=64, help='hidden units in MLP of KRNO kernel ')
    parser.add_argument('--n_integral_layers', type=int, default=3, help='no of integral_layers ')
    parser.add_argument('--n_hidden_layers', type=int, default=3, help='no of integral_layers in MLP of KRNO kernel ')
    parser.add_argument('--include_affine', type=bool, default=True, help='use affine in KRNO')
    parser.add_argument('--affine_in_first_layer', type=bool, default=False,
                         help='when seq_len and pred_len are same, this flag will turn on affine mapping in first layer')

    # FNO params
    parser.add_argument('--modes', type=int, default=6, help='Fourier modes')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.root_path = os.path.join(root_base_path, args.root_path)
    
    # args.modes = int(args.seq_len/2)

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.model == 'KRNO':
        args.learning_rate = 1e-3
        args.weight_decay  = 1e-4
        args.loss_pred_len = args.pred_len
        if not args.include_affine:
            args.affine_in_first_layer = False

    current_time = datetime.now()
    args.date_time    = current_time.strftime("%d_%m_%Y-%H_%M_%S")

    print('Args in experiment:')
    # print_args(args)

    run_exp(args)
