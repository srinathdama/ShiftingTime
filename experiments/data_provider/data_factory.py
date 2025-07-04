# MIT License
#
# Original work Copyright (c) 2021 THUML @ Tsinghua University
# Modified work Copyright (c) 2025 DACElab @ University of Toronto
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, \
    Sunspotsloader, GasRateCO2Loader, AirPassengersLoader, AusBeerLoader, MonthlyMilkLoader, \
         WineLoader, WoolyLoader, HeartRateLoader, Dataset_M4_KNF, CrytosDataset, TrajDataset
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'ECL':Dataset_Custom,
    'Exchange':Dataset_Custom,
    'Traffic':Dataset_Custom,
    'Weather':Dataset_Custom,
    'ILI':Dataset_Custom,
    'm4': Dataset_M4_KNF, # Dataset_M4, 
    'sunspots': Sunspotsloader,
    'GasRateCO2': GasRateCO2Loader,
    'AirPassengers': AirPassengersLoader,
    'AusBeer':AusBeerLoader,
    'MonthlyMilk':MonthlyMilkLoader,
    'Wine':WineLoader,
    'Wooly':WoolyLoader,
    'HeartRate':HeartRateLoader,
    'Cryptos':CrytosDataset,
    'Traj':TrajDataset
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 

    # pred_len = args.pred_len
    if args.pred_len != args.loss_pred_len:
        assert args.pred_len > args.loss_pred_len
        pred_len = min(args.pred_len, args.loss_pred_len)
    else:
        pred_len = args.pred_len

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = 1  # bsz=1 for evaluation
        step = args.step
    else:
        if hasattr(args, 'shuffle_flag'):
            shuffle_flag = args.shuffle_flag
        else:
            shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size  # bsz for train and valid
        step = args.step
    
    if hasattr(args, 'temp_batch_size'):
        batch_size = args.temp_batch_size
        del args.temp_batch_size

    if not hasattr(args, 'target'):
        args.target = None
    if not hasattr(args, 'val_train_ratio'):
        args.val_train_ratio = 0.33
    
    step = 0

    if args.data in ['AirPassengers', 'AusBeer', 'GasRateCO2', 
               'MonthlyMilk', 'sunspots', 'Wine', 'Wooly','HeartRate']:
        step = args.step
        seasonal_patterns=None
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, step, pred_len],
            features=args.features,
            target=args.target,
            seasonal_patterns=seasonal_patterns,
            val_train_ratio=args.val_train_ratio,
            detrend_flag=args.detrend_flag
        )
    elif args.data in ['m4', 'Cryptos', 'Traj']:
        drop_last = False
        batch_size = args.batch_size
        seasonal_patterns=args.seasonal_patterns
        # if flag in ['test', 'val']:
        if flag in ['test']:
            pred_len = args.test_output_length
        ## For M4 data, to use only the last one sequence in train data as validation set
        ## pass test_pred_len = args.test_output_length 
        test_pred_len = None #args.test_output_length # None
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, step, pred_len],
            features=args.features,
            target=args.target,
            seasonal_patterns=seasonal_patterns,
            val_train_ratio=args.val_train_ratio,
            test_pred_len = test_pred_len, # args.test_output_length,
            jumps = args.jumps
        )
    elif args.data in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2',
                     'ECL', 'Exchange', 'Traffic', 'Weather', 'ILI' ]:
        drop_last = True
        if flag == 'test':
            batch_size = 1
        else:
            batch_size = args.batch_size
        if flag in ['test']:
            pred_len = args.test_output_length
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, step, pred_len],
            features=args.features,
            target=args.target,
            val_train_ratio=args.val_train_ratio
        )
    else:
        seasonal_patterns=None
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, step, pred_len],
            features=args.features,
            target=args.target,
            seasonal_patterns=seasonal_patterns,
            val_train_ratio=args.val_train_ratio
        )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        #multiprocessing_context='fork',
        persistent_workers=True)
    return data_set, data_loader
