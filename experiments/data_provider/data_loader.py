# MIT License
#
# Original work Copyright (c) 2021 THUML @ Tsinghua University
# Modified work Copyright (c) 2025 DACElab @ University of Toronto
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None,
                 val_train_ratio=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 0
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)

            # time normalization
            # df_stamp = df_raw[['date']]
            # df_stamp['date'] = pd.to_datetime(df_stamp.date)
            # df_stamp['time_rel'] = df_stamp['date'] - df_stamp['date'][0]
            # df_stamp['relative_time'] = df_stamp['date'].apply(lambda x: x - pd.Timestamp(year=x.year, month=1, day=1))
            # df_stamp['time_rel_in_year'] = df_stamp['relative_time'].dt.total_seconds()/(3600*24*366)
            # data_stamp = df_stamp['time_rel_in_year'].values
            data_stamp = np.arange(len(df_raw))
        else:
            data = df_data.values

        if self.timeenc == 0:
            # df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            # df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            # df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = data_stamp[border1:border2]
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None,
                 val_train_ratio=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 0
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        data_stamp = np.arange(len(df_raw))
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            # df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            # df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            # df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            # df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            # df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            # data_stamp = df_stamp.drop(['date'], 1).values
            data_stamp = data_stamp[border1:border2]
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                   seasonal_patterns=None,
                    val_train_ratio=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 0
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features  = features
        self.target    = target
        self.scale     = scale
        self.timeenc   = timeenc
        self.data_freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        data_stamp = np.arange(len(df_raw))
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            # df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            # df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            # df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            # data_stamp = df_stamp.drop(['date'], 1).values
            data_stamp = data_stamp[border1:border2]
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Sunspotsloader(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='monthly-sunspots.csv',
                 target='Sunspots', scale=True, timeenc=0, data_freq=None,
                 seasonal_patterns=None, val_train_ratio=0.33,
                 detrend_flag = False):
        # size [seq_len, label_len, pred_len]
        # info
        if target is None:
            target='Sunspots'
        if size == None:
            self.seq_len = 24 
            self.step = 1
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.step = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc

        self.root_path = root_path
        self.data_path = data_path
        self.val_train_ratio = val_train_ratio
        self.detrend_flag = detrend_flag
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler() # MinMaxScaler() #StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        df_raw = df_raw.iloc[::4]

        _N     = len(df_raw)
        border1s = [0, int(0.6*_N) - self.seq_len, int(0.8*_N) - self.seq_len]
        border2s = [int(0.6*_N), int(0.8*_N), _N]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]].astype(float)

        if self.scale:
            # time 
            data_stamp = np.arange(len(df_raw), dtype=np.float32)
            train_data = df_data[border1s[0]:border2s[0]].values
            if self.detrend_flag:
                # Reshape for sklearn
                time = data_stamp[border1s[0]:border2s[0]].reshape(-1, 1)
                data = train_data.reshape(-1, 1)
                # Fit linear model
                self.lin_model = LinearRegression()
                self.lin_model.fit(time, data)
                # Detrend data
                trend = self.lin_model.predict(data_stamp.reshape(-1,1))
                detrended_data = df_data.values - trend
                detrended_train_data = detrended_data[border1s[0]:border2s[0]]
            else:
                detrended_train_data = train_data
                detrended_data       = df_data.values
            self.scaler.fit(detrended_train_data)
            data = self.scaler.transform(detrended_data)
        else:
            data = df_data.values

        if self.timeenc == 0:
            # df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            # df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            # df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = data_stamp[border1:border2]

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end:self.step]
        seq_y = self.data_y[r_begin:r_end:self.step]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.step]
        seq_y_mark = self.data_stamp[r_begin:r_end:self.step]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data, data_time=None):
        data = self.scaler.inverse_transform(data)
        if self.detrend_flag:
            trend = self.lin_model.predict(data_time)
            data  = trend + data
        return data


class GasRateCO2Loader(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='gasrate_co2.csv',
                 target='CO2%', scale=True, timeenc=0, data_freq=None,
                 seasonal_patterns=None, val_train_ratio=0.33,
                 detrend_flag = False):
        # size [seq_len, label_len, pred_len]
        # info
        if target is None:
            target='CO2%'
        if size == None:
            self.seq_len = 24 
            self.step = 0
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.step = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc

        self.root_path = root_path
        self.data_path = data_path
        self.val_train_ratio = val_train_ratio
        self.detrend_flag = detrend_flag
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler() # MinMaxScaler() #StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        _N     = len(df_raw)
        train_val_idx_max = int(0.8*_N)
        train_idx_max     = (0.8/(1+self.val_train_ratio))*_N
        train_idx_max     = int(train_idx_max)
        border1s = [0, train_idx_max - self.seq_len, train_val_idx_max - self.seq_len]
        border2s = [train_idx_max, train_val_idx_max, _N]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]].astype(float)

        if self.scale:
            # time 
            data_stamp = np.arange(len(df_raw), dtype=np.float32)
            train_data = df_data[border1s[0]:border2s[0]].values
            if self.detrend_flag:
                # Reshape for sklearn
                time = data_stamp[border1s[0]:border2s[0]].reshape(-1, 1)
                data = train_data.reshape(-1, 1)
                # Fit linear model
                self.lin_model = LinearRegression()
                self.lin_model.fit(time, data)
                # Detrend data
                trend = self.lin_model.predict(data_stamp.reshape(-1,1))
                detrended_data = df_data.values - trend
                detrended_train_data = detrended_data[border1s[0]:border2s[0]]
            else:
                detrended_train_data = train_data
                detrended_data       = df_data.values
            self.scaler.fit(detrended_train_data)
            data = self.scaler.transform(detrended_data)
        else:
            data = df_data.values

        if self.timeenc == 0:
            # df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            # df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            # df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = data_stamp[border1:border2]

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end:self.step]
        seq_y = self.data_y[r_begin:r_end:self.step]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.step]
        seq_y_mark = self.data_stamp[r_begin:r_end:self.step]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data, data_time=None):
        data = self.scaler.inverse_transform(data)
        if self.detrend_flag:
            trend = self.lin_model.predict(data_time)
            data  = trend + data
        return data


class AirPassengersLoader(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='AirPassengers.csv',
                 target='#Passengers', scale=True, timeenc=0,
                 seasonal_patterns=None, val_train_ratio=0.33,
                 detrend_flag = False):
        # size [seq_len, label_len, pred_len]
        # info
        if target is None:
            target='#Passengers'
        if size == None:
            self.seq_len = 24 
            self.step = 0
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.step = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc

        self.root_path = root_path
        self.data_path = data_path
        self.val_train_ratio = val_train_ratio
        self.detrend_flag = detrend_flag
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler() # MinMaxScaler() #StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        _N     = len(df_raw)
        train_val_idx_max = int(0.8*_N)
        train_idx_max     = (0.8/(1+self.val_train_ratio))*_N
        train_idx_max     = int(train_idx_max)
        border1s = [0, train_idx_max - self.seq_len, train_val_idx_max - self.seq_len]
        border2s = [train_idx_max, train_val_idx_max, _N]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]].astype(float)

        if self.scale:
            # time 
            data_stamp = np.arange(len(df_raw), dtype=np.float32)
            train_data = df_data[border1s[0]:border2s[0]].values
            if self.detrend_flag:
                # Reshape for sklearn
                time = data_stamp[border1s[0]:border2s[0]].reshape(-1, 1)
                data = train_data.reshape(-1, 1)
                # Fit linear model
                self.lin_model = LinearRegression()
                self.lin_model.fit(time, data)
                # Detrend data
                trend = self.lin_model.predict(data_stamp.reshape(-1,1))
                detrended_data = df_data.values - trend
                detrended_train_data = detrended_data[border1s[0]:border2s[0]]
            else:
                detrended_train_data = train_data
                detrended_data       = df_data.values
            self.scaler.fit(detrended_train_data)
            data = self.scaler.transform(detrended_data)
        else:
            data = df_data.values

        if self.timeenc == 0:
            # df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            # df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            # df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = data_stamp[border1:border2]

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end:self.step]
        seq_y = self.data_y[r_begin:r_end:self.step]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.step]
        seq_y_mark = self.data_stamp[r_begin:r_end:self.step]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data, data_time=None):
        data = self.scaler.inverse_transform(data)
        if self.detrend_flag:
            trend = self.lin_model.predict(data_time)
            data  = trend + data
        return data
    

class AusBeerLoader(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ausbeer.csv',
                 target='Y', scale=True, timeenc=0,
                 seasonal_patterns=None, val_train_ratio=0.33,
                 detrend_flag = False):
        # size [seq_len, label_len, pred_len]
        # info
        if target is None:
            target='Y'
        if size == None:
            self.seq_len = 24 
            self.step = 0
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.step = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc

        self.root_path = root_path
        self.data_path = data_path
        self.val_train_ratio = val_train_ratio
        self.detrend_flag = detrend_flag
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler() # MinMaxScaler() #StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        _N     = len(df_raw)
        train_val_idx_max = int(0.8*_N)
        train_idx_max     = (0.8/(1+self.val_train_ratio))*_N
        train_idx_max     = int(train_idx_max)
        border1s = [0, train_idx_max - self.seq_len, train_val_idx_max - self.seq_len]
        border2s = [train_idx_max, train_val_idx_max, _N]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]].astype(float)

        if self.scale:
            # time 
            data_stamp = np.arange(len(df_raw), dtype=np.float32)
            train_data = df_data[border1s[0]:border2s[0]].values
            if self.detrend_flag:
                # Reshape for sklearn
                time = data_stamp[border1s[0]:border2s[0]].reshape(-1, 1)
                data = train_data.reshape(-1, 1)
                # Fit linear model
                self.lin_model = LinearRegression()
                self.lin_model.fit(time, data)
                # Detrend data
                trend = self.lin_model.predict(data_stamp.reshape(-1,1))
                detrended_data = df_data.values - trend
                detrended_train_data = detrended_data[border1s[0]:border2s[0]]
            else:
                detrended_train_data = train_data
                detrended_data       = df_data.values
            self.scaler.fit(detrended_train_data)
            data = self.scaler.transform(detrended_data)
        else:
            data = df_data.values

        if self.timeenc == 0:
            # df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            # df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            # df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = data_stamp[border1:border2]

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end:self.step]
        seq_y = self.data_y[r_begin:r_end:self.step]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.step]
        seq_y_mark = self.data_stamp[r_begin:r_end:self.step]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data, data_time=None):
        data = self.scaler.inverse_transform(data)
        if self.detrend_flag:
            trend = self.lin_model.predict(data_time)
            data  = trend + data
        return data
    
class MonthlyMilkLoader(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='monthly-milk.csv',
                 target='Pounds per cow', scale=True, timeenc=0,
                 seasonal_patterns=None, val_train_ratio=0.33,
                 detrend_flag = False):
        # size [seq_len, label_len, pred_len]
        # info
        if target is None:
            target='Pounds per cow'
        if size == None:
            self.seq_len = 24 
            self.step = 0
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.step = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc

        self.root_path = root_path
        self.data_path = data_path
        self.val_train_ratio = val_train_ratio
        self.detrend_flag = detrend_flag
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler() # MinMaxScaler() #StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        _N     = len(df_raw)
        train_val_idx_max = int(0.8*_N)
        train_idx_max     = (0.8/(1+self.val_train_ratio))*_N
        train_idx_max     = int(train_idx_max)
        border1s = [0, train_idx_max - self.seq_len, train_val_idx_max - self.seq_len]
        border2s = [train_idx_max, train_val_idx_max, _N]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]].astype(float)

        if self.scale:
            # time 
            data_stamp = np.arange(len(df_raw), dtype=np.float32)
            train_data = df_data[border1s[0]:border2s[0]].values
            if self.detrend_flag:
                # Reshape for sklearn
                time = data_stamp[border1s[0]:border2s[0]].reshape(-1, 1)
                data = train_data.reshape(-1, 1)
                # Fit linear model
                self.lin_model = LinearRegression()
                self.lin_model.fit(time, data)
                # Detrend data
                trend = self.lin_model.predict(data_stamp.reshape(-1,1))
                detrended_data = df_data.values - trend
                detrended_train_data = detrended_data[border1s[0]:border2s[0]]
            else:
                detrended_train_data = train_data
                detrended_data       = df_data.values
            self.scaler.fit(detrended_train_data)
            data = self.scaler.transform(detrended_data)
        else:
            data = df_data.values

        if self.timeenc == 0:
            # df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            # df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            # df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = data_stamp[border1:border2]

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end:self.step]
        seq_y = self.data_y[r_begin:r_end:self.step]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.step]
        seq_y_mark = self.data_stamp[r_begin:r_end:self.step]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data, data_time=None):
        data = self.scaler.inverse_transform(data)
        if self.detrend_flag:
            trend = self.lin_model.predict(data_time)
            data  = trend + data
        return data
    
class WineLoader(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='wineind.csv',
                 target='Y', scale=True, timeenc=0,
                 seasonal_patterns=None, val_train_ratio=0.33,
                 detrend_flag = False):
        # size [seq_len, label_len, pred_len]
        # info
        if target is None:
            target='Y'
        if size == None:
            self.seq_len = 24 
            self.step = 0
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.step = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc

        self.root_path = root_path
        self.data_path = data_path
        self.val_train_ratio = val_train_ratio
        self.detrend_flag = detrend_flag
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler() # MinMaxScaler() #StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        _N     = len(df_raw)
        train_val_idx_max = int(0.8*_N)
        train_idx_max     = (0.8/(1+self.val_train_ratio))*_N
        train_idx_max     = int(train_idx_max)
        border1s = [0, train_idx_max - self.seq_len, train_val_idx_max - self.seq_len]
        border2s = [train_idx_max, train_val_idx_max, _N]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]].astype(float)

        if self.scale:
            # time 
            data_stamp = np.arange(len(df_raw), dtype=np.float32)
            train_data = df_data[border1s[0]:border2s[0]].values
            if self.detrend_flag:
                # Reshape for sklearn
                time = data_stamp[border1s[0]:border2s[0]].reshape(-1, 1)
                data = train_data.reshape(-1, 1)
                # Fit linear model
                self.lin_model = LinearRegression()
                self.lin_model.fit(time, data)
                # Detrend data
                trend = self.lin_model.predict(data_stamp.reshape(-1,1))
                detrended_data = df_data.values - trend
                detrended_train_data = detrended_data[border1s[0]:border2s[0]]
            else:
                detrended_train_data = train_data
                detrended_data       = df_data.values
            self.scaler.fit(detrended_train_data)
            data = self.scaler.transform(detrended_data)
        else:
            data = df_data.values

        if self.timeenc == 0:
            # df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            # df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            # df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = data_stamp[border1:border2]

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end:self.step]
        seq_y = self.data_y[r_begin:r_end:self.step]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.step]
        seq_y_mark = self.data_stamp[r_begin:r_end:self.step]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data, data_time=None):
        data = self.scaler.inverse_transform(data)
        if self.detrend_flag:
            trend = self.lin_model.predict(data_time)
            data  = trend + data
        return data
   

class WoolyLoader(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='woolyrnq.csv',
                 target='Y', scale=True, timeenc=0,
                 seasonal_patterns=None, val_train_ratio=0.33,
                 detrend_flag = False):
        # size [seq_len, label_len, pred_len]
        # info
        if target is None:
            target='Y'
        if size == None:
            self.seq_len = 24 
            self.step = 0
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.step = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc

        self.root_path = root_path
        self.data_path = data_path
        self.val_train_ratio = val_train_ratio
        self.detrend_flag = detrend_flag
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler() # MinMaxScaler() #StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        _N     = len(df_raw)
        train_val_idx_max = int(0.8*_N)
        train_idx_max     = (0.8/(1+self.val_train_ratio))*_N
        train_idx_max     = int(train_idx_max)
        border1s = [0, train_idx_max - self.seq_len, train_val_idx_max - self.seq_len]
        border2s = [train_idx_max, train_val_idx_max, _N]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]].astype(float)

        if self.scale:
            # time 
            data_stamp = np.arange(len(df_raw), dtype=np.float32)
            train_data = df_data[border1s[0]:border2s[0]].values
            if self.detrend_flag:
                # Reshape for sklearn
                time = data_stamp[border1s[0]:border2s[0]].reshape(-1, 1)
                data = train_data.reshape(-1, 1)
                # Fit linear model
                self.lin_model = LinearRegression()
                self.lin_model.fit(time, data)
                # Detrend data
                trend = self.lin_model.predict(data_stamp.reshape(-1,1))
                detrended_data = df_data.values - trend
                detrended_train_data = detrended_data[border1s[0]:border2s[0]]
            else:
                detrended_train_data = train_data
                detrended_data       = df_data.values
            self.scaler.fit(detrended_train_data)
            data = self.scaler.transform(detrended_data)
        else:
            data = df_data.values

        if self.timeenc == 0:
            # df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            # df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            # df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = data_stamp[border1:border2]

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end:self.step]
        seq_y = self.data_y[r_begin:r_end:self.step]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.step]
        seq_y_mark = self.data_stamp[r_begin:r_end:self.step]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data, data_time=None):
        data = self.scaler.inverse_transform(data)
        if self.detrend_flag:
            trend = self.lin_model.predict(data_time)
            data  = trend + data
        return data
   

class HeartRateLoader(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='heart_rate.csv',
                 target='Heart rate', scale=True, timeenc=0,
                 seasonal_patterns=None, val_train_ratio=0.33,
                 detrend_flag = False):
        # size [seq_len, label_len, pred_len]
        # info
        if target is None:
            target='Heart rate'
        if size == None:
            self.seq_len = 24 
            self.step = 0
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.step = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc

        self.root_path = root_path
        self.data_path = data_path
        self.val_train_ratio = val_train_ratio
        self.detrend_flag = detrend_flag
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler() # MinMaxScaler() #StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        df_raw = df_raw.iloc[::2]
        
        _N     = len(df_raw)
        train_val_idx_max = int(0.8*_N)
        train_idx_max     = (0.8/(1+self.val_train_ratio))*_N
        train_idx_max     = int(train_idx_max)
        border1s = [0, train_idx_max - self.seq_len, train_val_idx_max - self.seq_len]
        border2s = [train_idx_max, train_val_idx_max, _N]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]].astype(float)

        if self.scale:
            # time 
            data_stamp = np.arange(len(df_raw), dtype=np.float32)
            train_data = df_data[border1s[0]:border2s[0]].values
            if self.detrend_flag:
                # Reshape for sklearn
                time = data_stamp[border1s[0]:border2s[0]].reshape(-1, 1)
                data = train_data.reshape(-1, 1)
                # Fit linear model
                self.lin_model = LinearRegression()
                self.lin_model.fit(time, data)
                # Detrend data
                trend = self.lin_model.predict(data_stamp.reshape(-1,1))
                detrended_data = df_data.values - trend
                detrended_train_data = detrended_data[border1s[0]:border2s[0]]
            else:
                detrended_train_data = train_data
                detrended_data       = df_data.values
            self.scaler.fit(detrended_train_data)
            data = self.scaler.transform(detrended_data)
        else:
            data = df_data.values
        if self.timeenc == 0:
            # df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            # df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            # df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = data_stamp[border1:border2]

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end:self.step]
        seq_y = self.data_y[r_begin:r_end:self.step]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.step]
        seq_y_mark = self.data_stamp[r_begin:r_end:self.step]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data, data_time=None):
        data = self.scaler.inverse_transform(data)
        if self.detrend_flag:
            trend = self.lin_model.predict(data_time)
            data  = trend + data
        return data
   


class Dataset_M4_KNF(Dataset):
    """Dataset class for M4 dataset."""
    def __init__(
        self, root_path, flag='pred', size=None,
        features='S', data_path='ETTh1.csv',
        target='OT', scale=False, timeenc=0,
        seasonal_patterns='Weekly',
        jumps=1,  # The number of skipped steps when generating sliced samples
        non_dep_flag = False,
        val_train_ratio=0.1,
        test_pred_len = None
    ):
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.root_path = root_path
        direc       = os.path.join(self.root_path, 'train.npy')
        direc_test  = os.path.join(self.root_path, 'test.npy')
        seq_len = size[0]
        self.label_len = 0 
        self.step     = size[1]
        pred_len = size[2]

        freq = seasonal_patterns

        self.input_length = seq_len
        self.output_length = pred_len
        self.mode = flag
        self.non_dep_flag = non_dep_flag
        self.train_val_split = 1 - val_train_ratio
        self.test_pred_len = test_pred_len

        # Load training set
        self.data_lsts = np.load(direc, allow_pickle=True).item().get(freq)

        # First do global standardization
        self.ts_means, self.ts_stds = [], []
        for i, item in enumerate(self.data_lsts):
            avg, std = np.mean(item), np.std(item)
            self.ts_means.append(avg)
            self.ts_stds.append(std)
            self.data_lsts[i] = (self.data_lsts[i] - avg) / std

        self.ts_means = np.array(self.ts_means)
        self.ts_stds = np.array(self.ts_stds)

        lenghts_ = []
        if self.mode == "test":
            self.test_lsts = np.load(direc_test, allow_pickle=True).item().get(freq)
            for i, item in enumerate(self.test_lsts):
                ## raise error if len(item) < self.output_length
                ## test sequece len should be greater than self.output_length
                lenghts_.append(len(item))
                if len(item) < self.output_length:
                    print(f' {i}th test sequece len is greater than self.output_length - {self.output_length}')
                self.test_lsts[i] = (item - self.ts_means[i]) / self.ts_stds[i]
            self.ts_indices = list(range(len(self.test_lsts)))

        elif self.mode == "train" or "val":
            # shuffle slices before split
            # np.random.seed(123)
            # self.ts_indices = []
            # for i, item in enumerate(self.data_lsts):
            #     for j in range(0, len(item)-self.input_length-self.output_length, jumps):
            #         self.ts_indices.append((i, j))
            # np.random.shuffle(self.ts_indices)

            # # 90%-10% train-validation split
            # train_valid_split = int(len(self.ts_indices) * 0.9)
            # if self.mode == "train":
            #     self.ts_indices = self.ts_indices[:train_valid_split]
            # elif self.mode == "val":
            #     self.ts_indices = self.ts_indices[train_valid_split:]

            # 90%-10% train-validation split
            self.ts_indices_train = []
            self.ts_indices_val  = []
            if self.non_dep_flag:
                ## include only non dependent entries in the inputs based on the following paper
                ## https://www.sciencedirect.com/science/article/abs/pii/S0167947317302384
                # shuffle slices before split
                non_dep_ratio = 1
                if self.output_length>5:
                    eff_non_dep_len = 5
                else:
                    eff_non_dep_len = int(non_dep_ratio*self.output_length)
                np.random.seed(123)
                self.ts_indices = []
                for i, item in enumerate(self.data_lsts):
                    lenghts_.append(len(item))
                    train_val_idx   = np.arange(0, len(item)-self.input_length-self.output_length, jumps)
                    np.random.shuffle(train_val_idx)
                    train_valid_split = int(len(train_val_idx) * self.train_val_split)
                    train_idx       = list(train_val_idx[:train_valid_split])
                    val_idx         = list(train_val_idx[train_valid_split:])
                    # remove dependence idx from train_idx
                    for val_indice in val_idx:
                        dep_indices = np.arange(val_indice-eff_non_dep_len+1,
                                                    val_indice+eff_non_dep_len)
                        for dep_indice_ in dep_indices:
                            try:
                                train_idx.remove(dep_indice_)
                            except:
                                continue
                    for j in train_idx:
                        self.ts_indices_train.append((i, j))
                    for j in val_idx:
                        self.ts_indices_val.append((i, j))
            
            else:
                for i, item in enumerate(self.data_lsts):
                    lenghts_.append(len(item))
                    # print(i, len(item))
                    if self.test_pred_len is not None:
                        train_idx   = np.arange(0, len(item)-self.input_length-self.test_pred_len-self.output_length, jumps)
                        if len(train_idx) > 0:
                            train_idx_last  = train_idx[-1]
                            for j in train_idx:
                                if j < train_idx_last:
                                    self.ts_indices_train.append((i, j))
                                else:
                                    self.ts_indices_val.append((i, j))
                    else:
                        train_val_idx   = np.arange(0, len(item)-self.input_length-self.output_length, jumps)
                        if len(train_val_idx) > 0:
                            if self.train_val_split == 1:
                                train_idx_last  = train_val_idx[-1]
                            else:
                                train_idx_last  = train_val_idx[int(self.train_val_split*len(train_val_idx))]
                            for j in train_val_idx:
                                if j < train_idx_last:
                                    self.ts_indices_train.append((i, j))
                                else:
                                    self.ts_indices_val.append((i, j))

            
            if self.mode == "train":
                self.ts_indices = self.ts_indices_train
            elif self.mode == "val":
                self.ts_indices = self.ts_indices_val
        else:
            raise ValueError("Mode can only be one of train, val, test")
        print('min len -', min(lenghts_), 'max len -', max(lenghts_))

    def __len__(self):
        return len(self.ts_indices)

    def __getitem__(self, index):
        if self.mode == "test":
            x = self.data_lsts[index][-self.input_length:]
            y = self.test_lsts[index]

        else:
            i, j = self.ts_indices[index]
            x = self.data_lsts[i][j:j + self.input_length]
            if self.test_pred_len is not None and self.mode == 'val':
                y = self.data_lsts[i][j + self.input_length:j + self.input_length +
                                        self.test_pred_len]
            else:
                y = self.data_lsts[i][j + self.input_length:j + self.input_length +
                                        self.output_length]
            # y = self.data_lsts[i][j + self.input_length:j + self.input_length +
            #                             self.output_length]
        x_stamp = np.arange(len(x))
        y_stamp = np.arange(len(y))

        return x.reshape(-1,1), y.reshape(-1,1), x_stamp, y_stamp


class CrytosDataset(torch.utils.data.Dataset):
    """Dataset class for Cryptos data."""

    def __init__(
        self, root_path, flag='pred', size=None,
        features='S', data_path='ETTh1.csv',
        target='OT', scale=False, timeenc=0,
        seasonal_patterns='Weekly',
        jumps=100,  # The number of skipped steps when generating sliced samples
        non_dep_flag = False,
        val_train_ratio=0.1,
        test_pred_len = None
        ):

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.root_path = root_path
        direc       = os.path.join(self.root_path, 'train.npy')
        direc_test  = os.path.join(self.root_path, 'test.npy')
        seq_len = size[0]
        self.label_len = 0 
        self.step     = size[1]
        pred_len = size[2]

        freq = seasonal_patterns

        self.input_length = seq_len
        self.output_length = pred_len
        self.mode = flag
        self.non_dep_flag = non_dep_flag
        self.train_val_split = 1 - val_train_ratio

        # Load training set
        self.train_data = np.load(direc, allow_pickle=True)

        # First do global standardization
        self.ts_means, self.ts_stds = [], []
        for i, item in enumerate(self.train_data):
            avg, std = np.mean(
                item, axis=0, keepdims=True), np.std(
                    item, axis=0, keepdims=True)
            self.ts_means.append(avg)
            self.ts_stds.append(std)
            self.train_data[i] = (self.train_data[i] - avg) / std

        self.ts_means = np.concatenate(self.ts_means, axis=0)
        self.ts_stds = np.concatenate(self.ts_stds, axis=0)

        lenghts_ = []
        if self.mode == "test":
            self.test_lsts = np.load(direc_test, allow_pickle=True)
            for i, item in enumerate(self.test_lsts):
                lenghts_.append(len(item))
                self.test_lsts[i] = (self.test_lsts[i] -
                                    self.ts_means[i]) / self.ts_stds[i]

            # change the input length (< 100) will not affect the target output
            self.ts_indices = []
            for i, item in enumerate(self.test_lsts):
                for j in range(100, len(item) - self.output_length, self.output_length):
                    self.ts_indices.append((i, j))

        elif self.mode == "train" or "val":
            # # shuffle slices before split
            # np.random.seed(123)
            # self.ts_indices = []
            # for i, item in enumerate(self.train_data):
            #     for j in range(0, len(item)-self.input_length-self.output_length, jumps):
            #         self.ts_indices.append((i, j))

            # np.random.shuffle(self.ts_indices)

            # # 90%-10% train-validation split
            # train_valid_split = int(len(self.ts_indices) * 0.9)
            # if self.mode == "train":
            #     self.ts_indices = self.ts_indices[:train_valid_split]
            # elif self.mode == "valid":
            #     self.ts_indices = self.ts_indices[train_valid_split:]
            self.ts_indices_train = []
            self.ts_indices_val  = []
            if self.non_dep_flag:
                ## include only non dependent entries in the inputs based on the following paper
                ## https://www.sciencedirect.com/science/article/abs/pii/S0167947317302384
                # shuffle slices before split
                non_dep_ratio = 1
                if self.output_length>5:
                    eff_non_dep_len = 5
                else:
                    eff_non_dep_len = int(non_dep_ratio*self.output_length)
                np.random.seed(123)
                self.ts_indices = []
                for i, item in enumerate(self.train_data):
                    lenghts_.append(len(item))
                    train_val_idx   = np.arange(0, len(item)-self.input_length-self.output_length, jumps)
                    np.random.shuffle(train_val_idx)
                    train_valid_split = int(len(train_val_idx) * self.train_val_split)
                    train_idx       = list(train_val_idx[:train_valid_split])
                    val_idx         = list(train_val_idx[train_valid_split:])
                    # remove dependence idx from train_idx
                    for val_indice in val_idx:
                        dep_indices = np.arange(val_indice-eff_non_dep_len+1,
                                                    val_indice+eff_non_dep_len)
                        for dep_indice_ in dep_indices:
                            try:
                                train_idx.remove(dep_indice_)
                            except:
                                continue
                    for j in train_idx:
                        self.ts_indices_train.append((i, j))
                    for j in val_idx:
                        self.ts_indices_val.append((i, j))
            
            else:
                for i, item in enumerate(self.train_data):
                    lenghts_.append(len(item))
                    train_val_idx   = np.arange(0, len(item)-self.input_length-self.output_length, jumps)
                    if self.train_val_split == 1:
                        train_idx_last  = train_val_idx[-1]
                    else:
                        train_idx_last  = train_val_idx[int(self.train_val_split*len(train_val_idx))]
                    for j in train_val_idx:
                        if j < train_idx_last:
                            self.ts_indices_train.append((i, j))
                        else:
                            self.ts_indices_val.append((i, j))

            
            if self.mode == "train":
                self.ts_indices = self.ts_indices_train
            elif self.mode == "val":
                self.ts_indices = self.ts_indices_val
        else:
            raise ValueError("Mode can only be one of train, val, test")
        print('min len -', min(lenghts_), 'max len -', max(lenghts_))

    def __len__(self):
        return len(self.ts_indices)

    def __getitem__(self, index):
        if self.mode == "test":
            i, j = self.ts_indices[index]
            x = self.test_lsts[i][j - self.input_length:j]
            y = self.test_lsts[i][j:j + self.output_length]
        else:
            i, j = self.ts_indices[index]
            x = self.train_data[i][j:j + self.input_length]
            y = self.train_data[i][j + self.input_length:j + self.input_length +
                                 self.output_length]
        
        x_stamp = np.arange(len(x))
        y_stamp = np.arange(len(y))

        return x, y, x_stamp, y_stamp


class TrajDataset(torch.utils.data.Dataset):
    """Dataset class for NBA player trajectory data."""
    def __init__(
        self, root_path, flag='pred', size=None,
        features='S', data_path='ETTh1.csv',
        target='OT', scale=False, timeenc=0,
        seasonal_patterns='Weekly',
        jumps=2,  # The number of skipped steps when generating sliced samples
        non_dep_flag = False,
        val_train_ratio=0.1,
        test_pred_len = None
        ):

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.root_path = root_path
        direc       = os.path.join(self.root_path, 'train.npy')
        direc_test  = os.path.join(self.root_path, 'test.npy')
        seq_len = size[0]
        self.label_len = 0 
        self.step     = size[1]
        pred_len = size[2]

        freq = seasonal_patterns

        self.input_length = seq_len
        self.output_length = pred_len
        self.mode = flag
        self.non_dep_flag = non_dep_flag
        self.train_val_split = 1 - val_train_ratio

        # Load training set
        self.train_data = np.load(direc, allow_pickle=True)

        # First do global standardization
        self.ts_means, self.ts_stds = [], []
        for i, item in enumerate(self.train_data):
            avg = np.mean(item, axis=0, keepdims=True)
            std = np.std(item, axis=0, keepdims=True)
            self.ts_means.append(avg)
            self.ts_stds.append(std)
            self.train_data[i] = (item - avg) / std

        lenghts_ = []
        if self.mode == "test":
            self.ts_means, self.ts_stds = [], []
            self.test_lsts = np.load(direc_test, allow_pickle=True)
            for i, item in enumerate(self.test_lsts):
                lenghts_.append(len(item))
                avg = np.mean(item, axis=0, keepdims=True)
                std = np.std(item, axis=0, keepdims=True)
                self.ts_means.append(avg)
                self.ts_stds.append(std)
                self.test_lsts[i] = (self.test_lsts[i] - avg) / std

            # change the input length (<100) will not affect the target output
            self.ts_indices = []
            for i in range(len(self.test_lsts)):
                for j in range(50, 300 - self.output_length, 50):
                    self.ts_indices.append((i, j))
        elif self.mode == "train" or "val":
            # # shuffle slices before split
            # np.random.seed(123)
            # self.ts_indices = []
            # for i, item in enumerate(self.train_data):
            #     for j in range(0, len(item)-self.input_length-self.output_length, jumps):
            #         self.ts_indices.append((i, j))
            # np.random.shuffle(self.ts_indices)

            # # 90%-10% train-validation split
            # train_valid_split = int(len(self.ts_indices) * 0.9)
            # if self.mode == "train":
            #     self.ts_indices = self.ts_indices[:train_valid_split]
            # elif self.mode == "valid":
            #     self.ts_indices = self.ts_indices[train_valid_split:]
            self.ts_indices_train = []
            self.ts_indices_val  = []
            if self.non_dep_flag:
                ## include only non dependent entries in the inputs based on the following paper
                ## https://www.sciencedirect.com/science/article/abs/pii/S0167947317302384
                # shuffle slices before split
                non_dep_ratio = 1
                if self.output_length>5:
                    eff_non_dep_len = 5
                else:
                    eff_non_dep_len = int(non_dep_ratio*self.output_length)
                np.random.seed(123)
                self.ts_indices = []
                for i, item in enumerate(self.train_data):
                    lenghts_.append(len(item))
                    train_val_idx   = np.arange(0, len(item)-self.input_length-self.output_length, jumps)
                    np.random.shuffle(train_val_idx)
                    train_valid_split = int(len(train_val_idx) * self.train_val_split)
                    train_idx       = list(train_val_idx[:train_valid_split])
                    val_idx         = list(train_val_idx[train_valid_split:])
                    # remove dependence idx from train_idx
                    for val_indice in val_idx:
                        dep_indices = np.arange(val_indice-eff_non_dep_len+1,
                                                    val_indice+eff_non_dep_len)
                        for dep_indice_ in dep_indices:
                            try:
                                train_idx.remove(dep_indice_)
                            except:
                                continue
                    for j in train_idx:
                        self.ts_indices_train.append((i, j))
                    for j in val_idx:
                        self.ts_indices_val.append((i, j))
            
            else:
                for i, item in enumerate(self.train_data):
                    lenghts_.append(len(item))
                    train_val_idx   = np.arange(0, len(item)-self.input_length-self.output_length, jumps)
                    if self.train_val_split == 1:
                        train_idx_last  = train_val_idx[-1]
                    else:
                        train_idx_last  = train_val_idx[int(self.train_val_split*len(train_val_idx))]
                    for j in train_val_idx:
                        if j < train_idx_last:
                            self.ts_indices_train.append((i, j))
                        else:
                            self.ts_indices_val.append((i, j))

            
            if self.mode == "train":
                self.ts_indices = self.ts_indices_train
            elif self.mode == "val":
                self.ts_indices = self.ts_indices_val
        else:
            raise ValueError("Mode can only be one of train, valid, test")

        self.ts_means = np.concatenate(self.ts_means, axis=0)
        self.ts_stds = np.concatenate(self.ts_stds, axis=0)
        print('min len -', min(lenghts_), 'max len -', max(lenghts_))

    def __len__(self):
        return len(self.ts_indices)

    def __getitem__(self, index):
        if self.mode == "test":
            i, j = self.ts_indices[index]
            x = self.test_lsts[i][j - self.input_length:j]
            y = self.test_lsts[i][j:j + self.output_length]
        else:
            i, j = self.ts_indices[index]
            x = self.train_data[i][j:j + self.input_length]
            y = self.train_data[i][j + self.input_length:j + self.input_length +
                                 self.output_length]
        
        x_stamp = np.arange(len(x))
        y_stamp = np.arange(len(y))

        return x, y, x_stamp, y_stamp
