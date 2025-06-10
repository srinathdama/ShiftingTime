# MIT License
#
# Original work Copyright (c) 2021 THUML @ Tsinghua University
# Modified work Copyright (c) 2025 DACElab @ University of Toronto
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import darts.datasets
import pandas as pd

dataset_names = [
    'AirPassengersDataset', 
    'AusBeerDataset', 
    'AustralianTourismDataset', 
    'ETTh1Dataset', 
    'ETTh2Dataset', 
    'ETTm1Dataset', 
    'ETTm2Dataset', 
    'ElectricityConsumptionZurichDataset',
    'ElectricityDataset', 
    'EnergyDataset', 
    'ExchangeRateDataset', 
    'GasRateCO2Dataset', 
    'HeartRateDataset', 
    'ILINetDataset', 
    'IceCreamHeaterDataset', 
    'MonthlyMilkDataset', 
    'MonthlyMilkIncompleteDataset', 
    'SunspotsDataset', 
    'TaylorDataset', 
    'TemperatureDataset',
    'TrafficDataset', 
    'USGasolineDataset', 
    'UberTLCDataset', 
    'WeatherDataset', 
    'WineDataset', 
    'WoolyDataset',
]

def get_descriptions(w_references=False):
    descriptions = []
    for dsname in dataset_names:
        d = getattr(darts.datasets,dsname)().__doc__
        
        if w_references:
            descriptions.append(d)
            continue

        lines = []
        for l in d.split("\n"):
            if l.strip().startswith("References"):
                break
            if l.strip().startswith("Source"):
                break
            if l.strip().startswith("Obtained"):
                break
            lines.append(l)
        
        d = " ".join([x.strip() for x in lines]).strip()

        descriptions.append(d)

    return dict(zip(dataset_names,descriptions))

def get_dataset(dsname):
    darts_ds = getattr(darts.datasets,dsname)().load()
    if dsname=='GasRateCO2Dataset':
        darts_ds = darts_ds[darts_ds.columns[1]]
    series = darts_ds.pd_series()

    if dsname == 'SunspotsDataset':
        series = series.iloc[::4]
    if dsname =='HeartRateDataset':
        series = series.iloc[::2]
    return series

def get_single_variate_dataset(dsname):
    darts_ds = getattr(darts.datasets,dsname)().load()
    # print(dsname)
    if dsname=='GasRateCO2Dataset':
        darts_ds = darts_ds[darts_ds.columns[1]]
    elif dsname in ['ETTh1Dataset', 'ETTh2Dataset', 'ETTm1Dataset', 'ETTm2Dataset'] :
        darts_ds = darts_ds[darts_ds.columns[-1]] # get only oil temperature
    elif dsname=='ElectricityConsumptionZurichDataset':
        darts_ds = darts_ds[darts_ds.columns[0]] # get only Households & SMEs electricity consumption
    elif dsname=='ElectricityDataset':
        darts_ds = darts_ds[darts_ds.columns[0]] # get only first household consumption
    elif dsname=='ILINetDataset':
        darts_ds = darts_ds['ILITOTAL'] # get only Total number of ILI patients. 
    elif dsname=='TrafficDataset':
        darts_ds = darts_ds[darts_ds.columns[0]] # get only measureements from 1 sensors
    elif dsname=='UberTLCDataset':
        darts_ds = darts_ds[darts_ds.columns[0]] # get only data from one locationID
    elif dsname=='WeatherDataset':
        darts_ds = darts_ds[darts_ds.columns[1]] # get only temp indicator
    elif dsname=='EnergyDataset':
        darts_ds = darts_ds[darts_ds.columns[-1]] # get only data of one indicator
    if darts_ds.n_components > 1:
        darts_ds = darts_ds[darts_ds.columns[0]]
    series = darts_ds.pd_series()

    # if dsname == 'SunspotsDataset':
    #     series = series.iloc[::4]
    # if dsname =='HeartRateDataset':
    #     series = series.iloc[::2]
    return series

def get_datasets(n=-1,testfrac=0.2):
    datasets = [
        'AirPassengersDataset',
        'AusBeerDataset',
        'GasRateCO2Dataset', # multivariate
        'MonthlyMilkDataset',
        'SunspotsDataset', #very big, need to subsample?
        'WineDataset',
        'WoolyDataset',
        'HeartRateDataset',
    ]
    datas = []
    for i,dsname in enumerate(datasets):
        series = get_dataset(dsname)
        splitpoint = int(len(series)*(1-testfrac))
        
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]
        datas.append((train,test))
        if i+1==n:
            break
    return dict(zip(datasets,datas))

def get_all_datasets(n=-1,testfrac=0.2):
    datasets = [
            'AirPassengersDataset', 
            'AusBeerDataset', 
            'AustralianTourismDataset', 
            'ETTh1Dataset', 
            'ETTh2Dataset', 
            'ETTm1Dataset', 
            'ETTm2Dataset', 
            'ElectricityConsumptionZurichDataset',
            'ElectricityDataset', 
            'EnergyDataset', 
            'GasRateCO2Dataset', 
            'HeartRateDataset', 
            'ILINetDataset', 
            'IceCreamHeaterDataset', 
            'MonthlyMilkDataset', 
            'MonthlyMilkIncompleteDataset', 
            'SunspotsDataset', 
            'TaylorDataset', 
            'TemperatureDataset',
            'TrafficDataset', 
            'USGasolineDataset', 
            'UberTLCDataset', 
            'WeatherDataset', 
            'WineDataset', 
            'WoolyDataset',
        ]
    datas = []
    for i,dsname in enumerate(datasets):
        series = get_single_variate_dataset(dsname)
        splitpoint = int(len(series)*(1-testfrac))
        
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]
        datas.append((train,test))
        if i+1==n:
            break
    return dict(zip(datasets,datas))

def get_memorization_datasets(n=-1,testfrac=0.15, predict_steps=30):
    datasets = [
        'IstanbulTraffic',
        'TSMCStock',
        'TurkeyPower'
    ]
    datas = []
    for i,dsname in enumerate(datasets):
        with open(f"datasets/memorization/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        if predict_steps is not None:
            splitpoint = len(series)-predict_steps
        else:    
            splitpoint = int(len(series)*(1-testfrac))
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]
        datas.append((train,test))
        if i+1==n:
            break
    return dict(zip(datasets,datas))