# MIT License
#
# Original work Copyright (c) 2021 THUML @ Tsinghua University
# Modified work Copyright (c) 2025 DACElab @ University of Toronto
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
from collections import defaultdict
from collections.abc import Iterable
from matplotlib.ticker import MaxNLocator
import json

import itertools,operator,functools

plt.switch_backend('agg')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, val_loss_min=np.Inf,
                 train_loss_optimal = np.Inf):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = -val_loss_min
        self.early_stop = False
        self.val_loss_min = val_loss_min
        self.delta = delta
        self.cur_train_loss_optimal = train_loss_optimal

    def __call__(self, val_loss, model, path, cur_train_loss=None, epoch=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            if cur_train_loss is not None:
                self.cur_train_loss_optimal = cur_train_loss
            if epoch is not None:
                self.epoch = epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            if cur_train_loss is not None:
                self.cur_train_loss_optimal =cur_train_loss
            if epoch is not None:
                self.epoch = epoch
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        checkpoint = { 'model_state_dict': model.state_dict()}
        torch.save(checkpoint, path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

class CheckConvergence:
    def __init__(self, patience=3, verbose=False, rtol=0.01, atol=1e-6,
                 val_loss_min=np.Inf, train_loss_optimal = np.Inf):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.prev_train_loss = 1e5
        self.prev_val_loss   = 1e5
        self.val_loss_min = val_loss_min
        self.train_loss_optimal = train_loss_optimal
        self.early_stop = False
        self.rtol   = rtol
        self.atol   = atol

    def __call__(self, train_loss, model, model_optim, path, val_loss=None):
        
        convergence_flag = np.isclose(train_loss, self.prev_train_loss,
                                       rtol=self.rtol, atol= self.atol)
        
        rel_change = np.abs(train_loss-self.prev_train_loss)/np.abs(self.prev_train_loss)
        print(f'Rel change in train loss: {rel_change}')
        
        if convergence_flag:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0

        if val_loss is not None:
            # if train_loss < self.prev_train_loss and self.counter < 3: 
            # if train_loss < self.prev_train_loss and val_loss < self.val_loss_min:
            if val_loss < self.val_loss_min:
                self.save_checkpoint(train_loss, val_loss, model, model_optim, path)
                self.val_loss_min = val_loss
                self.train_loss_optimal = train_loss
            self.prev_val_loss   = val_loss
        else:
            # if train_loss < self.prev_train_loss and self.counter < 3: 
            if train_loss < self.prev_train_loss:
                self.save_checkpoint(train_loss, model, model_optim, path)
                self.train_loss_optimal = train_loss

        self.prev_train_loss = train_loss

    def save_checkpoint(self, train_loss, val_loss, model, model_optim, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        checkpoint = {  'train_loss':train_loss,
                        'val_loss':val_loss,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': model_optim.state_dict()
                        }
        # checkpoint = { 'model_state_dict': model.state_dict()}
        torch.save(checkpoint, path + '/' + 'checkpoint.pth')


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf', title=None, multiple_preds=None):
    """
    Results visualization
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(true, label='GroundTruth', linewidth=1)
    if preds is not None:
        ax.plot(preds, label='Prediction', linewidth=1)
    ax.grid(axis='x', which='major', linestyle='--', linewidth=0.2)
    if multiple_preds is not None:
        for j in range(multiple_preds.shape[1]):
            ax.plot(multiple_preds[:,j],linestyle=':' )
    ax.legend()
    ax.set_title(f'{title}')
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight')
    return fig, ax

def visual_all(trues, preds=None, name='./pic/test.pdf'):
    """
    Training Results visualization


    """
    seq_length = trues.shape[1]
    batch_no   = trues.shape[0]
    preds_      = preds[::seq_length,:,0].reshape(-1)
    trues_      = trues[::seq_length,:,0].reshape(-1)
    if preds_.shape[0] > batch_no:
        forecast_inds = preds_.shape[0] - batch_no
        preds       = np.concatenate((preds_, preds[-1,forecast_inds+1:,0]))
        trues       = np.concatenate((trues_, trues[-1,forecast_inds+1:,0]))
    else:
        preds       = preds_
        trues       = trues_
    # fig_width = min(20, len(preds) / seq_length)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(trues, label='GroundTruth', linewidth=1)
    if preds is not None:
        ax.plot(preds, label='Prediction', linewidth=1)
    # Adding grid lines
    # Generate tick positions based on seq_length
    # tick_positions = range(0, len(trues), seq_length)
    # # Set major ticks at every seq_length
    # ax.xticks(tick_positions)
    # Enable grid only for x-axis, aligning with each seq_length
    ax.grid(axis='x', which='major', linestyle='--', linewidth=0.2)
    ax.legend()
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight')
    return fig, ax

def club_axes(ax_train, ax_test, ax_val=None, name='./pic/test.pdf', path=None):

    x_train, y_train_gt   = ax_train.get_lines()[0].get_data()
    x_train, y_train_pred = ax_train.get_lines()[1].get_data()
    len_train             = len(x_train)

    if ax_val is not None:
        x_val, y_val_gt       = ax_val.get_lines()[0].get_data()
        x_val, y_val_pred     = ax_val.get_lines()[1].get_data()
        len_val               = len(x_val) 
    else:
        len_val               = 0
    val_idx_start         = len_train

    x_test, y_test_gt     = ax_test.get_lines()[0].get_data()
    x_test, y_test_pred   = ax_test.get_lines()[1].get_data()
    len_test              = len(x_test) 
    test_idx_start         = len_train + len_val

    fig, ax = plt.subplots(figsize=(10, 6))
    if ax_val is not None:
        ax.plot(np.concatenate((y_train_gt, y_val_gt, y_test_gt)), label='GroundTruth', linewidth=1, color='black')
    else:
        gt_data = np.concatenate((y_train_gt, y_test_gt))
        ax.plot(gt_data, label='GroundTruth', linewidth=1, color='black')
    ax.plot(np.arange(len_train), y_train_pred, linewidth=1, color='green')
    if ax_val is not None:
        ax.plot(np.arange(val_idx_start, test_idx_start), y_val_pred, linewidth=1, color='blue')
    test_idx   = np.arange(test_idx_start, test_idx_start+len_test)
    ax.plot(test_idx, y_test_pred, linewidth=1, color='red')
    ax.grid(axis='x', which='major', linestyle='--', linewidth=0.2)
    ax.legend()
    plt.tight_layout()
    if ax_val is None and path is not None:
        np.savez(os.path.join(path, 'pred_with_out_val.npz'),
                 gt_data = gt_data, test_idx=test_idx, y_test_pred=y_test_pred  )
    fig.savefig(name, format='pdf', dpi=1000, bbox_inches="tight")

def plot_train_val_test_together(train_results, test_results, val_results=None, name='./pic/test.pdf'):

    y_train_gt          = train_results['truth']
    if val_results is not None:
        y_val_gt            = val_results['truth']
    y_test_gt           = test_results['truth']

    # y_train_1step_pred  = train_results['predicted_forecast']
    # y_val_1step_pred    = val_results['predicted_forecast']
    # y_test_1step_pred   = test_results['predicted_forecast']

    y_train_1step_pred  = np.nanmean(train_results['all_predictions'], axis=1).reshape(-1)
    if val_results is not None:
        y_val_1step_pred    = np.nanmean(val_results['all_predictions'], axis=1).reshape(-1)
    y_test_1step_pred   = np.nanmean(test_results['all_predictions'], axis=1).reshape(-1)

    len_train           = len(y_train_gt)
    train_idx           = np.arange(len_train)
    val_idx_start       = len_train
    len_test            = len(y_test_gt)
    if val_results is not None:
        len_val         = len(y_val_gt)
    else:
        len_val         = 0
    test_idx_start      = len_train + len_val
    val_idx             = np.arange(val_idx_start, test_idx_start)
    test_idx            = np.arange(test_idx_start, test_idx_start+len_test)
    


    fig, ax = plt.subplots(figsize=(10, 6))
    # plot ground truth
    if val_results is not None:
        ax.plot(np.concatenate((y_train_gt, y_val_gt, y_test_gt)), label='GroundTruth', linewidth=1, color='black')
    else:
        ax.plot(np.concatenate((y_train_gt, y_test_gt)), label='GroundTruth', linewidth=1, color='black')
    # plot only the t+1 predictions
    ax.plot(train_idx, y_train_1step_pred,linestyle='-.', linewidth=1, color='green')
    if val_results is not None:
        ax.plot(val_idx, y_val_1step_pred,linestyle='-.', linewidth=1, color='blue')
    ax.plot(test_idx, y_test_1step_pred,linestyle='-.', linewidth=1, color='red')
    # plot all the predictions in the forecast (upto pred len)
    # multiple_preds = train_results['all_predictions']
    # for i in range(multiple_preds.shape[0]):
    #     ax.scatter((train_idx[i],)*multiple_preds.shape[1], multiple_preds[i,:],s=0.2 )
    # multiple_preds = val_results['all_predictions']
    # for i in range(multiple_preds.shape[0]):
    #     ax.scatter((val_idx[i],)*multiple_preds.shape[1], multiple_preds[i,:],s=0.2 )
    # multiple_preds = test_results['all_predictions']
    # for i in range(multiple_preds.shape[0]):
    #     ax.scatter((test_idx[i],)*multiple_preds.shape[1], multiple_preds[i,:],s=0.2 )
    
    # plot all the predictions in the forecast (upto pred len)
    multiple_preds = train_results['all_predictions']
    for j in range(multiple_preds.shape[1]):
        ax.plot(train_idx, multiple_preds[:,j], linestyle=':', linewidth=0.5)
    if val_results is not None:
        multiple_preds = val_results['all_predictions']
        for j in range(multiple_preds.shape[1]):
            ax.plot(val_idx, multiple_preds[:,j], linestyle=':', linewidth=0.5)
    multiple_preds = test_results['all_predictions']
    for j in range(multiple_preds.shape[1]):
        ax.plot(test_idx, multiple_preds[:,j], linestyle=':', linewidth=0.5)
    
    ax.grid(axis='x', which='major', linestyle='--', linewidth=0.2)
    ax.legend()
    plt.tight_layout()
    fig.savefig(name, format='pdf', dpi=1000, bbox_inches="tight")

    return fig


def plot_idx(input, pred, tgt, idx, file_name):
    
    len_input = len(input)
    tgt = np.concatenate([input, tgt], axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(len_input, len(tgt)), pred, 'b*-',  label='Pred', linewidth=1)
    ax.plot(tgt,'k^-', label='Target', linewidth=1)
    ax.axvline(x=len_input, color='r', linestyle='--')
    ax.legend()
    tick_positions = list(range(0, len(tgt), 2))
    ax.set_xticks(tick_positions)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f'Predictions of idx {idx}')
    plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred

def update_results_dict(train_results, val_results, test_results, output_metrics_final_model):
    
    output_metrics_final_model['train_mse'].append(train_results['mse'])
    output_metrics_final_model['train_nmse'].append(train_results['nmse'])
    output_metrics_final_model['train_mae'].append(train_results['mae'])
    output_metrics_final_model['train_nmae'].append(train_results['nmae'])

    if val_results is not None:
        output_metrics_final_model['val_mse'].append(val_results['mse'])
        output_metrics_final_model['val_nmse'].append(val_results['nmse'])
        output_metrics_final_model['val_mae'].append(val_results['mae'])
        output_metrics_final_model['val_nmae'].append(val_results['nmae'])

    output_metrics_final_model['test_mse'].append(test_results['mse'])
    output_metrics_final_model['test_nmse'].append(test_results['nmse'])
    output_metrics_final_model['test_mae'].append(test_results['mae'])
    output_metrics_final_model['test_nmae'].append(test_results['nmae'])

    return output_metrics_final_model



def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

class FixedNumpySeed(object):
    def __init__(self, seed):
        self.seed = seed
    def __enter__(self):
        self.np_rng_state = np.random.get_state()
        np.random.seed(self.seed)
        self.rand_rng_state = random.getstate()
        random.seed(self.seed)
    def __exit__(self, *args):
        np.random.set_state(self.np_rng_state)
        random.setstate(self.rand_rng_state)

class NoGetItLambdaDict(dict):
    """ Regular dict, but refuses to __getitem__ pretending
        the element is not there and throws a KeyError
        if the value is a non string iterable or a lambda """
    def __init__(self,d={}):
        super().__init__()
        for k,v in d.items():
            if isinstance(v,dict):
                self[k] = NoGetItLambdaDict(v)
            else:
                self[k] = v
    def __getitem__(self, key):
        value = super().__getitem__(key)
        if callable(value) and value.__name__ == "<lambda>":
            raise LookupError("You shouldn't try to retrieve lambda {} from this dict".format(value))
        if isinstance(value,Iterable) and not isinstance(value,(str,bytes,dict,tuple)):
            raise LookupError("You shouldn't try to retrieve iterable {} from this dict".format(value))
        return value

def sample_config(config_spec):
    """ Generates configs from the config spec.
        It will apply lambdas that depend on the config and sample from any
        iterables, make sure that no elements in the generated config are meant to 
        be iterable or lambdas, strings are allowed."""
    cfg_all = config_spec
    more_work=True
    i=0
    while more_work:
        cfg_all, more_work = _sample_config(cfg_all,NoGetItLambdaDict(cfg_all))
        i+=1
        if i>10: 
            raise RecursionError("config dependency unresolvable with {}".format(cfg_all))
    out = defaultdict(dict)
    out.update(cfg_all)
    return out

def _sample_config(config_spec,cfg_all):
    cfg = {}
    more_work = False
    for k,v in config_spec.items():
        if isinstance(v,dict):
            new_dict,extra_work = _sample_config(v,cfg_all)
            cfg[k] = new_dict
            more_work |= extra_work
        elif isinstance(v,Iterable) and not isinstance(v,(str,bytes,dict,tuple)):
            cfg[k] = random.choice(v)
        elif callable(v) and v.__name__ == "<lambda>":
            try:cfg[k] = v(cfg_all)
            except (KeyError, LookupError,Exception):
                cfg[k] = v # is used isntead of the variable it returns
                more_work = True
        else: cfg[k] = v
    return cfg, more_work

def flatten(d, parent_key='', sep='/'):
    """An invertible dictionary flattening operation that does not clobber objs"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict) and v: # non-empty dict
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten(d,sep='/'):
    """Take a dictionary with keys {'k1/k2/k3':v} to {'k1':{'k2':{'k3':v}}}
        as outputted by flatten """
    out_dict={}
    for k,v in d.items():
        if isinstance(k,str):
            keys = k.split(sep)
            dict_to_modify = out_dict
            for partial_key in keys[:-1]:
                try: dict_to_modify = dict_to_modify[partial_key]
                except KeyError:
                    dict_to_modify[partial_key] = {}
                    dict_to_modify = dict_to_modify[partial_key]
                # Base level reached
            if keys[-1] in dict_to_modify:
                dict_to_modify[keys[-1]].update(v)
            else:
                dict_to_modify[keys[-1]] = v
        else: out_dict[k]=v
    return out_dict

class grid_iter(object):
    """ Defines a length which corresponds to one full pass through the grid
        defined by grid variables in config_spec, but the iterator will continue iterating
        past that by repeating over the grid variables"""
    def __init__(self,config_spec,num_elements=-1,shuffle=True):
        self.cfg_flat = flatten(config_spec)
        is_grid_iterable = lambda v: (isinstance(v,Iterable) and not isinstance(v,(str,bytes,dict,tuple)))
        iterables = sorted({k:v for k,v in self.cfg_flat.items() if is_grid_iterable(v)}.items())
        if iterables: self.iter_keys,self.iter_vals = zip(*iterables)
        else: self.iter_keys,self.iter_vals = [],[[]]
        self.vals = list(itertools.product(*self.iter_vals))
        if shuffle:
            with FixedNumpySeed(0): random.shuffle(self.vals)
        self.num_elements = num_elements if num_elements>=0 else (-1*num_elements)*len(self)

    def __iter__(self):
        self.i=0
        self.vals_iter = iter(self.vals)
        return self
    def __next__(self):
        self.i+=1
        if self.i > self.num_elements: raise StopIteration
        if not self.vals: v = []
        else:
            try: v = next(self.vals_iter)
            except StopIteration:
                self.vals_iter = iter(self.vals)
                v = next(self.vals_iter)
        chosen_iter_params = dict(zip(self.iter_keys,v))
        self.cfg_flat.update(chosen_iter_params)
        return sample_config(unflatten(self.cfg_flat))
    def __len__(self):
        product = functools.partial(functools.reduce, operator.mul)
        return product(len(v) for v in self.iter_vals) if self.vals else 1
    


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

