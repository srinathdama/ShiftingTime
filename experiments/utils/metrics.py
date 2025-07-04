# MIT License
#
# Original work Copyright (c) 2021 THUML @ Tsinghua University
# Modified work Copyright (c) 2025 DACElab @ University of Toronto
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
# from jax import vmap
# import jax.numpy as jnp

def quantile_loss(target, pred, q):
    q_pred = jnp.quantile(pred, q, axis=0)
    return 2 * jnp.sum(
        jnp.abs((q_pred - target) * ((target <= q_pred) * 1.0 - q))
    )

def calculate_crps(target, pred, num_quantiles=20):
    quantiles = jnp.linspace(0, 1.0, num_quantiles+1)[1:]
    vec_quantile_loss = vmap(lambda q: quantile_loss(target, pred, q))
    crps = jnp.sum(vec_quantile_loss(quantiles))
    crps = crps / (jnp.sum(np.abs(target)) * len(quantiles))
    return crps

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


class Evaluator:

    def __init__(self):
        self.non_numerical_cols = [
            "serialized_history",
            "serialized_target",
            "serialized_prediction",
            "history_len",
            "num_channels",
            "example_num",
            "sample_num",
        ]

    def evaluate_df(self, gt_df, pred_df):
        cols = [c for c in gt_df.columns if c not in self.non_numerical_cols]
        num_channels = gt_df["num_channels"].iloc[0]
        history_len = gt_df["history_len"].iloc[0]
        gt_vals = gt_df[cols].to_numpy().reshape(len(gt_df), -1, num_channels) # (num_examples, history_len + target_len, num_channels)
        gt_vals = gt_vals[:, history_len:, :] # (num_examples, target_len, num_channels)
        
        cols = [c for c in pred_df.columns if c not in self.non_numerical_cols]
        num_channels = pred_df["num_channels"].iloc[0]
        pred_df = pred_df[cols + ["example_num"]]
        
        all_pred_vals = []
        for example_num in sorted(pred_df["example_num"].unique()):
            pred_vals = pred_df[pred_df["example_num"] == example_num][cols].to_numpy() # (num_samples, target_len * num_channels)
            pred_vals = pred_vals.reshape(pred_vals.shape[0], -1, num_channels) # (num_samples, target_len, num_channels)
            all_pred_vals.append(pred_vals)
           
        pred_vals = np.stack(all_pred_vals, axis=1) # (num_samples, num_examples, target_len, num_channels)
        assert gt_vals.shape == pred_vals.shape[1:]
        
        diff = (gt_vals[None] - pred_vals) # (num_samples, num_examples, target_len, num_channels)
        mse = np.mean(diff**2)
        mae = np.mean(np.abs(diff))
        crps = calculate_crps(gt_vals, pred_vals)

        return {
            "mse": mse,
            "mae": mae,
            "crps": crps,
        }
    
    def evaluate(self, gt, pred):
        ''' 
        gt: (batch_size, steps)
        pred: (batch_size, num_samples, steps)
        '''
        assert gt.shape == (pred.shape[0], pred.shape[2]), f"wrong shapes: gt.shape: {gt.shape}, pred.shape: {pred.shape}"
        diff = (gt[:, None, :] - pred) # (batch_size, num_samples, steps)
        mse = np.mean(diff**2)
        mae = np.mean(np.abs(diff))
        # std = np.std(gt, axis=1) + 1e-8 # (batch_size,)
        # normlized_diff = diff / std[:, None, None] # (batch_size, num_samples, steps)
        # nmse = np.mean(normlized_diff**2)
        # nmae = np.mean(np.abs(normlized_diff))
        nmse = mse/np.mean(gt**2)
        nmae = mae/np.mean(np.abs(gt))

        return {
            "nmse": nmse,
            "nmae": nmae,
            "mse": mse,
            "mae": mae,
        }
    

def process_testing_data(args, test_preds, test_tgts, test_set):
    # Denormalize the predictions
    if args.data == "m4" or args.data == "mini":
        test_preds = (test_preds * test_set.ts_stds.reshape(
            -1, 1, 1)) + test_set.ts_means.reshape(-1, 1, 1)
        test_tgts = (test_tgts * test_set.ts_stds.reshape(
            -1, 1, 1)) + test_set.ts_means.reshape(-1, 1, 1)

    elif args.data == "Cryptos":
        test_preds = test_preds.reshape(
            14, -1, args.test_output_length,
            8)  # 14 stocks x num_samples x #steps x 8 features
        test_tgts = test_tgts.reshape(14, -1, args.test_output_length, 8)
        stds = np.expand_dims(test_set.ts_stds, axis=(1, 2))
        means = np.expand_dims(test_set.ts_means, axis=(1, 2))
        test_preds = test_preds * stds + means
        test_tgts = test_tgts * stds + means

    else:
        stds = np.expand_dims(test_set.ts_stds, axis=(1, 2))
        means = np.expand_dims(test_set.ts_means, axis=(1, 2))

        test_preds = test_preds.reshape(len(means), -1, args.test_output_length, 2)
        test_tgts = test_tgts.reshape(len(means), -1, args.test_output_length, 2)

        test_preds = test_preds * stds + means
        test_tgts = test_tgts * stds + means
    
    return test_preds, test_tgts