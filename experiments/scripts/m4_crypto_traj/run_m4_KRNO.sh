#!/bin/bash

######## M4 experiments
# Run hyperparam tuning on M4 datasets

# python  scripts/m4_crypto_traj/run_KRNO_hyper_param_tuning.py --tmp_storage True --model KRNO --data m4 --root_path dataset/M4_KNF/ --seasonal_patterns Weekly --batch_size 512 --num_workers 1 --max_subprocesses 4

# python   scripts/m4_crypto_traj/run_KRNO_hyper_param_tuning.py --tmp_storage True --model KRNO --data m4 --root_path dataset/M4_KNF/ --seasonal_patterns Monthly --batch_size 512 --num_workers 1 --max_subprocesses 6

# python  scripts/m4_crypto_traj/run_KRNO_hyper_param_tuning.py --tmp_storage True --model KRNO --data m4 --root_path dataset/M4_KNF/ --seasonal_patterns Daily --batch_size 512 --num_workers 1 --max_subprocesses 4

# python  scripts/m4_crypto_traj/run_KRNO_hyper_param_tuning.py --tmp_storage True --model KRNO --data m4 --root_path dataset/M4_KNF/ --seasonal_patterns Hourly --batch_size 512 --num_workers 1 --max_subprocesses 4

# python  scripts/m4_crypto_traj/run_KRNO_hyper_param_tuning.py --tmp_storage True --model KRNO --data m4 --root_path dataset/M4_KNF/ --seasonal_patterns Quarterly --batch_size 512 --num_workers 2 --max_subprocesses 5

# python  scripts/m4_crypto_traj/run_KRNO_hyper_param_tuning.py --tmp_storage True --model KRNO --data m4 --root_path dataset/M4_KNF/ --seasonal_patterns Yearly --batch_size 512 --num_workers 2  --max_subprocesses 6


# Run M4 datasets

## Weekly data (SOTA)

# seed 2021: results from best model (6.068, 7.943, 6.821, 6.934)
# seed 2023: results from best model (5.553, 8.114, 7.337, 7.027)
# seed 2025: results from best model (5.756, 7.859, 6.915, 6.849)
# seed 2030: results from best model (5.799, 7.802, 6.823, 6.809) 
## average 6.90475
python  scripts/m4_crypto_traj/run_exp.py --data m4 --root_path dataset/M4_KNF/ --seasonal_patterns Weekly --batch_size 512 --seq_len 60 --pred_len 13 --loss_pred_len 13 --use_revin True --revin_affine True --learning_rate 0.001 --lifting_channels 128 --width 32 --hidden_units 64 --include_affine True --use_revin True --revin_affine True --train_epochs 30 --seed 2021 --jumps 3

## Daily data (SOTA)

# seed 2021: results from best model (1.716, 2.863, 4.266, 3.088)
# seed 2023: results from best model (1.72, 2.851, 4.253, 3.079)
# seed 2025: results from best model (1.712, 2.849, 4.254, 3.077)
# seed 2030: results from best model  (1.7, 2.831, 4.239, 3.062)
## average 3.0765
python  scripts/m4_crypto_traj/run_exp.py --data m4 --root_path dataset/M4_KNF/ --seasonal_patterns Daily --batch_size 512 --seq_len 45 --pred_len 14 --loss_pred_len 14 --use_revin True --revin_affine True --learning_rate 0.001 --lifting_channels 128 --width 8 --hidden_units 32 --include_affine True --train_epochs 30 --seed 2021 --jumps 5

## Monthly data 

python  scripts/m4_crypto_traj/run_exp.py --data m4 --root_path dataset/M4_KNF/ --seasonal_patterns Monthly --batch_size 1024 --num_workers 10 --seq_len 36 --pred_len 36 --loss_pred_len 18 --modes 18

echo 'done running experiments!'

