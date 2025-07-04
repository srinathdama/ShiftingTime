#!/bin/bash

######## Traj
# Run hyperparam tuning

# python scripts/m4_crypto_traj/run_KRNO_hyper_param_tuning.py --tmp_storage True --model KRNO --data Traj --root_path dataset/PlayerTraj/   --batch_size 512 --num_workers 2 --max_subprocesses 6


# Run Player Traj dataset

# seed 2021: results from best model (0.2684916292321474, 0.9264927846237809, 1.9496678256277344, 1.2558770196182478)
python scripts/m4_crypto_traj/run_exp.py --data Traj --root_path dataset/PlayerTraj/  --batch_size 512 --seq_len 30 --pred_len 15 --loss_pred_len 15 --use_revin True --revin_affine True --learning_rate 0.001 --lifting_channels 128 --width 32 --hidden_units 64 --include_affine True --use_revin True --revin_affine True --train_epochs 30 --seed 2021 --jumps 2 --run_id 0

# seed 2024: results from best model (0.30513452548360553, 0.9963021577616562, 2.0595829153172507, 1.3326173310781235)
python scripts/m4_crypto_traj/run_exp.py --data Traj --root_path dataset/PlayerTraj/  --batch_size 512 --seq_len 30 --pred_len 15 --loss_pred_len 15 --use_revin True --revin_affine True --learning_rate 0.001 --lifting_channels 128 --width 32 --hidden_units 64 --include_affine True --use_revin True --revin_affine True --train_epochs 30 --seed 2024 --jumps 2 --run_id 1

# seed 2025: results from best model (0.263408893154711, 0.9476585480328877, 2.013839980000883, 1.2939593545526262)
python scripts/m4_crypto_traj/run_exp.py --data Traj --root_path dataset/PlayerTraj/  --batch_size 512 --seq_len 30 --pred_len 15 --loss_pred_len 15 --use_revin True --revin_affine True --learning_rate 0.001 --lifting_channels 128 --width 32 --hidden_units 64 --include_affine True --use_revin True --revin_affine True --train_epochs 30 --seed 2025 --jumps 2 --run_id 2

# seed 2027: results from best model (0.24852085087841114, 0.9102258023059675, 1.946170284950793, 1.2487129911139607)
python scripts/m4_crypto_traj/run_exp.py --data Traj --root_path dataset/PlayerTraj/  --batch_size 512 --seq_len 30 --pred_len 15 --loss_pred_len 15 --use_revin True --revin_affine True --learning_rate 0.001 --lifting_channels 128 --width 32 --hidden_units 64 --include_affine True --use_revin True --revin_affine True --train_epochs 30 --seed 2027 --jumps 2 --run_id 3

# Player Traj. RMSE: (0.27138897 ± 0.03374555, 0.94516982 ± 0.05113233, 1.99231525 ± 0.06726766, 1.28279167 ± 0.04982566)

