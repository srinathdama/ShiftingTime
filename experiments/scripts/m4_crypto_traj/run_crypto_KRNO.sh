#!/bin/bash

######## Crypto
# Run hyperparam tuning on 

# python   scripts/m4_crypto_traj/run_KRNO_hyper_param_tuning.py --tmp_storage True --model KRNO --data Cryptos --root_path dataset/Cryptos/  --batch_size 512 --jumps 100 --num_workers 1 --max_subprocesses 1


# Run crypto dataset

# seed 2021: results from best model (0.005541750468231553, 0.007234551991288517, 0.007819525154677884, 0.007116036976529244)
python  scripts/m4_crypto_traj/run_exp.py --data Cryptos --root_path dataset/Cryptos/  --batch_size 512 --seq_len 80 --pred_len 15 --loss_pred_len 15 --use_revin True --revin_affine True --learning_rate 0.001 --lifting_channels 128 --width 32 --hidden_units 64 --include_affine True --use_revin True --revin_affine True --train_epochs 30 --seed 2021 --jumps 100 --run_id 0

# seed 2024: results from best model (0.005178846015949231, 0.006977673505826831, 0.007675954288566986, 0.0068796726766121585)
python  scripts/m4_crypto_traj/run_exp.py --data Cryptos --root_path dataset/Cryptos/  --batch_size 512 --seq_len 80 --pred_len 15 --loss_pred_len 15 --use_revin True --revin_affine True --learning_rate 0.001 --lifting_channels 128 --width 32 --hidden_units 64 --include_affine True --use_revin True --revin_affine True --train_epochs 30 --seed 2024 --jumps 100 --run_id 1

# seed 2025: results from best model (0.0052155901615299404, 0.007038914878429052, 0.007688453021619224, 0.006915296872561521)
python  scripts/m4_crypto_traj/run_exp.py --data Cryptos --root_path dataset/Cryptos/  --batch_size 512 --seq_len 80 --pred_len 15 --loss_pred_len 15 --use_revin True --revin_affine True --learning_rate 0.001 --lifting_channels 128 --width 32 --hidden_units 64 --include_affine True --use_revin True --revin_affine True --train_epochs 30 --seed 2025 --jumps 100 --run_id 2

# seed 2027: results from best model (0.0051585720893416315, 0.007025474993616176, 0.007681884455597184, 0.006906234688288656)
python  scripts/m4_crypto_traj/run_exp.py --data Cryptos --root_path dataset/Cryptos/  --batch_size 512 --seq_len 80 --pred_len 15 --loss_pred_len 15 --use_revin True --revin_affine True --learning_rate 0.001 --lifting_channels 128 --width 32 --hidden_units 64 --include_affine True --use_revin True --revin_affine True --train_epochs 30 --seed 2027 --jumps 100 --run_id 3

# Cryptos. Weighted RMSE: (0.00527369 ± 2.68060784e-04, 0.00706915 ± 1.65398149e-04, 0.00771645 ± 1.03070925e-04, 0.00695431 ± 1.61726673e-04)