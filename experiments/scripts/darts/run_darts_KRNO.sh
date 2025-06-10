#!/bin/bash

# Run hyperparam tuning on Dart datasets

python  run_KRNO_hyper_param_tuning.py --tmp_storage False  --model KRNO --data AirPassengers --root_path dataset/darts --data_path AirPassengers.csv --features S --max_subprocesses 6

python run_KRNO_hyper_param_tuning.py --tmp_storage False --model KRNO --data AusBeer --root_path dataset/darts --data_path ausbeer.csv --features S --target Y --max_subprocesses 6

python run_KRNO_hyper_param_tuning.py --tmp_storage False --model KRNO --data GasRateCO2 --root_path dataset/darts --data_path gasrate_co2.csv --features S --target CO2% --max_subprocesses 6

python  run_KRNO_hyper_param_tuning.py --tmp_storage False --model KRNO --data HeartRate --root_path dataset/darts --data_path heart_rate.csv --features S --max_subprocesses 8 #--batch_size 64

python  run_KRNO_hyper_param_tuning.py --tmp_storage False --model KRNO --data MonthlyMilk --root_path dataset/darts --data_path monthly-milk.csv --features S --max_subprocesses 5

python run_KRNO_hyper_param_tuning.py --tmp_storage False --model KRNO --data sunspots --root_path dataset/darts --data_path monthly-sunspots.csv --features S --target Sunspots --max_subprocesses 8 #--batch_size 48

python  run_KRNO_hyper_param_tuning.py --tmp_storage False --model KRNO --data Wine --root_path dataset/darts --data_path wineind.csv --features S --target Y --max_subprocesses 4

python  run_KRNO_hyper_param_tuning.py --tmp_storage False --model KRNO --data Wooly --root_path dataset/darts --data_path woolyrnq.csv --features S --target Y --max_subprocesses 4

echo 'done running experiments!'

