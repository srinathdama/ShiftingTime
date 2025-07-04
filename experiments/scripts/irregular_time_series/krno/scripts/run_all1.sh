#!/bin/bash

## activity
echo "Running activity"


### KRNO ###
patience=10
gpu=0

for seed in {1..5}
do

    echo $seed

    # python run_models.py \
    python run_models_new.py \
    --dataset activity --state 'def' --history 3000 \
    --patience $patience --batch_size 1 --lr 1e-3 \
    --seed $seed --gpu $gpu \
    --int_channels 20 --accumulation_steps 16
done
