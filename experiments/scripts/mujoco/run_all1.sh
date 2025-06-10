#!/bin/bash

echo "Running KRNO on MuJoCo data"

missing_rate=0.0
lr=0.001
for i in 0 1 2 3 4
    do
        nohup python -u run_mujoco.py  --seed $i --lr $lr  --missing_rate $missing_rate --time_seq 50 --y_seq 10 --epoch 500 \
                --step_mode 'valloss' --model krno > nohup_missrate_${missing_rate}_lr_${lr}_seed_${i}.out 2>&1 

    done

i=0
nohup python -u run_mujoco.py  --seed $i --lr $lr  --missing_rate $missing_rate --time_seq 50 --y_seq 10 --epoch 500 \
                --step_mode 'valloss' --model krno > nohup_missrate_${missing_rate}_lr_${lr}_seed_${i}_test_time.out 2>&1 