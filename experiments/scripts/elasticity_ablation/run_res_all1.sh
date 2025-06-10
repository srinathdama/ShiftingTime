#!/bin/bash

echo "Running ablation study on Elasticity data"
for res in 16 24 32 40 48
do
    for i in 0 1 2 3 4
    do
        nohup python train_script_krno.py --res $res --seed $i > nohup_kron_S_${res}_seed${i}.out 2>&1 
    done
done