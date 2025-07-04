#!/bin/bash

echo "Running KRNO on Elasticity data"

for i in 0 1 2 3 4
do
    nohup python train_script_krno.py --seed $i > nohup_kron_seed${i}.out 2>&1 
done