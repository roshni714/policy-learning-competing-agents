#!/bin/bash
seeds=(8 9)
MAX_ITER=100

### Toy Example Experiment
for SEED in "${seeds[@]}";
do
    krenew -t -- python train.py main --n 1000000 --perturbation_s 0.2 --perturbation_theta 0.025 --learning_rate 1. --max_iter $MAX_ITER --n_types 10 --save low_dim_7-19-22 --seed $SEED --gradient_type total_deriv
    krenew -t -- python train.py main --n 1000000 --perturbation_s 0.2 --perturbation_theta 0.025 --learning_rate 0.25 --max_iter $MAX_ITER --n_types 10 --save low_dim_7-19-22 --seed $SEED --gradient_type partial_deriv_loss_theta
done




