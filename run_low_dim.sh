#!/bin/bash
seeds=(0 1 2 3 4 5 6 7 8 9)
MAX_ITER=100

### Toy Example Experiment
for SEED in "${seeds[@]}";
do
    krenew -t -- python train.py main --n 1000000 --learning_rate 0.2 --perturbation_theta 0.025 --max_iter $MAX_ITER --n_types 10 --save low_dim_11-15-22 --seed $SEED --gradient_type naive_expected_total_deriv
    krenew -t -- python train.py main --n 1000000 --perturbation_s 0.2 --perturbation_theta 0.025 --learning_rate 0.2 --max_iter $MAX_ITER --n_types 10 --save low_dim_11-15-22 --seed $SEED --gradient_type total_deriv
    krenew -t -- python train.py main --n 1000000 --perturbation_s 0.2 --perturbation_theta 0.025 --learning_rate 0.2 --max_iter $MAX_ITER --n_types 10 --save low_dim_11-15-22 --seed $SEED --gradient_type partial_deriv_loss_theta
done




