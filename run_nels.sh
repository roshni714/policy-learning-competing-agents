#!/bin/bash
seeds=(0 1 2 3 4 5 6 7 8 9)
MAX_ITER=30

### High Dimensional Experiment
for SEED in "${seeds[@]}";
do
    krenew -t -- python train_beta.py main --nels --n 1000000 --d 9 --perturbation_s 0.2 --perturbation_beta 0.025 --learning_rate 0.1 --max_iter $MAX_ITER --save nels_10-27-22 --seed $SEED --gradient_type "total_deriv"
    krenew -t -- python train_beta.py main --nels --n 1000000 --d 9 --perturbation_s 0.2 --perturbation_beta 0.025 --learning_rate 0.1 --max_iter $MAX_ITER --save nels_10-27-22 --seed $SEED --gradient_type "partial_deriv_loss_beta"
done




