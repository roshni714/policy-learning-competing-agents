#!/bin/bash

# Example of running python script in a batch mode

#SBATCH -J high_dim
#SBATCH -p normal,dev
#SBATCH -c 1                            # CPU cores (up to 256 on normal partitioin)
#SBATCH -t 120:00
#SBATCH -o rollcount-%j.csv
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rsahoo@stanford.edu

# Run python script
seeds=(0 1 2 3 4 5 6 7 8 9)
MAX_ITER=100

### Toy Example Experiment
for SEED in "${seeds[@]}";
do
    python train_beta.py main --n 1000000 --d 10 --perturbation_s 0.2 --perturbation_beta 0.025 --learning_rate 0.5 --max_iter $MAX_ITER --n_types 10 --save high_dim_11-11-22 --seed $SEED --gradient_type "expected_gradient_beta_naive"
    python train_beta.py main --n 1000000 --d 10 --perturbation_s 0.2 --perturbation_beta 0.025 --learning_rate 0.5 --max_iter $MAX_ITER --n_types 10 --save high_dim_11-11-22 --seed $SEED --gradient_type "total_deriv"
    python train_beta.py main --n 1000000 --d 10 --perturbation_s 0.2 --perturbation_beta 0.025 --learning_rate 0.5 --max_iter $MAX_ITER --n_types 10 --save high_dim_11-11-22 --seed $SEED --gradient_type "partial_deriv_loss_beta"
done
