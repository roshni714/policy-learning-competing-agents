#!/bin/bash

# Example of running python script in a batch mode

#SBATCH -J low_dim
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
    python train.py main --n 1000000 --learning_rate 0.5 --perturbation_theta 0.025 --max_iter $MAX_ITER --n_types 10 --save low_dim_11-15-22 --seed $SEED --gradient_type naive_expected_total_deriv
    python train.py main --n 1000000 --perturbation_s 0.2 --perturbation_theta 0.025 --learning_rate 0.5 --max_iter $MAX_ITER --n_types 10 --save low_dim_11-15-22 --seed $SEED --gradient_type total_deriv
    python train.py main --n 1000000 --perturbation_s 0.2 --perturbation_theta 0.025 --learning_rate 0.5 --max_iter $MAX_ITER --n_types 10 --save low_dim_11-15-22 --seed $SEED --gradient_type partial_deriv_loss_theta
done

