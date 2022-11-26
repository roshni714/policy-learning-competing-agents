import itertools
import glob
import os


SBATCH_PREFACE = """#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 1
#SBATCH -p long
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="{}.sh"
#SBATCH --error="{}/{}_err.log"
#SBATCH --output="{}/{}_out.log"\n
"""

# constants for commands

OUTPUT_PATH = "/zfs/gsb/intermediate-yens/rsahoo/policy-learning-competing-agents/scripts"
SAVE_PATH = "/zfs/gsb/intermediate-yens/rsahoo/policy-learning-competing-agents/results"


def generate_nels():
    seeds = list(range(9))
    max_iters = 2
    loss_types = ["months_attended", "etas"]
    gradient_types = ["expected_gradient_beta_naive", "total_deriv", "partial_deriv_loss_beta"]

    for seed in seeds:
        exp_id = "nels_11-16-22"
        script_fn = os.path.join(OUTPUT_PATH, "{}_seed_{}.sh".format(exp_id, seed))
        with open(script_fn, "w") as f:
            print(SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id),file=f)

            for loss_type in loss_types:
                for gradient_type in gradient_types:
                    base_cmd = "python /zfs/gsb/intermediate-yens/rsahoo/policy-learning-competing-agents/train_beta.py main --nels "
                    new_cmd = (base_cmd
                        + "--n 1000000 --d 9 --perturbation_s 0.2 --perturbation_beta 0.025 --learning_rate 0.1 --max_iter 30 --save_dir {} --save nels_{}_11-16-22 --seed {} --gradient_type {}  --loss_type {}\n".format(SAVE_PATH, loss_type, seed, gradient_type, loss_type))
                    print(new_cmd, file=f)
                    print("sleep 1", file=f)

generate_nels()
