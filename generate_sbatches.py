import itertools
import glob
import os


SBATCH_PREFACE = """#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -c 1
#SBATCH --mem 10GB
#SBATCH -p normal
#SBATCH --exclude=yen11
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="{}.sh"
#SBATCH --error="{}/{}_err.log"
#SBATCH --output="{}/{}_out.log"\n
"""

# constants for commands

OUTPUT_PATH = (
    "/zfs/gsb/intermediate-yens/rsahoo/policy-learning-competing-agents/scripts"
)
SAVE_PATH = "/zfs/gsb/intermediate-yens/rsahoo/policy-learning-competing-agents/results"


def generate_nels():
    seeds = list(range(10))
    loss_types = ["socio_econ", "etas", "months_attended"]
    methods = ["total_deriv", "partial_deriv_loss_beta", "ewm"]
    for seed in seeds:
        exp_id = "nels_4-2-24_seed_{}".format(seed)
        script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
        with open(script_fn, "w") as f:
            print(
                SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id),
                file=f,
            )

            for loss_type in loss_types:
                for method in methods:
                    base_cmd = "python /zfs/gsb/intermediate-yens/rsahoo/policy-learning-competing-agents/train_beta.py main --nels "
                    new_cmd = (
                        base_cmd
                        + "--n 1000000 --d 9 --perturbation_s 0.2 --perturbation_beta 0.025 --learning_rate 0.1 --max_iter 50 --save_dir {} --save nels_{}_4-2-24 --seed {} --method {}  --loss_type {}\n".format(
                            SAVE_PATH, loss_type, seed, method, loss_type
                        )
                    )
                    print(new_cmd, file=f)
                    print("sleep 1", file=f)


def generate_nels_small():
    seeds = list(range(10))
    loss_types = ["months_attended", "etas", "socio_econ"]
    methods = ["total_deriv", "partial_deriv_loss_beta", "ewm"]
    for seed in seeds:
        exp_id = "nels_small_4-2-24_seed_{}".format(seed)
        script_fn = os.path.join(OUTPUT_PATH, "{}.sh".format(exp_id))
        with open(script_fn, "w") as f:
            print(
                SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id),
                file=f,
            )

            for loss_type in loss_types:
                for method in methods:
                    base_cmd = "python /zfs/gsb/intermediate-yens/rsahoo/policy-learning-competing-agents/train_beta.py main --nels "
                    new_cmd = (
                        base_cmd
                        + "--n 14915 --d 9 --perturbation_s 0.2 --perturbation_beta 0.025 --learning_rate 0.025 --max_iter 50 --save_dir {} --save nels_small_{}_4-2-24 --seed {} --method {}  --loss_type {}\n".format(
                            SAVE_PATH, loss_type, seed, method, loss_type
                        )
                    )
                    print(new_cmd, file=f)
                    print("sleep 1", file=f)


def generate_high_dim():
    seeds = list(range(10))
    methods = ["ewm", "total_deriv", "partial_deriv_loss_beta"]

    for seed in seeds:
        exp_id = "high_dim_12-13-23"
        script_fn = os.path.join(OUTPUT_PATH, "{}_seed_{}.sh".format(exp_id, seed))
        with open(script_fn, "w") as f:
            print(
                SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id),
                file=f,
            )

            for method in methods:
                base_cmd = "python /zfs/gsb/intermediate-yens/rsahoo/policy-learning-competing-agents/train_beta.py main "
                new_cmd = (
                    base_cmd
                    + "--n 1000000 --d 10 --n_types 10 --perturbation_s 0.2 --perturbation_beta 0.025 --learning_rate 0.5 --max_iter 50 --save_dir {} --save high_dim_12-13-23 --seed {} --method {}\n".format(
                        SAVE_PATH, seed, method
                    )
                )
                print(new_cmd, file=f)
                print("sleep 1", file=f)


def generate_low_dim():
    seeds = list(range(10))

    for seed in seeds:
        exp_id = "low_dim_12-13-23"
        script_fn = os.path.join(OUTPUT_PATH, "{}_seed_{}.sh".format(exp_id, seed))

        methods = ["ewm", "total_deriv", "partial_deriv_loss_theta"]
        with open(script_fn, "w") as f:
            print(
                SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id),
                file=f,
            )

            for method in methods:
                base_cmd = "python /zfs/gsb/intermediate-yens/rsahoo/policy-learning-competing-agents/train.py main "
                new_cmd = (
                    base_cmd
                    + "--n 1000000 --n_types 10 --perturbation_s 0.2 --perturbation_theta 0.025 --learning_rate 0.5 --max_iter 100 --save_dir {} --save low_dim_11-25-22 --seed {} --method {}\n".format(
                        SAVE_PATH, seed, method
                    )
                )
                print(new_cmd, file=f)
                print("sleep 1", file=f)


generate_nels()
generate_nels_small()
# generate_low_dim()
# generate_high_dim()
