#!/bin/bash

rm -rf /zfs/gsb/intermediate-yens/rsahoo/policy-learning-competing-agents/scripts
mkdir /zfs/gsb/intermediate-yens/rsahoo/policy-learning-competing-agents/scripts
python generate_sbatches.py 
 
 
for experiment in /zfs/gsb/intermediate-yens/rsahoo/policy-learning-competing-agents/scripts/*.sh
do
    echo $experiment
    chmod u+x $experiment
    sbatch $experiment
#    $experiment
    sleep 1
done

echo "Done"
