#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB

module purge;
source ./venv/bin/activate;
python wormcounter.py $1 $2 $3
