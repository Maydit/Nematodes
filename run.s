#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=wormcounter.out
#SBATCH --error=wormcounter.err
#SBATCH --mem=16GB

module purge;
source ./venv/bin/activate;
python wormcounter.py $1 $2 $3
