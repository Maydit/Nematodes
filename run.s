#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH --time=24:00:00

module purge;
source ./venv/bin/activate;
python wormcounter.py $1 $2 $3
