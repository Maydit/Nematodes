#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=wormcounter.out
#SBATCH --error=wormcounter.err
#SBATCH --partition=v100
#SBATCH --mem=64GB

module purge;
source ../venv/bin/activate;
python wormcounter.py ./img/"Sample Image.tiff" ./img/SI.csv --verbose
