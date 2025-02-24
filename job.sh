#!/usr/bin/bash

#SBATCH -J sod_inverse_design
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --ntasks=1

python -u inverse_design.py >& output
