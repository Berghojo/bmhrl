#!/usr/bin/env bash
#SBATCH --time=1000:00:00
#SBATCH --partition=students
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=./slurm_meteor.out

srun ./script_meteor.sh


