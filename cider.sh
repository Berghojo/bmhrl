#!/usr/bin/env bash
#SBATCH --time=100:00:00
#SBATCH --partition=students
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=./slurm_cider.out

srun ./script_cider.sh


