#!/usr/bin/env bash
#SBATCH --time=100:00:00
#SBATCH --partition=students
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=./slurm_bleu.out

srun ./script_bleu.sh


