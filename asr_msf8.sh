#!/bin/bash
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --job-name=asr_msf8
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --output=asr_msf8.out
#SBATCH --error=asr_msf8.err

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate /storage/project/r-gchou3-0/spanse30/ptt_env

python asr_msf8.py