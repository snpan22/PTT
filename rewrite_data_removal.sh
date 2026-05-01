#!/bin/bash
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --job-name=rewrite_removal_50
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=5:30:00
#SBATCH --output=rewrite_removal_50_%j.out
#SBATCH --error=rewrite_removal_50_%j.err

# Generate timestamp (YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")






module load anaconda3
eval "$(conda shell.bash hook)"
conda activate /storage/project/r-gchou3-0/spanse30/ptt_env

cd /storage/scratch1/9/spanse30/PTT
export PYTHONPATH=$PWD:$PYTHONPATH

srun -u python rewrite_dataset_removal.py \
    --cp_mf_root /storage/project/r-gchou3-0/spanse30/OpenPCDet \
    --cp_mf_cfg /storage/project/r-gchou3-0/spanse30/OpenPCDet/tools/cfgs/waymo_models/centerpoint_4frames.yaml \
    --cp_sf_cfg /storage/project/r-gchou3-0/spanse30/OpenPCDet/tools/cfgs/waymo_models/centerpoint.yaml \
    --save_dir_dataset datasets_waymo/removal_50_waymo \
    --log_file data_gen_logs/${TIMESTAMP}_rewrite_removal_50.log \
    --old_dataset_dir datasets/removal_50 \
    --annos_dir annos/removal_50 \

