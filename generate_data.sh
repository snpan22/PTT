#!/bin/bash
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --job-name=rem_50
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=9:00:00
#SBATCH --output=data_gen_logs/data_gen_removal_50_%j.out
#SBATCH --error=data_gen_logs/data_gen_removal_50_%j.err

# Generate timestamp (YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")






module load anaconda3
eval "$(conda shell.bash hook)"
conda activate /storage/project/r-gchou3-0/spanse30/ptt_env

cd /storage/scratch1/9/spanse30/PTT
export PYTHONPATH=$PWD:$PYTHONPATH

srun -u python generate_data_removal.py \
    --cp_mf_root /storage/project/r-gchou3-0/spanse30/OpenPCDet \
    --cp_mf_cfg /storage/project/r-gchou3-0/spanse30/OpenPCDet/tools/cfgs/waymo_models/centerpoint_4frames.yaml \
    --cp_mf_ckpt /storage/project/r-gchou3-0/spanse30/OpenPCDet/output/cfgs/custom_models/centerpoint_multiframe_waymo/default/ckpt/checkpoint_epoch_36.pth  \
    --cp_sf_cfg /storage/project/r-gchou3-0/spanse30/OpenPCDet/tools/cfgs/waymo_models/centerpoint.yaml \
    --cp_sf_ckpt /storage/project/r-gchou3-0/spanse30/OpenPCDet/output/cfgs/custom_models/centerpoint_singleframe_waymo/default/ckpt/checkpoint_epoch_30.pth  \
    --save_dir_dataset datasets/removal_50 \
    --save_dir_annos annos/removal_50 \
    --log_file data_gen_logs/${TIMESTAMP}_removal_50.log \
    --az_width 50 \
