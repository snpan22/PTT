#!/bin/bash
#SBATCH --account=gts-gchou3-paid
#SBATCH --job-name=generate_data_hard
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=11:00:00
#SBATCH --output=pla/data_gen_logs/rel_fixed_hard%j.out
#SBATCH --error=pla/data_gen_logs/rel_fixed_hard%j.err

# Generate timestamp (YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create output folder
# RESULTS_DIR=/storage/scratch1/9/spanse30/PTT/eval_runs/e8_${TIMESTAMP}
# mkdir -p $RESULTS_DIR




module load anaconda3
eval "$(conda shell.bash hook)"
conda activate /storage/project/r-gchou3-0/spanse30/ptt_env

cd /storage/scratch1/9/spanse30/PTT
export PYTHONPATH=$PWD:$PYTHONPATH

srun -u python generate_data_relative_pla.py \
    --cp_mf_root /storage/project/r-gchou3-0/spanse30/OpenPCDet \
    --cp_mf_cfg /storage/project/r-gchou3-0/spanse30/OpenPCDet/tools/cfgs/waymo_models/centerpoint_4frames.yaml \
    --cp_mf_ckpt /storage/project/r-gchou3-0/spanse30/OpenPCDet/output/cfgs/custom_models/centerpoint_multiframe_waymo/default/ckpt/checkpoint_epoch_36.pth  \
    --cp_sf_cfg /storage/project/r-gchou3-0/spanse30/OpenPCDet/tools/cfgs/waymo_models/centerpoint.yaml \
    --save_dir_dataset pla/datasets/rel_fixed_hard \
    --save_dir_annos pla/annos/rel_fixed_hard \
    --log_file pla/data_gen_logs/${TIMESTAMP}_rel_fixed_hard.log \
    --trace_file traces_hard.pkl 


#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --job-name=generate_data
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1