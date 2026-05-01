#!/bin/bash
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --job-name=ptt_e8
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=16:00:00

# Generate timestamp (YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create output folder
RESULTS_DIR=/storage/scratch1/9/spanse30/PTT/eval_runs/e8_${TIMESTAMP}
mkdir -p $RESULTS_DIR

#SBATCH --output=/dev/null
#SBATCH --error=/dev/null


module load anaconda3
eval "$(conda shell.bash hook)"
conda activate /storage/project/r-gchou3-0/spanse30/ptt_env

cd /storage/scratch1/9/spanse30/PTT
export PYTHONPATH=$PWD:$PYTHONPATH

srun -u python e8.py \
    --ptt_root /storage/scratch1/9/spanse30/PTT \
    --cp_root /storage/project/r-gchou3-0/spanse30/OpenPCDet \
    --ptt_cfg tools/cfgs/waymo_models/ptt_32frames.yaml  \
    --cp_cfg /storage/project/r-gchou3-0/spanse30/OpenPCDet/tools/cfgs/waymo_models/centerpoint_4frames.yaml \
    --ptt_ckpt output/cfgs/waymo_models/default/checkpoint_epoch_6.pth \
    --cp_ckpt /storage/project/r-gchou3-0/spanse30/OpenPCDet/output/cfgs/custom_models/centerpoint_multiframe_waymo/default/ckpt/checkpoint_epoch_36.pth  \
    --pred_save_dir experiments/segment_preds_e8 \
    --dataset_save_dir experiments/segment_datasets_e8 \
    --log_file $RESULTS_DIR/eval.log \
    --metrics_out $RESULTS_DIR/metrics.json \
    > $RESULTS_DIR/e8.out \
    2> $RESULTS_DIR/e8.err
