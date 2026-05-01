#!/bin/bash
#SBATCH --account=gts-gchou3-paid
#SBATCH --job-name=ptt_LOP_global_easy
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Generate timestamp (YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create output folder
RESULTS_DIR=/storage/scratch1/9/spanse30/PTT/LOP/metrics_pi
mkdir -p $RESULTS_DIR




module load anaconda3
eval "$(conda shell.bash hook)"
conda activate /storage/project/r-gchou3-0/spanse30/ptt_env


cd /storage/scratch1/9/spanse30/PTT
export PYTHONPATH=$PWD:$PYTHONPATH

srun python eval_ptt_final_pi_lop.py \
    --cfg_file /storage/scratch1/9/spanse30/PTT/tools/cfgs/waymo_models/ptt_32frames.yaml \
    --pred_dir /storage/scratch1/9/spanse30/PTT/LOP/preds_ptt/global_easy_B07_T044 \
    --ckpt  /storage/scratch1/9/spanse30/PTT/output/cfgs/waymo_models/default/checkpoint_epoch_6.pth \
    --dataset /storage/scratch1/9/spanse30/PTT/datasets/global_easy_d \
    --asr_path LOP/asr/ptt_global_easy_B07_T044.pkl \
    --lop_ckpt /storage/scratch1/9/spanse30/LOP/ckpts/lop_vehicle/lop_best.pt \
    --boundary 0.7 \
    --pillar_threshold 0.44 \
    --log_file $RESULTS_DIR/ptt_global_easy_B07_T044${TIMESTAMP}.log \
    --metrics_out $RESULTS_DIR/ptt_global_easy_B07_T044.json \
    > $RESULTS_DIR/slurm_ptt_global_easy_B07_T044${TIMESTAMP}.out \
    2> $RESULTS_DIR/slurm_ptt_global_easy_B07_T044${TIMESTAMP}.err

#SBATCH --account=gts-gchou3-ideasci23_dgx 
#SBATCH --job-name=ptt_window
#SBATCH --partition=gpu-h300
#SBATCH --gres=gpu:h300:1

#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --job-name=ptt_LOP_medium_04
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1