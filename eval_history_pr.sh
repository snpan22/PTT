#!/bin/bash
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --job-name=ptt_rem_20
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Generate timestamp (YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create output folder
RESULTS_DIR=/storage/scratch1/9/spanse30/PTT/metrics_local_final_ptt
mkdir -p $RESULTS_DIR




module load anaconda3
eval "$(conda shell.bash hook)"
conda activate /storage/project/r-gchou3-0/spanse30/ptt_env

cd /storage/scratch1/9/spanse30/PTT
export PYTHONPATH=$PWD:$PYTHONPATH



srun python eval_history_pr.py \
    --cfg_file tools/cfgs/waymo_models/ptt_32frames.yaml \
    --pred_dir /storage/scratch1/9/spanse30/PTT/preds_final_ptt/ptt_removal_20 \
    --timing_annos /storage/scratch1/9/spanse30/PTT/annos/removal_20 \
    --history_len 32 \
    --dataset /storage/scratch1/9/spanse30/PTT/datasets/removal_20 \
    --log_file $RESULTS_DIR/ptt_removal_20_local_${TIMESTAMP}.log \
    --metrics_out $RESULTS_DIR/ptt_removal_20_local.json\
    > $RESULTS_DIR/slurm_ptt_local_removal_20_${TIMESTAMP}.out \
    2> $RESULTS_DIR/slurm_ptt_local_removal_20_${TIMESTAMP}.err


#SBATCH --account=gts-gchou3-ideasci23_dgx 
#SBATCH --job-name=ptt_window
#SBATCH --partition=gpu-h300
#SBATCH --gres=gpu:h300:1