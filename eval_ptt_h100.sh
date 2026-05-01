#!/bin/bash
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --job-name=ptt_pla_easy
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=05:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Generate timestamp (YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create output folder
RESULTS_DIR=/storage/scratch1/9/spanse30/PTT/pla/metrics_pi
mkdir -p $RESULTS_DIR




module load anaconda3
eval "$(conda shell.bash hook)"
conda activate /storage/project/r-gchou3-0/spanse30/ptt_env

cd /storage/scratch1/9/spanse30/PTT
export PYTHONPATH=$PWD:$PYTHONPATH

# srun python eval_ptt_final.py \
#     --cfg_file tools/cfgs/waymo_models/ptt_32frames.yaml \
#     --ckpt output/cfgs/waymo_models/default/checkpoint_epoch_6.pth \
#     --dataset /storage/scratch1/9/spanse30/PTT/datasets/global_easy_d \
#     --pred_dir /storage/scratch1/9/spanse30/PTT/preds_final/ptt_global_easy \
#     --log_file $RESULTS_DIR/ptt_global_easy_${TIMESTAMP}.log \
#     --metrics_out $RESULTS_DIR/ptt_global_easy.json\
#     > $RESULTS_DIR/slurm_ptt_global_easy${TIMESTAMP}.out \
#     2> $RESULTS_DIR/slurm_ptt_global_easy${TIMESTAMP}.err

srun python eval_ptt_final_pi.py \
    --cfg_file tools/cfgs/waymo_models/ptt_32frames.yaml \
    --pred_dir /storage/scratch1/9/spanse30/PTT/pla/preds_ptt/global_easy \
    --ckpt output/cfgs/waymo_models/default/checkpoint_epoch_6.pth \
    --dataset /storage/scratch1/9/spanse30/PTT/pla/datasets/global_easy \
    --asr_path pla/asr/ptt_global_easy.pkl \
    --log_file $RESULTS_DIR/ptt_global_easy_${TIMESTAMP}.log \
    --metrics_out $RESULTS_DIR/ptt_global_easy.json \
    > $RESULTS_DIR/slurm_ptt_global_easy_${TIMESTAMP}.out \
    2> $RESULTS_DIR/slurm_ptt_global_easy_${TIMESTAMP}.err



# srun python eval_history_decay_ptt.py \
#     --cfg_file tools/cfgs/waymo_models/ptt_32frames.yaml \
#     --pred_dir /storage/scratch1/9/spanse30/PTT/preds_final/ptt_global_easy \
#     --timing_annos /storage/scratch1/9/spanse30/PTT/annos/global_easy_a \
#     --detector ptt \
#     --dataset /storage/scratch1/9/spanse30/PTT/datasets/global_easy_d \
#     --log_file $RESULTS_DIR/ptt_global_easy_decay_${TIMESTAMP}.log \
#     --metrics_out $RESULTS_DIR/ptt_global_easy_decay\
#     > $RESULTS_DIR/slurm_ptt_global_easy_decay_${TIMESTAMP}.out \
#     2> $RESULTS_DIR/slurm_ptt_global_easy_decay_${TIMESTAMP}.err

# srun python eval_ptt_clean_sanity.py \
#     --cfg_file tools/cfgs/waymo_models/ptt_32frames.yaml \
#     --result_pkl /storage/project/r-gchou3-0/spanse30/OpenPCDet/output/cfgs/custom_models/centerpoint_multiframe_waymo/default/eval/epoch_36/val/default/result_fixed.pkl \
#     --ckpt output/cfgs/waymo_models/default/checkpoint_epoch_6.pth \
#     --pred_dir /storage/scratch1/9/spanse30/PTT/preds/ptt_clean_updated \
#     --log_file $RESULTS_DIR/ptt_clean_updated_${TIMESTAMP}.log \
#     --metrics_out $RESULTS_DIR/ptt_clean_updated_${TIMESTAMP}.json \
#     > $RESULTS_DIR/slurm_ptt_clean_updated_${TIMESTAMP}.out \
#     2> $RESULTS_DIR/slurm_ptt_clean_updated_${TIMESTAMP}.err

#/storage/project/r-gchou3-0/spanse30/OpenPCDet/output/cfgs/custom_models/centerpoint_multiframe_waymo/default/eval/epoch_36/val/default/result_fixed.pkl
#/storage/project/r-gchou3-0/OpenPCDet/output/cfgs/custom_models/centerpoint_multiframe_waymo/default/eval/epoch_36/val/default/result_fixed.pkl 

#SBATCH --account=gts-gchou3-ideasci23_dgx 
#SBATCH --job-name=ptt_window
#SBATCH --partition=gpu-h300
#SBATCH --gres=gpu:h300:1