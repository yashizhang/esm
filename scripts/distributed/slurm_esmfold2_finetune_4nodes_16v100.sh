#!/usr/bin/env bash
#SBATCH --job-name=esmfold2-ft
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}
export WORLD_SIZE=$((SLURM_NNODES * 4))
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=${WANDB_MODE:-offline}

MANIFEST=${MANIFEST:-data/sample_nanobody10/manifest.csv}
CACHE_INDEX=${CACHE_INDEX:-data/sample_nanobody10/cache_index.jsonl}
MODEL_CHECKPOINT=${MODEL_CHECKPOINT:?Set MODEL_CHECKPOINT to local or HF ESMFold2 checkpoint}
OUTPUT_DIR=${OUTPUT_DIR:-runs/esmfold2_nanobody_16v100_${SLURM_JOB_ID}}
PYTHON=${PYTHON:-python}
read -r -a PYTHON_CMD <<< "$PYTHON"

srun --kill-on-bad-exit=1 \
  "${PYTHON_CMD[@]}" scripts/esmfold2_frozen_feature_finetune.py \
  --manifest "$MANIFEST" \
  --cache-index "$CACHE_INDEX" \
  --model-checkpoint "$MODEL_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --stage "${STAGE:-1}" \
  --max-steps "${MAX_STEPS:-100}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS:-1}" \
  --distributed ddp \
  --ddp-find-unused-parameters "${DDP_FIND_UNUSED_PARAMETERS:-true}" \
  --dist-backend nccl \
  --precision auto \
  --wandb-mode "${WANDB_MODE:-offline}" \
  --wandb-project "${WANDB_PROJECT:-esmfold2-nanobody-finetune}" \
  --wandb-run-name "${WANDB_RUN_NAME:-esmfold2-16v100-${SLURM_JOB_ID}}" \
  --log-every-n-steps "${LOG_EVERY_N_STEPS:-1}" \
  --wandb-log-every-n-steps "${WANDB_LOG_EVERY_N_STEPS:-1}" \
  --seed "${SEED:-0}"
