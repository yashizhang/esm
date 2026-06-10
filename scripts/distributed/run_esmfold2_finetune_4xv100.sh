#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=${WANDB_MODE:-offline}

MANIFEST=${MANIFEST:-data/sample_nanobody10/manifest.csv}
CACHE_INDEX=${CACHE_INDEX:-data/sample_nanobody10/cache_index.jsonl}
MODEL_CHECKPOINT=${MODEL_CHECKPOINT:?Set MODEL_CHECKPOINT to local or HF ESMFold2 checkpoint}
OUTPUT_DIR=${OUTPUT_DIR:-runs/esmfold2_nanobody_4xv100}

torchrun --standalone --nproc_per_node=4 \
  scripts/esmfold2_frozen_feature_finetune.py \
  --manifest "$MANIFEST" \
  --cache-index "$CACHE_INDEX" \
  --model-checkpoint "$MODEL_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --stage "${STAGE:-1}" \
  --max-steps "${MAX_STEPS:-20}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS:-1}" \
  --distributed ddp \
  --ddp-find-unused-parameters "${DDP_FIND_UNUSED_PARAMETERS:-true}" \
  --precision auto \
  --wandb-mode "${WANDB_MODE:-offline}" \
  --wandb-project "${WANDB_PROJECT:-esmfold2-nanobody-finetune}" \
  --wandb-run-name "${WANDB_RUN_NAME:-esmfold2-4xv100-smoke}" \
  --log-every-n-steps "${LOG_EVERY_N_STEPS:-1}" \
  --wandb-log-every-n-steps "${WANDB_LOG_EVERY_N_STEPS:-1}" \
  --seed "${SEED:-0}"
