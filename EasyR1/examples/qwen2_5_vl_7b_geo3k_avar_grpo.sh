#!/bin/bash
# AVAR GRPO Training Script for Qwen2.5-VL
#
# This script trains a VLM using GRPO with AVAR visual attention reward.
#
# AVAR Parameters:
#   - use_avar_attn: Enable AVAR visual attention reward
#   - avar_lambda_v: Visual attention reward weight (default: 0.3)
#   - avar_lambda_f: Format reward weight (default: 0.1)
#   - avar_layers: Layer indices for attention extraction (e.g., "0,1,2,3")
#
# Note: AVAR requires attention weights, which increases memory usage.
#       Consider using fewer layers (avar_layers) for longer sequences.

set -x

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_geo_avar_grpo \
    trainer.n_gpus_per_node=8 \
    algorithm.use_avar_attn=true \
    algorithm.avar_lambda_v=0.3 \
    algorithm.avar_lambda_f=0.1 \
    algorithm.avar_layers="0,1,2,3,4,5,6,7" \
    algorithm.avar_epsilon=1e-6
