#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Setup Verl + vLLM environment, run GSM8K Qwen0.5B ppo example, then test with vLLM

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VERL_REPO="https://github.com/volcengine/verl.git"
VERL_BRANCH="main"
VERL_DIR="${REPO_ROOT}/verl"
TARGET_DIR="${VERL_DIR}/examples/data_preprocess"
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
MODEL_DIR="${VERL_DIR}/models/Qwen2.5-0.5B-Instruct"
train_epochs=2
data_dir="${VERL_DIR}/gsm8k"
n_gpus_per_node=8
nnodes=1

echo "VERL_REPO=${VERL_REPO}"
echo "VERL_BRANCH=${VERL_BRANCH}"
echo "VERL_DIR=${VERL_DIR}"
echo "TARGET_DIR=${TARGET_DIR}"
echo "MODEL_ID=${MODEL_ID}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "train_epochs=${train_epochs}"
echo "data_dir=${data_dir}"
echo "n_gpus_per_node=${n_gpus_per_node}"
echo "nnodes=${nnodes}"

echo "===== Setting up Verl environment ====="

if [ -d "${VERL_DIR}" ]; then
    echo "Verl exists, skip clone"
else
    git clone --branch "${VERL_BRANCH}" --single-branch "${VERL_REPO}" "${VERL_DIR}"
fi

# Install UV if not available
if ! command -v uv &> /dev/null; then
    echo "Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

echo "Entering ${VERL_DIR} ..."
cd "${VERL_DIR}"
uv pip install --no-deps -e .
uv pip install -e .[vllm]

echo "Entering ${TARGET_DIR} ..."
cd "${TARGET_DIR}"
echo "Running gsm8k.py "
python3 gsm8k.py --local_save_dir "${data_dir}"; 

echo "===== gsm8k.py preprocessing completed! ====="

echo "===== Downloading model: ${MODEL_ID} ====="
echo "Target directory: ${MODEL_DIR}"
huggingface-cli download "${MODEL_ID}" --resume-download --local-dir "${MODEL_DIR}"
echo "===== Downloading model: ${MODEL_ID} completed! ====="
echo "===== Starting PPO Training ====="
python3 -m verl.trainer.main_ppo \
 data.train_files="${data_dir}/train.parquet" \
 data.val_files="${data_dir}/train.parquet" \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=512 \
 actor_rollout_ref.model.path="${MODEL_DIR}" \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path="${MODEL_DIR}" \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=tensorboard \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node="${n_gpus_per_node}" \
 trainer.nnodes="${nnodes}" \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs="${train_epochs}" 

echo "===== End PPO Training ====="

echo "===== Model Restoration ====="

# steps_per_epoch = 7473 samples(GSM8K: ~7473 samples) / 256 global batch size ≈ 29
step=$((29 * train_epochs))
merge_LOCAL_DIR="${TARGET_DIR}/checkpoints/verl_examples/gsm8k/global_step_${step}/actor"
merge_TARGET_DIR="${TARGET_DIR}/checkpoints/verl_examples/gsm8k/global_step_${step}/actor_hf"

python "${VERL_DIR}/scripts/legacy_model_merger.py" merge \
  --backend fsdp \
  --local_dir "${merge_LOCAL_DIR}" \
  --target_dir "${merge_TARGET_DIR}"

CUDA_VISIBLE_DEVICES=0,1,2,3 lm_eval --model hf \
  --model_args pretrained="${merge_TARGET_DIR}",trust_remote_code=True \
  --tasks gsm8k \
  --batch_size auto \
  --apply_chat_template True \
  --output_path results_ppo.json

echo "=====Test completed! ====="