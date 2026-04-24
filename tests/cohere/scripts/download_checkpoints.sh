#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# download_checkpoints.sh
#
# Pre-fetch all model checkpoints (and other heavy artifacts) required by the
# various vLLM test groups in CI so that the real test job can run fully offline
# and not hammer huggingface. Requires GCP authentication.
#
# Usage:  ./download_checkpoints.sh <test_group> [models]
#         where <test_group> ∈ {cpu, fast_check, model_arch, lm_eval,
#                               bee_eval, performance, guided_generation,
#                               speculative_decoding, vision, models}
#         models is optional, comma-separated list (default: "command-r7b,command-a")
#         - Used by eval and performance to download specific model checkpoints
#         - Use test_group="models" to download only the models specified in MODELS for all tasks
#
# Environment Variables:
#   ENGINES_DIR     - Directory for model checkpoints (default: /home/runner/_work/engines)
#   HF_CACHE_DIR    - Directory for HuggingFace cache (default: /home/runner/_work/hf_cache)
#   MODELS          - Comma-separated list of models to download (can also be passed as argument)
#
###############################################################################

# Configuration: Allow overriding checkpoint directories via environment variables
ENGINES_DIR="${ENGINES_DIR:-/home/runner/_work/engines}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/home/runner/_work/hf_cache}"
GITHUB_WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"

setup_directories () {
    # Ensure the necessary directories exist
    echo $GITHUB_WORKSPACE
    mkdir -p "$HF_CACHE_DIR"
    mkdir -p "$ENGINES_DIR"
    echo "Using ENGINES_DIR: $ENGINES_DIR"
    echo "Using HF_CACHE_DIR: $HF_CACHE_DIR"

    # Configure gcloud storage retry settings to handle transient network errors
    gcloud config set storage/max_retries 3
    gcloud config set storage/base_retry_delay 5
    gcloud config set storage/max_retry_delay 60
    gcloud config set storage/exponential_sleep_multiplier 2
}

# Model checkpoints live under a common GCS prefix + model name.
MODEL_PATH_PREFIX="${MODEL_PATH_PREFIX:-gs://cohere-model-efficiency-ci/engines/}"

get_checkpoint_url () {
    local MODEL_NAME="$1"
    if [[ "$MODEL_PATH_PREFIX" != */ ]]; then
        echo "${MODEL_PATH_PREFIX}/${MODEL_NAME}"
    else
        echo "${MODEL_PATH_PREFIX}${MODEL_NAME}"
    fi
}

is_public_hf_model () {
    local MODEL_NAME="$1"
    [[ "$MODEL_NAME" == */* ]]
}

download_model_if_missing () {
    local MODEL_NAME="$1"
    local CHECKPOINT_URL

    CHECKPOINT_URL=$(get_checkpoint_url "$MODEL_NAME")

    if [[ ! -d "${ENGINES_DIR}/${MODEL_NAME}" ]]; then
        echo "==> Downloading ${MODEL_NAME} model checkpoint from ${CHECKPOINT_URL}"
        mkdir -p "${ENGINES_DIR}/${MODEL_NAME}"
        gcloud storage cp -r ${CHECKPOINT_URL}/* "${ENGINES_DIR}/${MODEL_NAME}/"
    else
        echo "${MODEL_NAME} model checkpoint already exists, skipping download."
    fi
}

download_model_always () {
    local MODEL_NAME="$1"
    local CHECKPOINT_URL

    CHECKPOINT_URL=$(get_checkpoint_url "$MODEL_NAME")

    echo "==> Downloading ${MODEL_NAME} model checkpoint from ${CHECKPOINT_URL}"
    mkdir -p "${ENGINES_DIR}/${MODEL_NAME}"
    gcloud storage cp -r ${CHECKPOINT_URL}/* "${ENGINES_DIR}/${MODEL_NAME}/"
}

download_and_untar () {
    local TAR_NAME="hub_${TEST_GROUP}.tar"
    local OBJ="gs://cohere-model-efficiency-ci/hf_cache/${TAR_NAME}"

    echo "Downloading checkpoints for test group: $TEST_GROUP"
    mkdir -p "$HF_CACHE_DIR/hub"

    if gcloud storage cp "$OBJ" "$HF_CACHE_DIR/" && \
        tar -xf "$HF_CACHE_DIR/$TAR_NAME" -C "$HF_CACHE_DIR/hub" && \
        rm "$HF_CACHE_DIR/$TAR_NAME"; then
        echo "Checkpoint download and extraction complete."
    else
        echo "Failed to download or extract checkpoints for $TEST_GROUP" >&2
        return 1
    fi
}

# Per-group download functions
download_fast_check () {
    echo "==> Downloading checkpoints for fast_check suite"
    download_and_untar
}

download_eval () {
    echo "==> Downloading checkpoints for eval suite (MODELS=${MODELS})"

    # Split models by comma and iterate
    IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"

    for MODEL_NAME in "${MODEL_ARRAY[@]}"; do
        # Trim whitespace
        MODEL_NAME=$(echo "$MODEL_NAME" | xargs)

        echo "==> Downloading checkpoint for model: ${MODEL_NAME}"

        # Create destination directory and download the checkpoint if it doesn't exist
        download_model_if_missing "$MODEL_NAME"
    done
}

download_performance () {
    echo "==> Downloading checkpoints for performance benchmarks (MODELS=${MODELS})"
    echo "==> Skipping .safetensors files (using dummy weights for performance tests)"

    # Split models by comma and iterate
    IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"

    for MODEL_NAME in "${MODEL_ARRAY[@]}"; do
        # Trim whitespace
        MODEL_NAME=$(echo "$MODEL_NAME" | xargs)

        if is_public_hf_model "$MODEL_NAME"; then
            echo "==> Skipping checkpoint pre-download for public Hugging Face model: ${MODEL_NAME}"
            continue
        fi

        echo "==> Downloading checkpoint for model: ${MODEL_NAME}"

        # Create destination directory and download the checkpoint, excluding .safetensors files
        CHECKPOINT_URL=$(get_checkpoint_url "$MODEL_NAME")
        mkdir -p "${ENGINES_DIR}/${MODEL_NAME}"
        gcloud storage rsync -r -x '.*\.safetensors$' ${CHECKPOINT_URL}/ "${ENGINES_DIR}/${MODEL_NAME}/"
    done
}

download_guided_generation () {
    echo "==> Downloading checkpoints for guided_generation demo"
    
    # Define the models needed for guided generation (c4-25a218t_fp8_eagle_l5 contains base at root + draft under eagle/)
    local GUIDED_GENERATION_MODELS=(
        "command-a_fp8"
        "command-a_fp8_draft"
        "command-r35b_fp8"
        "command-a-reasoning_fp8"
        "c4-25a218t_fp8_eagle_l5"
    )

    # Download each model using the checkpoint mapping
    for MODEL_NAME in "${GUIDED_GENERATION_MODELS[@]}"; do
        echo "==> Processing model: ${MODEL_NAME}"

        # Create destination directory and download the checkpoint if it doesn't exist
        download_model_if_missing "$MODEL_NAME"
    done
}

download_speculative_decoding () {
    echo "==> Downloading target checkpoints for speculative decoding"
    download_model_if_missing "command-a_fp8"
    # c4-25a218t_fp8_eagle_l5 contains base at root + draft under eagle/
    download_model_if_missing "c4-25a218t_fp8_eagle_l5"

    echo "==> Downloading draft checkpoints for speculative decoding"
    download_model_always "command-a_fp8_draft"
}

download_vision () {
    echo "==> Downloading vision model checkpoint"
    download_model_if_missing "command-a-vision_fp8"
}

download_model_arch_c5_3a30t_assets () {
    echo "==> Downloading checkpoint for c5 sanity check"
    download_model_if_missing "c5-3a30t_fp8"
}

download_model_arch_reward_assets () {
    # download reward v4.3.0 checkpoint if it doesn't exist
    if [[ ! -d "${ENGINES_DIR}/reward_v430" ]]; then
        echo "==> Downloading reward model checkpoint"
        mkdir -p "${ENGINES_DIR}/reward_v430"
        gcloud storage cp -r gs://cohere-model-efficiency-ci/engines/reward_111B_v4.3.0/poseidon/* "${ENGINES_DIR}/reward_v430"
    else
        echo "Reward model checkpoint already exists, skipping download."
    fi
}

download_model_arch () {
    echo "==> Downloading checkpoints for model architecture suite"
    download_model_arch_reward_assets
    download_model_arch_c5_3a30t_assets
}


download_models () {
    echo "==> Downloading all checkpoints needed by all test groups"

    # Download all models needed by the all test groups
    download_eval
    download_vision
    download_speculative_decoding
    download_guided_generation
    download_model_arch_reward_assets
    download_model_arch_c5_3a30t_assets

    echo "==> All required models downloaded successfully"
}

# Dispatch
run_downloads () {
    case "${TEST_GROUP}" in
        cpu)               echo "==> CPU tests require no model downloads"; return 0 ;;
        model_arch)        download_model_arch        ;;
        quantization_32bit_logits) download_model_arch_c5_3a30t_assets ;;
        model_arch_reward) download_model_arch_reward_assets ;;
        model_arch_c5_3a30t)     download_model_arch_c5_3a30t_assets     ;;
        bee_sample_tb_check)     download_model_arch_c5_3a30t_assets     ;;
        fast_check)        download_fast_check        ;;
        lm_eval)           download_eval              ;;
        bee_eval)          download_eval              ;;
        performance)       download_performance       ;;
        guided_generation) download_guided_generation ;;
        speculative_decoding) download_speculative_decoding ;;
        vision)            download_vision            ;;
        models)            download_models            ;;
        *)
            echo "Unknown group '${TEST_GROUP}'"
            echo "Valid groups: cpu, fast_check, model_arch, model_arch_reward, model_arch_c5_3a30t, bee_sample_tb_check, quantization, quantization_32bit_logits, lm_eval, bee_eval, performance, guided_generation, speculative_decoding, vision, models"
            exit 1
            ;;
    esac
}

# Entry point
if [[ $# -lt 1 ]] || [[ $# -gt 2 ]]; then
    echo "Usage: $0 <test_group> [models]"
    echo "  models can also be set via MODELS environment variable"
    exit 1
fi

TEST_GROUP="$1"
# Use argument if provided, otherwise fall back to environment variable
if [[ $# -eq 2 ]]; then
    MODELS="$2"
else
    MODELS="${MODELS:-}"
fi

setup_directories
run_downloads "$TEST_GROUP"

echo "🏁  All requested checkpoints are present."
