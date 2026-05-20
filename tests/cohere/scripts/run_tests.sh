#!/bin/bash

# Run tests based on the provided test group.
# We assume the testing environment is already set up,
# i.e. via `setup_tests.sh` or similar.
# needs periodic manual sync with upstream `test-pipeline.yaml`

# Configuration: Allow overriding directories via environment variables
export ENGINES_DIR="${ENGINES_DIR:-/root/engines}"
export VLLM_WORKSPACE="${VLLM_WORKSPACE:-/vllm-workspace}"
export BEE_DIR="${BEE_DIR:-/app/cohere/apiary/bee}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/output}"
export UNIT_SUMMARY_FILE_NAME="${UNIT_SUMMARY_FILE_NAME:-unit_results_summary.json}"
# automatically hardware_profiles
export VLLM_ENABLE_COHERE_AUTO_CONFIG=1
echo "Using ENGINES_DIR: $ENGINES_DIR"
echo "Using VLLM_WORKSPACE: $VLLM_WORKSPACE"
echo "Using BEE_DIR: $BEE_DIR"
echo "Using OUTPUT_DIR: $OUTPUT_DIR"
mkdir -p ${OUTPUT_DIR}

# Print GPU usage information (supports both NVIDIA and AMD GPUs)
print_gpu_usage() {
    echo "=========================================="
    echo "GPU Usage Information"
    echo "=========================================="
    if command -v nvidia-smi &> /dev/null; then
        echo "Detected NVIDIA GPU(s):"
        nvidia-smi
    elif command -v rocm-smi &> /dev/null; then
        echo "Detected AMD GPU(s):"
        rocm-smi
    else
        echo "Warning: No GPU monitoring tool found (neither nvidia-smi nor rocm-smi)"
    fi
    echo "=========================================="
}

# Clean up stale vLLM shared memory buffers that may have been left behind
# by previous runs. This prevents "FileExistsError: [Errno 17] File exists"
# when starting a new vLLM instance with mm_processor_cache_type="shm".
cleanup_vllm_shm() {
    echo "Cleaning up stale vLLM shared memory buffers..."
    # Remove any VLLM shared memory files in /dev/shm
    for shm_file in /dev/shm/VLLM_*; do
        if [ -e "$shm_file" ]; then
            echo "Removing stale shared memory: $shm_file"
            rm -f "$shm_file"
        fi
    done
    echo "Shared memory cleanup complete."
}

resolve_model_path() {
    local model_name="$1"
    local local_model_path="${ENGINES_DIR}/${model_name}"

    if [[ -d "$local_model_path" ]]; then
        echo "$local_model_path"
    else
        echo "$model_name"
    fi
}

sanitize_model_label() {
    local model_name="$1"
    echo "$model_name" | sed 's#[/:]#_#g'
}

run_cpu_tests() {
    echo "Running CPU tests (no GPU required)..."

    # ── Auto-discovered upstream tests ──
    # Runs every test marked @pytest.mark.cpu_test in directories known to
    # contain CPU tests. New tests added within these directories are picked
    # up automatically — no need to list individual files.
    pytest -v -s \
        v1/core/ v1/kv_connector/unit/ v1/test_serial_utils.py \
        v1/structured_output/ v1/streaming_input/ v1/metrics/ \
        multimodal/ tool_use/ \
        models/test_vision.py models/test_utils.py \
        kernels/quantization/test_scaled_mm_kernel_selection.py \
        test_inputs.py test_outputs.py \
        -m cpu_test \
        `# marked cpu_test but imports GPUModelRunner, which pulls in CUDA deps at import time` \
        --deselect v1/streaming_input/test_gpu_model_runner_v2_streaming.py \
        `# requires ffmpeg to extract audio from video; not available in CI container` \
        --deselect multimodal/media/test_audio.py::test_audio_media_io_from_video

    # ── Tests that are CPU-safe but not marked with cpu_test ──

    # Cohere-specific CPU tests
    pytest -v -s cohere/cpu

    # Tool parser unit tests (parsers only, no LLM required)
    pytest -v -s entrypoints/openai/tool_parsers \
        --ignore=entrypoints/openai/tool_parsers/test_openai_tool_parser.py \
        --ignore=entrypoints/openai/tool_parsers/test_hermes_tool_parser.py \
        `# requires RemoteOpenAIServer (GPU) but collected in CPU-only parser run` \
        --deselect entrypoints/openai/tool_parsers/test_granite4_tool_parser.py::test_stop_sequence_interference

    # V1 output dataclass tests
    pytest -v -s v1/core/test_output.py

    # Logits processor correctness tests
    pytest -v -s v1/logits_processors/test_correctness.py
}

run_fast_check() {
    # for amd, unset this for unit tests
    unset VLLM_V1_USE_PREFILL_DECODE_ATTENTION
    echo "Running fast_check tests..."

    # V1 Core Test (5min)
    pytest -v -s v1/core

    # Basic Correctness Test (20min)
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    pytest -v -s basic_correctness/test_cumem.py
    pytest -v -s basic_correctness/test_basic_correctness.py
    pytest -v -s basic_correctness/test_cpu_offload.py

    # Entrypoints Unit Tests (5min)
    pytest -v -s entrypoints/openai/tool_parsers
    pytest -v -s entrypoints/ --ignore=entrypoints/llm --ignore=entrypoints/openai --ignore=entrypoints/offline_mode --ignore=entrypoints/test_chat_utils.py  --ignore=entrypoints/pooling

    # Entrypoints Integration Test (LLM) (30min)
    # disable encode + embedding
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    pytest -v -s entrypoints/llm \
        --ignore=entrypoints/llm/test_generate.py \
        --ignore=entrypoints/llm/test_collective_rpc.py \
        --ignore=entrypoints/llm/test_lazy_outlines.py \
        --ignore=entrypoints/llm/test_encode.py \
        --ignore=entrypoints/llm/test_embedding.py

    pytest -v -s entrypoints/llm/test_generate.py # it needs a clean process
    pytest -v -s entrypoints/offline_mode # Needs to avoid interference with other tests

    # Entrypoints Integration Test (API Server) # 100min
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    # below is not in our local branch yet
    # PYTHONPATH=${VLLM_WORKSPACE} pytest -v -s entrypoints/openai/test_collective_rpc.py # PYTHONPATH is needed to import custom Worker extension
    # lora, rerank, embed are ignored to save time
    pytest -v -s entrypoints/openai \
        --ignore=entrypoints/openai/test_chat_with_tool_reasoning.py \
        --ignore=entrypoints/openai/test_oot_registration.py \
        --ignore=entrypoints/openai/test_tensorizer_entrypoint.py \
        --ignore=entrypoints/openai/correctness/ \
        --ignore=entrypoints/openai/test_collective_rpc.py \
        --ignore=entrypoints/openai/tool_parsers/ \
        --ignore=*lora* --ignore=*rerank* \
        --ignore=entrypoints/openai/test_embedding.py \ 
        --ignore=entrypoints/openai/test_embedding_dimensions.py \ 
        --ignore=entrypoints/openai/test_embedding_long_text.py


    pytest -v -s entrypoints/test_chat_utils.py
}

run_model_arch_logits_checks() {
    echo "Running model architecture logits checks..."
    cd "${VLLM_WORKSPACE}"

    if ! pytest -v -s tests/cohere/test_logits_processor.py; then
        echo "Model architecture logits checks failed (test_logits_processor)."
        return 1
    fi

    # FP32 logits consistency test: compare generation with and without fp32
    # logits projection. Requires the C5 model checkpoint.
    MODEL_DIR=${ENGINES_DIR}/c5-3a30t_fp8
    if ! C5_MODEL_DIR=$MODEL_DIR pytest -v -s tests/cohere/test_c5_fp32_logits.py; then
        echo "Model architecture logits checks failed (test_c5_fp32_logits)."
        return 1
    fi

    echo "Model architecture logits checks passed."
    return 0
}

run_lm_eval() {
    echo "Running eval tests with TP_SIZE=${TP_SIZE} and MODELS=${MODELS}..."
    git config --global --add safe.directory ${VLLM_WORKSPACE}

    # need editable install for RULER
    # cd ../../
    # mkdir lm-evaluation-harness && cd lm-evaluation-harness
    # git init && git remote add origin https://github.com/EleutherAI/lm-evaluation-harness.git
    # git fetch --depth 1 origin e4a7b69fe0fc6cb430e12cf15c4109bf28185124 && git checkout FETCH_HEAD
    # uv pip install --system -e ".[ruler]"

    # run lm eval
    cd ${VLLM_WORKSPACE}/.buildkite/lm-eval-harness
    export VLLM_WORKER_MULTIPROC_METHOD=spawn

    # Load model-to-config mapping from JSON file
    EVAL_MAP_PATH="${VLLM_WORKSPACE}/tests/cohere/configs/model_eval_map.json"
    MODELS_LIST="${MODELS}"

    # Split models by comma and iterate
    IFS=',' read -ra MODEL_ARRAY <<< "$MODELS_LIST"

    for MODEL_NAME in "${MODEL_ARRAY[@]}"; do
        # Trim whitespace
        MODEL_NAME=$(echo "$MODEL_NAME" | xargs)

        echo "Running eval for model: ${MODEL_NAME}"

        # Get config files array for this model from JSON
        CONFIG_YAMLS=$(jq -r --arg model "$MODEL_NAME" '.[$model] // empty | .[]' "$EVAL_MAP_PATH")

        if [ -z "$CONFIG_YAMLS" ]; then
            # TODO: Revert this to Error when we are using C4 for vision, reasoning eval 
            # Now we are using command-a vision and reasoning for vision and reasoning test,
            # which are not in the model_eval_map.json
            echo "Warning: Model ${MODEL_NAME} not found in ${EVAL_MAP_PATH}"

        else
            export LM_EVAL_TP_SIZE=$TP_SIZE

            # Loop through each config YAML for this model
            while IFS= read -r CONFIG_YAML; do
                echo "=== RUNNING CONFIG: $CONFIG_YAML WITH TP SIZE: $TP_SIZE ==="

                # Clean up stale shared memory before each test run
                cleanup_vllm_shm

                export TEST_DATA_FILE=$PWD/configs/${CONFIG_YAML}
                pytest -s ${VLLM_WORKSPACE}/tests/cohere/test_lm_eval_correctness.py

                if [[ $? == 0 ]]; then
                    echo "=== PASSED CONFIG: ${CONFIG_YAML} ==="
                else
                    echo "=== FAILED CONFIG: ${CONFIG_YAML} ==="
                    exit 1
                fi
            done <<< "$CONFIG_YAMLS"
        fi
    done
}

run_bee_eval() {
    echo "Running eval tests with TP_SIZE=${TP_SIZE} and MODELS=${MODELS}..."
    # required for bee eval report uploading
    git config --global --add safe.directory ${VLLM_WORKSPACE}

    MODELS_LIST="${MODELS}"

    # Split models by comma and iterate
    IFS=',' read -ra MODEL_ARRAY <<< "$MODELS_LIST"

    echo "Installing Bee"
    cd ${BEE_DIR}
    uv sync --python 3.12 --no-editable --extra all || { echo "Failed to install Bee dependencies"; exit 1; }
    echo "Bee installed"
    cd ${VLLM_WORKSPACE}

    for MODEL_NAME in "${MODEL_ARRAY[@]}"; do
        # Trim whitespace
        MODEL_NAME=$(echo "$MODEL_NAME" | xargs)
        # Build model path from model name
        MODEL_PATH="${ENGINES_DIR}/${MODEL_NAME}"

        echo "Running eval for model: ${MODEL_NAME}"

        # Clean up stale shared memory before each model eval
        cleanup_vllm_shm

        EVAL_NAME="${MODEL_NAME}_tp${TP_SIZE}"
        bash tests/cohere/scripts/run-bee-eval.sh "$EVAL_NAME" "$MODEL_PATH"
    done

    python3 tests/cohere/scripts/convert-eval-results-to-json.py
    python3 tests/cohere/scripts/check_bee_eval.py && echo "Command succeeded" || exit 1
    cp results/eval_results_summary.json ${OUTPUT_DIR}/

    mkdir -p results/co-bench
    python3 tests/cohere/scripts/convert-eval-results-to-co-bench.py
    cp -r results/co-bench ${OUTPUT_DIR}/

}

run_bee_samples() {
    echo "Running bee sample checks with TP_SIZE=${TP_SIZE} and MODELS=${MODELS}..."

    cd ${VLLM_WORKSPACE}
    source tests/cohere/scripts/run-helper.sh
    check_gpus

    MODELS_LIST="${MODELS}"
    IFS=',' read -ra MODEL_ARRAY <<< "$MODELS_LIST"

    for MODEL_NAME in "${MODEL_ARRAY[@]}"; do
        MODEL_NAME=$(echo "$MODEL_NAME" | xargs)
        MODEL_PATH="${ENGINES_DIR}/${MODEL_NAME}"

        echo "=== Bee samples: ${MODEL_NAME} (tp=${TP_SIZE}) ==="
        cleanup_vllm_shm

        # Build server command
        local think_start think_end
        think_start=$(python3 -c "from vllm.cohere.guided_decoding.cohere_constants import START_THINKING_TOKEN; print(START_THINKING_TOKEN)")
        think_end=$(python3 -c "from vllm.cohere.guided_decoding.cohere_constants import END_THINKING_TOKEN; print(END_THINKING_TOKEN)")
        local reasoning_json="{\"reasoning_start_str\":\"${think_start}\",\"reasoning_end_str\":\"${think_end}\"}"
        local parsers="--reasoning-parser cohere_command4 --enable-auto-tool-choice --tool-call-parser cohere_command4"
        local server_cmd="vllm serve ${MODEL_PATH} --tensor-parallel-size ${TP_SIZE} --served-model-name ${MODEL_NAME} --disable-log-stats --mm-processor-cache-type shm ${parsers} --reasoning-config '${reasoning_json}'"

        if [[ "${MODEL_NAME}" == *"eagle"* ]]; then
            local eagle_path="${MODEL_PATH}/eagle"
            local spec_json="{\"method\":\"eagle\",\"model\":\"${eagle_path}\",\"num_speculative_tokens\":3,\"draft_tensor_parallel_size\":${TP_SIZE}}"
            server_cmd="${server_cmd} --speculative-config '${spec_json}'"
        fi

        echo "Server command: ${server_cmd}"
        bash -c "${server_cmd}" &
        server_pid=$!

        if wait_for_server; then
            echo "vLLM server is up."
        else
            echo "vLLM server failed to start."
            kill -9 $server_pid 2>/dev/null
            kill_gpu_processes
            exit 1
        fi

        BEE_MODEL=${MODEL_NAME} \
        BEE_DATA_DIR=tests/cohere/bee_eval_data \
        BEE_OUTPUT_JSON=${OUTPUT_DIR}/bee_samples_${MODEL_NAME}.json \
        ENABLE_THINKING_BUDGET=${ENABLE_THINKING_BUDGET:-0} \
        BEE_MIN_SCORE_OVERRIDES=${BEE_MIN_SCORE_OVERRIDES:-} \
        PYTHONUNBUFFERED=1 \
            pytest -v -s tests/cohere/test_bee_samples.py
        local test_result=$?

        kill -9 $server_pid 2>/dev/null
        kill_gpu_processes

        if [ $test_result -ne 0 ]; then
            echo "=== FAILED: Bee samples for ${MODEL_NAME} ==="
            exit 1
        fi
        echo "=== PASSED: Bee samples for ${MODEL_NAME} ==="
    done
}

run_performance() {
    echo "Running performance benchmarks with TP_SIZE=${TP_SIZE} and MODELS=${MODELS}..."
    if [[ -z "${BENCHMARK_OUTPUT_LEN:-}" ]]; then
        echo "Error: BENCHMARK_OUTPUT_LEN environment variable is required for performance benchmarks."
        exit 1
    fi

    cd ../

    MODELS_LIST="${MODELS}"

    # Split models by comma and iterate
    IFS=',' read -ra MODEL_ARRAY <<< "$MODELS_LIST"

    for MODEL_NAME in "${MODEL_ARRAY[@]}"; do
        # Trim whitespace
        MODEL_NAME=$(echo "$MODEL_NAME" | xargs)

        echo "Running performance benchmark for model: ${MODEL_NAME}"

        MODEL_PATH="$(resolve_model_path "$MODEL_NAME")"
        SAFE_MODEL_NAME="$(sanitize_model_label "$MODEL_NAME")"

        # Run benchmark with descriptive model name for results
        BENCHMARK_NAME="${SAFE_MODEL_NAME}_tp${TP_SIZE}_out${BENCHMARK_OUTPUT_LEN}"
        bash tests/cohere/scripts/run-performance-benchmarks.sh "$BENCHMARK_NAME" "$MODEL_PATH"
    done

    cat benchmarks/results/benchmark_results.md

    # copy for post-processing and upload
    cp benchmarks/results/benchmark_results_summary.json ${OUTPUT_DIR}/benchmark_results_summary.json
}

run_guided_generation() {
    echo "Running guided generation tests..."
    local errors=0

    cd cohere
    python3 test_handle_token_thinking.py || errors=1

    cd ../..
    BLS_MODEL_DIR=${ENGINES_DIR}/c5-3a30t_fp8
    BLS_DRAFT_MODEL_DIR=${ENGINES_DIR}/c5-3a30t_eagle_bf16

    cd tests/cohere
    export PYTHONPATH="${VLLM_WORKSPACE}:${PYTHONPATH}"

    # GG (JSON, tools, long context) with thinking tokens — non-SD BLS
    echo "Running GG merged (JSON + tools + long-context) non-SD with BLS model: $BLS_MODEL_DIR"
    VLLM_WORKER_MULTIPROC_METHOD=spawn python3 test_guided_generation.py --suite merged --model "$BLS_MODEL_DIR" --tensor_parallel_size 1 --mode non-speculative || errors=1

    # GG melody — SD BLS
    echo "Running test_guided_generation_melody SD with BLS model: $BLS_MODEL_DIR"
    python3 test_guided_generation_melody.py --mode "speculative" --model="$BLS_MODEL_DIR" --tensor_parallel_size 1 --draft_model $BLS_DRAFT_MODEL_DIR --num_spec_tokens 3 --draft_tp 1 || errors=1
    echo "Running test_guided_generation_tools_melody SD with BLS model: $BLS_MODEL_DIR"
    python3 test_guided_generation_tools_melody.py --mode "speculative" --model="$BLS_MODEL_DIR" --tensor_parallel_size 1 --draft_model $BLS_DRAFT_MODEL_DIR --num_spec_tokens 3 --draft_tp 1 || errors=1

    # GG + MM + Spec + Thinking budget sweep — SD BLS
    echo "Running GG + MM + TB sweep SD with BLS model: $BLS_MODEL_DIR"
    cd ../..
    VLLM_WORKER_MULTIPROC_METHOD=spawn python3 tests/cohere/test_guided_generation_vision_spec_async.py --model $BLS_MODEL_DIR --draft_model $BLS_DRAFT_MODEL_DIR --num_spec_tokens 3 --draft_tp 1 --tensor-parallel-size 1 --mode "speculative" --thinking-budgets 500 1000 5000 || errors=1
    cd tests/cohere

    exit $errors
}

run_thinking_budget() {
    echo "Running thinking budget tests..."
    local errors=0

    BLS_MODEL_DIR=${ENGINES_DIR}/c5-3a30t_fp8
    BLS_DRAFT_MODEL_DIR=${ENGINES_DIR}/c5-3a30t_eagle_bf16

    cd cohere
    export PYTHONPATH="${VLLM_WORKSPACE}:${PYTHONPATH}"

    # Thinking budget — non-SD BLS
    echo "Running thinking budget non-SD with BLS model: $BLS_MODEL_DIR"
    python3 test_thinking_budget.py --reasoning_mode "reasoning" --model $BLS_MODEL_DIR --tensor_parallel_size 1 --mode "non-speculative" || errors=1

    # Thinking budget — SD BLS
    echo "Running thinking budget SD with BLS model: $BLS_MODEL_DIR"
    python3 test_thinking_budget.py --reasoning_mode "reasoning" --model $BLS_MODEL_DIR --tensor_parallel_size 2 --draft_model $BLS_DRAFT_MODEL_DIR --num_spec_tokens 3 --draft_tp 2 --mode "speculative" || errors=1

    exit $errors
}

run_speculative_decoding() {
    echo "Running speculative decoding tests..."

    # originally we are in vllm-cohere/tests directory
    cd ../

    local errors=0

    # -----------------------------------------------------------------
    # Helper functions for acceptance-length quality gates
    # -----------------------------------------------------------------
    extract_mean_acceptance_length() {
        local log_file="$1"
        awk -F'[:,]' '/mean acceptance length/ {print $2; exit}' "$log_file" | xargs
    }

    check_absolute_acceptance_length() {
        local observed_al="$1"
        local expected_al="$2"
        local relative_tolerance="$3"
        python3 -c "obs=float('$observed_al'); exp=float('$expected_al'); tol=float('$relative_tolerance'); print(obs * (1 + tol) >= exp and obs * (1 - tol) <= exp)"
    }

    check_relative_acceptance_length_delta() {
        local al_off="$1"
        local al_on="$2"
        local relative_tolerance="$3"
        python3 -c "off=float('$al_off'); on=float('$al_on'); tol=float('$relative_tolerance'); print(abs(on - off) / off <= tol if off != 0 else abs(on - off) <= tol)"
    }

    # -----------------------------------------------------------------
    # C4 multimodal speculative decoding gates:
    # 1. Absolute acceptance-length quality gate.
    # 2. fp32-logits compatibility gate.
    # -----------------------------------------------------------------
    c4_absolute_relative_tolerance=0.1
    c4_fp32_relative_tolerance=0.1
    C4_TP=4
    C4_TARGET_MODEL_DIR=${ENGINES_DIR}/c4-25a218t_fp8_eagle_l5
    C4_DRAFT_MODEL_DIR=${ENGINES_DIR}/c4-25a218t_fp8_eagle_l5/eagle
    C4_EXPECTED_AL=2.5
    C4_LOG_OFF=${OUTPUT_DIR}/speculative_decoding_c4_mm_fp32_off.log
    C4_LOG_ON=${OUTPUT_DIR}/speculative_decoding_c4_mm_fp32_on.log

    run_c4_spec_decode_case() {
        local fp32_mode="$1"
        local log_file="$2"

        VLLM_USE_V1=1 \
        VLLM_USE_LOGITS_FP32_COMPUTATION="$fp32_mode" \
        python3 examples/offline_inference/spec_decode.py \
            --method eagle \
            --model-dir "$C4_TARGET_MODEL_DIR" \
            --eagle-dir "$C4_DRAFT_MODEL_DIR" \
            --num_spec_tokens 3 \
            --tp "$C4_TP" \
            --num-prompts 12 \
            --custom-mm-prompts \
            &> "$log_file"
        cat "$log_file"
    }

    echo "Running C4 multimodal speculative decoding absolute AL gate (VLLM_USE_LOGITS_FP32_COMPUTATION=0)"
    run_c4_spec_decode_case 0 "$C4_LOG_OFF"
    c4_mal_off=$(extract_mean_acceptance_length "$C4_LOG_OFF")
    if [ -z "$c4_mal_off" ]; then
        echo "C4 speculative decoding test failed: missing mean acceptance length in $C4_LOG_OFF."
        exit 1
    fi

    c4_absolute_passed=$(check_absolute_acceptance_length "$c4_mal_off" "$C4_EXPECTED_AL" "$c4_absolute_relative_tolerance")
    if [ "$c4_absolute_passed" != "True" ]; then
        echo "C4 speculative decoding test failed. Expected acceptance length of $C4_EXPECTED_AL within relative tolerance $c4_absolute_relative_tolerance, got $c4_mal_off."
        exit 1
    fi
    echo "C4 speculative decoding test passed (AL=$c4_mal_off, expected=$C4_EXPECTED_AL, tol=$c4_absolute_relative_tolerance)."

    echo "Running C4 multimodal fp32 logits compatibility gate (VLLM_USE_LOGITS_FP32_COMPUTATION=1)"
    run_c4_spec_decode_case 1 "$C4_LOG_ON"
    c4_mal_on=$(extract_mean_acceptance_length "$C4_LOG_ON")
    if [ -z "$c4_mal_on" ]; then
        echo "C4 fp32 compatibility test failed: missing mean acceptance length in $C4_LOG_ON."
        exit 1
    fi

    c4_compatible=$(check_relative_acceptance_length_delta "$c4_mal_off" "$c4_mal_on" "$c4_fp32_relative_tolerance")
    if [ "$c4_compatible" != "True" ]; then
        echo "C4 speculative decoding fp32-logits compatibility test failed (AL off=$c4_mal_off, on=$c4_mal_on, tol=$c4_fp32_relative_tolerance)."
        exit 1
    fi
    echo "C4 speculative decoding fp32-logits compatibility test passed (AL off=$c4_mal_off, on=$c4_mal_on, tol=$c4_fp32_relative_tolerance)."

    # -----------------------------------------------------------------
    # C3 text speculative decoding gates:
    # 1. Absolute acceptance-length quality gate.
    # 2. fp32-logits compatibility gate.
    # -----------------------------------------------------------------
    c3_absolute_relative_tolerance=0.1
    c3_fp32_relative_tolerance=0.1
    C3_TP=2
    C3_TARGET_MODEL_DIR=${ENGINES_DIR}/command-a_fp8
    C3_DRAFT_MODEL_DIR=${ENGINES_DIR}/command-a_fp8_draft
    C3_EXPECTED_AL=2.34
    C3_LOG_OFF=${OUTPUT_DIR}/speculative_decoding_c3_fp32_off.log
    C3_LOG_ON=${OUTPUT_DIR}/speculative_decoding_c3_fp32_on.log

    run_c3_spec_decode_case() {
        local fp32_mode="$1"
        local log_file="$2"

        VLLM_USE_V1=1 \
        VLLM_USE_LOGITS_FP32_COMPUTATION="$fp32_mode" \
        python3 examples/offline_inference/spec_decode.py \
            --method eagle \
            --model-dir "$C3_TARGET_MODEL_DIR" \
            --eagle-dir "$C3_DRAFT_MODEL_DIR" \
            --num_spec_tokens 3 \
            --tp "$C3_TP" \
            --dataset-name hf \
            --dataset-path philschmid/mt-bench \
            --num-prompts 80 \
            &> "$log_file"
        cat "$log_file"
    }

    echo "Running C3 speculative decoding absolute AL gate (VLLM_USE_LOGITS_FP32_COMPUTATION=0)"
    run_c3_spec_decode_case 0 "$C3_LOG_OFF"
    c3_mal_off=$(extract_mean_acceptance_length "$C3_LOG_OFF")
    if [ -z "$c3_mal_off" ]; then
        echo "C3 speculative decoding test failed: missing mean acceptance length in $C3_LOG_OFF."
        exit 1
    fi

    c3_absolute_passed=$(check_absolute_acceptance_length "$c3_mal_off" "$C3_EXPECTED_AL" "$c3_absolute_relative_tolerance")
    if [ "$c3_absolute_passed" != "True" ]; then
        echo "C3 speculative decoding test failed. Expected acceptance length of $C3_EXPECTED_AL within relative tolerance $c3_absolute_relative_tolerance, got $c3_mal_off."
        exit 1
    fi
    echo "C3 speculative decoding test passed (AL=$c3_mal_off, expected=$C3_EXPECTED_AL, tol=$c3_absolute_relative_tolerance)."

    echo "Running C3 fp32 logits compatibility gate (VLLM_USE_LOGITS_FP32_COMPUTATION=1)"
    run_c3_spec_decode_case 1 "$C3_LOG_ON"
    c3_mal_on=$(extract_mean_acceptance_length "$C3_LOG_ON")
    if [ -z "$c3_mal_on" ]; then
        echo "C3 fp32 compatibility test failed: missing mean acceptance length in $C3_LOG_ON."
        exit 1
    fi

    c3_compatible=$(check_relative_acceptance_length_delta "$c3_mal_off" "$c3_mal_on" "$c3_fp32_relative_tolerance")
    if [ "$c3_compatible" != "True" ]; then
        echo "Speculative decoding fp32-logits compatibility test failed (AL off=$c3_mal_off, on=$c3_mal_on, tol=$c3_fp32_relative_tolerance)."
        exit 1
    fi
    echo "Speculative decoding fp32-logits compatibility test passed (AL off=$c3_mal_off, on=$c3_mal_on, tol=$c3_fp32_relative_tolerance)."

    # -----------------------------------------------------------------
    # Request cancellation sweeps (BLS model)
    # -----------------------------------------------------------------
    BLS_MODEL_DIR=${ENGINES_DIR}/c5-3a30t_fp8
    BLS_DRAFT_MODEL_DIR=${ENGINES_DIR}/c5-3a30t_eagle_bf16

    cd tests/cohere
    export PYTHONPATH="${VLLM_WORKSPACE}:${PYTHONPATH}"

    # Request cancellation sweep [32, 64] — SD BLS
    echo "Running request cancellation sweep (32, 64) SD with BLS model: $BLS_MODEL_DIR"
    python3 test_request_cancellation.py --model=$BLS_MODEL_DIR --tp-size 1 --draft_model $BLS_DRAFT_MODEL_DIR --num-requests 32 64 || errors=1

    # Request cancellation sweep [32, 64] — non-SD BLS
    echo "Running request cancellation sweep (32, 64) non-SD with BLS model: $BLS_MODEL_DIR"
    python3 test_request_cancellation.py --model=$BLS_MODEL_DIR --tp-size 1 --disable-spec --num-requests 32 64 || errors=1

    exit $errors
}

run_vision() {
    echo "Running vision tests..."

    # originally we are in vllm-cohere/tests directory
    cd ../

    # set environment variables
    VISION_MODEL_DIR=${ENGINES_DIR}/command-a-vision_fp8
    export VLLM_WORKER_MULTIPROC_METHOD=spawn

    echo "Vision model directory: $VISION_MODEL_DIR"

    # Run vision test with the downloaded model
    # The test checks if the model can identify "duck" and "lion" in images
    python3 tests/cohere/test_vision.py --model $VISION_MODEL_DIR --tensor-parallel-size 2

    if [ $? -eq 0 ]; then
        echo "Vision tests passed."
        exit 0
    else
        echo "Vision tests failed."
        exit 1
    fi
}

run_model_arch_c5_3a30t_checks() {
    echo "Running model architecture c5 checks..."

    cd "${VLLM_WORKSPACE}"

    # set environment variables
    MODEL_DIR=${ENGINES_DIR}/c5-3a30t_fp8
    export VLLM_WORKER_MULTIPROC_METHOD=spawn

    MODELS=c5-3a30t_fp8 TP_SIZE=1 run_bee_samples || return 1

    echo "Model architecture c5 checks passed."
    return 0
}

run_model_arch_c5_lora_checks() {
    echo "Running c5 LoRA serving sanity check..."

    cd "${VLLM_WORKSPACE}"

    MODEL_DIR=${ENGINES_DIR}/c5-3a30t-petfatt-bf16
    export VLLM_WORKER_MULTIPROC_METHOD=spawn

    # If C5_LORA_DIR is not provided, generate a zero-weight synthetic adapter
    # from the model config so the full vLLM LoRA path is still exercised.
    _LORA_DIR="${C5_LORA_DIR:-}"
    if [[ -z "${_LORA_DIR}" ]]; then
        _DUMMY_LORA_DIR="${OUTPUT_DIR}/c5_dummy_lora"
        echo "C5_LORA_DIR not set — generating a zero-weight dummy LoRA at ${_DUMMY_LORA_DIR}"
        if python3 tests/cohere/scripts/create_dummy_lora.py \
               --model-dir "$MODEL_DIR" \
               --output-dir "$_DUMMY_LORA_DIR"; then
            _LORA_DIR="$_DUMMY_LORA_DIR"
        else
            echo "Error: dummy LoRA creation failed."
            return 1
        fi
    fi

    echo "Running c5 LoRA serving sanity check with model: $MODEL_DIR"
    if ! C5_MODEL_DIR=$MODEL_DIR C5_LORA_DIR=$_LORA_DIR pytest -v -s tests/cohere/test_c5_lora.py; then
        echo "Model architecture c5 LoRA checks failed."
        return 1
    fi

    echo "Model architecture c5 LoRA checks passed."
    return 0
}

run_bee_sample_tb_check() {
    echo "Running bee sample thinking-budget check..."

    cd "${VLLM_WORKSPACE}"

    export VLLM_WORKER_MULTIPROC_METHOD=spawn

    MODELS=c5-3a30t_fp8 TP_SIZE=1 ENABLE_THINKING_BUDGET=1 run_bee_samples || return 1

    echo "Bee sample thinking-budget check passed."
    return 0
}

run_model_arch_reward_checks() {
    echo "Running model architecture reward checks..."

    cd "${VLLM_WORKSPACE}"

    # set environment variables
    REWARD_MODEL_DIR=${ENGINES_DIR}/reward_v430
    export VLLM_WORKER_MULTIPROC_METHOD=spawn

    echo "Reward model directory: $REWARD_MODEL_DIR"

    # Run reward test with the downloaded model
    # The test checks reward score
    if python3 tests/cohere/test_reward.py --model $REWARD_MODEL_DIR --tensor-parallel-size 2; then
        echo "Model architecture reward checks passed."
        return 0
    else
        echo "Model architecture reward checks failed."
        return 1
    fi
}

run_model_arch() {
    echo "Running model architecture tests..."
    local failed=0

    run_model_arch_logits_checks || failed=1
    run_model_arch_reward_checks || failed=1
    run_model_arch_c5_3a30t_checks || failed=1
    run_model_arch_c5_lora_checks || failed=1

    if [[ $failed -ne 0 ]]; then
        echo "Model architecture tests FAILED."
        exit 1
    fi
    echo "Model architecture tests passed."
}

run_c4_sanity_check() {
    echo "Running c4 sanity check tests..."
    # c4-25a218t_fp8_eagle_l5 contains base at root + draft under eagle/
    C4_MODEL_DIR=${ENGINES_DIR}/c4-25a218t_fp8_eagle_l5
    echo "Running c4 sanity check tests with model: $C4_MODEL_DIR"

    python3 tests/cohere/test_corrupted_tokens.py --model  $C4_MODEL_DIR --tensor_parallel_size 4 --num_iterations 4

    if [ $? -eq 0 ]; then
        echo "C4 sanity check tests passed."
        exit 0
    else
        echo "C4 sanity check tests failed."
        exit 1
    fi
}

run_tests() {
    if [[ -z "${TEST_GROUP:-}" ]]; then
        echo "Error: TEST_GROUP environment variable is not set"
        echo "Available test groups: cpu_check, fast_check, model_arch, model_arch_logits, model_arch_reward, model_arch_c5_3a30t, model_arch_c5_lora, quantization, quantization_32bit_logits, GG, guided_generation, thinking_budget, bee_sample_tb_check, lm_eval, bee_eval, bee_samples, performance, speculative_decoding, vision, c4_sanity_check"
        exit 1
    fi

    # Print GPU usage before running tests
    print_gpu_usage

    case $TEST_GROUP in
        cpu_check)
            run_cpu_tests
            ;;
        fast_check)
            run_fast_check
            ;;
        model_arch)
            run_model_arch
            ;;
        quantization_32bit_logits)
            run_model_arch_logits_checks
            ;;
        model_arch_reward)
            run_model_arch_reward_checks
            ;;
        model_arch_c5_3a30t)
            run_model_arch_c5_3a30t_checks
            ;;
        model_arch_c5_lora)
            run_model_arch_c5_lora_checks
            ;;
        lm_eval)
            run_lm_eval
            ;;
        bee_eval)
            run_bee_eval
            ;;
        bee_samples)
            run_bee_samples
            ;;
        performance)
            run_performance
            ;;
        guided_generation)
            run_guided_generation
            ;;
        bee_sample_tb_check)
            run_bee_sample_tb_check
            ;;
        thinking_budget)
            run_thinking_budget
            ;;
        speculative_decoding)
            run_speculative_decoding
            ;;
        vision)
            run_vision
            ;;
        c4_sanity_check)
            run_c4_sanity_check
            ;;
        *)
            echo "Unknown test group: $TEST_GROUP"
            echo "Available test groups: cpu_check, fast_check, model_arch, model_arch_logits, model_arch_reward, model_arch_c5_3a30t, model_arch_c5_lora, quantization, quantization_32bit_logits, GG, guided_generation, thinking_budget, bee_sample_tb_check, lm_eval, bee_eval, bee_samples, performance, speculative_decoding, vision, c4_sanity_check"
            exit 1
            ;;
    esac
}

if ! run_tests; then
    echo "Tests FAILED."
    exit 1
fi

echo "All tests passed successfully."
