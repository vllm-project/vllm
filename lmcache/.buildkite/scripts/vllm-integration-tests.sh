#!/usr/bin/bash
#
# This test script runs integration tests for the LMCache integration with vLLM.
# A lmcache/vllm-openai container image is built by this script from the LMCache code base
# the script is running from and the latest nightly build of vLLM. It is therefore using the
# latest of both code bases to build the image which it then performs tests on.
#
# It’s laid out as follows:
# - UTILITIES:  utility functions
# - TESTS:      test functions
# - SETUP:      environment setup steps
# - MAIN:       test execution steps
#
# It requires the following to be installed to run:
# - curl
# - docker engine (daemon running)
# - NVIDIA Container Toolkit:
#   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
#
# Note: The script should be run from the LMCache code base root.
# Note: L4 CI runners cannot use Flash Infer

set -e
trap 'cleanup $?' EXIT INT TERM

CID=
PREFILLER_CID=
DECODER_CID=
CID1=
CID2=
HF_TOKEN=
SERVER_WAIT_TIMEOUT=180
PORT=
PORT1=
PORT2=

#############
# UTILITIES #
#############

cleanup() {
    local code="${1:-0}"

    echo "→ Cleaning up Docker containers and ports..." >&2

    # Clean up container IDs if defined
    for cid_var in CID PREFILLER_CID DECODER_CID CID1 CID2; do
        local cid="${!cid_var:-}"
        if [[ -n "$cid" ]]; then
            echo "  - Killing and removing container: $cid" >&2
            docker kill "$cid" >&2 || true
            docker rm "$cid" >&2 || true
            printf -v "$cid_var" ''
        fi
    done

    # Clean up ports if defined
    for port_var in PORT PORT1 PORT2; do
        local port="${!port_var:-}"
        if [[ -n "$port" ]]; then
            echo "  - Killing and removing port: $port" >&2
            fuser -k "${port}/tcp" >&2 || true
            if [[ "$port_var" != "PORT" ]]; then
                printf -v "$port_var" ''
            fi
        fi
    done
    
    # Wait for GPU memory to be fully released
    echo "  - Waiting 5 seconds for GPU memory to be released..." >&2
    sleep 5
}

find_available_port() {
    local start_port=${1:-8000}
    local port=$start_port

    while [ $port -lt 65536 ]; do
        # Check if port is available using netstat
        if ! netstat -tuln 2>/dev/null | grep -q ":${port} "; then
            # Double-check by trying to bind to the port with nc
            if timeout 1 bash -c "</dev/tcp/127.0.0.1/${port}" 2>/dev/null; then
                # Port is in use, try next one
                ((port++))
                continue
            else
                # Port is available
                echo $port
                return 0
            fi
        fi
        ((port++))
    done

    echo "ERROR: No available ports found starting from $start_port" >&2
    return 1
}

build_lmcache_vllmopenai_image() {
    cp example_build.sh test-build.sh
    chmod 755 test-build.sh
    ./test-build.sh
}

wait_for_openai_api_server() {
    local port="$1"
    local model="$2"
    local cid="$3"

    if ! timeout "$SERVER_WAIT_TIMEOUT" bash -c "
        echo \"Curl /v1/models endpoint\"
        until curl -s 127.0.0.1:${port}/v1/models \
                | grep -Fq \"\\\"id\\\":\\\"${model}\\\"\"; do
            sleep 30
        done
        echo \"Model ${model} is available on ${port}\"
    "; then
        echo "OpenAI API server did not start"
        docker logs $cid
        return 1
    fi
}

run_lmcache_vllmopenai_container() {
    local docker="$1"
    local vllm="$2"
    local cfg_name="$3"
    LOGFILE="/tmp/build_${BUILD_ID}_${cfg_name}.log"

    # Ensure host directory exists for socket mapping
    mkdir -p "/tmp/lmcache_internal_api_server/${PORT}"

    # Pick the GPUs based on config
    gpu_count=$(yq -r '.docker.gpu_count // 1' "$cfg_file")
    source "$ORIG_DIR/.buildkite/scripts/pick-free-gpu.sh" 40000 "$gpu_count"
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        echo "❌ Failed to select $gpu_count GPU(s)"
        exit 1
    fi
    best_gpu="${CUDA_VISIBLE_DEVICES}"

    # docker args
    docker_args=(
        --runtime nvidia
        --network host
        --volume ~/.cache/huggingface:/root/.cache/huggingface
        --volume "${CONFIG_DIR}/lmcache_configs:/etc/lmcache:ro"
        --volume /tmp/lmcache_internal_api_server/${PORT}:/tmp/lmcache_internal_api_server
        --env VLLM_USE_FLASHINFER_SAMPLER=0
        --env HF_TOKEN="$HF_TOKEN"
    )
    
    # Handle GPU assignment based on count
    if [ "$gpu_count" -gt 1 ]; then
        # Multi-GPU: expose all and use CUDA_VISIBLE_DEVICES to restrict
        docker_args+=(--gpus all)
        docker_args+=(--env "CUDA_VISIBLE_DEVICES=${best_gpu}")
    else
        # Single GPU: use device isolation
        docker_args+=(--gpus "device=${best_gpu}")
    fi
    while IFS= read -r e; do
        [[ -n $e ]] && docker_args+=(--env "$e")
    done < <(yq -r '.env[]?' <<<"$docker")

    # vllm args
    vllm_model="$(yq -r '.model' <<<"$vllm")"
    mapfile -t vllm_cli_args < <(yq -r '.args // [] | .[]' <<<"$vllm")
    cmd_args=(
        lmcache/vllm-openai:build-latest
        "$vllm_model"
    )
    cmd_args+=("${vllm_cli_args[@]}")
    cmd_args+=("--port" "$PORT")

    # Start docker
    CID=$(
        docker run -d \
            "${docker_args[@]}" \
            "${cmd_args[@]}"
    )

    wait_for_openai_api_server "$PORT" "$vllm_model" "$CID"

    # Logging
    touch "$LOGFILE"
    docker logs -f "$CID" >>"$LOGFILE" 2>&1 &
    LOG_PID=$!
}

run_pd_lmcache() {
    local prefiller_docker="$1"
    local prefiller_vllm="$2"
    local decoder_docker="$3"
    local decoder_vllm="$4"
    local cfg_name="$5"
    PREFILLER_LOGFILE="/tmp/build_${BUILD_ID}_${cfg_name}_prefiller.log"
    DECODER_LOGFILE="/tmp/build_${BUILD_ID}_${cfg_name}_decoder.log"

    # Pick 2 free GPUs for prefiller and decoder
    source "$ORIG_DIR/.buildkite/scripts/pick-free-gpu.sh" 40000 2
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        echo "❌ Failed to select 2 free GPUs"
        exit 1
    fi
    IFS=',' read -ra SELECTED_GPUS <<< "$CUDA_VISIBLE_DEVICES"
    if [ ${#SELECTED_GPUS[@]} -ne 2 ]; then
        echo "❌ Expected 2 GPUs, but got ${#SELECTED_GPUS[@]}: ${CUDA_VISIBLE_DEVICES}"
        exit 1
    fi
    GPU_PREFILLER="${SELECTED_GPUS[0]}"
    GPU_DECODER="${SELECTED_GPUS[1]}"
    echo "Selected GPU ${GPU_PREFILLER} for prefiller, GPU ${GPU_DECODER} for decoder"

    # Ensure host directory exists for socket mapping
    mkdir -p "/tmp/lmcache_internal_api_server/${PORT1}"
    mkdir -p "/tmp/lmcache_internal_api_server/${PORT2}"

    ########## Prefiller ##########
    # docker args
    prefiller_docker_args=(
        --runtime nvidia
        --network host
        --gpus "device=${GPU_PREFILLER}"
        --volume ~/.cache/huggingface:/root/.cache/huggingface
        --volume /tmp/lmcache_internal_api_server/${PORT1}:/tmp/lmcache_internal_api_server
        --env VLLM_USE_FLASHINFER_SAMPLER=0
        --env HF_TOKEN="$HF_TOKEN"
        --env UCX_TLS=cuda_ipc,cuda_copy,tcp
        --ipc host
        --shm-size 4G
    )
    while IFS= read -r e; do
        [[ -n $e ]] && prefiller_docker_args+=(--env "$e")
    done < <(yq -r '.env[]?' <<<"$prefiller_docker")
    proxy=$(yq -er '."proxy-port"' <<<"$prefiller_docker" 2>/dev/null)
    prefiller_docker_args+=(--env "LMCACHE_PD_PROXY_PORT=$proxy")

    # vllm args
    prefiller_vllm_model="$(yq -r '.model' <<<"$prefiller_vllm")"
    mapfile -t prefiller_vllm_cli_args < <(yq -r '.args // [] | .[]' <<<"$prefiller_vllm")
    prefiller_cmd_args=(
        lmcache/vllm-openai:build-latest
        "$prefiller_vllm_model"
    )
    prefiller_cmd_args+=("${prefiller_vllm_cli_args[@]}")
    prefiller_cmd_args+=("--port" "$PORT1")

    # Start docker
    PREFILLER_CID=$(
        docker run -d \
            "${prefiller_docker_args[@]}" \
            "${prefiller_cmd_args[@]}"
    )

    # Health check
    wait_for_openai_api_server "$PORT1" "$prefiller_vllm_model" "$PREFILLER_CID"

    # Logging
    touch "$PREFILLER_LOGFILE"
    docker logs -f "$PREFILLER_CID" >>"$PREFILLER_LOGFILE" 2>&1 &

    ########## Decoder ##########
    # docker args
    decoder_docker_args=(
        --runtime nvidia
        --network host
        --gpus "device=${GPU_DECODER}"
        --volume ~/.cache/huggingface:/root/.cache/huggingface
        --volume /tmp/lmcache_internal_api_server/${PORT2}:/tmp/lmcache_internal_api_server
        --env VLLM_USE_FLASHINFER_SAMPLER=0
        --env HF_TOKEN="$HF_TOKEN"
        --env UCX_TLS=cuda_ipc,cuda_copy,tcp
        --ipc host
        --shm-size 4G
    )
    while IFS= read -r e; do
        [[ -n $e ]] && decoder_docker_args+=(--env "$e")
    done < <(yq -r '.env[]?' <<<"$decoder_docker")
    init=$(yq -er '."init-port"' <<<"$decoder_docker" 2>/dev/null)
    decoder_docker_args+=(--env "LMCACHE_PD_PEER_INIT_PORT=$init")
    alloc=$(yq -er '."alloc-port"' <<<"$decoder_docker" 2>/dev/null)
    decoder_docker_args+=(--env "LMCACHE_PD_PEER_ALLOC_PORT=$alloc")

    # vllm args
    decoder_vllm_model="$(yq -r '.model' <<<"$decoder_vllm")"
    mapfile -t decoder_vllm_cli_args < <(yq -r '.args // [] | .[]' <<<"$decoder_vllm")
    decoder_cmd_args=(
        lmcache/vllm-openai:build-latest
        "$decoder_vllm_model"
    )
    decoder_cmd_args+=("${decoder_vllm_cli_args[@]}")
    decoder_cmd_args+=("--port" "$PORT2")

    # Start docker
    DECODER_CID=$(
        docker run -d \
            "${decoder_docker_args[@]}" \
            "${decoder_cmd_args[@]}"
    )

    # Health check
    wait_for_openai_api_server "$PORT2" "$decoder_vllm_model" "$DECODER_CID"

    # Logging
    touch "$DECODER_LOGFILE"
    docker logs -f "$DECODER_CID" >>"$DECODER_LOGFILE" 2>&1 &

    ########## Proxy ##########
    if [ ! -d ".venv" ]; then
        UV_PYTHON=python3 uv -q venv
    fi
    source .venv/bin/activate
    uv pip install -r "$ORIG_DIR/requirements/build.txt" > /dev/null 2>&1
    uv pip install torch==2.7.1 httpx fastapi uvicorn requests > /dev/null 2>&1
    uv pip install -e "$ORIG_DIR" --no-build-isolation > /dev/null 2>&1
    # Start proxy
    python3 "$ORIG_DIR/examples/disagg_prefill/disagg_proxy_server.py" \
        --port "$PORT" \
        --prefiller-port "$PORT1" \
        --decoder-port "$PORT2" \
        --decoder-init-port "$init" \
        --decoder-alloc-port "$alloc" \
        --proxy-port "$proxy" \
        > "/tmp/build_${BUILD_ID}_${cfg_name}_proxy.log" 2>&1 &
    sleep 10
}

run_p2p_lmcache() {
    local docker1="$1"
    local vllm1="$2"
    local docker2="$3"
    local vllm2="$4"
    local cfg_name="$5"
    LOGFILE1="/tmp/build_${BUILD_ID}_${cfg_name}1.log"
    LOGFILE2="/tmp/build_${BUILD_ID}_${cfg_name}2.log"

    # Pick 2 free GPUs for instance 1 and instance 2
    source "$ORIG_DIR/.buildkite/scripts/pick-free-gpu.sh" 40000 2
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        echo "❌ Failed to select 2 free GPUs"
        exit 1
    fi
    IFS=',' read -ra SELECTED_GPUS <<< "$CUDA_VISIBLE_DEVICES"
    if [ ${#SELECTED_GPUS[@]} -ne 2 ]; then
        echo "❌ Expected 2 GPUs, but got ${#SELECTED_GPUS[@]}: ${CUDA_VISIBLE_DEVICES}"
        exit 1
    fi
    GPU_INSTANCE1="${SELECTED_GPUS[0]}"
    GPU_INSTANCE2="${SELECTED_GPUS[1]}"
    echo "Selected GPU ${GPU_INSTANCE1} for instance 1, GPU ${GPU_INSTANCE2} for instance 2"

    # Ensure host directory exists for socket mapping
    mkdir -p "/tmp/lmcache_internal_api_server/${PORT1}"
    mkdir -p "/tmp/lmcache_internal_api_server/${PORT2}"

    ########## Instance 1 ##########
    # docker args
    docker1_args=(
        --runtime nvidia
        --network host
        --gpus "device=${GPU_INSTANCE1}"
        --volume ~/.cache/huggingface:/root/.cache/huggingface
        --volume /tmp/lmcache_internal_api_server/${PORT1}:/tmp/lmcache_internal_api_server
        --env VLLM_USE_FLASHINFER_SAMPLER=0
        --env HF_TOKEN="$HF_TOKEN"
        --env UCX_TLS=tcp
        --ipc host
        --shm-size 4G
    )
    while IFS= read -r e; do
        [[ -n $e ]] && docker1_args+=(--env "$e")
    done < <(yq -r '.env[]?' <<<"$docker1")
    pull=$(yq -er '."pull-port"' <<<"$docker1" 2>/dev/null)
    docker1_args+=(--env "LMCACHE_CONTROLLER_PULL_URL=localhost:$pull")
    reply=$(yq -er '."reply-port"' <<<"$docker1" 2>/dev/null)
    docker1_args+=(--env "LMCACHE_CONTROLLER_REPLY_URL=localhost:$reply")

    # vllm args
    vllm1_model="$(yq -r '.model' <<<"$vllm1")"
    mapfile -t vllm1_cli_args < <(yq -r '.args // [] | .[]' <<<"$vllm1")
    cmd_args1=(
        lmcache/vllm-openai:build-latest
        "$vllm1_model"
    )
    cmd_args1+=("${vllm1_cli_args[@]}")
    cmd_args1+=("--port" "$PORT1")

    ##### Controller part start #####
    if [ ! -d ".venv" ]; then
        UV_PYTHON=python3 uv -q venv
    fi
    source .venv/bin/activate
    uv pip install -r "$ORIG_DIR/requirements/build.txt" > /dev/null 2>&1
    uv pip install torch==2.7.1 httpx fastapi uvicorn > /dev/null 2>&1
    uv pip install -e "$ORIG_DIR" --no-build-isolation > /dev/null 2>&1
    # Start controller
    PYTHONHASHSEED=123 lmcache_controller \
        --host localhost \
        --port "$PORT" \
        --monitor-ports "{\"pull\": ${pull}, \"reply\": ${reply}}" \
        > "/tmp/build_${BUILD_ID}_${cfg_name}_controller.log" 2>&1 &
    sleep 10
    ##### Controller part end #####

    # Start docker
    CID1=$(
        docker run -d \
            "${docker1_args[@]}" \
            "${cmd_args1[@]}"
    )

    # Health check
    wait_for_openai_api_server "$PORT1" "$vllm1_model" "$CID1"

    # Logging
    touch "$LOGFILE1"
    docker logs -f "$CID1" >>"$LOGFILE1" 2>&1 &

    ########## Instance 2 ##########
    # docker args
    docker2_args=(
        --runtime nvidia
        --network host
        --gpus "device=${GPU_INSTANCE2}"
        --volume ~/.cache/huggingface:/root/.cache/huggingface
        --volume /tmp/lmcache_internal_api_server/${PORT2}:/tmp/lmcache_internal_api_server
        --env VLLM_USE_FLASHINFER_SAMPLER=0
        --env HF_TOKEN="$HF_TOKEN"
        --env UCX_TLS=tcp
        --ipc host
        --shm-size 4G
    )
    while IFS= read -r e; do
        [[ -n $e ]] && docker2_args+=(--env "$e")
    done < <(yq -r '.env[]?' <<<"$docker2")
    pull=$(yq -er '."pull-port"' <<<"$docker2" 2>/dev/null)
    docker2_args+=(--env "LMCACHE_CONTROLLER_PULL_URL=localhost:$pull")
    reply=$(yq -er '."reply-port"' <<<"$docker2" 2>/dev/null)
    docker2_args+=(--env "LMCACHE_CONTROLLER_REPLY_URL=localhost:$reply")

    # vllm args
    vllm2_model="$(yq -r '.model' <<<"$vllm2")"
    mapfile -t vllm2_cli_args < <(yq -r '.args // [] | .[]' <<<"$vllm2")
    cmd_args2=(
        lmcache/vllm-openai:build-latest
        "$vllm2_model"
    )
    cmd_args2+=("${vllm2_cli_args[@]}")
    cmd_args2+=("--port" "$PORT2")

    # Start docker
    CID2=$(
        docker run -d \
            "${docker2_args[@]}" \
            "${cmd_args2[@]}"
    )

    # Health check
    wait_for_openai_api_server "$PORT2" "$vllm2_model" "$CID2"

    # Logging
    touch "$LOGFILE2"
    docker logs -f "$CID2" >>"$LOGFILE2" 2>&1 &
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo " "
    echo "Options:"
    echo "  --hf-token|-hft              HuggingFace access token for downloading model(s)"
    echo "  --server-wait-timeout|-swt   Wait time in seconds for vLLM OpenAI server to start"
    echo "  --help|-h                    Print usage"
    echo "  --configs|-c                 Path to a file containing one config filename per line (required)"
    echo "  --tests|-t                   Test mode"
}

#########
# TESTS #
#########

check_memory_leak() {
    local port="$1"

    # Socket path on host: /tmp/lmcache_internal_api_server/{port}/socket_7000
    local socket_path="/tmp/lmcache_internal_api_server/${port}/socket_7000"

    # Get use_hot from /conf endpoint
    local use_hot
    use_hot=$(curl -s --unix-socket "$socket_path" "http://localhost/conf" 2>/dev/null | jq -r '.local_cpu // false')
    if [ -z "$use_hot" ] || [ "$use_hot" = "null" ]; then
        use_hot="false"
    fi

    echo "→ Checking memory leak on socket_path $socket_path (use_hot=$use_hot)..."

    # Fetch metrics from the prometheus endpoint via unix socket
    local metrics
    metrics=$(curl -s --unix-socket "$socket_path" "http://localhost/metrics" 2>/dev/null)
    if [ -z "$metrics" ]; then
        echo "ERROR: Failed to fetch metrics from socket_path $socket_path"
        return 1
    fi

    # Extract metric values
    local local_cpu_hot_cache_count
    local active_memory_objs_count
    local pinned_memory_objs_count
    local pin_monitor_pinned_objects_count

    local_cpu_hot_cache_count=$(echo "$metrics" | grep -E '^lmcache:local_cpu_hot_cache_count\b' | awk '{print $2}' | head -n 1)
    active_memory_objs_count=$(echo "$metrics" | grep -E '^lmcache:active_memory_objs_count\b' | awk '{print $2}' | head -n 1)
    pinned_memory_objs_count=$(echo "$metrics" | grep -E '^lmcache:pinned_memory_objs_count\b' | awk '{print $2}' | head -n 1)
    pin_monitor_pinned_objects_count=$(echo "$metrics" | grep -E '^lmcache:pin_monitor_pinned_objects_count\b' | awk '{print $2}' | head -n 1)

    # Default to 0 if not found
    local_cpu_hot_cache_count=${local_cpu_hot_cache_count:-0}
    active_memory_objs_count=${active_memory_objs_count:-0}
    pinned_memory_objs_count=${pinned_memory_objs_count:-0}
    pin_monitor_pinned_objects_count=${pin_monitor_pinned_objects_count:-0}

    # Convert to integer (remove decimal part if any)
    local_cpu_hot_cache_count=$(printf "%.0f" "$local_cpu_hot_cache_count")
    active_memory_objs_count=$(printf "%.0f" "$active_memory_objs_count")
    pinned_memory_objs_count=$(printf "%.0f" "$pinned_memory_objs_count")
    pin_monitor_pinned_objects_count=$(printf "%.0f" "$pin_monitor_pinned_objects_count")

    echo "  local_cpu_hot_cache_count: $local_cpu_hot_cache_count"
    echo "  active_memory_objs_count: $active_memory_objs_count"
    echo "  pinned_memory_objs_count: $pinned_memory_objs_count"
    echo "  pin_monitor_pinned_objects_count: $pin_monitor_pinned_objects_count"

    local has_leak=false

    # Check pinned_memory_objs_count must be 0
    if [ "$pinned_memory_objs_count" -ne 0 ]; then
        echo "ERROR: Memory leak detected - pinned_memory_objs_count ($pinned_memory_objs_count) should be 0"
        has_leak=true
    fi

    # Check based on use_hot setting
    if [ "$use_hot" = "false" ] || [ "$use_hot" = "False" ]; then
        # use_hot is False: both local_cpu_hot_cache_count and active_memory_objs_count should be 0
        if [ "$local_cpu_hot_cache_count" -ne 0 ]; then
            echo "ERROR: Memory leak detected - local_cpu_hot_cache_count ($local_cpu_hot_cache_count) should be 0 when use_hot=false"
            has_leak=true
        fi
        if [ "$active_memory_objs_count" -ne 0 ]; then
            echo "ERROR: Memory leak detected - active_memory_objs_count ($active_memory_objs_count) should be 0 when use_hot=false"
            has_leak=true
        fi
    else
        # use_hot is True: active_memory_objs_count should equal local_cpu_hot_cache_count
        if [ "$active_memory_objs_count" -ne "$local_cpu_hot_cache_count" ]; then
            echo "ERROR: Memory leak detected - active_memory_objs_count ($active_memory_objs_count) should equal local_cpu_hot_cache_count ($local_cpu_hot_cache_count) when use_hot=true"
            has_leak=true
        fi
    fi

    if [ "$has_leak" = true ]; then
        echo "$metrics"
        return 1
    fi

    echo "  Memory leak check passed!"
    return 0
}

test_vllmopenai_server_with_lmcache_integrated() {
    local model="$1"

    http_status_code=$(
        curl --max-time 60 http://localhost:${PORT}/v1/completions \
            -w "%{http_code}" -o response-file.txt \
            -H "Content-Type: application/json" \
            -d '{
                "model": "'"$model"'",
                "prompt": "<|begin_of_text|><|system|>\nYou are a helpful AI assistant.\n<|user|>\nWhat is the capital of France?\n<|assistant|>",
                "max_tokens": 100,
                "temperature": 0.7
            }'
    )

    if [ "$http_status_code" -ne 200 ]; then
        echo "Model prompt request from OpenAI API server failed, HTTP status code: ${http_status_code}."
        cat response-file.txt
        docker logs -n 20 $CID
        return 1
    else
        echo "Model prompt request from OpenAI API server succeeded"
        cat response-file.txt
    fi
}

run_long_doc_qa() {
    local workload_config="$1"
    local port="$2"
    local check_warmup_round_time_per_prompt="${3:-"false"}"
    local check_query_ttft_per_prompt="${4:-"false"}"
    local check_query_round_time_per_prompt="${5:-"false"}"
    local feature_type="${6:-"dummy"}"
    local need_upload="${7:-"false"}"

    echo "→ Running long_doc_qa with customed workload config:"
    printf '%s\n' "$workload_config"

    local workload_args=()
    mapfile -d '' -t workload_args < <(
    jq -j '
        to_entries[]
        | select(.value != null and (.value|tostring) != "")
        | (
            if (.value|type) == "boolean" then
            if .value then ["--\(.key)", "\u0000"] else [] end
            elif (.value|type) == "array" then
            .value[]
            | ["--\(.key)", "\u0000",
                (if (type=="string") then . else tostring end),
                "\u0000"]
            else
            ["--\(.key)", "\u0000",
            (if ((.value|type)=="string") then .value else (.value|tostring) end),
            "\u0000"]
            end
        )
        | .[]
    ' <<<"$workload_config"
    )

    if [ ! -d ".venv" ]; then
        UV_PYTHON=python3 uv -q venv
    fi
    source .venv/bin/activate
    uv -q pip install openai pandas matplotlib
    json=$(
        python3 "$ORIG_DIR/benchmarks/long_doc_qa/long_doc_qa.py" \
            "${workload_args[@]}" \
            --port="$port" \
            --output="response.txt" \
            --json-output \
            2>>response.txt | tail -n 1
    )
    query_ttft_per_prompt=$(echo "$json" | jq -r '.query_ttft_per_prompt')
    query_round_time_per_prompt=$(echo "$json" | jq -r '.query_round_time_per_prompt')
    warmup_round_time_per_prompt=$(echo "$json" | jq -r '.warmup_round_time_per_prompt')

    if [ "$need_upload" = "true" ]; then
        local baseline_path="$ORIG_DIR/benchmarks/long_doc_qa/$feature_type.json"
        echo "$json"
        printf '%s\n' "$json" > "$baseline_path"

        git config user.email "$USER_EMAIL"
        git config user.name "$USER_NAME"
        git add "$baseline_path"
        git commit -m "Update long_doc_qa baseline: $feature_type.json" || true
        if ! git remote get-url internal >/dev/null 2>&1; then
            git remote add internal git@github.com:LMCache/LMCache.git
        fi
        git push internal +HEAD:benchmarks-main >/dev/null 2>&1
        return 0
    fi

    # Fetch branch
    git fetch origin benchmarks-main >/dev/null 2>&1 || true

    # Load baseline from branch
    baseline_json=$(git show origin/benchmarks-main:benchmarks/long_doc_qa/$feature_type.json 2>/dev/null || echo "")

    # Check if baseline exists, skip comparisons if not
    if [[ -z "$baseline_json" ]] || ! echo "$baseline_json" | jq -e . >/dev/null 2>&1; then
        if [[ "$feature_type" != "dummy" ]]; then
            echo "⚠️  No baseline found for $feature_type.json - skipping performance comparisons"
            echo "   This is expected for newly added configs. Baseline will be generated on next nightly run."
            echo "   Current metrics: TTFT=$query_ttft_per_prompt, Latency=$query_round_time_per_prompt, Warmup=$warmup_round_time_per_prompt"
        fi
        return 0
    fi

    # Extract baseline numbers
    expected_query_ttft_per_prompt=$(echo "$baseline_json" | jq -r '.query_ttft_per_prompt')
    expected_query_round_time_per_prompt=$(echo "$baseline_json" | jq -r '.query_round_time_per_prompt')
    expected_warmup_round_time_per_prompt=$(echo "$baseline_json" | jq -r '.warmup_round_time_per_prompt')

    if [ "$check_query_ttft_per_prompt" = "true" ]; then
        echo "Expected query ttft per prompt: $expected_query_ttft_per_prompt"
        echo "Actual query ttft per prompt: $query_ttft_per_prompt"
        awk -v expected="$expected_query_ttft_per_prompt" -v actual="$query_ttft_per_prompt" 'BEGIN {
            if (actual > expected * 1.2) {
                print "Query ttft per prompt requirement not met (>20% overhead)"
                exit 1
            } else {
                print "Query ttft per prompt requirement met"
            }
        }'
    fi

    if [ "$check_query_round_time_per_prompt" = "true" ]; then
        echo "Expected query round time per prompt: $expected_query_round_time_per_prompt"
        echo "Actual query round time per prompt: $query_round_time_per_prompt"
        awk -v expected="$expected_query_round_time_per_prompt" -v actual="$query_round_time_per_prompt" 'BEGIN {
            if (actual > expected * 1.2) {
                print "Query round time per prompt requirement not met (>20% overhead)"
                exit 1
            } else {
                print "Query round time per prompt requirement met"
            }
        }'
    fi

    if [ "$check_warmup_round_time_per_prompt" = "true" ]; then
        echo "Expected warmup round time per prompt: $expected_warmup_round_time_per_prompt"
        echo "Actual warmup round time per prompt: $warmup_round_time_per_prompt"
        awk -v expected="$expected_warmup_round_time_per_prompt" -v actual="$warmup_round_time_per_prompt" 'BEGIN {
            if (actual > expected * 1.2) {
                print "Warmup round time per prompt requirement not met (>20% overhead)"
                exit 1
            } else {
                print "Warmup round time per prompt requirement met"
            }
        }'
    fi
}

#########
# SETUP #
#########

while [ $# -gt 0 ]; do
    case "$1" in
    --configs* | -c*)
        if [[ "$1" != *=* ]]; then shift; fi
        configs_arg="${1#*=}"
        ;;
    --hf-token* | -hft*)
        if [[ "$1" != *=* ]]; then shift; fi
        HF_TOKEN="${1#*=}"
        ;;
    --server-wait-timeout* | -swt*)
        if [[ "$1" != *=* ]]; then shift; fi
        SERVER_WAIT_TIMEOUT="${1#*=}"
        if ! [[ "$SERVER_WAIT_TIMEOUT" =~ ^[0-9]+$ ]]; then
            echo "server-wait-timeout is wait time in seconds - integer only"
            exit 1
        fi
        ;;
    --help | -h)
        usage
        exit 0
        ;;
    *)
        printf >&2 "Error: Invalid argument\n"
        usage
        exit 1
        ;;
    esac
    shift
done

ORIG_DIR="$PWD"
CONFIG_DIR="${ORIG_DIR}/.buildkite/configs"

# Read the configs argument (always a file with one config per line)
if [[ ! -f "$configs_arg" ]]; then
    echo "Error: --configs file not found: $configs_arg" >&2
    exit 1
fi
mapfile -t CONFIG_NAMES < <(
    sed 's/[[:space:]]\+$//' "$configs_arg"
)

# Find an available port starting from 8000
PORT=$(find_available_port 8000)
if [ $? -ne 0 ]; then
    echo "Failed to find an available port"
    exit 1
fi
echo "Using port $PORT to send or receive requests."

# Need to run from docker directory
cd docker/

# Create the container image
build_lmcache_vllmopenai_image

########
# MAIN #
########

for cfg_name in "${CONFIG_NAMES[@]}"; do
    echo -e "\033[1;33m===== Testing LMCache with ${cfg_name} =====\033[0m"
    cfg_file="${CONFIG_DIR}/${cfg_name}"

    # Start engine
    feature_type=$(yq -r '.feature.type // ""' "$cfg_file")
    if [[ "$feature_type" == "pd" ]]; then
        PORT1=$(find_available_port 8100)
        prefiller_docker_args="$(yq '.["docker-prefiller"]' "$cfg_file")"
        prefiller_vllm_args="$(yq '.["vllm-prefiller"]' "$cfg_file")"
        PORT2=$(find_available_port 8200)
        decoder_docker_args="$(yq '.["docker-decoder"]' "$cfg_file")"
        decoder_vllm_args="$(yq '.["vllm-decoder"]' "$cfg_file")"
        run_pd_lmcache "$prefiller_docker_args" "$prefiller_vllm_args" "$decoder_docker_args" "$decoder_vllm_args" "$cfg_name" 
        model="$(yq -r '.["vllm-prefiller"].model' "$cfg_file")"
    elif [[ "$feature_type" == "p2p" ]]; then
        PORT1=$(find_available_port 8177)
        docker1_args="$(yq '.["docker1"]' "$cfg_file")"
        vllm1_args="$(yq '.["vllm1"]' "$cfg_file")"
        PORT2=$(find_available_port 8277)
        docker2_args="$(yq '.["docker2"]' "$cfg_file")"
        vllm2_args="$(yq '.["vllm2"]' "$cfg_file")"
        run_p2p_lmcache "$docker1_args" "$vllm1_args" "$docker2_args" "$vllm2_args" "$cfg_name" 
        model="$(yq -r '.["vllm1"].model' "$cfg_file")"
    elif [[ -z "$feature_type" ]]; then
        docker_args="$(yq '.docker' "$cfg_file")"
        vllm_args="$(yq '.vllm' "$cfg_file")"
        run_lmcache_vllmopenai_container "$docker_args" "$vllm_args" "$cfg_name"
        model="$(yq -r '.vllm.model' "$cfg_file")"
    fi
    
    # Send request
    test_mode="$(yq -r '.workload.type' "$cfg_file")"
    if [ "$test_mode" = "dummy" ]; then
        test_vllmopenai_server_with_lmcache_integrated "$model"
    elif [ "$test_mode" = "long_doc_qa" ]; then
        workload_yaml="$(yq "(.workload * {\"model\": \"$model\"}) | del(.type)" "$cfg_file")"
        cfg_json="$(yq '.' "$cfg_file")"
        check_warmup_round_time_per_prompt=$(
            jq -e '(.["checking-fields"] // []) | index("warmup_round_time_per_prompt") != null' \
                <<< "$cfg_json" >/dev/null && echo true || echo false
        )
        check_query_round_time_per_prompt=$(
            jq -e '(.["checking-fields"] // []) | index("query_round_time_per_prompt") != null' \
                <<< "$cfg_json" >/dev/null && echo true || echo false
        )
        check_query_ttft_per_prompt=$(
            jq -e '(.["checking-fields"] // []) | index("warmup_ttft_per_prompt") != null' \
                <<< "$cfg_json" >/dev/null && echo true || echo false
        )
        if [[ "$feature_type" == "p2p" ]]; then
            run_long_doc_qa "$workload_yaml" "$PORT1"
            run_long_doc_qa "$workload_yaml" "$PORT2" "$check_warmup_round_time_per_prompt" "$check_query_ttft_per_prompt" "$check_query_round_time_per_prompt" "${cfg_name%.yaml}" "$NEED_UPLOAD"
        else
            run_long_doc_qa "$workload_yaml" "$PORT" "$check_warmup_round_time_per_prompt" "$check_query_ttft_per_prompt" "$check_query_round_time_per_prompt" "${cfg_name%.yaml}" "$NEED_UPLOAD"
        fi
    fi

    # Check memory leak after test
    sleep 15
    echo "→ Checking for memory leaks..."
    if [[ "$feature_type" == "pd" ]]; then
        # Check both prefiller and decoder instances
        if ! check_memory_leak "$PORT1"; then
            echo "Memory leak check failed for prefiller (port $PORT1)"
            exit 1
        fi
        if ! check_memory_leak "$PORT2"; then
            echo "Memory leak check failed for decoder (port $PORT2)"
            exit 1
        fi
    elif [[ "$feature_type" == "p2p" ]]; then
        # TODO: p2p check_memory_leak has a known bug, skip for now
        echo "⚠️  Skipping memory leak check for p2p case: check_memory_leak has a known bug that is being fixed."
    elif [[ "$cfg_name" == "multi_device.yaml" || "$cfg_name" == "layerwise.yaml" ]]; then
        echo "⚠️  Skipping memory leak check for $cfg_name case as it's a flaky test while run check_memory_leak check."
    else
        # Single instance
        if ! check_memory_leak "$PORT"; then
            echo "Memory leak check failed for $cfg_name"
            exit 1
        fi
    fi

    cleanup 0
done

exit 0
