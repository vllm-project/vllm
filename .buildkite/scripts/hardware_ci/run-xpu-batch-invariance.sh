#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

model="ibm-research/PowerMoE-3b"
port=8000
num_decode_tokens=128
server_pid=
server_log=

# XPU MoE kernels may not safely handle the -1 routes emitted for padding.
export VLLM_MOE_SKIP_PADDING=0

cd "$repo_root"

cleanup_server() {
    local attempt

    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true

        for ((attempt = 0; attempt < 30; attempt++)); do
            if ! kill -0 "$server_pid" 2>/dev/null; then
                break
            fi
            sleep 1
        done

        if kill -0 "$server_pid" 2>/dev/null; then
            echo "vLLM did not stop after 30 seconds; sending SIGKILL" >&2
            kill -KILL "$server_pid" 2>/dev/null || true
        fi

        wait "$server_pid" 2>/dev/null || true
    fi

    server_pid=
}

on_exit() {
    local status=$?

    cleanup_server
    if [[ "$status" -ne 0 && -n "$server_log" && -f "$server_log" ]]; then
        if command -v buildkite-agent >/dev/null 2>&1; then
            buildkite-agent artifact upload "$server_log" ||
                echo "Failed to upload $server_log as a Buildkite artifact" >&2
        fi
    fi
    [[ -z "$server_log" ]] || rm -f "$server_log"
    exit "$status"
}
trap on_exit EXIT

wait_for_server() {
    for _ in $(seq 1 600); do
        if curl -sf "http://127.0.0.1:$port/health" >/dev/null; then
            return
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "vLLM exited before becoming healthy; see the server log above." >&2
            return 1
        fi
        sleep 1
    done
    echo "Timed out waiting for vLLM server health check" >&2
    return 1
}

request_outputs() {
    local prompts_json=$1
    local expected_choices=$2
    local response

    response=$(printf \
        '{"model":"%s","prompt":%s,"temperature":0.7,"top_p":0.95,"seed":0,"max_tokens":%s,"logprobs":64,"ignore_eos":true,"return_token_ids":true}' \
        "$model" "$prompts_json" "$num_decode_tokens" |
        curl --fail --silent --show-error \
            --max-time 1200 \
            "http://127.0.0.1:$port/v1/completions" \
            -H "Content-Type: application/json" \
            --data-binary @-)

    local -a token_id_arrays logprob_arrays
    mapfile -t token_id_arrays < <(
        grep -o '"token_ids":\[[^]]*\]' <<<"$response" |
            sed 's/^"token_ids":\[//; s/\]$//'
    )
    mapfile -t logprob_arrays < <(
        grep -o '"token_logprobs":\[[^]]*\]' <<<"$response" |
            sed 's/^"token_logprobs":\[//; s/\]$//'
    )
    if [[ "${#token_id_arrays[@]}" -ne "$expected_choices" ]] ||
        [[ "${#logprob_arrays[@]}" -ne "$expected_choices" ]]; then
        echo "Expected $expected_choices choices with token IDs and logits" >&2
        return 1
    fi
    local index
    local array
    local value
    local -a values
    for index in "${!logprob_arrays[@]}"; do
        array="${logprob_arrays[index]}"
        IFS=, read -r -a values <<<"$array"
        if [[ "${#values[@]}" -ne "$num_decode_tokens" ]]; then
            echo "Completion returned an unexpected number of decode steps" >&2
            return 1
        fi
        IFS=, read -r -a values <<<"${token_id_arrays[index]}"
        if [[ "${#values[@]}" -ne "$num_decode_tokens" ]]; then
            echo "Completion returned an unexpected number of generated tokens" >&2
            return 1
        fi
        for value in "${values[@]}"; do
            if [[ ! "$value" =~ ^[0-9]+$ ]]; then
                echo "Completion returned an invalid token ID" >&2
                return 1
            fi
        done
        IFS=, read -r -a values <<<"$array"
        for value in "${values[@]}"; do
            if [[ ! "$value" =~ ^-?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
                echo "Completion returned non-finite raw logits" >&2
                return 1
            fi
        done
        printf '%s|%s\n' "${token_id_arrays[index]}" "${logprob_arrays[index]}"
    done
}

make_prompt() {
    local length=$1
    local offset=${2:-0}
    local i
    local prompt="Use the following system-design context to answer the question. Scenario $offset:"
    for ((i = 0; i < length; i++)); do
        prompt+=" context"
    done
    prompt+="\\n\\nQuestion: How should batch-invariant inference behave when this same request is decoded alongside unrelated requests?\\nAnswer:"
    printf '"%s"' "$prompt"
}

first_difference() {
    local actual=$1
    local expected=$2
    local position=$3
    local actual_tokens=${actual%%|*}
    local expected_tokens=${expected%%|*}
    local actual_logits=${actual#*|}
    local expected_logits=${expected#*|}
    local -a actual_token_values
    local -a expected_token_values
    local -a actual_logit_values
    local -a expected_logit_values
    local step

    IFS=, read -r -a actual_token_values <<<"$actual_tokens"
    IFS=, read -r -a expected_token_values <<<"$expected_tokens"
    IFS=, read -r -a actual_logit_values <<<"$actual_logits"
    IFS=, read -r -a expected_logit_values <<<"$expected_logits"
    for ((step = 0; step < num_decode_tokens; step++)); do
        if [[ "${actual_token_values[step]}" != "${expected_token_values[step]}" ]]; then
            printf 'batch position %s: step=%s, baseline token=%s, actual token=%s' \
                "$position" \
                "$step" \
                "${expected_token_values[step]}" \
                "${actual_token_values[step]}"
            return
        fi
        if [[ "${actual_logit_values[step]}" != "${expected_logit_values[step]}" ]]; then
            printf 'batch position %s: step=%s, token=%s, baseline logit=%s, actual logit=%s' \
                "$position" \
                "$step" \
                "${expected_token_values[step]}" \
                "${expected_logit_values[step]}" \
                "${actual_logit_values[step]}"
            return
        fi
    done
}

run_e2e() {
    local needle
    local batch
    local baseline_output
    local batch_response
    local difference
    local position
    local -a fillers
    local -a batch_outputs
    local -a differences=()

    echo "Testing PowerMoE-3b: TP=2, VLLM_BATCH_INVARIANT=1, VLLM_MOE_SKIP_PADDING=0"
    server_log=$(mktemp "${TMPDIR:-/tmp}/xpu-batch-invariance.XXXXXX.log")
    VLLM_BATCH_INVARIANT=1 vllm serve "$model" \
        --tensor-parallel-size 2 \
        --max-model-len 1024 \
        --max-num-seqs 16 \
        --max-num-batched-tokens 512 \
        --enable-chunked-prefill \
        --no-enable-prefix-caching \
        --enforce-eager \
        --seed 0 \
        --max-logprobs 64 \
        --logprobs-mode raw_logits \
        --port "$port" > >(tee "$server_log") 2>&1 &
    server_pid=$!
    wait_for_server

    needle=$(make_prompt 257)
    local filler_lengths=(7 31 63 95 127 159 191 223 255 287 319 383 447)
    for position in "${!filler_lengths[@]}"; do
        fillers+=( "$(make_prompt "${filler_lengths[position]}" "$((position + 1))")" )
    done
    batch="[$needle,${fillers[0]},${fillers[1]},${fillers[2]},${fillers[3]},${fillers[4]},${fillers[5]},$needle,${fillers[6]},${fillers[7]},${fillers[8]},${fillers[9]},${fillers[10]},${fillers[11]},${fillers[12]},$needle]"

    echo "Requesting BS=1 baseline ($num_decode_tokens decode tokens)..."
    baseline_output=$(request_outputs "$needle" 1)
    echo "Received BS=1 baseline."
    echo "Requesting BS=16 batch ($((16 * num_decode_tokens)) decode tokens)..."
    batch_response=$(request_outputs "$batch" 16)
    mapfile -t batch_outputs <<<"$batch_response"
    echo "Received BS=16 batch; comparing generated tokens and logits..."

    for position in 0 7 15; do
        difference=$(first_difference "${batch_outputs[position]}" "$baseline_output" "$position")
        if [[ -n "$difference" ]]; then
            differences+=("$difference")
        fi
    done

    if [[ "${#differences[@]}" -ne 0 ]]; then
        echo "Batch-invariant PowerMoE-3b output changed: ${differences[*]}" >&2
        return 1
    fi
    echo "PowerMoE-3b batch invariance check passed"
    cleanup_server
    rm -f "$server_log"
    server_log=
}

run_e2e
