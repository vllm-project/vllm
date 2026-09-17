#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

model="${VLLM_TEST_POWERMOE_MODEL:-ibm/PowerMoE-3b}"
port="${VLLM_TEST_PORT:-8000}"
server_pid=
server_log=
decode_token=11
num_decode_tokens=128

cd "$repo_root"

cleanup_server() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid"
        wait "$server_pid" || true
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
            return 1
        fi
        sleep 1
    done
    echo "Timed out waiting for vLLM server health check" >&2
    return 1
}

request_logits() {
    local prompts_json=$1
    local expected_choices=$2
    local response

    response=$(printf \
        '{"model":"%s","prompt":%s,"temperature":0,"seed":0,"max_tokens":%s,"logprobs":64,"ignore_eos":true,"allowed_token_ids":[%s]}' \
        "$model" "$prompts_json" "$num_decode_tokens" "$decode_token" |
        curl --fail --silent --show-error \
            --max-time 1200 \
            "http://127.0.0.1:$port/v1/completions" \
            -H "Content-Type: application/json" \
            --data-binary @-)

    mapfile -t logprob_arrays < <(
        grep -o '"token_logprobs":\[[^]]*\]' <<<"$response" |
            sed 's/^"token_logprobs":\[//; s/\]$//'
    )
    if [[ "${#logprob_arrays[@]}" -ne "$expected_choices" ]]; then
        echo "Expected $expected_choices choices, received ${#logprob_arrays[@]}" >&2
        return 1
    fi
    local array
    local value
    local -a values
    for array in "${logprob_arrays[@]}"; do
        IFS=, read -r -a values <<<"$array"
        if [[ "${#values[@]}" -ne "$num_decode_tokens" ]]; then
            echo "Completion returned an unexpected number of decode steps" >&2
            return 1
        fi
        for value in "${values[@]}"; do
            if [[ ! "$value" =~ ^-?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
                echo "Completion returned non-finite raw logits" >&2
                return 1
            fi
        done
        printf '%s\n' "$array"
    done
}

make_prompt() {
    local length=$1
    local offset=${2:-0}
    local i
    local -a tokens

    for ((i = 0; i < length; i++)); do
        tokens+=( $((3 + (i + offset) % 253)) )
    done
    local IFS=,
    printf '[%s]' "${tokens[*]}"
}

first_difference() {
    local actual=$1
    local expected=$2
    local position=$3
    local -a actual_values
    local -a expected_values
    local step

    IFS=, read -r -a actual_values <<<"$actual"
    IFS=, read -r -a expected_values <<<"$expected"
    for ((step = 0; step < num_decode_tokens; step++)); do
        if [[ "${actual_values[step]}" != "${expected_values[step]}" ]]; then
            printf 'batch position %s: step=%s, token=%s, baseline=%s, actual=%s' \
                "$position" \
                "$step" \
                "$decode_token" \
                "${expected_values[step]}" \
                "${actual_values[step]}"
            return
        fi
    done
}

run_e2e() {
    local mode=$1
    local expectation=$2
    local batch_invariant=0
    local needle
    local batch
    local baseline_logits
    local batch_logits_output
    local difference
    local position
    local -a fillers
    local -a batch_logits
    local -a differences

    if [[ "$mode" == "batch-invariant" ]]; then
        batch_invariant=1
    fi

    server_log=$(mktemp "${TMPDIR:-/tmp}/xpu-batch-invariance-${mode}.XXXXXX.log")
    VLLM_BATCH_INVARIANT="$batch_invariant" vllm serve "$model" \
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

    echo "Requesting BS=1 baseline (257 prefill tokens, $num_decode_tokens decode tokens)..."
    baseline_logits=$(request_logits "$needle" 1)
    echo "Received BS=1 baseline."
    echo "Requesting BS=16 batch (3358 prefill tokens, $((16 * num_decode_tokens)) decode tokens)..."
    batch_logits_output=$(request_logits "$batch" 16)
    mapfile -t batch_logits <<<"$batch_logits_output"
    echo "Received BS=16 batch; comparing logits..."

    for position in 0 7 15; do
        difference=$(first_difference "${batch_logits[position]}" "$baseline_logits" "$position")
        if [[ -n "$difference" ]]; then
            differences+=("$difference")
        fi
    done

    if [[ "$expectation" == "identical" ]] && [[ "${#differences[@]}" -ne 0 ]]; then
        echo "Batch-invariant PowerMoE logits changed: ${differences[*]}" >&2
        return 1
    fi
    if [[ "$expectation" == "different" ]] &&
        [[ "${#differences[@]}" -eq 0 ]]; then
        echo "PowerMoE produced identical BS=1 and BS=16 logits without batch-invariant mode" >&2
        return 1
    fi
    if [[ "${#differences[@]}" -ne 0 ]]; then
        echo "BS=1 vs BS=16 differences: ${differences[*]}"
    fi
    echo "PowerMoE logits were $expectation as expected"
    cleanup_server
    rm -f "$server_log"
    server_log=
}

run_e2e batch-variant different
run_e2e batch-invariant identical
