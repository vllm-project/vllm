#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -Eeuo pipefail
readonly MODEL="Qwen/Qwen3-0.6B"
readonly MODEL_REVISION="c1899de289a04d12100db370d81485cdf75e47ca"
readonly EXPECTED_TOKEN_ID="12095"
readonly EXPECTED_TEXT=" Paris"
readonly RUN_LIMIT_S=3300
readonly CLEANUP_RESERVE_S=300
if (( $# != 2 )); then
    echo "usage: $0 IMAGE_TAG BUILDKITE_COMMIT" >&2
    exit 2
fi
readonly IMAGE_TAG="$1"
readonly EXPECTED_COMMIT="$2"
readonly DEADLINE_EPOCH="$(( $(date +%s) + RUN_LIMIT_S ))"
RUN_ID="" CONTAINER_NAME="" CONTAINER_ID=""
GPU_UUID="" PORT_ONE="" PORT_TWO=""
LINK_REMAP_BASELINE=""
SNAPSHOT_PIDS=()
die() {
    echo "initialized snapshot E2E: $*" >&2
    exit 1
}

run() {
    local label="$1" maximum_s="$2" available limit
    shift 2
    available="$((DEADLINE_EPOCH - $(date +%s) - CLEANUP_RESERVE_S))"
    (( available > 0 )) || die "no time remains before the cleanup reserve"
    (( available < maximum_s )) && limit="$available" || limit="$maximum_s"
    echo "--- ${label} (timeout ${limit}s)" >&2
    timeout --signal=TERM --kill-after=30s "${limit}s" "$@"
}

link_remaps() {
    find /dev/shm -maxdepth 1 -name 'link_remap.*' \
        -printf '%p|%D|%i|%m|%s\n' | sort
}

gpu_pids() {
    nvidia-smi --id="$GPU_UUID" --query-compute-apps=pid \
        --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'
}

port_open() {
    [[ -n "$(ss -H -ltn "sport = :$1")" ]]
}

state_clean() {
    local pid port
    for pid in "${SNAPSHOT_PIDS[@]}"; do
        [[ ! -e "/proc/$pid" ]] || return 1
    done
    [[ -z "$(gpu_pids)" ]] || return 1
    [[ "$(link_remaps)" == "$LINK_REMAP_BASELINE" ]] || return 1
    for port in "$PORT_ONE" "$PORT_TWO"; do
        [[ -z "$port" ]] || ! port_open "$port" || return 1
    done
}

wait_clean() {
    local _attempt
    for ((_attempt = 0; _attempt < 60; _attempt++)); do
        state_clean && return 0
        sleep 1
    done
    return 1
}

container_names() {
    timeout 15s docker container ls --all --format '{{.Names}}'
}

cleanup() {
    local failed=0 names="" actual="" expected=""
    if [[ -n "$CONTAINER_NAME" ]]; then
        names="$(container_names 2>/dev/null)" || return 1
        if grep -Fxq "$CONTAINER_NAME" <<< "$names"; then
            actual="$(docker container inspect \
                --format '{{.Id}}|{{index .Config.Labels "ai.vllm.snapshot.e2e.run"}}' \
                "$CONTAINER_NAME" 2>/dev/null)" || return 1
            expected="${CONTAINER_ID:-${actual%%|*}}|$RUN_ID"
            if [[ "$actual" != "$expected" ]]; then
                echo "refusing to remove a container with mismatched identity" >&2
                failed=1
            else
                timeout 60s docker rm --force "$CONTAINER_NAME" >/dev/null \
                    || failed=1
            fi
        fi
        names="$(container_names 2>/dev/null)" || return 1
        grep -Fxq "$CONTAINER_NAME" <<< "$names" && failed=1
    fi
    if [[ -n "$PORT_TWO" ]] && ! wait_clean; then
        echo "process, GPU, port, or link-remap residue remained" >&2
        failed=1
    fi
    return "$failed"
}

on_exit() {
    local status=$?
    trap - EXIT INT TERM
    set +e
    cleanup || status=1
    exit "$status"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

parse_inspect() {
    timeout 30s python3 -c '
import json, math, sys
data = json.load(sys.stdin)
model, revision, gpu_uuid, token_id, text = sys.argv[1:]
expected = {
    "boundary": "post-engine-init-reloadable-state-released",
    "model": model, "model_revision": revision,
    "gpu_uuid": gpu_uuid, "oracle_token_ids": [int(token_id)],
    "oracle_text": text,
}
bad = next((field for field, value in expected.items() if data.get(field) != value), None)
if bad:
    raise SystemExit(f"inspect field {bad} mismatch: {data.get(bad)!r}")
logprob = data.get("oracle_sampled_token_logprob")
if type(logprob) is not float or not math.isfinite(logprob):
    raise SystemExit("inspect logprob is not finite")
pids = data.get("process_tree")
if not pids:
    raise SystemExit("inspect process_tree is empty")
for pid in pids:
    print(f"pid={pid}")
print(f"oracle_logprob={logprob}")
' "$MODEL" "$MODEL_REVISION" "$GPU_UUID" "$EXPECTED_TOKEN_ID" \
        "$EXPECTED_TEXT"
}

assert_oracle() {
    local port="$1" ordinal="$2" response
    response="$(run "restore ${ordinal} public completion" 120 curl \
        --fail --silent --show-error --max-time 120 \
        --header 'Content-Type: application/json' \
        --data "{\"model\":\"${MODEL}\",\"prompt\":\"The capital of France is\",\"temperature\":0,\"seed\":0,\"min_tokens\":1,\"max_tokens\":1,\"logprobs\":0,\"return_token_ids\":true}" \
        "http://127.0.0.1:${port}/v1/completions")"
    printf '%s' "$response" | timeout 30s python3 -c '
import json, math, sys
choice = json.load(sys.stdin)["choices"][0]
token_id, text, expected, ordinal = sys.argv[1:]
values = choice["logprobs"]["token_logprobs"]
actual = float(values[0]) if len(values) == 1 else math.nan
if (choice["token_ids"] != [int(token_id)] or choice["text"] != text
        or not math.isfinite(actual) or abs(actual - float(expected)) > 1e-3):
    raise SystemExit(f"public oracle mismatch: {choice!r}")
print(f"restore={ordinal} token_id={token_id} text={text!r} logprob={actual}")
' "$EXPECTED_TOKEN_ID" "$EXPECTED_TEXT" "$INSPECT_LOGPROB" "$ordinal"
}

[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] \
    || die "host must be Linux x86_64"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
    || die "BUILDKITE_COMMIT must be a full lowercase commit hash"
[[ "$(< /proc/sys/kernel/io_uring_disabled)" == "2" ]] \
    || die "kernel.io_uring_disabled must already equal 2"
run "Docker preflight" 60 docker info >/dev/null

VISIBLE_GPU="${NVIDIA_VISIBLE_DEVICES:-}"
if [[ -z "$VISIBLE_GPU" || "$VISIBLE_GPU" == "all" || "$VISIBLE_GPU" == "void" ]]; then
    VISIBLE_GPU="${CUDA_VISIBLE_DEVICES:-}"
fi
[[ -n "$VISIBLE_GPU" && "$VISIBLE_GPU" != *","* ]] \
    || die "the runner must expose exactly one selected GPU"
GPU_ROW="$(nvidia-smi --id="$VISIBLE_GPU" \
    --query-gpu=name,uuid,mig.mode.current,memory.total \
    --format=csv,noheader,nounits)"
IFS=',' read -r GPU_NAME GPU_UUID MIG_MODE GPU_MEMORY_MIB <<< "$GPU_ROW"
for variable in GPU_NAME GPU_UUID MIG_MODE GPU_MEMORY_MIB; do
    printf -v "$variable" '%s' "${!variable#"${!variable%%[![:space:]]*}"}"
done
[[ "$GPU_NAME" == *H200* && "$GPU_UUID" == GPU-* && "$MIG_MODE" == Disabled ]] \
    || die "selected device is not a full non-MIG H200: $GPU_ROW"
(( GPU_MEMORY_MIB >= 130000 )) || die "selected H200 does not expose full memory"
[[ -z "$(gpu_pids)" ]] || die "selected H200 is not idle"

RUN_ID="snapshot-e2e-$(date +%s)-$$-$RANDOM"
CONTAINER_NAME="vllm-$RUN_ID"
read -r PORT_ONE PORT_TWO < <(python3 -c 'import socket; s=[socket.socket(),socket.socket()]; [x.bind(("127.0.0.1",0)) for x in s]; print(*(x.getsockname()[1] for x in s))')
LINK_REMAP_BASELINE="$(link_remaps)"

run "pull exact candidate image" 900 docker pull --platform linux/amd64 \
    "$IMAGE_TAG" >/dev/null
IFS='|' read -r IMAGE_ID IMAGE_PLATFORM OCI_COMMIT VLLM_COMMIT < <(
    docker image inspect \
        --format '{{.Id}}|{{.Os}}/{{.Architecture}}|{{index .Config.Labels "org.opencontainers.image.revision"}}|{{index .Config.Labels "ai.vllm.build.commit"}}' \
        "$IMAGE_TAG"
)
[[ "$IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ && "$IMAGE_PLATFORM" == linux/amd64 ]] \
    || die "candidate image identity is invalid"
[[ "$OCI_COMMIT" == "$EXPECTED_COMMIT" && "$VLLM_COMMIT" == "$EXPECTED_COMMIT" ]] \
    || die "candidate image commit labels do not match"
CONTAINER_ID="$(run "start exact candidate container" 90 docker run --detach \
    --name "$CONTAINER_NAME" --label "ai.vllm.snapshot.e2e.run=$RUN_ID" \
    --gpus "device=$GPU_UUID" --user 0 --privileged --pid=host --ipc=host \
    --network=host --env CUDA_VISIBLE_DEVICES=0 --env HF_HUB_DISABLE_TELEMETRY=1 \
    --env VLLM_NO_USAGE_STATS=1 --env VLLM_USE_V2_MODEL_RUNNER=1 \
    --entrypoint sleep "$IMAGE_ID" infinity)"
[[ "$CONTAINER_ID" =~ ^[0-9a-f]{64}$ ]] || die "invalid container ID"
run "prepare private artifact root" 60 docker exec "$CONTAINER_NAME" sh -c \
    "mkdir -m 0700 /e2e && mkdir -m 0700 /e2e/hf && df -Pk / | awk 'NR == 2 && \$4 >= 10485760 {ok=1} END {exit !ok}'"
run "verify snapshot runtime" 60 docker exec "$CONTAINER_NAME" sh -c \
    'criu --version; test -x /usr/local/sbin/cuda-checkpoint; test -f /usr/local/lib/criu/cuda_plugin.so'
run "prefetch pinned public model" 900 docker exec --env HF_HOME=/e2e/hf \
    "$CONTAINER_NAME" python3 -c \
    'from pathlib import Path; from huggingface_hub import snapshot_download; import sys; p=snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2]); assert Path(p).name == sys.argv[2]' \
    "$MODEL" "$MODEL_REVISION"

OFFLINE_EXEC=(docker exec --env HF_HOME=/e2e/hf --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 --env VLLM_SNAPSHOT_TIMEOUT_S=900 \
    "$CONTAINER_NAME")
run "create compact initialized snapshot" 1200 "${OFFLINE_EXEC[@]}" \
    vllm snapshot create "$MODEL" --revision "$MODEL_REVISION" \
    --tokenizer-revision "$MODEL_REVISION" --snapshot-dir /e2e/artifact \
    --dtype float16 --max-model-len 512 --gpu-memory-utilization 0.50
wait_clean || die "create left process, GPU, port, or link-remap residue"

INSPECT_JSON="$(run "inspect compact initialized snapshot" 120 \
    "${OFFLINE_EXEC[@]}" vllm snapshot inspect /e2e/artifact)"
INSPECT_RECEIPT="$(printf '%s' "$INSPECT_JSON" | parse_inspect)"
mapfile -t SNAPSHOT_PIDS < <(sed -n 's/^pid=//p' <<< "$INSPECT_RECEIPT")
INSPECT_LOGPROB="$(sed -n 's/^oracle_logprob=//p' <<< "$INSPECT_RECEIPT")"
(( ${#SNAPSHOT_PIDS[@]} > 0 )) || die "inspect returned no PIDs"
state_clean || die "create left recorded state alive"
INSPECT_SHA="$(printf '%s' "$INSPECT_JSON" | sha256sum | awk '{print $1}')"
echo "$INSPECT_RECEIPT"
echo "public_inspect_sha256=$INSPECT_SHA"

run "restore compact snapshot 1" 900 "${OFFLINE_EXEC[@]}" \
    vllm snapshot restore /e2e/artifact --host 127.0.0.1 --port "$PORT_ONE"
port_open "$PORT_ONE" || die "restore 1 did not bind its port"
[[ -n "$(gpu_pids)" ]] || die "restore 1 did not engage the selected GPU"
assert_oracle "$PORT_ONE" 1
run "stop container after restore 1" 60 docker stop --time 20 "$CONTAINER_NAME" >/dev/null
[[ "$(docker container inspect --format '{{.Id}}' "$CONTAINER_NAME")" == "$CONTAINER_ID" ]] \
    || die "docker stop replaced the container"
wait_clean || die "restore 1 teardown left residue"

run "start the same container" 60 docker start "$CONTAINER_NAME" >/dev/null
[[ "$(docker container inspect --format '{{.Id}}' "$CONTAINER_NAME")" == "$CONTAINER_ID" ]] \
    || die "docker start replaced the container"
INSPECT_AFTER="$(run "inspect after container restart" 120 \
    "${OFFLINE_EXEC[@]}" vllm snapshot inspect /e2e/artifact)"
[[ "$(printf '%s' "$INSPECT_AFTER" | sha256sum | awk '{print $1}')" == "$INSPECT_SHA" ]] \
    || die "public inspect output changed across stop/start"
run "restore compact snapshot 2" 900 "${OFFLINE_EXEC[@]}" \
    vllm snapshot restore /e2e/artifact --host 127.0.0.1 --port "$PORT_TWO"
port_open "$PORT_TWO" || die "restore 2 did not bind its port"
[[ -n "$(gpu_pids)" ]] || die "restore 2 did not engage the selected GPU"
assert_oracle "$PORT_TWO" 2
run "stop container after restore 2" 60 docker stop --time 20 "$CONTAINER_NAME" >/dev/null
wait_clean || die "restore 2 teardown left residue"
cleanup || die "exact cleanup failed"
echo "container_id=$CONTAINER_ID restore_1_port=$PORT_ONE restore_2_port=$PORT_TWO"
echo "result=PASS"
