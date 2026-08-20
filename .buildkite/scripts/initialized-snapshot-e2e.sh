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
    local label="$1" maximum_s="$2" now available limit started status
    shift 2
    now="$(date +%s)"
    available="$((DEADLINE_EPOCH - now - CLEANUP_RESERVE_S))"
    (( available > 0 )) || die "no time remains before the cleanup reserve"
    limit="$maximum_s"
    (( available < limit )) && limit="$available"
    started="$now"
    echo "--- ${label} (timeout ${limit}s)" >&2
    if timeout --signal=TERM --kill-after=30s "${limit}s" "$@"; then
        status=0
    else
        status=$?
    fi
    echo "${label}: status=${status} seconds=$(( $(date +%s) - started ))" >&2
    return "$status"
}

link_remaps() {
    find /dev/shm -maxdepth 1 -name 'link_remap.*' \
        -printf '%p|%D|%i|%m|%s\n' | sort
}

gpu_pids() {
    nvidia-smi --id="$GPU_UUID" --query-compute-apps=pid \
        --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'
}

gpu_state() {
    local expected="$1" pids
    pids="$(gpu_pids)" || return 1
    [[ "$expected" == idle && -z "$pids" ]] \
        || [[ "$expected" == engaged && -n "$pids" ]]
}

port_state() {
    local port="$1" expected="$2" listeners
    listeners="$(ss -H -ltn "sport = :$port")" || return 1
    [[ "$expected" == closed && -z "$listeners" ]] \
        || [[ "$expected" == open && -n "$listeners" ]]
}

state_clean() {
    local pid port current
    for pid in "${SNAPSHOT_PIDS[@]}"; do
        [[ ! -e "/proc/$pid" ]] || return 1
    done
    gpu_state idle || return 1
    current="$(link_remaps)" || return 1
    [[ "$current" == "$LINK_REMAP_BASELINE" ]] || return 1
    for port in "$PORT_ONE" "$PORT_TWO"; do
        port_state "$port" closed || return 1
    done
}

wait_clean() {
    local mode="$1" deadline="$(( $(date +%s) + 60 ))"
    local work_deadline="$((DEADLINE_EPOCH - CLEANUP_RESERVE_S))"
    if [[ "$mode" == "work" ]] && (( work_deadline < deadline )); then
        deadline="$work_deadline"
    fi
    while (( $(date +%s) < deadline )); do
        state_clean && return 0
        sleep 1
    done
    return 1
}

container_names() {
    timeout 15s docker container ls --all --format '{{.Names}}'
}

cleanup() {
    local failed=0 names="" actual_id="" actual_label=""
    if [[ -n "$CONTAINER_NAME" ]]; then
        if ! names="$(container_names 2>/dev/null)"; then
            echo "could not query Docker during cleanup" >&2
            failed=1
        elif grep -Fxq "$CONTAINER_NAME" <<< "$names"; then
            actual_id="$(docker container inspect --format '{{.Id}}' \
                "$CONTAINER_NAME" 2>/dev/null || true)"
            actual_label="$(docker container inspect \
                --format '{{index .Config.Labels "ai.vllm.snapshot.e2e.run"}}' \
                "$CONTAINER_NAME" 2>/dev/null || true)"
            if [[ "$actual_label" != "$RUN_ID" ]] \
                || { [[ -n "$CONTAINER_ID" ]] && [[ "$actual_id" != "$CONTAINER_ID" ]]; }; then
                echo "refusing to remove a container with mismatched identity" >&2
                failed=1
            else
                timeout 60s docker rm --force "$CONTAINER_NAME" >/dev/null \
                    || failed=1
            fi
        fi
        if ! names="$(container_names 2>/dev/null)"; then
            echo "could not verify Docker cleanup" >&2
            failed=1
        elif grep -Fxq "$CONTAINER_NAME" <<< "$names"; then
            echo "exact container remained after cleanup: $CONTAINER_NAME" >&2
            failed=1
        fi
    fi
    if [[ -n "$PORT_TWO" ]] && ! wait_clean cleanup; then
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

choose_port() {
    local ordinal="$1" seed offset candidate
    seed="$(printf '%s' "${RUN_ID}:${ordinal}" | cksum | awk '{print $1}')"
    for ((offset = 0; offset < 128; offset++)); do
        candidate="$((20000 + (seed + offset) % 30000))"
        if [[ "$candidate" != "$PORT_ONE" ]] \
            && port_state "$candidate" closed; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

parse_inspect() {
    timeout 30s python3 -c '
import json, math, sys
data = json.load(sys.stdin)
model, revision, gpu_uuid, token_id, text = sys.argv[1:]
expected = {
    "schema_version": 1, "boundary": "post-engine-init-reloadable-state-released",
    "model": model, "served_model_name": model, "model_revision": revision,
    "tokenizer_revision": revision, "gpu_uuid": gpu_uuid,
    "oracle_token_ids": [int(token_id)], "oracle_text": text,
}
for field, value in expected.items():
    if data.get(field) != value:
        raise SystemExit(f"inspect field {field} mismatch: {data.get(field)!r}")
logprob = data.get("oracle_sampled_token_logprob")
if type(logprob) is not float or not math.isfinite(logprob):
    raise SystemExit("inspect logprob is not finite")
if type(data.get("artifact_bytes")) is not int or data["artifact_bytes"] <= 0:
    raise SystemExit("inspect artifact_bytes is not positive")
pids, holders = data.get("process_tree"), data.get("cuda_holders")
if (not isinstance(pids, list) or not pids or len(pids) != len(set(pids))
        or any(type(pid) is not int or pid <= 0 for pid in pids)):
    raise SystemExit("inspect process_tree is invalid")
if not isinstance(holders, list) or not holders or not set(holders).issubset(pids):
    raise SystemExit("inspect cuda_holders is invalid")
for pid in pids:
    print(f"pid={pid}")
print(f"artifact_bytes={data['"'"'artifact_bytes'"'"']}")
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
if choice["token_ids"] != [int(token_id)] or choice["text"] != text:
    raise SystemExit(f"public token/text mismatch: {choice!r}")
if len(values) != 1 or type(values[0]) not in (int, float):
    raise SystemExit("public completion did not return one logprob")
actual = float(values[0])
if not math.isfinite(actual) or abs(actual - float(expected)) > 1e-3:
    raise SystemExit(f"public logprob mismatch: {actual} != {expected}")
print(f"restore={ordinal} token_id={token_id} text={text!r} logprob={actual}")
' "$EXPECTED_TOKEN_ID" "$EXPECTED_TEXT" "$INSPECT_LOGPROB" "$ordinal"
}

for command in docker nvidia-smi timeout curl ss sha256sum find sort sed grep \
    awk cksum python3; do
    command -v "$command" >/dev/null 2>&1 || die "required command not found: $command"
done
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
    --query-gpu=name,uuid,mig.mode.current,driver_version,memory.total \
    --format=csv,noheader,nounits)"
IFS=',' read -r GPU_NAME GPU_UUID MIG_MODE DRIVER_VERSION GPU_MEMORY_MIB \
    <<< "$GPU_ROW"
for variable in GPU_NAME GPU_UUID MIG_MODE DRIVER_VERSION GPU_MEMORY_MIB; do
    printf -v "$variable" '%s' "${!variable#"${!variable%%[![:space:]]*}"}"
done
[[ "$GPU_NAME" == *H200* && "$GPU_UUID" == GPU-* && "$MIG_MODE" == Disabled ]] \
    || die "selected device is not a full non-MIG H200: $GPU_ROW"
gpu_state idle || die "selected H200 is not idle or could not be queried"

RUN_ID="snapshot-e2e-$(date +%s)-$$-$RANDOM"
CONTAINER_NAME="vllm-$RUN_ID"
[[ "$CONTAINER_NAME" =~ ^[a-z0-9-]{1,128}$ ]] || die "invalid container name"
names="$(container_names)" || die "could not query Docker containers"
grep -Fxq "$CONTAINER_NAME" <<< "$names" && die "container name collision"
PORT_ONE="$(choose_port 1)" || die "could not allocate the first port"
PORT_TWO="$(choose_port 2)" || die "could not allocate the second port"
LINK_REMAP_BASELINE="$(link_remaps)"

echo "gpu_name=$GPU_NAME gpu_uuid=$GPU_UUID gpu_memory_mib=$GPU_MEMORY_MIB"
echo "driver_version=$DRIVER_VERSION model_revision=$MODEL_REVISION"
run "pull exact candidate image" 900 docker pull --platform linux/amd64 \
    "$IMAGE_TAG" >/dev/null
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE_TAG")"
IMAGE_PLATFORM="$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$IMAGE_TAG")"
OCI_COMMIT="$(docker image inspect \
    --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
    "$IMAGE_TAG")"
VLLM_COMMIT="$(docker image inspect \
    --format '{{index .Config.Labels "ai.vllm.build.commit"}}' "$IMAGE_TAG")"
[[ "$IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ && "$IMAGE_PLATFORM" == linux/amd64 ]] \
    || die "candidate image identity is invalid"
[[ "$OCI_COMMIT" == "$EXPECTED_COMMIT" && "$VLLM_COMMIT" == "$EXPECTED_COMMIT" ]] \
    || die "candidate image commit labels do not match"
echo "image_id=$IMAGE_ID image_commit=$EXPECTED_COMMIT platform=$IMAGE_PLATFORM"

CONTAINER_ID="$(run "start exact candidate container" 90 docker run --detach \
    --name "$CONTAINER_NAME" --label "ai.vllm.snapshot.e2e.run=$RUN_ID" \
    --gpus "device=$GPU_UUID" --user 0 --privileged --pid=host --ipc=host \
    --network=host --env CUDA_VISIBLE_DEVICES=0 --env HF_HUB_DISABLE_TELEMETRY=1 \
    --env VLLM_NO_USAGE_STATS=1 --env VLLM_USE_V2_MODEL_RUNNER=1 \
    --entrypoint sleep "$IMAGE_ID" infinity)"
[[ "$CONTAINER_ID" =~ ^[0-9a-f]{64}$ ]] || die "invalid container ID"
[[ "$(docker container inspect --format '{{.Id}}' "$CONTAINER_NAME")" == "$CONTAINER_ID" ]] \
    || die "container identity changed after launch"
[[ "$(docker container inspect \
    --format '{{.HostConfig.Privileged}}|{{.HostConfig.PidMode}}|{{.HostConfig.IpcMode}}|{{.HostConfig.NetworkMode}}|{{len .Mounts}}' \
    "$CONTAINER_NAME")" == "true|host|host|host|0" ]] \
    || die "container topology is invalid"
CONTAINER_ENV="$(docker container inspect \
    --format '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER_NAME")"
grep -Eq '^(HF_TOKEN|HUGGING_FACE_HUB_TOKEN|HUGGINGFACE_HUB_TOKEN)=' \
    <<< "$CONTAINER_ENV" && die "candidate container has an HF token"
run "prepare private artifact root" 60 docker exec "$CONTAINER_NAME" sh -c \
    "mkdir -m 0700 /e2e && mkdir -m 0700 /e2e/hf && df -Pk / | awk 'NR == 2 && \$4 >= 10485760 {ok=1} END {exit !ok}'"
run "verify snapshot runtime" 60 docker exec "$CONTAINER_NAME" sh -c \
    'criu --version; test -x /usr/local/sbin/cuda-checkpoint; test -f /usr/local/lib/criu/cuda_plugin.so'
echo "cuda_version=$(docker exec "$CONTAINER_NAME" printenv CUDA_VERSION)"
run "prefetch pinned public model" 900 docker exec --env HF_HOME=/e2e/hf \
    "$CONTAINER_NAME" python3 -c \
    'from pathlib import Path; from huggingface_hub import snapshot_download; import sys; p=snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2]); assert Path(p).name == sys.argv[2]' \
    "$MODEL" "$MODEL_REVISION"

OFFLINE_EXEC=(docker exec --env HF_HOME=/e2e/hf --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 --env HF_HUB_DISABLE_TELEMETRY=1 \
    --env VLLM_NO_USAGE_STATS=1 --env VLLM_SNAPSHOT_TIMEOUT_S=900 \
    "$CONTAINER_NAME")
run "create compact initialized snapshot" 1200 "${OFFLINE_EXEC[@]}" \
    vllm snapshot create "$MODEL" --revision "$MODEL_REVISION" \
    --tokenizer-revision "$MODEL_REVISION" --snapshot-dir /e2e/artifact \
    --dtype float16 --max-model-len 512 --gpu-memory-utilization 0.50
wait_clean work || die "create left process, GPU, port, or link-remap residue"

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
port_state "$PORT_ONE" open || die "restore 1 did not bind its port"
gpu_state engaged || die "restore 1 did not engage the selected GPU"
assert_oracle "$PORT_ONE" 1
run "stop container after restore 1" 60 docker stop --time 20 "$CONTAINER_NAME" >/dev/null
[[ "$(docker container inspect --format '{{.Id}}' "$CONTAINER_NAME")" == "$CONTAINER_ID" ]] \
    || die "docker stop replaced the container"
wait_clean work || die "restore 1 teardown left residue"

run "start the same container" 60 docker start "$CONTAINER_NAME" >/dev/null
[[ "$(docker container inspect --format '{{.Id}}' "$CONTAINER_NAME")" == "$CONTAINER_ID" ]] \
    || die "docker start replaced the container"
INSPECT_AFTER="$(run "inspect after container restart" 120 \
    "${OFFLINE_EXEC[@]}" vllm snapshot inspect /e2e/artifact)"
[[ "$(printf '%s' "$INSPECT_AFTER" | sha256sum | awk '{print $1}')" == "$INSPECT_SHA" ]] \
    || die "public inspect output changed across stop/start"
run "restore compact snapshot 2" 900 "${OFFLINE_EXEC[@]}" \
    vllm snapshot restore /e2e/artifact --host 127.0.0.1 --port "$PORT_TWO"
port_state "$PORT_TWO" open || die "restore 2 did not bind its port"
gpu_state engaged || die "restore 2 did not engage the selected GPU"
assert_oracle "$PORT_TWO" 2
run "stop container after restore 2" 60 docker stop --time 20 "$CONTAINER_NAME" >/dev/null
wait_clean work || die "restore 2 teardown left residue"
cleanup || die "exact cleanup failed"
state_clean || die "final state is not clean"
echo "container_id=$CONTAINER_ID restore_1_port=$PORT_ONE restore_2_port=$PORT_TWO"
echo "result=PASS"
