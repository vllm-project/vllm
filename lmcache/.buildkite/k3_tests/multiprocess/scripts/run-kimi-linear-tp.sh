#!/usr/bin/env bash
# Cross-TP-rank KV correctness for a hybrid MLA + linear-attention model
# (Kimi-Linear) served with tensor parallelism, across a vLLM restart.
#
# Why this test exists:
#   Kimi-Linear is hybrid -- MLA full-attention layers (whose KV latent is
#   REPLICATED across TP ranks) plus Kimi-Delta-Attention linear-attention
#   layers (whose recurrent/conv "mamba" state is SHARDED across TP ranks).
#   LMCache's MLA optimization ("store KV once on rank 0, share it with every
#   TP rank") is only valid when all cached state is replicated. Applying it to
#   this hybrid model made every TP rank load rank 0's mamba shard on a cache
#   hit, so a request served from LMCache produced a different (wrong) output
#   than the same request computed from scratch. See
#   lmcache/integration/vllm/utils.py::mla_only.
#
# This test is self-contained: it launches its own LMCache server + a TP=2 vLLM
# (both GPUs) instead of using launch-processes.sh / wait-for-servers.sh, since
# it needs tensor parallelism and trust-remote-code. PIDs are written to the
# shared PID_FILE so the dispatcher's cleanup.sh trap still tears everything down.
#
# Flow (2 GPUs, TP=2):
#   1. Launch the LMCache server (kept alive for the whole test) + vLLM.
#   2. vLLM run: send one long deterministic (greedy) completion request; vLLM
#      computes it from scratch and populates LMCache. Capture output A.
#   3. Restart vLLM (kill + relaunch), keeping the LMCache server running. The
#      fresh vLLM has an empty prefix cache, so the request's prefix KV --
#      including the per-rank linear-attention state -- must be served by
#      LMCache. (A prefix-cache reset would not work here: vLLM's
#      /reset_prefix_cache does not evict mamba state, so a full restart is the
#      only way to force LMCache to serve the linear-attention state.)
#   4. LMCache retrieve run: send the identical request. Capture output B.
#   5. Assert A == B. A broken cross-rank restore hands the non-zero ranks the
#      wrong state shard, which diverges the greedy decode -- so the outputs
#      differ. The stored state is byte-reloaded (not recomputed), so a correct
#      run is deterministic and the two outputs match exactly.
#   6. Assert LMCache actually served retrieves in run 2 (non-vacuous).
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

# ── Configuration ───────────────────────────────────────────
MODEL="${MODEL:-moonshotai/Kimi-Linear-48B-A3B-Instruct}"
LMCACHE_PORT="${LMCACHE_PORT:-6555}"
VLLM_PORT="${VLLM_PORT:-8000}"
BUILD_ID="${BUILD_ID:-local_$$}"
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"
LMCACHE_LOG="/tmp/build_${BUILD_ID}_lmcache.log"

# vLLM block size == LMCache chunk size for this model. 'align' requires
# block_size <= max_num_batched_tokens < 2*block_size so every prefill step
# advances exactly one block (one reusable linear-attention snapshot per chunk).
CHUNK_SIZE="${CHUNK_SIZE:-944}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1500}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
# Readiness timeout per vLLM launch. This is owned by the test (a 48B TP-shard
# load is slow) and deliberately does NOT reuse MAX_WAIT_SECONDS, which
# run-single-test.sh pre-exports to 300s -- that would shadow the value here.
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-900}"
# Seconds to wait for vLLM's GPU memory to be released after a restart before
# relaunching (avoids an OOM racing the dying process).
GPU_RELEASE_TIMEOUT="${GPU_RELEASE_TIMEOUT:-180}"

# Tokens to generate per request. Greedy (temperature 0); a divergence in the
# resumed state shows up within the first few tokens, but a longer generation
# makes an accidental match astronomically unlikely.
MAX_TOKENS="${MAX_TOKENS:-128}"
# Seconds to let async LMCache stores drain before restarting vLLM.
STORE_DRAIN_SECONDS="${STORE_DRAIN_SECONDS:-20}"

RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
TP_DIR="$RESULTS_DIR/kimi_linear_tp"
mkdir -p "$TP_DIR"
PROMPT_FILE="$TP_DIR/prompt.txt"
OUT_A="$TP_DIR/output_vllm_run.txt"
OUT_B="$TP_DIR/output_retrieve_run.txt"

VLLM_PID=""

echo "=== Kimi-Linear cross-TP-rank KV correctness test (vLLM restart) ==="
echo "Model: $MODEL"
echo "LMCache port: $LMCACHE_PORT | vLLM port: $VLLM_PORT | TP=2"
echo "Chunk size: $CHUNK_SIZE | max_num_batched_tokens: $MAX_NUM_BATCHED_TOKENS"
echo "Results dir: $TP_DIR"
echo ""

# Launch a TP=2 vLLM+LMCache instance and wait until it serves.
# Arguments:
#   $1 log_file - where to redirect vLLM stdout/stderr.
# Sets the global VLLM_PID and appends it to PID_FILE for cleanup.
launch_vllm() {
    local log_file="$1"
    echo "=== Launching vLLM (Kimi-Linear, TP=2, port $VLLM_PORT) ==="
    echo "Log: $log_file"
    # Save and unset VLLM_PORT: vLLM's internal get_open_port() would otherwise
    # collide with the serving port for torch.distributed.
    local saved_port="$VLLM_PORT"
    unset VLLM_PORT

    vllm serve "$MODEL" \
        --tensor-parallel-size 2 \
        --trust-remote-code \
        --enforce-eager \
        --no-enable-flashinfer-autotune \
        --enable-prefix-caching \
        --moe-backend triton \
        --mamba-cache-mode align \
        --max-model-len auto \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --port "$saved_port" \
        --block-size 944 \
        --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\", \"kv_role\":\"kv_both\", \"kv_load_failure_policy\": \"recompute\", \"kv_connector_extra_config\": {\"lmcache.mp.port\": $LMCACHE_PORT, \"lmcache.mp.mq_timeout\": 30}}" \
        > "$log_file" 2>&1 &
    VLLM_PID=$!
    echo "$VLLM_PID" >> "$PID_FILE"
    export VLLM_PORT="$saved_port"
    echo "vLLM started (PID=$VLLM_PID)"

    if ! wait_for_server "$VLLM_PORT" "$VLLM_READY_TIMEOUT" "$log_file"; then
        echo "vLLM failed to start."
        return 1
    fi
    echo ""
}

# Poll a GPU's used memory until it drops below a threshold (i.e. the previous
# vLLM instance's weights/KV have been freed) or a timeout elapses.
# Arguments:
#   $1 gpu_index      - GPU to poll.
#   $2 threshold_mib  - consider "released" once used memory is below this.
wait_for_gpu_release() {
    local gpu_index="$1"
    local threshold_mib="$2"
    local deadline=$(( $(date +%s) + GPU_RELEASE_TIMEOUT ))
    echo "=== Waiting for GPU $gpu_index memory to be released (< ${threshold_mib} MiB) ==="
    while [ "$(date +%s)" -lt "$deadline" ]; do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            -i "$gpu_index" 2>/dev/null | tr -d ' ' || echo 999999)
        if [ -n "$used" ] && [ "$used" -lt "$threshold_mib" ]; then
            echo "GPU $gpu_index memory released (used=${used} MiB)."
            echo ""
            return 0
        fi
        sleep 3
    done
    echo "WARNING: GPU $gpu_index still busy after ${GPU_RELEASE_TIMEOUT}s; relaunching anyway."
    echo ""
}

# Kill the current vLLM instance (keeping the LMCache server), then wait for its
# GPU memory to be freed so the relaunch has room for the 48B weights.
stop_vllm() {
    echo "=== Stopping vLLM (PID=$VLLM_PID), keeping LMCache server alive ==="
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
        # Give vLLM time to shut its TP workers down cleanly.
        local wait_deadline=$(( $(date +%s) + 60 ))
        while [ "$(date +%s)" -lt "$wait_deadline" ] && kill -0 "$VLLM_PID" 2>/dev/null; do
            sleep 2
        done
        kill -9 "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    # Free the serving port in case a child socket lingers.
    fuser -k "${VLLM_PORT}/tcp" 2>/dev/null || true
    # GPU 1 is used exclusively by vLLM (the LMCache server pins its CUDA
    # context on GPU 0), so its memory dropping to near-idle is a clean signal
    # that the old vLLM (and its TP workers) are fully gone.
    wait_for_gpu_release 1 2000
}

# Send one greedy completion request and write the generated text to a file.
# Uses only the Python stdlib so no extra client dependency is required.
send_completion() {
    local out_file="$1"
    local run_name="$2"
    echo "=== Sending completion ($run_name) on port $VLLM_PORT ==="
    python3 - "$VLLM_PORT" "$MODEL" "$PROMPT_FILE" "$MAX_TOKENS" "$out_file" <<'PYEOF'
import json
import sys
import urllib.request

port, model, prompt_file, max_tokens, out_file = sys.argv[1:6]
prompt = open(prompt_file).read()
body = json.dumps(
    {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": int(max_tokens),
        "seed": 0,
    }
).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=600) as resp:
    data = json.load(resp)
text = data["choices"][0]["text"]
with open(out_file, "w") as f:
    f.write(text)
print(f"  generated {len(text)} chars")
PYEOF
    echo "$run_name completed"
    echo ""
}

# Count completed LMCache retrieves in the server log (proves run 2 was served
# by LMCache, so the comparison can't pass vacuously by recomputing).
count_retrieves() {
    [ -f "$LMCACHE_LOG" ] || { echo 0; return; }
    grep -c "Retrieved" "$LMCACHE_LOG" 2>/dev/null || true
}

# ── 1. Launch LMCache MP server (kept alive across the vLLM restart) ──
echo "=== Launching LMCache MP server (port $LMCACHE_PORT) ==="
lmcache server \
    --host localhost \
    --port "$LMCACHE_PORT" \
    --chunk-size "$CHUNK_SIZE" \
    --l1-size-gb 80 \
    --eviction-policy LRU \
    --max-workers 4 \
    > "$LMCACHE_LOG" 2>&1 &
LMCACHE_PID=$!
echo "$LMCACHE_PID" >> "$PID_FILE"
echo "LMCache MP server started (PID=$LMCACHE_PID)"
sleep 10

# ── 2. Build a long, deterministic prompt, then ask for a summary ──
# A ~7-8k word document (well over the several-thousand-token span needed for
# the request to store multiple linear-attention snapshots) built by repeating
# a fixed paragraph, so the input is identical on every run.
python3 - "$PROMPT_FILE" <<'PYEOF'
import sys

prompt_file = sys.argv[1]
paragraph = (
    "The poll() system call waits for one of a set of file descriptors to "
    "become ready to perform I/O. The set of file descriptors to be monitored "
    "is specified in the fds argument, which is an array of pollfd structures. "
    "The caller should specify the number of items in the fds array in nfds. "
    "The timeout argument specifies the number of milliseconds that poll() "
    "should block waiting for a file descriptor to become ready. The call will "
    "block until either a file descriptor becomes ready, the call is "
    "interrupted by a signal handler, or the timeout expires. "
)
# ~97 words/paragraph * 80 ~= 7.8k words.
with open(prompt_file, "w") as f:
    f.write(paragraph * 80)
    f.write("\n\nSummarize the manual page text above:")
PYEOF
echo "Prompt built ($(wc -w < "$PROMPT_FILE") words)."
echo ""

# ── 3. vLLM run: compute from scratch, populating LMCache ───
launch_vllm "/tmp/build_${BUILD_ID}_vllm.log"
send_completion "$OUT_A" "vLLM run"

echo "Waiting ${STORE_DRAIN_SECONDS}s for LMCache stores to drain..."
sleep "$STORE_DRAIN_SECONDS"
retrieves_before=$(count_retrieves)

# ── 4. Restart vLLM (keep LMCache) so the prefix must come from LMCache ──
stop_vllm
launch_vllm "/tmp/build_${BUILD_ID}_vllm_restart.log"

# ── 5. Retrieve run: fresh vLLM APC is empty -> LMCache serves the KV ─
send_completion "$OUT_B" "LMCache retrieve run"
retrieves_after=$(count_retrieves)

# ── 6. Compare outputs and verify LMCache was actually used ──
echo "============================================"
echo "=== Verifying cross-TP-rank KV correctness ==="
echo "============================================"
echo "LMCache retrieves logged: before=${retrieves_before}, after=${retrieves_after}"

failed=0

if cmp -s "$OUT_A" "$OUT_B"; then
    echo "PASS: vLLM-run and LMCache-retrieve outputs are identical."
else
    echo "FAILED: outputs differ between the cold run and the LMCache-served run."
    echo "--- vLLM run (first 400 chars) ---"
    head -c 400 "$OUT_A"; echo
    echo "--- LMCache retrieve run (first 400 chars) ---"
    head -c 400 "$OUT_B"; echo
    failed=1
fi

if [ "$retrieves_after" -le "$retrieves_before" ]; then
    echo "FAILED: LMCache served no retrieves during the retrieve run "
    echo "        (before=${retrieves_before}, after=${retrieves_after}); the "
    echo "        comparison would be vacuous."
    failed=1
fi

if [ "$failed" -ne 0 ]; then
    exit 1
fi

echo ""
echo "============================================"
echo "=== Kimi-Linear cross-TP-rank KV test passed ==="
echo "  outputs identical; LMCache served $((retrieves_after - retrieves_before)) retrieves."
echo "============================================"
