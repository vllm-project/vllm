#!/bin/bash
# KV digest smoke validation: clean path, byte-corruption injection, fail
# routing, and the nixl_integration accuracy regression, on one node + toy
# proxy. Logs land in ~/pd1p1d/{p,d,proxy}.log.
#
# TP variants: PREFILLER_TP_SIZE / DECODER_TP_SIZE with matching
# PREFILLER_GPUS / DECODER_GPUS lists, e.g. TP4:
#   PREFILLER_TP_SIZE=4 DECODER_TP_SIZE=4 \
#   PREFILLER_GPUS=0,1,2,3 DECODER_GPUS=4,5,6,7 bash kv_digest_smoke.sh
set -u
export PATH=$HOME/vllm/.venv/bin:$HOME/.local/bin:$PATH
cd ~

P_TP=${PREFILLER_TP_SIZE:-1}
D_TP=${DECODER_TP_SIZE:-1}
P_GPUS=${PREFILLER_GPUS:-0}
D_GPUS=${DECODER_GPUS:-1}

# enable_kv_digest=true on BOTH sides: P computes digests, D verifies them.
KT='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_connector_extra_config":{"enable_kv_digest":true}}'
KD='{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"enable_kv_digest":true}}'

wait_health() {  # $1=port
  for _ in $(seq 1 60); do
    [ "$(curl -s -m 3 http://localhost:"$1"/health -o /dev/null -w '%{http_code}' 2>/dev/null)" = "200" ] && return 0
    sleep 5
  done
  echo "FAIL: port $1 never healthy"; return 1
}

launch_p() {
  CUDA_VISIBLE_DEVICES="$P_GPUS" UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5559 VLLM_LOGGING_LEVEL=DEBUG \
    setsid ~/vllm/.venv/bin/vllm serve Qwen/Qwen3-0.6B --port 8100 --tensor-parallel-size "$P_TP" \
    --gpu-memory-utilization 0.2 \
    --enforce-eager --kv-transfer-config "$KT" < /dev/null > ~/pd1p1d/p.log 2>&1 &
}
launch_d() {  # $@=extra env assignments (e.g. VLLM_NIXL_DIGEST_CORRUPT=1)
  env CUDA_VISIBLE_DEVICES="$D_GPUS" UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5659 VLLM_LOGGING_LEVEL=DEBUG "$@" \
    setsid ~/vllm/.venv/bin/vllm serve Qwen/Qwen3-0.6B --port 8200 --tensor-parallel-size "$D_TP" \
    --gpu-memory-utilization 0.2 \
    --enforce-eager --kv-transfer-config "$KD" < /dev/null > ~/pd1p1d/d.log 2>&1 &
}
launch_proxy() {
  setsid ~/vllm/.venv/bin/python ~/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
    --port 8192 --prefiller-hosts localhost --prefiller-ports 8100 \
    --decoder-hosts localhost --decoder-ports 8200 < /dev/null > ~/pd1p1d/proxy.log 2>&1 &
}
probe() {  # $1=label
  curl -s -m 60 http://localhost:8192/v1/chat/completions -H "Content-Type: application/json" \
    -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":15,\"temperature\":0}" \
    -o /dev/null -w "%{http_code}"
}

# ── Stage 1: clean path, NO fault injected ──────────────────────────────
# Validates the digest pipeline end to end (P computes -> rides
# kv_transfer_params -> D verifies) and that it never false-positives.
echo "=== stage 1: clean path ==="
launch_p; launch_d; launch_proxy
wait_health 8100 && wait_health 8200 || exit 1
sleep 3
for i in 1 2 3; do echo "clean req$i: $(probe "clean check $i")"; done
sleep 2
M=$(grep -c "KV digest mismatch" ~/pd1p1d/d.log || true)
W=$(grep -c "omitting remote_block_digests" ~/pd1p1d/p.log || true)
echo "CLEAN_PATH mismatches=$M (want 0) omit_warnings=$W (want 0)"

# ── Stage 2: byte-corruption fault injection ────────────────────────────
# Fault: VLLM_NIXL_DIGEST_CORRUPT=1 flips one byte of the first received
# block on D after the NIXL READ, simulating silent transport corruption.
# Expect: D's verification catches it (mismatch ERROR in d.log); the request
# still completes (log-only mode).
echo "=== stage 2: corruption injection ==="
pkill -f "kv_cons""umer"; sleep 5
launch_d "VLLM_NIXL_DIGEST_CORRUPT=1"
wait_health 8200 || exit 1
echo "corrupt req: $(probe corrupt-probe)"
sleep 2
M=$(grep -c "KV digest mismatch" ~/pd1p1d/d.log || true)
echo "CORRUPT_PATH mismatches=$M (want >=1)"

# ── Stage 3: failure routing on mismatch ────────────────────────────────
# Same injected fault as stage 2, plus VLLM_NIXL_DIGEST_FAIL=1: the request
# must be FAILED via kv_load_failure_policy instead of generating from
# corrupt KV.
# Expect: "KV digest verification failed" and "Failing 1 request(s) due to
# KV load failure" in d.log; the client gets an error, not an answer.
echo "=== stage 3: fail routing on mismatch ==="
pkill -f "kv_cons""umer"; sleep 5
launch_d "VLLM_NIXL_DIGEST_CORRUPT=1" "VLLM_NIXL_DIGEST_FAIL=1"
wait_health 8200 || exit 1
echo "fail-routing req: $(probe fail-routing-probe)"
sleep 2
M=$(grep -c "KV digest verification failed" ~/pd1p1d/d.log || true)
F=$(grep -c "Failing 1 request(s) due to KV load failure" ~/pd1p1d/d.log || true)
echo "FAIL_ROUTING verify_failures=$M requests_failed=$F (want >=1 each)"

# ── Stage 4: accuracy regression, NO fault injected ─────────────────────
# Real accuracy eval (lm-eval) with digests on: the feature must not change
# model output quality.
echo "=== stage 4: accuracy regression ==="
pkill -f "kv_cons""umer"; sleep 5
launch_d
wait_health 8200 || exit 1
cd ~/vllm && TEST_MODEL=Qwen/Qwen3-0.6B .venv/bin/python -m pytest -s -x \
  tests/v1/kv_connector/nixl_integration/test_accuracy.py 2>&1 | tail -3

echo "=== smoke validation done ==="
