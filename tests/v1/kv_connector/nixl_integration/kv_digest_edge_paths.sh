#!/bin/bash
# Edge-path validation for the KV digest prototype. NO fault is injected;
# these exercise code paths the happy path never reaches:
#
#   Stage 1 (partial read): D already holds the prompt prefix in its LOCAL
#     prefix cache (warmed by a direct-to-D request), so a subsequent PD
#     request for the same prefix makes D NIXL-read only the missing suffix
#     blocks. Digest verification must align to that trailing slice.
#     Pass criteria: 0 mismatches AND evidence the read was partial
#     (num_external_tokens < full prompt tokens in d.log).
#
#   Stage 2 (chunked prefill): P runs with --max-num-batched-tokens 32, so a
#     ~120-token prompt prefills in multiple chunks. Digests must be computed
#     once, after the LAST chunk, covering the full block list.
#     Pass criteria: 200, coherent output, digests present, 0 mismatches.
#
# Logs land in ~/pd1p1d/{p,d,proxy}.log.
set -u
export PATH=$HOME/vllm/.venv/bin:$HOME/.local/bin:$PATH
cd ~
KT='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_connector_extra_config":{"enable_kv_digest":true}}'
KD='{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"enable_kv_digest":true}}'

wait_health() {  # $1=port
  for _ in $(seq 1 60); do
    [ "$(curl -s -m 3 http://localhost:"$1"/health -o /dev/null -w '%{http_code}' 2>/dev/null)" = "200" ] && return 0
    sleep 5
  done
  echo "FAIL: port $1 never healthy"; return 1
}

# $@ = extra vllm args for P (e.g. launch_p --max-num-batched-tokens 32)
launch_p() {
  CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5559 VLLM_LOGGING_LEVEL=DEBUG \
    setsid ~/vllm/.venv/bin/vllm serve Qwen/Qwen3-0.6B --port 8100 --gpu-memory-utilization 0.2 \
    --enforce-eager --kv-transfer-config "$KT" "$@" < /dev/null > ~/pd1p1d/p.log 2>&1 &
}
launch_d() {
  CUDA_VISIBLE_DEVICES=1 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5659 VLLM_LOGGING_LEVEL=DEBUG \
    setsid ~/vllm/.venv/bin/vllm serve Qwen/Qwen3-0.6B --port 8200 --gpu-memory-utilization 0.2 \
    --enforce-eager --kv-transfer-config "$KD" < /dev/null > ~/pd1p1d/d.log 2>&1 &
}
launch_proxy() {
  setsid ~/vllm/.venv/bin/python ~/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
    --port 8192 --prefiller-hosts localhost --prefiller-ports 8100 \
    --decoder-hosts localhost --decoder-ports 8200 < /dev/null > ~/pd1p1d/proxy.log 2>&1 &
}

# 64-token shared prefix (4 full blocks), tails differ per request.
SHARED="The quick brown fox jumps over the lazy dog near the river bank under the old oak tree while the sun sets behind the distant hills and birds return to their nests. "

echo "=== stage 1: D-side prefix cache partial read ==="
pkill -f "vllm s""erve"; pkill -f "toy_pro""xy"; sleep 5
launch_p; launch_d; launch_proxy
wait_health 8100 && wait_health 8200 || exit 1
sleep 3
# Warm D's local prefix cache directly (no kv_transfer_params -> plain local run).
curl -s -m 60 http://localhost:8200/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"${SHARED}Warmup tail: count to five.\"}],\"max_tokens\":8,\"temperature\":0}" \
  -o /dev/null -w "warm: %{http_code}\n"
# Victim through the proxy: same prefix, different tail.
curl -s -m 60 http://localhost:8192/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"${SHARED}Victim tail: name a color.\"}],\"max_tokens\":8,\"temperature\":0}" \
  -o /dev/null -w "victim: %{http_code}\n"
sleep 2
M=$(grep -c "KV digest mismatch" ~/pd1p1d/d.log || true)
# Evidence the read was partial: the victim's external token count should be
# the suffix only (warm prompt is ~70 tokens; full fetch would be ~70).
E=$(grep -o "update_state_after_alloc: num_external_tokens=[0-9]*" ~/pd1p1d/d.log | tail -1)
echo "PARTIAL_READ mismatches=$M (want 0) last_external='$E'"

echo "=== stage 2: chunked prefill ==="
pkill -f "kv_pro""ducer"; sleep 5
launch_p --max-num-batched-tokens 32
wait_health 8100 || exit 1
sleep 3
LONG_PROMPT=$(printf 'word%.0s ' $(seq 1 110))
CODE=$(curl -s -m 120 http://localhost:8192/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"${LONG_PROMPT} Question: what is 2+2?\"}],\"max_tokens\":16,\"temperature\":0}" \
  -o /dev/null -w "%{http_code}")
sleep 2
M=$(grep -c "KV digest mismatch" ~/pd1p1d/d.log || true)
W=$(grep -c "omitting remote_block_digests" ~/pd1p1d/p.log || true)
# Chunking evidence: the request's computed-token count advances in 32-token
# steps in P's scheduler debug logs (0 -> 32 -> 64 -> 96 -> done).
C=$(grep -o "get_num_new_matched_tokens: num_computed_tokens=[0-9]*" ~/pd1p1d/p.log | sort | uniq -c | tr '\n' ' ')
echo "CHUNKED_PREFILL http=$CODE mismatches=$M (want 0) omit_warnings=$W (want 0)"
echo "chunk steps seen on P: $C"
echo "=== edge-path validation done ==="
