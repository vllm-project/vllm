#!/bin/bash
# Concurrency false-positive control for the KV digest prototype.
#
# NO fault is injected and no pathological config is used: 240 concurrent
# unique-prompt requests with digests on. This is the no-fault baseline under
# load - interleaved requests, batched digest computation, many requests
# finishing in the same engine step. The digest must stay silent: any
# mismatch here is a false positive in our code, not a corruption.
#
# (History: this used to include a second arm with a pathological 0.1s lease.
# As executed that arm expired zero leases - a NIXL transfer takes ~2ms, far
# under 100ms - so it tested the same thing as the control arm. Early lease
# expiry done right lives in kv_digest_race.sh.)
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

launch_p() {
  CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5559 VLLM_LOGGING_LEVEL=DEBUG \
    setsid ~/vllm/.venv/bin/vllm serve Qwen/Qwen3-0.6B --port 8100 --gpu-memory-utilization 0.2 \
    --enforce-eager --kv-transfer-config "$KT" < /dev/null > ~/pd1p1d/p.log 2>&1 &
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

pkill -f "vllm s""erve"; pkill -f "toy_pro""xy"; sleep 5
launch_p; launch_d; launch_proxy
wait_health 8100 && wait_health 8200 || exit 1
sleep 3

# 240 unique multi-block prompts, 16-way parallel.
seq 1 240 | xargs -P 16 -I{} sh -c '
  curl -s -m 120 http://localhost:8192/v1/chat/completions -H "Content-Type: application/json" \
    -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a 60-word story about city number {} and its famous bridge built in year {}\"}],\"max_tokens\":48,\"temperature\":0.7}" \
    -o /dev/null -w "%{http_code}\n"' > /tmp/load_codes.txt 2>&1
echo "load done: $(sort /tmp/load_codes.txt | uniq -c | tr '\n' ' ')"
sleep 3
M=$(grep -c "KV digest mismatch" ~/pd1p1d/d.log || true)
W=$(grep -c "omitting remote_block_digests" ~/pd1p1d/p.log || true)
echo "CONCURRENCY mismatches=$M (want 0) omit_warnings=$W (want 0)"
echo "=== concurrency validation done ==="
