#!/bin/bash
# Race validation: force a REAL read-after-release/reassign race and check
# the digest catches it.
#
# Injected fault: none at the byte level. Instead we manufacture the timing
# race that corrupts data in production:
#   1. P lease (50ms) << D time-to-read (2s, via VLLM_NIXL_DEBUG_RECV_DELAY_MS
#      which sleeps on D after handshake and before posting the NIXL READ),
#      so P releases the victim's blocks first (read-after-RELEASE).
#   2. P's KV pool is 24 blocks (--num-gpu-blocks-override 24), so the
#      allocator wraps around after ~12 prefills (~1.2s of churn).
#   3. Churn goes DIRECTLY to P (prefill-only requests, bypassing D's
#      delay-throttled pipeline) with prefix caching off, so freed victim
#      blocks are reallocated and overwritten before D's READ lands
#      (read-after-REASSIGN).
#
# Expected: some victims caught, with BOTH digest halves differing (the whole
# block was overwritten by another request, not a byte flip). Not every
# victim mismatches - a READ that lands before the overwrite passes
# legitimately; that timing lottery is the nature of the race.
# Logs land in ~/pd1p1d/{p,d,proxy}.log.
#
# TP variants: PREFILLER_TP_SIZE / DECODER_TP_SIZE with matching
# PREFILLER_GPUS / DECODER_GPUS lists, e.g. TP4:
#   PREFILLER_TP_SIZE=4 DECODER_TP_SIZE=4 \
#   PREFILLER_GPUS=0,1,2,3 DECODER_GPUS=4,5,6,7 bash kv_digest_race.sh
set -u
export PATH=$HOME/vllm/.venv/bin:$HOME/.local/bin:$PATH
cd ~

P_TP=${PREFILLER_TP_SIZE:-1}
D_TP=${DECODER_TP_SIZE:-1}
P_GPUS=${PREFILLER_GPUS:-0}
D_GPUS=${DECODER_GPUS:-1}
KP='"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_connector_extra_config":{"enable_kv_digest":true,"kv_lease_duration":0.05}'
KD='"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"enable_kv_digest":true}'

wait_health() {
  for _ in $(seq 1 60); do
    [ "$(curl -s -m 3 http://localhost:"$1"/health -o /dev/null -w '%{http_code}' 2>/dev/null)" = "200" ] && return 0
    sleep 5
  done
  echo "FAIL: port $1 never healthy"; return 1
}

pkill -f "vllm s""erve"; pkill -f "toy_pro""xy"; sleep 5

# P: 50ms block lease, 24-block pool, prefix caching off (cache hits would
# otherwise let blocks keep their content across frees).
CUDA_VISIBLE_DEVICES="$P_GPUS" UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5559 VLLM_LOGGING_LEVEL=DEBUG \
  setsid ~/vllm/.venv/bin/vllm serve Qwen/Qwen3-0.6B --port 8100 --tensor-parallel-size "$P_TP" \
  --gpu-memory-utilization 0.1 --num-gpu-blocks-override 24 --max-model-len 256 \
  --no-enable-prefix-caching --enforce-eager \
  --kv-transfer-config "{$KP}" < /dev/null > ~/pd1p1d/p.log 2>&1 &
# D: digests on; the 2s pre-READ delay is what opens the race window.
CUDA_VISIBLE_DEVICES="$D_GPUS" UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5659 VLLM_LOGGING_LEVEL=DEBUG \
  VLLM_NIXL_DEBUG_RECV_DELAY_MS=2000 \
  setsid ~/vllm/.venv/bin/vllm serve Qwen/Qwen3-0.6B --port 8200 --tensor-parallel-size "$D_TP" \
  --gpu-memory-utilization 0.1 --max-model-len 256 --no-enable-prefix-caching --enforce-eager \
  --kv-transfer-config "{$KD}" < /dev/null > ~/pd1p1d/d.log 2>&1 &
setsid ~/vllm/.venv/bin/python ~/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
  --port 8192 --prefiller-hosts localhost --prefiller-ports 8100 \
  --decoder-hosts localhost --decoder-ports 8200 < /dev/null > ~/pd1p1d/proxy.log 2>&1 &

wait_health 8100 && wait_health 8200 || exit 1
sleep 3

# Churn P's allocator directly: prefill-only requests, unique prompts.
churn() {
  local i=0
  while [ "$i" -lt 400 ]; do
    i=$((i+1))
    curl -s -m 30 http://localhost:8100/v1/chat/completions -H "Content-Type: application/json" \
      -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"churn $RANDOM-$i-$(date +%s%N) padding padding padding padding\"}],\"max_tokens\":1}" \
      -o /dev/null 2>&1
  done
}
churn & CHURN1=$!
churn & CHURN2=$!
sleep 1

# Victims through the proxy: D reads each ~2s after P's 50ms lease freed the
# blocks, by which time the churn has overwritten them (pool wraparound ~1.2s).
for v in $(seq 1 12); do
  curl -s -m 120 http://localhost:8192/v1/chat/completions -H "Content-Type: application/json" \
    -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"Victim $v: name a fruit\"}],\"max_tokens\":8,\"temperature\":0}" \
    -o /dev/null -w "victim $v: %{http_code}\n"
done
kill $CHURN1 $CHURN2 2>/dev/null
sleep 3
echo "P expired releases: $(grep -ac 'Releasing expired KV blocks' ~/pd1p1d/p.log || true)"
echo "D mismatches: $(grep -ac 'KV digest mismatch' ~/pd1p1d/d.log || true)"
grep -a "KV digest" ~/pd1p1d/d.log | head -5
echo "=== race v4 done ==="
