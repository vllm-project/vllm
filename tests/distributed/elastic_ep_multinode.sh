#!/bin/bash
# Elastic EP across two NVLink domains (NIX-1762 step 6, test plan EL-M1 to EL-M3): on every node of a 4-node Slurm
# job (one task per node, two nodes in each of two NVLink domains) start one Ray cluster, then run pytest with the
# given arguments on node 0 against it (LEADER_ADDRESS set, so test_elastic_ep.py joins it), e.g.
#   VLLM_TEST_ELASTIC_EP_INITIAL_DP=8 VLLM_TEST_ELASTIC_EP_TARGET_DP=16 VLLM_TEST_ELASTIC_EP_DP_LOCAL=4 \
#     VLLM_NIXL_EP_V2_DOMAIN_SIZE=8 srun --ntasks-per-node=1 bash tests/distributed/elastic_ep_multinode.sh \
#     -s "tests/distributed/test_elastic_ep.py::test_elastic_ep_scaling[cuda_graphs_heavy_nixl_ep_v2]"
# The pytest arguments are passed as they are (test ids, --junitxml, -k ...); the exit code is pytest's on node 0.
# VLLM_RAY_DP_PLACEMENT_NODE_IPS comes from each node's NVLink fabric (nvidia-smi -q: ClusterUUID and CliqueId; two
# NVL72 racks can share a CliqueId), node 0's domain first, so EP rank // (GPUs per domain) is the domain. Slurm's
# block (scontrol show topology, else the harness's EP_NVLBLK) must agree. Anything else refuses before Ray starts:
# a wrong order makes nixl_ep_v2's commit fail under the pause. Never falls back to the SLURM_JOB_NODELIST order.
# The caller provides an offline model cache for test_elastic_ep.py's bare repo id and the GSM8K files, if needed.
# Env [default]: ELASTIC_EP_OUT_ROOT [/workspace/logs/vllm/elastic_ep], a directory every node sees; the nodes meet
#   in <root>/<job id>/ (node<N>: ip fabric block; order; refused; done). ELASTIC_EP_RAY_WAIT_S [600].
set -uo pipefail
HERE=$(dirname "$(realpath "$0")")
TREE=$(realpath "$HERE/../..")
NODE=${SLURM_NODEID:-0} NNODES=${SLURM_NNODES:-1}
D=${ELASTIC_EP_OUT_ROOT:-/workspace/logs/vllm/elastic_ep}/${SLURM_JOB_ID:?run under Slurm}
WAIT_S=${ELASTIC_EP_RAY_WAIT_S:-600} PORT=6379
mkdir -p "$D"
log() { echo "[elastic_ep_multinode] node $NODE: $*"; }

# This node: its address (as Ray will advertise it), GPUs, NVLink domain and Slurm block.
host=$(hostname -s)
ip=$(getent ahostsv4 "$host" | awk 'NR == 1 {print $1}')
gpus=$(nvidia-smi -L | grep -c '^GPU ')
# One "ClusterUUID/CliqueId" per GPU; every GPU of the node must agree.
fabric=$(nvidia-smi -q | awk -F' *: *' '$1 ~ /^ *CliqueId$/ {c = $2} $1 ~ /^ *ClusterUUID$/ {u = $2}
  c != "" && u != "" {print u "/" c; c = u = ""}' | sort -u | paste -sd,)
if command -v scontrol >/dev/null; then
  block=$(scontrol show topology "$host" 2>/dev/null | grep -o -m1 'BlockName=[^ ]*' | cut -d= -f2)
else
  block=$(awk -v h="$host" '$1 == h {print $2}' <<< "${EP_NVLBLK:-}")
fi
echo "${ip:--} $gpus ${fabric:--} ${block:--} $host" > "$D/node$NODE.tmp" && mv "$D/node$NODE.tmp" "$D/node$NODE"
log "$(cat "$D/node$NODE")"

if (( NODE > 0 )); then
  until [[ -e $D/order || -e $D/refused ]]; do sleep 2; done
  [[ -e $D/refused ]] && { log "refused: $(cat "$D/refused")"; exit 1; }
  read -r head_ip _ < "$D/order"
  ray start --address="$head_ip:$PORT" --node-ip-address="$ip" --num-gpus="$gpus" --block &
  pid=$!
  until [[ -e $D/done ]] || ! kill -0 $pid 2>/dev/null; do sleep 5; done
  ray stop --force >/dev/null 2>&1
  kill $pid 2>/dev/null
  wait $pid 2>/dev/null
  exit 0
fi

refuse() { echo "$*" > "$D/refused"; log "refused: $*"; exit 1; }
trap 'touch "$D/done"' EXIT
t0=$SECONDS
until (( $(compgen -G "$D/node[0-9]*" | grep -vc '\.tmp$') == NNODES )); do
  (( SECONDS - t0 > 300 )) && refuse "only $(compgen -G "$D/node[0-9]*" | wc -l) of $NNODES nodes reported"
  sleep 2
done
# Node 0, its domain's other node, then the other domain's two nodes.
order=$(python - "$D" "$NNODES" <<'EOF'
import sys
from collections import defaultdict

d, nnodes = sys.argv[1], int(sys.argv[2])
nodes = [open(f"{d}/node{n}").read().split() for n in range(nnodes)]
for ip, _, fabric, _, host in nodes:
    if ip == "-":
        sys.exit(f"{host}: no address")
    uuid = fabric.split("/")[0]
    if fabric == "-" or "," in fabric or uuid.strip("0-") in ("", "N/A"):
        sys.exit(f"{host}: no single NVLink fabric ({fabric})")
domains = defaultdict(list)
for node in nodes:
    domains[node[2]].append(node)
if nnodes != 4 or sorted(map(len, domains.values())) != [2, 2]:
    sys.exit(f"want two NVLink domains of 2 nodes, got {dict(domains)}")
blocks = [{node[3] for node in group} for group in domains.values()]
if "-" in set().union(*blocks):
    print("no Slurm block on some node: cross-check skipped", file=sys.stderr)
elif any(len(b) != 1 for b in blocks) or blocks[0] == blocks[1]:
    sys.exit(f"Slurm blocks disagree with the NVLink domains: {dict(domains)}")
first = domains[nodes[0][2]]
first.sort(key=lambda node: node is not nodes[0])
rest = next(group for group in domains.values() if group is not first)
print(" ".join(node[0] for node in first + rest))
EOF
) || refuse "node order: see above"
read -r head_ip _ <<< "$order"
[[ $head_ip == "$ip" ]] || refuse "node 0's address $ip is not first in '$order'"
log "domains: $order"

ray start --head --port=$PORT --node-ip-address="$ip" --num-gpus="$gpus" --disable-usage-stats ||
  refuse "ray start --head failed"
echo "$order" > "$D/order"
total=$(awk '{s += $2} END {print s}' "$D"/node[0-9]*)
python - "$total" "$WAIT_S" <<'EOF' || { log "Ray did not reach the job's GPUs"; ray stop --force; exit 1; }
import sys
import time

import ray

want, wait_s = float(sys.argv[1]), float(sys.argv[2])
ray.init(address="auto", logging_level="ERROR")
deadline = time.monotonic() + wait_s
while (have := ray.cluster_resources().get("GPU", 0)) < want:
    if time.monotonic() > deadline:
        sys.exit(f"Ray has {have} of {want} GPUs after {wait_s} s")
    time.sleep(2)
print(f"[elastic_ep_multinode] Ray has {have} GPUs")
EOF

export LEADER_ADDRESS=$ip VLLM_RAY_DP_PLACEMENT_NODE_IPS=${order// /,}
log "LEADER_ADDRESS=$LEADER_ADDRESS VLLM_RAY_DP_PLACEMENT_NODE_IPS=$VLLM_RAY_DP_PLACEMENT_NODE_IPS"
cd "$TREE" || exit 1
python -m pytest "$@"
rc=$?
ray stop --force >/dev/null 2>&1
exit $rc
