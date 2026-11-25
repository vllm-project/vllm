#!/usr/bin/env bash
# Runs on the dev machine. Creates one RTX 4090 pod, then runs pod_setup.sh
# on it (vLLM install + model download). Prints the ssh command at the end.
# Tear down with pod_down.sh: delete, never just stop.
#
# Env overrides:
#   CLOUDS="COMMUNITY SECURE" cloud types to try in order ($0.34/hr vs $0.89/hr)
#   MIN_CUDA=13.0            minimum host CUDA (driver) version
#   BRANCH=inkling2-mini-sm89
#   MIN_BALANCE=1.00         refuse to start below this balance (USD)
#   POD_ID=<id>              rerun setup on an existing pod instead of creating one
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
POD_NAME=${POD_NAME:-inkling2-4090}
CLOUDS=${CLOUDS:-COMMUNITY SECURE}
MIN_CUDA=${MIN_CUDA:-13.0}
BRANCH=${BRANCH:-inkling2-mini-sm89}
MIN_BALANCE=${MIN_BALANCE:-1.00}
GPU="NVIDIA GeForce RTX 4090"
# Only cu132 precompiled wheels are published. CUDA minor-version
# compatibility lets them run on any CUDA 13.x driver; this image ships the
# matching toolkit for any JIT.
WHEEL_VARIANT=cu132
IMAGE=runpod/pytorch:2.2.0-cu1320-torch2150-ubuntu2404
SSH_KEY=${SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}
REPO=suyashpatel98/vllm

t0=$(date +%s)
stamp() { echo "[up +$(( $(date +%s) - t0 ))s] $*"; }

balance=$(runpodctl user -o json | jq -r .clientBalance)
stamp "balance \$$balance"
if awk -v b="$balance" -v m="$MIN_BALANCE" 'BEGIN{exit !(b < m)}'; then
  echo "balance below MIN_BALANCE=$MIN_BALANCE; refusing" >&2
  exit 1
fi

POD_ID=${POD_ID:-}
existing=$(runpodctl pod list -o json \
  | jq -r --arg n "$POD_NAME" '.[] | select(.name == $n) | .id')
if [[ -n "$existing" && "$existing" != "$POD_ID" ]]; then
  echo "pod $POD_NAME already exists ($existing); run pod_down.sh first" >&2
  exit 1
fi

# The fork's main mirrors an upstream commit, so the branch's merge base
# with it is the upstream commit whose precompiled wheel matches.
branch_sha=$(gh api "repos/$REPO/branches/$BRANCH" --jq .commit.sha)
WHEEL_COMMIT=$(gh api "repos/$REPO/compare/main...$BRANCH" \
  --jq .merge_base_commit.sha)
meta="https://wheels.vllm.ai/$WHEEL_COMMIT/$WHEEL_VARIANT/vllm/metadata.json"
if ! curl -sfo /dev/null "$meta"; then
  echo "no precompiled wheel at $meta" >&2
  exit 1
fi
stamp "branch $BRANCH @ ${branch_sha:0:10}, wheel ${WHEEL_COMMIT:0:10}/$WHEEL_VARIANT"

pod_json=""
[[ -n "$POD_ID" ]] && pod_json=$(runpodctl pod get "$POD_ID" -o json) && CLOUDS=""
for cloud in $CLOUDS; do
  stamp "creating $cloud pod (host CUDA >= $MIN_CUDA)"
  if pod_json=$(runpodctl pod create \
      --name "$POD_NAME" \
      --gpu-id "$GPU" \
      --cloud-type "$cloud" \
      --image "$IMAGE" \
      --container-disk-in-gb 80 \
      --min-cuda-version "$MIN_CUDA" \
      --ports 22/tcp \
      --wait-for running \
      --wait-timeout 15m); then
    break
  fi
  echo "$pod_json"
  pod_json=""
  # A --wait timeout keeps the pod; never start a second one.
  if runpodctl pod list -o json | jq -e --arg n "$POD_NAME" \
      'any(.[]; .name == $n)' >/dev/null; then
    echo "a $POD_NAME pod exists after a failed create; run pod_down.sh" >&2
    exit 1
  fi
done
if [[ -z "$pod_json" ]]; then
  echo "no 4090 available in: $CLOUDS" >&2
  exit 1
fi
pod_id=$(jq -r .id <<<"$pod_json")
stamp "pod $pod_id running at \$$(jq -r .costPerHr <<<"$pod_json")/hr"

# Direct TCP ssh (the ssh.runpod.io proxy can't run piped scripts).
ssh_ip="" ssh_port=""
for _ in $(seq 60); do
  read -r ssh_ip ssh_port < <(runpodctl pod get "$pod_id" -o json \
    | jq -r '"\(.ssh.ip // "") \(.ssh.port // "")"' 2>/dev/null || echo "")
  [[ -n "$ssh_ip" && -n "$ssh_port" ]] && break
  sleep 5
done
if [[ -z "$ssh_ip" ]]; then
  echo "pod $pod_id has no public ssh port; run pod_down.sh" >&2
  exit 1
fi
SSH=(ssh -i "$SSH_KEY" -p "$ssh_port" -o StrictHostKeyChecking=no
     -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30
     "root@$ssh_ip")
until "${SSH[@]}" true 2>/dev/null; do sleep 5; done
stamp "ssh up: ${SSH[*]}"

"${SSH[@]}" "BRANCH=$BRANCH WHEEL_COMMIT=$WHEEL_COMMIT WHEEL_VARIANT=$WHEEL_VARIANT bash -s" \
  <"$HERE/pod_setup.sh"

stamp "ready. connect with:"
echo "${SSH[*]}"
