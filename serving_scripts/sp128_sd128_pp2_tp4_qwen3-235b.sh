#!/usr/bin/env bash
#SBATCH --nodelist=htc-g[059-060]
#SBATCH --job-name=vllm-qwen3-235b-tp4-pp2
#SBATCH --nodes=2
#SBATCH --partition=short
#SBATCH --gres=gpu:h100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=02:00:00
#SBATCH --output=results/%x-%j.out
#SBATCH --error=results/%x-%j.err
#SBATCH --mail-user=jason.miller@eng.ox.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=engs-glass
#SBATCH --qos=priority

set -euo pipefail

SCRIPT_VERSION="arc-ray-qwen3-235b-a22b-tp4-pp2-workers-nsys-no-api-wrapper-live-copy-v1"

# Configuration:
#   2 nodes x 4 H100s/node = 8 total GPUs
#   TP=4 within each node
#   PP=2 across the two nodes
#
# Layout:
#   htc-g059: 4-GPU TP group for PP stage 0
#   htc-g060: 4-GPU TP group for PP stage 1
#
# This variant intentionally DOES NOT wrap the vLLM API server in:
#   nsys profile python -m vllm.entrypoints.openai.api_server
#
# It still enables vLLM worker-side Nsight via:
#   --ray-workers-use-nsight
#
# It also enables your patched per-iteration/per-comm NVTX path via:
#   VLLM_ITERATION_NVTX=1
#
# Important Nsight copy behavior:
#   - Ignore empty.nsys-rep and zero-byte placeholder reports.
#   - Keep Ray alive after the workload/server stops while worker Nsight
#     has time to finalize real worker_process_*.nsys-rep files.
#   - Run a live sidecar copier inside each Ray srun step. This matches the
#     manual rescue that worked: SSH into the node, find the current Ray session,
#     and copy before Slurm tmp cleanup.
#   - For ARC/Slurm, worker reports may appear under either:
#       /tmp/ray/session_*/logs/nsight/*
#       /tmp/slurm-${SLURM_JOB_ID}/tmp/ray/session_*/logs/nsight/*
#     Do not trust /tmp/ray/session_latest; it can be stale.

DEBUG_SLURM_SCRIPT="${DEBUG_SLURM_SCRIPT:-0}"
NSYS_COPY_DEBUG="${NSYS_COPY_DEBUG:-0}"

# Post-run fallback copy should not hang the batch forever.
SRUN_COPY_TIMEOUT="${SRUN_COPY_TIMEOUT:-480s}"

# Server shutdown should not hang forever.
SERVER_SHUTDOWN_TIMEOUT_S="${SERVER_SHUTDOWN_TIMEOUT_S:-420}"

# Live sidecar copy interval inside each Ray srun step.
WORKER_NSYS_LIVE_COPY_INTERVAL="${WORKER_NSYS_LIVE_COPY_INTERVAL:-2}"

# After stopping the vLLM API server, keep Ray alive this long while waiting
# for real worker_process_*.nsys-rep files to appear and be copied.
WORKER_NSYS_FINALIZE_WAIT_S="${WORKER_NSYS_FINALIZE_WAIT_S:-300}"
WORKER_NSYS_FINALIZE_POLL_S="${WORKER_NSYS_FINALIZE_POLL_S:-5}"

# Treat files smaller than this as placeholders, not useful reports.
MIN_WORKER_NSYS_REP_BYTES="${MIN_WORKER_NSYS_REP_BYTES:-1024}"

slurm_debug() {
  if [ "${DEBUG_SLURM_SCRIPT}" = "1" ]; then
    echo "[slurm-debug] $*" >&2
  fi
}

nsight_debug() {
  if [ "${NSYS_COPY_DEBUG}" = "1" ] || [ "${DEBUG_SLURM_SCRIPT}" = "1" ]; then
    echo "[nsight-debug] $*"
  fi
}

nsight_copy_msg() {
  echo "[nsight-copy] $*"
}

run_with_timeout() {
  local limit="$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "${limit}" "$@"
  else
    "$@"
  fi
}

wait_for_pid_or_kill() {
  local pid="$1"
  local label="$2"
  local timeout_s="${3:-420}"

  if [ -z "${pid}" ] || ! kill -0 "${pid}" 2>/dev/null; then
    echo "${label}: pid ${pid:-<empty>} is not running"
    return 0
  fi

  local elapsed=0
  while kill -0 "${pid}" 2>/dev/null && [ "${elapsed}" -lt "${timeout_s}" ]; do
    sleep 1
    elapsed=$((elapsed + 1))
  done

  if kill -0 "${pid}" 2>/dev/null; then
    echo "${label}: still alive after ${timeout_s}s; sending TERM"
    kill -TERM "${pid}" 2>/dev/null || true
    sleep 10
  fi

  if kill -0 "${pid}" 2>/dev/null; then
    echo "${label}: still alive after TERM; sending KILL"
    kill -KILL "${pid}" 2>/dev/null || true
  fi

  wait "${pid}" 2>/dev/null || true
}

nsight_summarize_dest() {
  local node="$1"
  local dest="${TRACE_RUN_DIR}/ray_worker_nsight/${node}"

  nsight_copy_msg "dest summary for ${node}: ${dest}"
  if [ -d "${dest}" ]; then
    ls -la "${dest}/" 2>/dev/null || true
    find "${dest}" -type f \( -name '*.nsys-rep' -o -name '*.qdstrm' \) \
      -exec stat -c '[nsight-copy]   %n %s bytes' {} \; 2>/dev/null || true
  else
    nsight_copy_msg "  (no dest directory yet)"
  fi
}

# Return success only if every expected node has at least EXPECTED_WORKER_REPORTS_PER_NODE
# real non-empty worker_process_*.nsys-rep files copied into
# TRACE_RUN_DIR/ray_worker_nsight/<node>.
wait_for_real_worker_nsys_reports() {
  local timeout_s="${1:-300}"
  local poll_s="${2:-5}"
  local elapsed=0

  if [ "${NSYS_ENABLE:-1}" != "1" ] || [ "${NSYS_PROFILE_WORKERS:-1}" != "1" ]; then
    echo "[nsight-copy] worker Nsight disabled; skipping wait for worker reports."
    return 0
  fi

  echo "[nsight-copy] waiting up to ${timeout_s}s for real non-empty worker_process_*.nsys-rep files..."
  echo "[nsight-copy] minimum accepted .nsys-rep size: ${MIN_WORKER_NSYS_REP_BYTES} bytes"
  echo "[nsight-copy] expected nodes: ${HEAD_NODE} ${WORKER_NODES}"
  echo "[nsight-copy] expected worker reports per node: ${EXPECTED_WORKER_REPORTS_PER_NODE}"

  while [ "${elapsed}" -lt "${timeout_s}" ]; do
    local all_ok=1
    local missing_nodes=""

    for node in ${HEAD_NODE} ${WORKER_NODES}; do
      local dest="${TRACE_RUN_DIR}/ray_worker_nsight/${node}"
      local n

      n="$(
        find "${dest}" -maxdepth 1 -type f \
          -name '*worker_process*.nsys-rep' \
          ! -name '*empty*' \
          -size +"${MIN_WORKER_NSYS_REP_BYTES}"c \
          -print 2>/dev/null | wc -l | tr -d ' '
      )"

      if [ "${n:-0}" -lt "${EXPECTED_WORKER_REPORTS_PER_NODE}" ]; then
        all_ok=0
        missing_nodes="${missing_nodes} ${node}(${n:-0}/${EXPECTED_WORKER_REPORTS_PER_NODE})"
      fi
    done

    if [ "${all_ok}" = "1" ]; then
      echo "[nsight-copy] found expected real non-empty worker_process_*.nsys-rep files on every node."
      find "${TRACE_RUN_DIR}/ray_worker_nsight" -type f \
        \( -name '*.nsys-rep' -o -name '*.qdstrm' \) \
        -printf '[nsight-copy]   %p %s bytes\n' 2>/dev/null | sort || true
      return 0
    fi

    echo "[nsight-copy] worker reports not ready yet (${elapsed}/${timeout_s}s); missing:${missing_nodes}"
    find "${TRACE_RUN_DIR}/ray_worker_nsight" -type f \
      \( -name '*.nsys-rep' -o -name '*.qdstrm' \) \
      -printf '[nsight-copy]   seen %p %s bytes\n' 2>/dev/null | sort || true

    sleep "${poll_s}"
    elapsed=$((elapsed + poll_s))
  done

  echo "[nsight-copy] WARNING: timed out waiting for expected real non-empty worker_process_*.nsys-rep files."
  echo "[nsight-copy] Current worker Nsight destination contents:"
  find "${TRACE_RUN_DIR}/ray_worker_nsight" -type f \
    \( -name '*.nsys-rep' -o -name '*.qdstrm' \) \
    -printf '[nsight-copy]   %p %s bytes\n' 2>/dev/null | sort || true

  return 1
}

# Copy worker Nsight files from inside a live Ray srun step.
#
# This is deliberately more restrictive than "find every /tmp/ray/session_*":
# /tmp/ray often contains stale old sessions from previous jobs. We only copy:
#
#   1. current-job Slurm-local tmp:
#      /tmp/slurm-${SLURM_JOB_ID}/tmp/ray/session_*/logs/nsight/*
#
#   2. /tmp/ray sessions that are referenced by currently live Ray/vLLM/nsys
#      processes in this srun step.
#
# This matches what worked manually: SSH into the node while the allocation is
# alive, find the real current session path, and copy before Slurm tmp cleanup.
#
# IMPORTANT:
#   We intentionally ignore empty.nsys-rep and *_empty.nsys-rep. Those are
#   zero-byte placeholders and not usable Nsight reports.
copy_ray_nsight_locally_once() {
  set +e

  local node
  node="$(hostname -s 2>/dev/null || hostname)"

  local dest="${TRACE_RUN_DIR}/ray_worker_nsight/${node}"
  local session_record_dir="${TRACE_RUN_DIR}/ray_session_names"
  local session_record="${session_record_dir}/${node}.txt"

  mkdir -p "${dest}" "${session_record_dir}" 2>/dev/null || true

  local live_sessions
  live_sessions="$(
    ps -eo args 2>/dev/null |
      grep -E 'ray|vllm|EngineCore|worker_process|nsys' |
      grep -oE 'session_[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}_[0-9]+_[0-9]+' |
      sort -u
  )"

  if [ -n "${live_sessions}" ]; then
    printf '%s\n' ${live_sessions} | sort -u > "${session_record}" 2>/dev/null || true
  fi

  local candidate_dirs=""
  local d s

  # Always safe: this path is keyed by the current Slurm job id.
  for d in /tmp/slurm-${SLURM_JOB_ID}/tmp/ray/session_*/logs/nsight; do
    [ -d "${d}" ] && candidate_dirs="${candidate_dirs}
${d}"
  done

  # Only trust /tmp/ray for process-referenced live sessions.
  for s in ${live_sessions}; do
    [ -d "/tmp/ray/${s}/logs/nsight" ] && candidate_dirs="${candidate_dirs}
/tmp/ray/${s}/logs/nsight"

    [ -d "/tmp/slurm-${SLURM_JOB_ID}/tmp/ray/${s}/logs/nsight" ] && candidate_dirs="${candidate_dirs}
/tmp/slurm-${SLURM_JOB_ID}/tmp/ray/${s}/logs/nsight"
  done

  candidate_dirs="$(printf '%s\n' "${candidate_dirs}" | sed '/^$/d' | sort -u)"

  if [ "${NSYS_COPY_DEBUG:-0}" = "1" ] || [ "${DEBUG_SLURM_SCRIPT:-0}" = "1" ]; then
    echo "[nsight-copy] live copy on ${node} -> ${dest}"
    echo "[nsight-copy] live sessions:"
    if [ -n "${live_sessions}" ]; then
      printf '%s\n' ${live_sessions} | sed 's/^/[nsight-copy]   /'
    else
      echo "[nsight-copy]   none"
    fi

    echo "[nsight-copy] candidate dirs:"
    if [ -n "${candidate_dirs}" ]; then
      printf '%s\n' "${candidate_dirs}" | sed 's/^/[nsight-copy]   /'
    else
      echo "[nsight-copy]   none"
    fi
  fi

  if [ -n "${candidate_dirs}" ]; then
    while IFS= read -r d; do
      [ -d "${d}" ] || continue

      find "${d}" -maxdepth 1 -type f \
        \( \
          \( -name '*.nsys-rep' ! -name 'empty.nsys-rep' ! -name '*_empty.nsys-rep' ! -name '*empty*' -size +"${MIN_WORKER_NSYS_REP_BYTES}"c \) \
          -o \
          \( -name '*.qdstrm' ! -name '*empty*' -size +0c \) \
        \) \
        -print0 2>/dev/null |
        while IFS= read -r -d '' f; do
          local base session out tmpout
          base="$(basename "${f}")"
          session="$(echo "${f}" | sed -n 's#^.*/\(session_[^/]*\)/logs/nsight/.*#\1#p')"
          [ -n "${session}" ] || session=unknown_session

          out="${dest}/${node}_${session}_${base}"
          tmpout="${out}.tmp"

          # Copy via tmp + rename so the destination is not left half-written
          # if the source is still growing.
          cp -f "${f}" "${tmpout}" 2>/dev/null &&
            mv -f "${tmpout}" "${out}" 2>/dev/null ||
            rm -f "${tmpout}" 2>/dev/null || true

          if [ "${NSYS_COPY_DEBUG:-0}" = "1" ] || [ "${DEBUG_SLURM_SCRIPT:-0}" = "1" ]; then
            stat -c '[nsight-copy] copied %n %s bytes' "${out}" 2>/dev/null || true
          fi
        done
    done <<< "${candidate_dirs}"
  fi
}

copy_ray_nsight_locally_final() {
  set +e

  local node
  node="$(hostname -s 2>/dev/null || hostname)"
  local dest="${TRACE_RUN_DIR}/ray_worker_nsight/${node}"

  echo "[nsight-copy] final in-step copy on ${node}"
  copy_ray_nsight_locally_once

  echo "[nsight-copy] final in-step dest summary for ${node}: ${dest}"
  find "${dest}" -maxdepth 1 -type f \( -name '*.nsys-rep' -o -name '*.qdstrm' \) \
    -printf '[nsight-copy]   %p %s bytes\n' 2>/dev/null | sort || true
}

start_ray_nsight_live_copier() {
  set +e

  local node
  node="$(hostname -s 2>/dev/null || hostname)"

  echo "[nsight-copy] starting live worker Nsight sidecar on ${node}; interval=${WORKER_NSYS_LIVE_COPY_INTERVAL:-2}s"

  (
    set +e
    while true; do
      copy_ray_nsight_locally_once
      sleep "${WORKER_NSYS_LIVE_COPY_INTERVAL:-2}"
    done
  ) &
  NSYS_LIVE_COPIER_PID="$!"

  cleanup_live_copier() {
    set +e
    trap - EXIT TERM INT

    copy_ray_nsight_locally_final

    if [ -n "${NSYS_LIVE_COPIER_PID:-}" ] && kill -0 "${NSYS_LIVE_COPIER_PID}" 2>/dev/null; then
      kill "${NSYS_LIVE_COPIER_PID}" 2>/dev/null || true
      wait "${NSYS_LIVE_COPIER_PID}" 2>/dev/null || true
    fi

    copy_ray_nsight_locally_final
  }

  trap cleanup_live_copier EXIT TERM INT
}

# Post-run fallback copy. This is not the primary mechanism; the live sidecar is.
#
# IMPORTANT:
#   Do not blindly copy all /tmp/ray/session_* files. That picked up stale
#   reports in earlier debugging. This fallback searches:
#     1. current-job /tmp/slurm-${SLURM_JOB_ID}/tmp/ray/session_*/logs/nsight
#     2. /tmp/ray/<session names recorded while Ray was live>/logs/nsight
#
# Also intentionally skip empty.nsys-rep and zero-byte reports.
copy_ray_nsight_from_node() {
  local NODE="$1"
  local dest="${TRACE_RUN_DIR}/ray_worker_nsight/${NODE}"
  local session_record="${TRACE_RUN_DIR}/ray_session_names/${NODE}.txt"
  local srun_status=0

  nsight_copy_msg "=== fallback copy start: ${NODE} -> ${dest} ==="

  set +e
  run_with_timeout "${SRUN_COPY_TIMEOUT}" \
    srun \
      --nodelist "${NODE}" \
      --nodes=1 \
      --ntasks=1 \
      --ntasks-per-node=1 \
      --cpus-per-task=1 \
      --mem=1G \
      bash -lc "
        set +e
        mkdir -p '${dest}'

        echo '[nsight-copy] fallback on '\$(hostname)' expected ${NODE}'
        echo '[nsight-copy] dest=${dest}'
        echo '[nsight-copy] session_latest ->' \$(readlink -f /tmp/ray/session_latest 2>/dev/null || echo MISSING)

        candidate_dirs=''

        for d in /tmp/slurm-${SLURM_JOB_ID}/tmp/ray/session_*/logs/nsight; do
          [ -d \"\$d\" ] && candidate_dirs=\"\${candidate_dirs}
\$d\"
        done

        if [ -f '${session_record}' ]; then
          echo '[nsight-copy] recorded sessions from ${session_record}:'
          sed 's/^/[nsight-copy]   /' '${session_record}' 2>/dev/null || true

          while read -r s; do
            [ -n \"\$s\" ] || continue
            [ -d \"/tmp/ray/\${s}/logs/nsight\" ] && candidate_dirs=\"\${candidate_dirs}
/tmp/ray/\${s}/logs/nsight\"
            [ -d \"/tmp/slurm-${SLURM_JOB_ID}/tmp/ray/\${s}/logs/nsight\" ] && candidate_dirs=\"\${candidate_dirs}
/tmp/slurm-${SLURM_JOB_ID}/tmp/ray/\${s}/logs/nsight\"
          done < '${session_record}'
        else
          echo '[nsight-copy] no recorded session file: ${session_record}'
        fi

        candidate_dirs=\$(printf '%s\n' \"\${candidate_dirs}\" | sed '/^$/d' | sort -u)

        echo '[nsight-copy] fallback candidate dirs:'
        if [ -n \"\${candidate_dirs}\" ]; then
          printf '%s\n' \"\${candidate_dirs}\" | sed 's/^/[nsight-copy]   /'
        else
          echo '[nsight-copy]   none'
        fi

        if [ -n \"\${candidate_dirs}\" ]; then
          while read -r d; do
            [ -d \"\$d\" ] || continue

            echo '[nsight-copy] candidate report files in '\$d':'
            find \"\$d\" -maxdepth 1 -type f \
              \( \
                \( -name '*.nsys-rep' ! -name 'empty.nsys-rep' ! -name '*_empty.nsys-rep' ! -name '*empty*' -size +${MIN_WORKER_NSYS_REP_BYTES}c \) \
                -o \
                \( -name '*.qdstrm' ! -name '*empty*' -size +0c \) \
              \) \
              -printf '[nsight-copy] candidate %p %s bytes\n' 2>/dev/null | sort || true

            find \"\$d\" -maxdepth 1 -type f \
              \( \
                \( -name '*.nsys-rep' ! -name 'empty.nsys-rep' ! -name '*_empty.nsys-rep' ! -name '*empty*' -size +${MIN_WORKER_NSYS_REP_BYTES}c \) \
                -o \
                \( -name '*.qdstrm' ! -name '*empty*' -size +0c \) \
              \) \
              -print0 2>/dev/null |
              while IFS= read -r -d '' f; do
                base=\$(basename \"\$f\")
                session=\$(echo \"\$f\" | sed -n 's#^.*/\(session_[^/]*\)/logs/nsight/.*#\1#p')
                [ -n \"\$session\" ] || session=unknown_session

                out='${dest}'/\"\$(hostname -s)_\${session}_\${base}\"
                tmpout=\"\${out}.tmp\"

                cp -f \"\$f\" \"\$tmpout\" 2>/dev/null &&
                  mv -f \"\$tmpout\" \"\$out\" 2>/dev/null ||
                  rm -f \"\$tmpout\" 2>/dev/null || true
              done
          done <<< \"\${candidate_dirs}\"
        fi

        echo '[nsight-copy] dest after fallback copy:'
        find '${dest}' -maxdepth 1 -type f \( -name '*.nsys-rep' -o -name '*.qdstrm' \) \
          -printf '[nsight-copy]   %p %s bytes\n' 2>/dev/null | sort || true
      "
  srun_status=$?
  set -e

  nsight_copy_msg "=== fallback copy end: ${NODE} srun_status=${srun_status} ==="
  nsight_summarize_dest "${NODE}"
}

# SP = prompt / prefill token bucket
# SD = decode / output tokens per request
SP="${SP:-128}"
SD="${SD:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-25}"
REQUEST_RATE="${REQUEST_RATE:-1.0}"

export NSYS_ENABLE="${NSYS_ENABLE:-1}"

# Worker Nsight is the target. API-server outer wrapper is intentionally removed.
export NSYS_PROFILE_WORKERS="${NSYS_PROFILE_WORKERS:-1}"
export NSYS_PROFILE_RAY="${NSYS_PROFILE_RAY:-0}"

export HEAD_NODE
HEAD_NODE="$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)"

export WORKER_NODES
WORKER_NODES="$(scontrol show hostnames "$SLURM_NODELIST" | tail -n+2)"

echo "=== vLLM multi-node Qwen3-235B host job ==="
echo "SCRIPT_VERSION=${SCRIPT_VERSION}"
echo "Date: $(date -Is 2>/dev/null || date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "HEAD_NODE=${HEAD_NODE}"
echo "WORKER_NODES=${WORKER_NODES}"
slurm_debug "SLURM_NTASKS=${SLURM_NTASKS:-} SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-}"
slurm_debug "Full nodelist: $(scontrol show hostnames "${SLURM_NODELIST}" 2>/dev/null | tr '\n' ' ')"

# ARC/some clusters return link-local IPv6 records for Slurm hostnames.
# Ray/vLLM need a routable node address here, so resolve an IPv4 address.
resolve_host_ip() {
  local nodename="$1"
  local ip=""
  local method=""

  pick_ipv4() {
    awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {print; exit}'
  }

  ip=$(dig +short "${nodename}" 2>/dev/null | pick_ipv4 || true)
  if [ -n "${ip}" ]; then
    method="dig_ipv4"
  fi

  if [ -z "${ip}" ]; then
    ip=$(getent hosts "${nodename}" 2>/dev/null | awk '{print $1}' | pick_ipv4 || true)
    [ -n "${ip}" ] && method="getent_ipv4"
  fi

  if [ -z "${ip}" ]; then
    ip=$(
      srun --nodelist="${nodename}" --nodes=1 --ntasks=1 \
        --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
        bash -c "hostname -I 2>/dev/null | tr ' ' '\n' | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {print; exit}'" 2>/dev/null || true
    )
    [ -n "${ip}" ] && method="srun_hostname_I_ipv4"
  fi

  slurm_debug "resolve_host_ip(${nodename}) -> ${ip:-<empty>} [${method:-failed}]"
  printf '%s' "${ip}"
}

interface_for_ip() {
  local target_ip="$1"
  ip -o -4 addr show 2>/dev/null | awk -v target="${target_ip}" '
    {
      split($4, addr, "/")
      if (addr[1] == target) {
        print $2
        exit
      }
    }'
}

interface_has_ip() {
  local iface="$1"
  local target_ip="$2"
  ip -o -4 addr show dev "${iface}" 2>/dev/null | awk -v target="${target_ip}" '
    {
      split($4, addr, "/")
      if (addr[1] == target) {
        found = 1
        exit
      }
    }
    END { exit(found ? 0 : 1) }'
}

configure_socket_ifnames() {
  local target_ip="$1"
  local set_nccl="${2:-1}"
  local iface="${GLOO_SOCKET_IFNAME:-}"

  if [ -n "${iface}" ] && ! interface_has_ip "${iface}" "${target_ip}"; then
    echo "Ignoring GLOO_SOCKET_IFNAME=${iface}; it does not own ${target_ip} on $(hostname)." >&2
    iface=""
  fi

  if [ -z "${iface}" ]; then
    iface="$(interface_for_ip "${target_ip}")"
  fi

  if [ -z "${iface}" ]; then
    echo "Error: could not find a network interface for ${target_ip} on $(hostname)." >&2
    ip -o -4 addr show >&2 || true
    exit 1
  fi

  export GLOO_SOCKET_IFNAME="${iface}"

  if [ "${set_nccl}" = "1" ]; then
    local nccl_iface="${NCCL_SOCKET_IFNAME:-}"

    if [ -n "${nccl_iface}" ] && ! interface_has_ip "${nccl_iface}" "${target_ip}"; then
      echo "Ignoring NCCL_SOCKET_IFNAME=${nccl_iface}; it does not own ${target_ip} on $(hostname)." >&2
      nccl_iface=""
    fi

    export NCCL_SOCKET_IFNAME="${nccl_iface:-${iface}}"
  fi

  echo "Socket interface for ${target_ip} on $(hostname): GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<unset>}"
}

export HEAD_NODE_IP
HEAD_NODE_IP="$(resolve_host_ip "${HEAD_NODE}")"
if [ -z "${HEAD_NODE_IP}" ]; then
  echo "Error: could not resolve an IPv4 address for head node ${HEAD_NODE}." >&2
  exit 1
fi

echo "HEAD_NODE_IP=${HEAD_NODE_IP}"
export VLLM_HOST_IP="${HEAD_NODE_IP}"
echo "VLLM_HOST_IP=${VLLM_HOST_IP}"
configure_socket_ifnames "${HEAD_NODE_IP}" 0

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET="${NCCL_NET:-IB}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eno,ens,enp,eth,ib}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,COLL,P2P,TUNING}"

export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"

export RAY_PORT="${RAY_PORT:-6378}"
export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"
echo "RAY_PORT=${RAY_PORT} RAY_ADDRESS=${RAY_ADDRESS}"

module purge
module load Anaconda3/2025.06-1
module load CUDA/12.9.0

# === Trace output directory ===
TRACE_BASE="${TRACE_BASE:-/data/engs-glass/catz0932/inference-traces/vllm/results}"
TRACE_RUN_DIR="${TRACE_BASE}/${SLURM_JOB_ID}"

mkdir -p "${TRACE_RUN_DIR}/nsight"
mkdir -p "${TRACE_RUN_DIR}/nccl_logs"
mkdir -p "${TRACE_RUN_DIR}/ray_worker_nsight"
mkdir -p "${TRACE_RUN_DIR}/ray_session_names"

# === Nsight Systems ===
export NSYS_ENABLE="${NSYS_ENABLE:-1}"
export NSYS_DIR="${TRACE_RUN_DIR}/nsight"
export NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt,cudnn,cublas}"
export NSYS_DELAY="${NSYS_DELAY:-0}"

# Per-iteration + per-comm NVTX ranges on Ray GPU workers.
# Requires your patched vLLM with iteration_phase_nvtx.py / comm_nvtx_mark().
export VLLM_ITERATION_NVTX="${VLLM_ITERATION_NVTX:-1}"

# Sampled KV block residency metrics (lifetime, idle-before-evict, reuse gaps).
export VLLM_KV_CACHE_METRICS="${VLLM_KV_CACHE_METRICS:-1}"
export VLLM_KV_CACHE_METRICS_SAMPLE="${VLLM_KV_CACHE_METRICS_SAMPLE:-0.01}"

# === NCCL logs ===
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_FILE="${TRACE_RUN_DIR}/nccl_logs/nccl_%h_%p.log"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

VENV_DIR="${REPO_ROOT}/.venv"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-235B-A22B-Instruct-2507}"
HOST="${HOST:-${HEAD_NODE_IP}}"
PORT="${PORT:-8000}"

# Recommended layout for 2 nodes x 4 GPUs/node.
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NUM_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-2}}"
TOTAL_GPUS="$((GPUS_PER_NODE * NUM_NODES))"

# TP stays inside each node; PP spans the two nodes.
TP="${TP:-${GPUS_PER_NODE}}"
PP="${PP:-${NUM_NODES}}"
EP="${EP:-1}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

CPUS_PER_TASK="${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-1}}"
SERVE_SCRIPT="${REPO_ROOT}/serving_scripts/serve_ShareGPT_multi_node.sh"

# With TP=4 and PP=2, expect one Ray worker report per GPU, i.e. 4 per node.
EXPECTED_WORKER_REPORTS_PER_NODE="${EXPECTED_WORKER_REPORTS_PER_NODE:-${GPUS_PER_NODE}}"

echo "TRACE_RUN_DIR=${TRACE_RUN_DIR}"
echo "NSYS_DIR=${NSYS_DIR}"
echo "NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE}"
echo "NSYS_TRACE=${NSYS_TRACE}"
echo "NSYS_DELAY=${NSYS_DELAY}"
echo "VLLM_ITERATION_NVTX=${VLLM_ITERATION_NVTX}"
echo "VLLM_KV_CACHE_METRICS=${VLLM_KV_CACHE_METRICS}"
echo "VLLM_KV_CACHE_METRICS_SAMPLE=${VLLM_KV_CACHE_METRICS_SAMPLE}"
echo "NSYS_ENABLE=${NSYS_ENABLE}"
echo "NSYS_PROFILE_WORKERS=${NSYS_PROFILE_WORKERS}"
echo "NSYS_PROFILE_RAY=${NSYS_PROFILE_RAY}"
echo "API_SERVER_OUTER_NSYS_WRAPPER=0"
echo "NSYS_COPY_DEBUG=${NSYS_COPY_DEBUG}"
echo "SRUN_COPY_TIMEOUT=${SRUN_COPY_TIMEOUT}"
echo "SERVER_SHUTDOWN_TIMEOUT_S=${SERVER_SHUTDOWN_TIMEOUT_S}"
echo "WORKER_NSYS_LIVE_COPY_INTERVAL=${WORKER_NSYS_LIVE_COPY_INTERVAL}"
echo "WORKER_NSYS_FINALIZE_WAIT_S=${WORKER_NSYS_FINALIZE_WAIT_S}"
echo "WORKER_NSYS_FINALIZE_POLL_S=${WORKER_NSYS_FINALIZE_POLL_S}"
echo "MIN_WORKER_NSYS_REP_BYTES=${MIN_WORKER_NSYS_REP_BYTES}"
echo "EXPECTED_WORKER_REPORTS_PER_NODE=${EXPECTED_WORKER_REPORTS_PER_NODE}"
echo "nsys path: $(command -v nsys || echo '<not found>')"
nsys --version || true

echo "REPO_ROOT=${REPO_ROOT}"
echo "VENV_DIR=${VENV_DIR}"

if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

PYTHON_PATH="$(command -v python)"
EXPECTED_PYTHON="${VENV_DIR}/bin/python"
echo "Using python: ${PYTHON_PATH}"

if [ "${PYTHON_PATH}" != "${EXPECTED_PYTHON}" ]; then
  echo "Error: python did not resolve to venv interpreter." >&2
  echo "Expected: ${EXPECTED_PYTHON}" >&2
  echo "Got:      ${PYTHON_PATH}" >&2
  exit 1
fi

echo "After venv: python=$(command -v python) ray=$(command -v ray 2>/dev/null || echo '<not on PATH>')"
slurm_debug "PATH=${PATH}"
slurm_debug "pip install starting (cuda + build + editable vllm)..."

python -m pip install -U pip
python -m pip install -r "${REPO_ROOT}/requirements/cuda.txt"
python -m pip install -r "${REPO_ROOT}/requirements/build/cuda.txt"

RAY_REQUIREMENT="${RAY_REQUIREMENT:-ray[cgraph]>=2.48.0}"
echo "Installing Ray requirement: ${RAY_REQUIREMENT}"
python -m pip install "${RAY_REQUIREMENT}"

(
  cd "${REPO_ROOT}" || exit 1
  export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-1}"
  python -m pip install -e . ${VLLM_PIP_INSTALL_EXTRA_ARGS:-}
)

RAY_BIN="${VENV_DIR}/bin/ray"
if [ ! -x "${RAY_BIN}" ]; then
  echo "Error: ray binary not found at ${RAY_BIN}. Install ray into this venv." >&2
  echo "Hint: source \"${VENV_DIR}/bin/activate\" && python -m pip install ray" >&2
  exit 1
fi
echo "Using RAY_BIN=${RAY_BIN}"

export VLLM_TARGET_DEVICE=cuda
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"

echo "=== runtime knobs ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT} TP=${TP} PP=${PP} EP=${EP}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE} NUM_NODES=${NUM_NODES} TOTAL_GPUS=${TOTAL_GPUS} CPUS_PER_TASK=${CPUS_PER_TASK}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN} MAX_NUM_SEQS=${MAX_NUM_SEQS} MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE} NCCL_NET=${NCCL_NET} NCCL_IB_HCA=${NCCL_IB_HCA}"
echo "NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "NCCL_DEBUG=${NCCL_DEBUG} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
echo "SERVE_SCRIPT=${SERVE_SCRIPT}"

if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is set"
else
  echo "HF_TOKEN is not set"
fi

slurm_debug "VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE} VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM}"

SERVER_STEP_PID=""
HEAD_RAY_PID=""
WORKER_RAY_PIDS=""

cleanup() {
  set +e

  if [ -n "${SERVER_STEP_PID}" ] && kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
    kill "${SERVER_STEP_PID}" 2>/dev/null || true
    wait "${SERVER_STEP_PID}" 2>/dev/null || true
  fi

  if [ -n "${HEAD_RAY_PID}" ] && kill -0 "${HEAD_RAY_PID}" 2>/dev/null; then
    kill "${HEAD_RAY_PID}" 2>/dev/null || true
    wait "${HEAD_RAY_PID}" 2>/dev/null || true
  fi

  for pid in ${WORKER_RAY_PIDS}; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

echo "=== Ray head (background srun) ==="
echo "Starting head node ${HEAD_NODE}..."

RAY_HEAD_CMD="$(
  declare -f interface_for_ip
  declare -f interface_has_ip
  declare -f configure_socket_ifnames
  declare -f copy_ray_nsight_locally_once
  declare -f copy_ray_nsight_locally_final
  declare -f start_ray_nsight_live_copier
)
source \"${VENV_DIR}/bin/activate\"
unset GLOO_SOCKET_IFNAME

export TRACE_RUN_DIR='${TRACE_RUN_DIR}'
export SLURM_JOB_ID='${SLURM_JOB_ID:-unknown}'
export NSYS_COPY_DEBUG='${NSYS_COPY_DEBUG}'
export DEBUG_SLURM_SCRIPT='${DEBUG_SLURM_SCRIPT}'
export WORKER_NSYS_LIVE_COPY_INTERVAL='${WORKER_NSYS_LIVE_COPY_INTERVAL}'
export MIN_WORKER_NSYS_REP_BYTES='${MIN_WORKER_NSYS_REP_BYTES}'
export VLLM_ITERATION_NVTX='${VLLM_ITERATION_NVTX}'
export NSYS_ENABLE='${NSYS_ENABLE}'
export NSYS_TRACE='${NSYS_TRACE}'
export NSYS_DELAY='${NSYS_DELAY}'

start_ray_nsight_live_copier

export VLLM_HOST_IP=${HEAD_NODE_IP}
configure_socket_ifnames \"${HEAD_NODE_IP}\" 0

if [ \"${NSYS_ENABLE}\" = \"1\" ] && [ \"${NSYS_PROFILE_RAY}\" = \"1\" ]; then
  echo \"Profiling Ray head process with Nsight Systems\"
  nsys profile \\
    --force-overwrite=true \\
    --trace=\"${NSYS_TRACE}\" \\
    --sample=none \\
    --delay=\"${NSYS_DELAY}\" \\
    --output=\"${NSYS_DIR}/ray_head_${HEAD_NODE}\" \\
    \"${RAY_BIN}\" start --block \\
      --head \\
      --node-ip-address=${HEAD_NODE_IP} \\
      --port=${RAY_PORT} \\
      --num-gpus=${GPUS_PER_NODE} \\
      --num-cpus=${CPUS_PER_TASK}
else
  \"${RAY_BIN}\" start --block \\
    --head \\
    --node-ip-address=${HEAD_NODE_IP} \\
    --port=${RAY_PORT} \\
    --num-gpus=${GPUS_PER_NODE} \\
    --num-cpus=${CPUS_PER_TASK}
fi"

srun \
  --nodelist "${HEAD_NODE}" \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --gpus-per-task="${GPUS_PER_NODE}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  bash -lc "${RAY_HEAD_CMD}" &
HEAD_RAY_PID=$!

echo "HEAD_RAY_PID=${HEAD_RAY_PID}"
sleep 20

if [ -n "${WORKER_NODES}" ]; then
  echo "=== Ray workers ==="
  echo "Starting worker nodes..."

  for WORKER in ${WORKER_NODES}; do
    WORKER_IP="$(resolve_host_ip "${WORKER}")"

    if [ -z "${WORKER_IP}" ]; then
      echo "Error: could not resolve IP for worker ${WORKER}." >&2
      exit 1
    fi

    echo "Starting worker node: ${WORKER} with IP ${WORKER_IP}"

    RAY_WORKER_CMD="$(
      declare -f interface_for_ip
      declare -f interface_has_ip
      declare -f configure_socket_ifnames
      declare -f copy_ray_nsight_locally_once
      declare -f copy_ray_nsight_locally_final
      declare -f start_ray_nsight_live_copier
)
source \"${VENV_DIR}/bin/activate\"
unset GLOO_SOCKET_IFNAME

export TRACE_RUN_DIR='${TRACE_RUN_DIR}'
export SLURM_JOB_ID='${SLURM_JOB_ID:-unknown}'
export NSYS_COPY_DEBUG='${NSYS_COPY_DEBUG}'
export DEBUG_SLURM_SCRIPT='${DEBUG_SLURM_SCRIPT}'
export WORKER_NSYS_LIVE_COPY_INTERVAL='${WORKER_NSYS_LIVE_COPY_INTERVAL}'
export MIN_WORKER_NSYS_REP_BYTES='${MIN_WORKER_NSYS_REP_BYTES}'
export VLLM_ITERATION_NVTX='${VLLM_ITERATION_NVTX}'
export NSYS_ENABLE='${NSYS_ENABLE}'
export NSYS_TRACE='${NSYS_TRACE}'
export NSYS_DELAY='${NSYS_DELAY}'

start_ray_nsight_live_copier

export VLLM_HOST_IP=${WORKER_IP}
configure_socket_ifnames \"${WORKER_IP}\" 0

if [ \"${NSYS_ENABLE}\" = \"1\" ] && [ \"${NSYS_PROFILE_RAY}\" = \"1\" ]; then
  echo \"Profiling Ray worker process ${WORKER} with Nsight Systems\"
  nsys profile \\
    --force-overwrite=true \\
    --trace=\"${NSYS_TRACE}\" \\
    --sample=none \\
    --delay=\"${NSYS_DELAY}\" \\
    --output=\"${NSYS_DIR}/ray_worker_${WORKER}\" \\
    \"${RAY_BIN}\" start --block \\
      --address=${HEAD_NODE_IP}:${RAY_PORT} \\
      --node-ip-address=${WORKER_IP} \\
      --num-gpus=${GPUS_PER_NODE} \\
      --num-cpus=${CPUS_PER_TASK}
else
  \"${RAY_BIN}\" start --block \\
    --address=${HEAD_NODE_IP}:${RAY_PORT} \\
    --node-ip-address=${WORKER_IP} \\
    --num-gpus=${GPUS_PER_NODE} \\
    --num-cpus=${CPUS_PER_TASK}
fi"

    srun \
      --nodelist "${WORKER}" \
      --nodes=1 \
      --ntasks=1 \
      --ntasks-per-node=1 \
      --gpus-per-task="${GPUS_PER_NODE}" \
      --cpus-per-task="${CPUS_PER_TASK}" \
      bash -lc "${RAY_WORKER_CMD}" &

    WORKER_RAY_PIDS="${WORKER_RAY_PIDS} $!"
    echo "Worker Ray step pid: $! (WORKER_RAY_PIDS=${WORKER_RAY_PIDS})"
  done

  sleep 20
fi

echo "=== ray status ==="
echo "Checking cluster status..."

"${RAY_BIN}" status || echo "Warning: ray status failed; continuing with Python Ray node check."

python - <<'PY'
import ray

ray.init(address="auto")
nodes = ray.nodes()
print("Ray nodes:")
for node in nodes:
    print(
        f"  {node.get('NodeManagerAddress')} "
        f"alive={node.get('Alive')} "
        f"resources={node.get('Resources')}"
    )
PY

echo "=== vLLM api_server (background process; no outer nsys wrapper) ==="
echo "Starting vLLM server on head node WITHOUT outer Nsight wrapper..."

VLLM_TRACE_FLAGS=()
if [ "${NSYS_ENABLE}" = "1" ] && [ "${NSYS_PROFILE_WORKERS}" = "1" ]; then
  VLLM_TRACE_FLAGS+=(
    --ray-workers-use-nsight
    --enable-layerwise-nvtx-tracing
    --enable-logging-iteration-details
  )
fi

VLLM_KV_CACHE_FLAGS=()
if [ "${VLLM_KV_CACHE_METRICS}" = "1" ]; then
  VLLM_KV_CACHE_FLAGS+=(
    --kv-cache-metrics
    --kv-cache-metrics-sample "${VLLM_KV_CACHE_METRICS_SAMPLE}"
  )
fi

echo "API_SERVER_OUTER_NSYS_WRAPPER=0"
if [ "${NSYS_ENABLE}" = "1" ] && [ "${NSYS_PROFILE_WORKERS}" = "1" ]; then
  nsight_copy_msg "worker Nsight enabled via --ray-workers-use-nsight"
  nsight_copy_msg "VLLM_ITERATION_NVTX=${VLLM_ITERATION_NVTX}; patched comm NVTX should emit prefill/decode message records"
  nsight_copy_msg "live sidecar copies current-job Slurm tmp plus live /tmp/ray sessions"
  nsight_copy_msg "empty.nsys-rep and zero-byte reports are ignored"
fi

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ID}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --distributed-executor-backend ray \
  --tensor-parallel-size "${TP}" \
  --pipeline-parallel-size "${PP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enforce-eager \
  "${VLLM_TRACE_FLAGS[@]}" \
  "${VLLM_KV_CACHE_FLAGS[@]}" \
  --disable-custom-all-reduce &

SERVER_STEP_PID=$!

echo "Started vLLM server process (pid=${SERVER_STEP_PID}). Waiting for /health ..."

_health_wait_n=0
until curl -fsS "http://${HEAD_NODE_IP}:${PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
    echo "Server process exited before becoming ready." >&2
    wait "${SERVER_STEP_PID}" || true
    exit 1
  fi

  _health_wait_n=$((_health_wait_n + 1))

  if [ "${DEBUG_SLURM_SCRIPT}" = "1" ] || [ "$((_health_wait_n % 12))" -eq 0 ]; then
    echo "Still waiting for http://${HEAD_NODE_IP}:${PORT}/health (attempt ${_health_wait_n}) ..."
  fi

  sleep 5
done
unset _health_wait_n

echo "Server is healthy. Running ${SERVE_SCRIPT} ..."
echo "SP=${SP} SD=${SD}"

HOST="${HEAD_NODE_IP}" PORT="${PORT}" MODEL_ID="${MODEL_ID}" \
  SP="${SP}" SD="${SD}" \
  HEAD_NODE_IP="${HEAD_NODE_IP}" \
  GPUS_PER_NODE="${GPUS_PER_NODE}" CPUS_PER_TASK="${CPUS_PER_TASK}" \
  RAY_PORT="${RAY_PORT}" bash "${SERVE_SCRIPT}" "${SLURM_JOB_ID}" "${HEAD_NODE}"

# -----------------------------------------------------------------------------
# Clean shutdown and trace collection.
#
# Important ordering:
#   1. Stop vLLM API server.
#      There is no outer API-server nsys wrapper in this script.
#
#   2. Keep Ray alive while worker Nsight finalizes.
#      The live sidecar in each Ray srun step keeps copying from:
#        /tmp/slurm-${SLURM_JOB_ID}/tmp/ray/session_*/logs/nsight
#        /tmp/ray/<live-session>/logs/nsight
#
#      This is the key fix. Previously Ray was killed too soon, so only
#      empty.nsys-rep placeholders were copied.
#
#   3. Only after the wait, stop Ray head/worker srun --block steps.
#
#   4. Run a plain, non-overlapped fallback copy.
#
# Do not call extra diagnostic srun steps before Ray exits.
# Do not use --overlap in the copy path.
# -----------------------------------------------------------------------------

echo "Workload finished. Stopping vLLM server process cleanly..."

if [ -n "${SERVER_STEP_PID}" ] && kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
  echo "Sending SIGINT to vLLM server pid=${SERVER_STEP_PID}..."
  kill -INT "${SERVER_STEP_PID}" 2>/dev/null || true
  wait_for_pid_or_kill "${SERVER_STEP_PID}" "vLLM API server" "${SERVER_SHUTDOWN_TIMEOUT_S}"
  SERVER_STEP_PID=""
fi

echo "NSYS_DIR after server shutdown:"
ls -la "${NSYS_DIR}/" 2>/dev/null || true

echo "Keeping Ray alive so worker Nsight can finalize real reports..."
wait_for_real_worker_nsys_reports "${WORKER_NSYS_FINALIZE_WAIT_S}" "${WORKER_NSYS_FINALIZE_POLL_S}" || true

echo "Stopping Ray background srun steps after worker Nsight finalize wait..."

if [ -n "${HEAD_RAY_PID}" ] && kill -0 "${HEAD_RAY_PID}" 2>/dev/null; then
  kill -TERM "${HEAD_RAY_PID}" 2>/dev/null || true
  wait "${HEAD_RAY_PID}" 2>/dev/null || true
fi
HEAD_RAY_PID=""

for pid in ${WORKER_RAY_PIDS}; do
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
done
WORKER_RAY_PIDS=""

echo "Waiting briefly for Ray/Nsight files to flush..."
sleep 10

echo "Running fallback Ray worker Nsight copy..."
mkdir -p "${TRACE_RUN_DIR}/ray_worker_nsight"

copy_ray_nsight_from_node "${HEAD_NODE}"
for WORKER in ${WORKER_NODES}; do
  copy_ray_nsight_from_node "${WORKER}"
done

nsight_copy_msg "=== final Nsight artifact summary ==="
nsight_copy_msg "API server trace dir, expected empty/no api-server nsys in this variant: ${NSYS_DIR}"
ls -la "${NSYS_DIR}/" 2>/dev/null || true

for NODE in ${HEAD_NODE} ${WORKER_NODES}; do
  nsight_summarize_dest "${NODE}"
done

echo "Trace files:"
find "${TRACE_RUN_DIR}" -maxdepth 5 -type f -printf "%p %s bytes\n" 2>/dev/null || true