#!/usr/bin/env bash
#SBATCH --nodelist=htc-g[059-060]
#SBATCH --job-name=r32_sp128_sd128_pp2_tp4_qwen3_30b
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

SCRIPT_VERSION="arc-ray-qwen3-30b-a3b-r32-sp128-sd128-tp4-pp2-raytmp-gated-nsys-v2"

# -----------------------------------------------------------------------------
# Debug / timeout knobs
# -----------------------------------------------------------------------------

DEBUG_SLURM_SCRIPT="${DEBUG_SLURM_SCRIPT:-0}"
NSYS_COPY_DEBUG="${NSYS_COPY_DEBUG:-0}"

SRUN_COPY_TIMEOUT="${SRUN_COPY_TIMEOUT:-480s}"
SERVER_SHUTDOWN_TIMEOUT_S="${SERVER_SHUTDOWN_TIMEOUT_S:-420}"

HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1200}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-2400}"

RAY_CLIENT_PORT="${RAY_CLIENT_PORT:-10001}"
RAY_TCP_PREFLIGHT_TIMEOUT_S="${RAY_TCP_PREFLIGHT_TIMEOUT_S:-360}"
RAY_HEAD_START_SLEEP_S="${RAY_HEAD_START_SLEEP_S:-30}"
RAY_WORKER_START_SLEEP_S="${RAY_WORKER_START_SLEEP_S:-30}"

RAY_CLUSTER_READY_TIMEOUT_S="${RAY_CLUSTER_READY_TIMEOUT_S:-600}"
RAY_CLUSTER_READY_POLL_S="${RAY_CLUSTER_READY_POLL_S:-10}"

WORKER_NSYS_LIVE_COPY_INTERVAL="${WORKER_NSYS_LIVE_COPY_INTERVAL:-2}"
WORKER_NSYS_FINALIZE_WAIT_S="${WORKER_NSYS_FINALIZE_WAIT_S:-900}"
WORKER_NSYS_FINALIZE_POLL_S="${WORKER_NSYS_FINALIZE_POLL_S:-5}"
MIN_WORKER_NSYS_REP_BYTES="${MIN_WORKER_NSYS_REP_BYTES:-1024}"

# -----------------------------------------------------------------------------
# Workload knobs
# -----------------------------------------------------------------------------

SP="${SP:-128}"
SD="${SD:-128}"

NUM_PROMPTS="${NUM_PROMPTS:-32}"
REQUEST_RATE="${REQUEST_RATE:-1}"
BURSTINESS="${BURSTINESS:-1.0}"
SEED="${SEED:-100}"

ENDPOINT="${ENDPOINT:-/v1/completions}"

AUTO_GENERATE_DATASET="${AUTO_GENERATE_DATASET:-1}"
DATASET_NAME="${DATASET_NAME:-Aeala/ShareGPT_Vicuna_unfiltered}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
DATASET_MAX_PER_BUCKET="${DATASET_MAX_PER_BUCKET:-1000}"
DATASET_MAX_SOURCE_ROWS="${DATASET_MAX_SOURCE_ROWS:-200000}"

DATASET_PATH="${DATASET_PATH:-}"

SAVE_RESULT="${SAVE_RESULT:-0}"
SAVE_DETAILED="${SAVE_DETAILED:-0}"
RESULT_ROOT="${RESULT_ROOT:-}"
RESULT_DIR="${RESULT_DIR:-}"
RESULT_FILENAME="${RESULT_FILENAME:-}"

# -----------------------------------------------------------------------------
# Profiling / runtime env
# -----------------------------------------------------------------------------

export NSYS_ENABLE="${NSYS_ENABLE:-1}"
export NSYS_PROFILE_WORKERS="${NSYS_PROFILE_WORKERS:-1}"
export NSYS_PROFILE_RAY="${NSYS_PROFILE_RAY:-0}"

export RAY_USAGE_STATS_ENABLED="${RAY_USAGE_STATS_ENABLED:-1}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"

export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
export VLLM_LOG_STATS_INTERVAL="${VLLM_LOG_STATS_INTERVAL:-1}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"

# Needed for vLLM/Ray/Nsight multiprocessing behavior.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# Ray temp layout:
# Ray sees a short path:
#   /tmp/vray-${SLURM_JOB_ID}-${node}
# that points to real job-local Slurm tmp:
#   /tmp/slurm-${SLURM_JOB_ID}/tmp/ray-${SLURM_JOB_ID}-${node}
RAY_TMP_LINK_PARENT="${RAY_TMP_LINK_PARENT:-/tmp}"
RAY_TMP_PREFIX="${RAY_TMP_PREFIX:-vray-${SLURM_JOB_ID}}"

RAY_PLASMA_DIRECTORY="${RAY_PLASMA_DIRECTORY:-/dev/shm}"
RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-200000000000}"

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

slurm_debug() {
  if [ "${DEBUG_SLURM_SCRIPT}" = "1" ]; then
    echo "[slurm-debug] $*" >&2
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

# -----------------------------------------------------------------------------
# Network helpers
# -----------------------------------------------------------------------------

resolve_host_ip() {
  local nodename="$1"
  local ip=""

  pick_ipv4() {
    awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {print; exit}'
  }

  ip="$(dig +short "${nodename}" 2>/dev/null | pick_ipv4 || true)"
  if [ -z "${ip}" ]; then
    ip="$(getent hosts "${nodename}" 2>/dev/null | awk '{print $1}' | pick_ipv4 || true)"
  fi
  if [ -z "${ip}" ]; then
    ip="$(
      srun --nodelist="${nodename}" --nodes=1 --ntasks=1 \
        --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
        bash -c "hostname -I 2>/dev/null | tr ' ' '\n' | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {print; exit}'" 2>/dev/null || true
    )"
  fi

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

configure_socket_ifnames() {
  local target_ip="$1"
  local iface=""

  iface="$(interface_for_ip "${target_ip}")"

  if [ -z "${iface}" ]; then
    echo "Error: could not find a network interface owning ${target_ip} on $(hostname)." >&2
    ip -o -4 addr show >&2 || true
    exit 1
  fi

  export GLOO_SOCKET_IFNAME="${iface}"
  export NCCL_SOCKET_IFNAME="${iface}"
  export VLLM_HOST_IP="${target_ip}"
  export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"

  echo "Socket interface for ${target_ip} on $(hostname): GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} VLLM_HOST_IP=${VLLM_HOST_IP}"
}

network_debug_snapshot() {
  local label="$1"

  echo "=== network debug: ${label} ==="
  echo "hostname=$(hostname)"
  echo "hostname -s=$(hostname -s 2>/dev/null || true)"
  echo "hostname -I=$(hostname -I 2>/dev/null || true)"
  echo "HEAD_NODE=${HEAD_NODE:-}"
  echo "WORKER_NODES=${WORKER_NODES:-}"
  echo "HEAD_NODE_IP=${HEAD_NODE_IP:-}"
  echo "VLLM_HOST_IP=${VLLM_HOST_IP:-}"
  echo "HOST=${HOST:-}"
  echo "PORT=${PORT:-}"
  echo "RAY_ADDRESS=${RAY_ADDRESS:-}"
  echo "RAY_PORT=${RAY_PORT:-}"
  echo "RAY_CLIENT_PORT=${RAY_CLIENT_PORT:-}"
  echo "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-}"
  echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}"
  echo "NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-}"
  echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-}"
  echo "NCCL_NET=${NCCL_NET:-}"
  echo "NCCL_IB_HCA=${NCCL_IB_HCA:-}"
  echo "--- ip -o -4 addr show ---"
  ip -o -4 addr show 2>/dev/null || true
  echo "--- nvidia-smi -L ---"
  nvidia-smi -L 2>/dev/null || true
  echo "=== end network debug: ${label} ==="
}

# -----------------------------------------------------------------------------
# Ray tmp helpers
# -----------------------------------------------------------------------------

discover_arc_node_tmp_parent() {
  local slurm_tmp_with_tmp="/tmp/slurm-${SLURM_JOB_ID}/tmp"
  local slurm_tmp_root="/tmp/slurm-${SLURM_JOB_ID}"

  mkdir -p "${slurm_tmp_with_tmp}" 2>/dev/null || true
  if [ -d "${slurm_tmp_with_tmp}" ]; then
    printf '%s' "${slurm_tmp_with_tmp}"
    return 0
  fi

  mkdir -p "${slurm_tmp_root}" 2>/dev/null || true
  if [ -d "${slurm_tmp_root}" ]; then
    printf '%s' "${slurm_tmp_root}"
    return 0
  fi

  echo "Error: could not create/find Slurm tmp for job ${SLURM_JOB_ID}" >&2
  exit 1
}

ray_tmp_link_for_node() {
  local node="$1"
  printf '%s/%s-%s' "${RAY_TMP_LINK_PARENT}" "${RAY_TMP_PREFIX}" "${node}"
}

ray_tmp_real_dir_for_node() {
  local node="$1"
  local parent
  parent="$(discover_arc_node_tmp_parent)"
  printf '%s/ray-%s-%s' "${parent}" "${SLURM_JOB_ID}" "${node}"
}

ensure_ray_tmp_for_node() {
  local node="$1"
  local parent root link

  parent="$(discover_arc_node_tmp_parent)"
  root="$(ray_tmp_real_dir_for_node "${node}")"
  link="$(ray_tmp_link_for_node "${node}")"

  rm -rf "${root}" "${link}" 2>/dev/null || true
  mkdir -p "${parent}" "${root}" "${root}/spill" "${root}/py_tmp"
  ln -sfn "${root}" "${link}"

  export RAY_TMPDIR="${link}"
  export RAY_SPILL_DIR="${link}/spill"
  export TMPDIR="${link}/py_tmp"

  echo "[ray-tmp] node=${node}"
  echo "[ray-tmp] real_root=${root}"
  echo "[ray-tmp] short_link=${link} -> $(readlink -f "${link}" 2>/dev/null || echo MISSING)"
  echo "[ray-tmp] RAY_TMPDIR=${RAY_TMPDIR}"
  echo "[ray-tmp] TMPDIR=${TMPDIR}"
  echo "[ray-tmp] RAY_PLASMA_DIRECTORY=${RAY_PLASMA_DIRECTORY}"
  echo "[ray-tmp] RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY}"
  df -h /tmp "${parent}" "${root}" "${link}" "${RAY_PLASMA_DIRECTORY}" 2>/dev/null || true
}

# -----------------------------------------------------------------------------
# Nsight copy helpers
# -----------------------------------------------------------------------------

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
      grep -E 'ray|vllm|EngineCore|worker_process|nsys|vray' |
      grep -oE 'session_[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}_[0-9]+_[0-9]+' |
      sort -u
  )"

  if [ -n "${live_sessions}" ]; then
    printf '%s\n' ${live_sessions} | sort -u > "${session_record}" 2>/dev/null || true
  fi

  local candidate_dirs=""
  local d s ray_real

  if [ -n "${RAY_TMPDIR:-}" ]; then
    for d in "${RAY_TMPDIR}"/session_*/logs/nsight "${RAY_TMPDIR}"/session_latest/logs/nsight; do
      [ -d "${d}" ] && candidate_dirs="${candidate_dirs}
${d}"
    done

    ray_real="$(readlink -f "${RAY_TMPDIR}" 2>/dev/null || true)"
    if [ -n "${ray_real}" ]; then
      for d in "${ray_real}"/session_*/logs/nsight "${ray_real}"/session_latest/logs/nsight; do
        [ -d "${d}" ] && candidate_dirs="${candidate_dirs}
${d}"
      done
    fi
  fi

  for d in \
    /tmp/vray-${SLURM_JOB_ID}-${node}/session_*/logs/nsight \
    /tmp/vray-${SLURM_JOB_ID}-${node}/session_latest/logs/nsight \
    /tmp/slurm-${SLURM_JOB_ID}/tmp/ray-${SLURM_JOB_ID}-${node}/session_*/logs/nsight \
    /tmp/slurm-${SLURM_JOB_ID}/tmp/ray-${SLURM_JOB_ID}-${node}/session_latest/logs/nsight \
    /tmp/ray/session_*/logs/nsight; do
    [ -d "${d}" ] && candidate_dirs="${candidate_dirs}
${d}"
  done

  for s in ${live_sessions}; do
    [ -d "/tmp/ray/${s}/logs/nsight" ] && candidate_dirs="${candidate_dirs}
/tmp/ray/${s}/logs/nsight"
    [ -d "/tmp/vray-${SLURM_JOB_ID}-${node}/${s}/logs/nsight" ] && candidate_dirs="${candidate_dirs}
/tmp/vray-${SLURM_JOB_ID}-${node}/${s}/logs/nsight"
    [ -d "/tmp/slurm-${SLURM_JOB_ID}/tmp/ray-${SLURM_JOB_ID}-${node}/${s}/logs/nsight" ] && candidate_dirs="${candidate_dirs}
/tmp/slurm-${SLURM_JOB_ID}/tmp/ray-${SLURM_JOB_ID}-${node}/${s}/logs/nsight"
  done

  candidate_dirs="$(printf '%s\n' "${candidate_dirs}" | sed '/^$/d' | sort -u)"

  if [ "${NSYS_COPY_DEBUG}" = "1" ] || [ "${DEBUG_SLURM_SCRIPT}" = "1" ]; then
    echo "[nsight-copy] live copy on ${node} -> ${dest}"
    echo "[nsight-copy] RAY_TMPDIR=${RAY_TMPDIR:-<unset>}"
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

          cp -f "${f}" "${tmpout}" 2>/dev/null &&
            mv -f "${tmpout}" "${out}" 2>/dev/null ||
            rm -f "${tmpout}" 2>/dev/null || true
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

wait_for_real_worker_nsys_reports() {
  local timeout_s="${1:-300}"
  local poll_s="${2:-5}"
  local elapsed=0

  if [ "${NSYS_ENABLE:-1}" != "1" ] || [ "${NSYS_PROFILE_WORKERS:-1}" != "1" ]; then
    echo "[nsight-copy] worker Nsight disabled; skipping wait for worker reports."
    return 0
  fi

  echo "[nsight-copy] waiting up to ${timeout_s}s for real non-empty worker_process_*.nsys-rep files..."
  echo "[nsight-copy] expected nodes: ${HEAD_NODE} ${WORKER_NODES}"

  while [ "${elapsed}" -lt "${timeout_s}" ]; do
    local all_ok=1
    local missing_nodes=""

    for node in ${HEAD_NODE} ${WORKER_NODES}; do
      local dest="${TRACE_RUN_DIR}/ray_worker_nsight/${node}"

      if ! find "${dest}" -maxdepth 1 -type f \
        -name '*worker_process*.nsys-rep' \
        ! -name '*empty*' \
        -size +"${MIN_WORKER_NSYS_REP_BYTES}"c \
        -print -quit 2>/dev/null | grep -q .; then
        all_ok=0
        missing_nodes="${missing_nodes} ${node}"
      fi
    done

    if [ "${all_ok}" = "1" ]; then
      echo "[nsight-copy] found real non-empty worker_process_*.nsys-rep on every node."
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

  echo "[nsight-copy] WARNING: timed out waiting for real non-empty worker_process_*.nsys-rep files."
  return 1
}

copy_ray_nsight_from_node() {
  local NODE="$1"
  local dest="${TRACE_RUN_DIR}/ray_worker_nsight/${NODE}"
  local session_record="${TRACE_RUN_DIR}/ray_session_names/${NODE}.txt"
  local srun_status=0

  nsight_copy_msg "=== fallback copy start: ${NODE} -> ${dest} ==="

  set +e
  run_with_timeout "${SRUN_COPY_TIMEOUT}" \
    srun \
      --overlap \
      --nodelist "${NODE}" \
      --nodes=1 \
      --ntasks=1 \
      --ntasks-per-node=1 \
      --cpus-per-task=1 \
      --mem=1G \
      bash -lc "
        set +e
        mkdir -p '${dest}'

        node=\$(hostname -s 2>/dev/null || hostname)
        echo '[nsight-copy] fallback on '\${node}' expected ${NODE}'
        echo '[nsight-copy] dest=${dest}'

        candidate_dirs=''

        for d in \
          /tmp/vray-${SLURM_JOB_ID}-${NODE}/session_*/logs/nsight \
          /tmp/vray-${SLURM_JOB_ID}-${NODE}/session_latest/logs/nsight \
          /tmp/slurm-${SLURM_JOB_ID}/tmp/ray-${SLURM_JOB_ID}-${NODE}/session_*/logs/nsight \
          /tmp/slurm-${SLURM_JOB_ID}/tmp/ray-${SLURM_JOB_ID}-${NODE}/session_latest/logs/nsight \
          /tmp/ray/session_*/logs/nsight; do
          [ -d \"\$d\" ] && candidate_dirs=\"\${candidate_dirs}
\$d\"
        done

        vray_real=\$(readlink -f /tmp/vray-${SLURM_JOB_ID}-${NODE} 2>/dev/null || true)
        if [ -n \"\${vray_real}\" ]; then
          for d in \"\${vray_real}\"/session_*/logs/nsight \"\${vray_real}\"/session_latest/logs/nsight; do
            [ -d \"\$d\" ] && candidate_dirs=\"\${candidate_dirs}
\$d\"
          done
        fi

        if [ -f '${session_record}' ]; then
          while read -r s; do
            [ -n \"\$s\" ] || continue
            [ -d \"/tmp/ray/\${s}/logs/nsight\" ] && candidate_dirs=\"\${candidate_dirs}
/tmp/ray/\${s}/logs/nsight\"
            [ -d \"/tmp/vray-${SLURM_JOB_ID}-${NODE}/\${s}/logs/nsight\" ] && candidate_dirs=\"\${candidate_dirs}
/tmp/vray-${SLURM_JOB_ID}-${NODE}/\${s}/logs/nsight\"
            [ -d \"/tmp/slurm-${SLURM_JOB_ID}/tmp/ray-${SLURM_JOB_ID}-${NODE}/\${s}/logs/nsight\" ] && candidate_dirs=\"\${candidate_dirs}
/tmp/slurm-${SLURM_JOB_ID}/tmp/ray-${SLURM_JOB_ID}-${NODE}/\${s}/logs/nsight\"
          done < '${session_record}'
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

                out='${dest}'/\"\${node}_\${session}_\${base}\"
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

# -----------------------------------------------------------------------------
# Ray readiness / shutdown helpers
# -----------------------------------------------------------------------------

wait_for_ray_cluster_ready() {
  local expected_nodes="$1"
  local expected_gpus="$2"
  local timeout_s="${3:-600}"
  local poll_s="${4:-10}"

  echo "=== Waiting for Ray cluster readiness ==="
  echo "Expected alive nodes: ${expected_nodes}"
  echo "Expected GPUs:        ${expected_gpus}"
  echo "RAY_ADDRESS=${RAY_ADDRESS}"

  export RAY_TMPDIR
  RAY_TMPDIR="$(ray_tmp_link_for_node "${HEAD_NODE}")"
  export TMPDIR="${RAY_TMPDIR}/py_tmp"
  mkdir -p "${TMPDIR}" 2>/dev/null || true

  echo "Ray readiness driver RAY_TMPDIR=${RAY_TMPDIR} -> $(readlink -f "${RAY_TMPDIR}" 2>/dev/null || echo MISSING)"
  echo "Ray readiness driver TMPDIR=${TMPDIR}"

  EXPECTED_RAY_NODES="${expected_nodes}" \
  EXPECTED_RAY_GPUS="${expected_gpus}" \
  RAY_READY_TIMEOUT_S="${timeout_s}" \
  RAY_READY_POLL_S="${poll_s}" \
  python -u <<'PY'
import os
import sys
import time
import ray

address = os.environ["RAY_ADDRESS"]
expected_nodes = int(os.environ["EXPECTED_RAY_NODES"])
expected_gpus = float(os.environ["EXPECTED_RAY_GPUS"])
timeout_s = int(os.environ["RAY_READY_TIMEOUT_S"])
poll_s = int(os.environ["RAY_READY_POLL_S"])

deadline = time.time() + timeout_s
last = None

while time.time() < deadline:
    try:
        ray.init(address=address, ignore_reinit_error=True)
        nodes = [n for n in ray.nodes() if n.get("Alive")]
        resources = ray.cluster_resources()
        available = ray.available_resources()

        print("Ray readiness check:", flush=True)
        print(f"  alive_nodes={len(nodes)}/{expected_nodes}", flush=True)
        print(f"  cluster_resources={resources}", flush=True)
        print(f"  available_resources={available}", flush=True)

        for n in nodes:
            print(
                f"  node={n.get('NodeManagerAddress')} "
                f"resources={n.get('Resources')}",
                flush=True,
            )

        gpu_count = float(resources.get("GPU", 0.0))
        if len(nodes) >= expected_nodes and gpu_count >= expected_gpus:
            print("Ray cluster is ready.", flush=True)
            ray.shutdown()
            sys.exit(0)

        last = f"alive_nodes={len(nodes)}, gpu_count={gpu_count}"

    except Exception as e:
        last = repr(e)
        print(f"Ray readiness error: {last}", flush=True)

    try:
        ray.shutdown()
    except Exception:
        pass

    time.sleep(poll_s)

print(f"ERROR: Ray cluster did not become ready. Last state: {last}", file=sys.stderr)
sys.exit(1)
PY
}

stop_ray_cleanly_on_node() {
  local NODE="$1"
  local node_tmp

  node_tmp="$(ray_tmp_link_for_node "${NODE}")"

  echo "=== stopping Ray cleanly on ${NODE} ==="

  set +e
  run_with_timeout 240s \
    srun \
      --overlap \
      --nodelist "${NODE}" \
      --nodes=1 \
      --ntasks=1 \
      --ntasks-per-node=1 \
      --cpus-per-task=2 \
      --mem=4G \
      bash -lc "
        set +e
        source '${VENV_DIR}/bin/activate'

        export RAY_TMPDIR='${node_tmp}'
        export TMPDIR='${node_tmp}/py_tmp'
        mkdir -p \"\${TMPDIR}\" 2>/dev/null || true

        echo '[ray-stop] on '\$(hostname)
        echo '[ray-stop] RAY_TMPDIR='\"\${RAY_TMPDIR}\"
        echo '[ray-stop] before:'
        ps -u '$USER' -f | egrep 'nsys|ray::|raylet|gcs_server|worker_process|vllm|EngineCore' | grep -v grep || true

        ray stop || true

        echo '[ray-stop] sleeping 90s for Nsight export...'
        sleep 90

        ray stop --force || true

        echo '[ray-stop] after:'
        ps -u '$USER' -f | egrep 'nsys|ray::|raylet|gcs_server|worker_process|vllm|EngineCore' | grep -v grep || true

        echo '[ray-stop] nsight files:'
        find '${node_tmp}'/session_*/logs/nsight '${node_tmp}'/session_latest/logs/nsight \
          -maxdepth 1 -type f \( -name '*.nsys-rep' -o -name '*.qdstrm' \) \
          -printf '%p %s bytes\n' 2>/dev/null | sort || true
      "
  set -e
}

# -----------------------------------------------------------------------------
# Main setup
# -----------------------------------------------------------------------------

export HEAD_NODE
HEAD_NODE="$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)"

export WORKER_NODES
WORKER_NODES="$(scontrol show hostnames "$SLURM_NODELIST" | tail -n+2)"

echo "=== vLLM multi-node host job ==="
echo "SCRIPT_VERSION=${SCRIPT_VERSION}"
echo "Date: $(date -Is 2>/dev/null || date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "HEAD_NODE=${HEAD_NODE}"
echo "WORKER_NODES=${WORKER_NODES}"

export HEAD_NODE_IP
HEAD_NODE_IP="$(resolve_host_ip "${HEAD_NODE}")"
if [ -z "${HEAD_NODE_IP}" ]; then
  echo "Error: could not resolve IPv4 address for head node ${HEAD_NODE}." >&2
  exit 1
fi

echo "HEAD_NODE_IP=${HEAD_NODE_IP}"

module purge
module load Anaconda3/2025.06-1
module load CUDA/12.9.0

TRACE_BASE="${TRACE_BASE:-/data/engs-glass/catz0932/inference-traces/vllm/results}"
TRACE_RUN_DIR="${TRACE_BASE}/${SLURM_JOB_ID}"

mkdir -p "${TRACE_RUN_DIR}/nsight"
mkdir -p "${TRACE_RUN_DIR}/nccl_logs"
mkdir -p "${TRACE_RUN_DIR}/ray_worker_nsight"
mkdir -p "${TRACE_RUN_DIR}/ray_session_names"
mkdir -p "${TRACE_RUN_DIR}/ray_logs"

export NSYS_DIR="${TRACE_RUN_DIR}/nsight"
export NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt,cudnn,cublas}"
export NSYS_DELAY="${NSYS_DELAY:-0}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET="${NCCL_NET:-IB}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,COLL,P2P,TUNING}"
export NCCL_DEBUG_FILE="${TRACE_RUN_DIR}/nccl_logs/nccl_%h_%p.log"

export RAY_PORT="${RAY_PORT:-6378}"
export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"

echo "RAY_PORT=${RAY_PORT} RAY_ADDRESS=${RAY_ADDRESS}"
echo "RAY_CLIENT_PORT=${RAY_CLIENT_PORT}"
echo "TRACE_RUN_DIR=${TRACE_RUN_DIR}"
echo "NSYS_DIR=${NSYS_DIR}"
echo "NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE}"
echo "NSYS_TRACE=${NSYS_TRACE}"
echo "NSYS_DELAY=${NSYS_DELAY}"
echo "NSYS_ENABLE=${NSYS_ENABLE}"
echo "NSYS_PROFILE_WORKERS=${NSYS_PROFILE_WORKERS}"
echo "NSYS_PROFILE_RAY=${NSYS_PROFILE_RAY}"
echo "VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD}"
echo "VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL}"
echo "VLLM_LOG_STATS_INTERVAL=${VLLM_LOG_STATS_INTERVAL}"
echo "VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S}"
echo "HEALTH_TIMEOUT_S=${HEALTH_TIMEOUT_S}"
echo "TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG}"
echo "RAY_USAGE_STATS_ENABLED=${RAY_USAGE_STATS_ENABLED}"
echo "RAY_TMP_LINK_PARENT=${RAY_TMP_LINK_PARENT}"
echo "RAY_TMP_PREFIX=${RAY_TMP_PREFIX}"
echo "RAY_PLASMA_DIRECTORY=${RAY_PLASMA_DIRECTORY}"
echo "RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY}"
echo "nsys path: $(command -v nsys || echo '<not found>')"
nsys --version || true

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

VENV_DIR="${REPO_ROOT}/.venv"
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

INSTALL_DEPS="${INSTALL_DEPS:-0}"

if [ "${INSTALL_DEPS}" = "1" ]; then
  echo "INSTALL_DEPS=1; installing dependencies and editable vLLM..."

  python -m pip install -U pip
  python -m pip install -r "${REPO_ROOT}/requirements/cuda.txt"
  python -m pip install -r "${REPO_ROOT}/requirements/build/cuda.txt"

  RAY_REQUIREMENT="${RAY_REQUIREMENT:-ray[cgraph]>=2.48.0}"
  echo "Installing Ray requirement: ${RAY_REQUIREMENT}"
  python -m pip install "${RAY_REQUIREMENT}"

  (
    cd "${REPO_ROOT}" || exit 1
    export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-1}"
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM:-0.19.2.dev0}"
    export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.19.2.dev0}"

    install_ok=0
    for attempt in 1 2 3; do
      echo "Editable vLLM install attempt ${attempt}/3..."
      if python -m pip install -e . ${VLLM_PIP_INSTALL_EXTRA_ARGS:-}; then
        install_ok=1
        break
      fi
      echo "Editable install failed on attempt ${attempt}; retrying in 30s..."
      sleep 30
    done

    if [ "${install_ok}" != "1" ]; then
      echo "Error: editable vLLM install failed after 3 attempts." >&2
      exit 1
    fi
  )
else
  echo "INSTALL_DEPS=${INSTALL_DEPS}; skipping pip install steps and using existing venv."
fi

RAY_BIN="${VENV_DIR}/bin/ray"
if [ ! -x "${RAY_BIN}" ]; then
  echo "Error: ray binary not found at ${RAY_BIN}. Install ray into this venv." >&2
  exit 1
fi

echo "After venv: python=$(command -v python) ray=${RAY_BIN}"

export VLLM_TARGET_DEVICE=cuda
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
HOST="${HOST:-${HEAD_NODE_IP}}"
PORT="${PORT:-8000}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NUM_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-2}}"
TP="${TP:-4}"
PP="${PP:-2}"
EP="${EP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
CPUS_PER_TASK="${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-1}}"
SERVE_SCRIPT="${REPO_ROOT}/serving_scripts/serve_ShareGPT_multi_node.sh"

if [ -z "${RESULT_ROOT}" ]; then
  RESULT_ROOT="${REPO_ROOT}/traces/bench_results"
fi

if [ -z "${RESULT_DIR}" ]; then
  MODEL_SLUG_FOR_RESULT="$(printf '%s' "${MODEL_ID}" | sed -E 's#[/:]+#_#g; s#[^A-Za-z0-9._-]+#_#g')"
  RESULT_DIR="${RESULT_ROOT}/${MODEL_SLUG_FOR_RESULT}_sp${SP}_sd${SD}_np${NUM_PROMPTS}_rr${REQUEST_RATE}_job${SLURM_JOB_ID}"
fi

if [ -z "${RESULT_FILENAME}" ]; then
  RESULT_FILENAME="sharegpt_sp${SP}_sd${SD}_np${NUM_PROMPTS}_rr${REQUEST_RATE}_job${SLURM_JOB_ID}.json"
fi

export MODEL_ID HOST PORT GPUS_PER_NODE NUM_NODES TP PP EP MAX_MODEL_LEN CPUS_PER_TASK SERVE_SCRIPT
export SP SD NUM_PROMPTS REQUEST_RATE BURSTINESS SEED ENDPOINT
export AUTO_GENERATE_DATASET DATASET_NAME DATASET_SPLIT DATASET_MAX_PER_BUCKET DATASET_MAX_SOURCE_ROWS DATASET_PATH
export SAVE_RESULT SAVE_DETAILED RESULT_ROOT RESULT_DIR RESULT_FILENAME

configure_socket_ifnames "${HEAD_NODE_IP}"
network_debug_snapshot "batch-before-ray"

echo "=== runtime knobs ==="
echo "MODEL_ID=${MODEL_ID} HOST=${HOST} PORT=${PORT} TP=${TP} PP=${PP} EP=${EP}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE} NUM_NODES=${NUM_NODES} CPUS_PER_TASK=${CPUS_PER_TASK}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE} NCCL_NET=${NCCL_NET} NCCL_IB_HCA=${NCCL_IB_HCA}"
echo "NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} VLLM_HOST_IP=${VLLM_HOST_IP}"
echo "NCCL_DEBUG=${NCCL_DEBUG} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
echo "SERVE_SCRIPT=${SERVE_SCRIPT}"

echo "=== benchmark knobs owned by host script ==="
echo "SP=${SP} SD=${SD}"
echo "NUM_PROMPTS=${NUM_PROMPTS}"
echo "REQUEST_RATE=${REQUEST_RATE}"
echo "BURSTINESS=${BURSTINESS}"
echo "SEED=${SEED}"
echo "ENDPOINT=${ENDPOINT}"
echo "AUTO_GENERATE_DATASET=${AUTO_GENERATE_DATASET}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "DATASET_SPLIT=${DATASET_SPLIT}"
echo "DATASET_MAX_PER_BUCKET=${DATASET_MAX_PER_BUCKET}"
echo "DATASET_MAX_SOURCE_ROWS=${DATASET_MAX_SOURCE_ROWS}"
echo "DATASET_PATH=${DATASET_PATH:-<derived by serving file>}"
echo "SAVE_RESULT=${SAVE_RESULT}"
echo "SAVE_DETAILED=${SAVE_DETAILED}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "RESULT_FILENAME=${RESULT_FILENAME}"

if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is set"
else
  echo "HF_TOKEN is not set"
fi

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

# -----------------------------------------------------------------------------
# Start Ray head
# -----------------------------------------------------------------------------

echo "=== Ray head (background srun) ==="
echo "Starting head node ${HEAD_NODE}..."

RAY_HEAD_CMD="$(
  declare -f interface_for_ip
  declare -f configure_socket_ifnames
  declare -f network_debug_snapshot
  declare -f discover_arc_node_tmp_parent
  declare -f ray_tmp_link_for_node
  declare -f ray_tmp_real_dir_for_node
  declare -f ensure_ray_tmp_for_node
  declare -f copy_ray_nsight_locally_once
  declare -f copy_ray_nsight_locally_final
  declare -f start_ray_nsight_live_copier
)
source \"${VENV_DIR}/bin/activate\"

export HEAD_NODE='${HEAD_NODE}'
export WORKER_NODES='${WORKER_NODES}'
export HEAD_NODE_IP='${HEAD_NODE_IP}'
export HOST='${HOST}'
export PORT='${PORT}'
export RAY_PORT='${RAY_PORT}'
export RAY_ADDRESS='${RAY_ADDRESS}'
export RAY_CLIENT_PORT='${RAY_CLIENT_PORT}'

export TRACE_RUN_DIR='${TRACE_RUN_DIR}'
export SLURM_JOB_ID='${SLURM_JOB_ID:-unknown}'
export NSYS_COPY_DEBUG='${NSYS_COPY_DEBUG}'
export DEBUG_SLURM_SCRIPT='${DEBUG_SLURM_SCRIPT}'
export WORKER_NSYS_LIVE_COPY_INTERVAL='${WORKER_NSYS_LIVE_COPY_INTERVAL}'
export MIN_WORKER_NSYS_REP_BYTES='${MIN_WORKER_NSYS_REP_BYTES}'

export RAY_USAGE_STATS_ENABLED='${RAY_USAGE_STATS_ENABLED}'
export RAY_DEDUP_LOGS='${RAY_DEDUP_LOGS}'
export VLLM_WORKER_MULTIPROC_METHOD='${VLLM_WORKER_MULTIPROC_METHOD}'
export VLLM_LOGGING_LEVEL='${VLLM_LOGGING_LEVEL}'
export VLLM_LOG_STATS_INTERVAL='${VLLM_LOG_STATS_INTERVAL}'
export VLLM_ENGINE_READY_TIMEOUT_S='${VLLM_ENGINE_READY_TIMEOUT_S}'
export TORCH_DISTRIBUTED_DEBUG='${TORCH_DISTRIBUTED_DEBUG}'

export NCCL_IB_DISABLE='${NCCL_IB_DISABLE}'
export NCCL_NET='${NCCL_NET}'
export NCCL_IB_HCA='${NCCL_IB_HCA}'
export NCCL_SOCKET_FAMILY='${NCCL_SOCKET_FAMILY}'
export NCCL_DEBUG='${NCCL_DEBUG}'
export NCCL_DEBUG_SUBSYS='${NCCL_DEBUG_SUBSYS}'
export NCCL_DEBUG_FILE='${NCCL_DEBUG_FILE}'

export NSYS_ENABLE='${NSYS_ENABLE}'
export NSYS_PROFILE_RAY='${NSYS_PROFILE_RAY}'
export NSYS_TRACE='${NSYS_TRACE}'
export NSYS_DELAY='${NSYS_DELAY}'
export NSYS_DIR='${NSYS_DIR}'

export RAY_TMP_LINK_PARENT='${RAY_TMP_LINK_PARENT}'
export RAY_TMP_PREFIX='${RAY_TMP_PREFIX}'
export RAY_PLASMA_DIRECTORY='${RAY_PLASMA_DIRECTORY}'
export RAY_OBJECT_STORE_MEMORY='${RAY_OBJECT_STORE_MEMORY}'

unset GLOO_SOCKET_IFNAME
unset NCCL_SOCKET_IFNAME
configure_socket_ifnames \"${HEAD_NODE_IP}\"
network_debug_snapshot \"ray-head-before-start\"

ensure_ray_tmp_for_node '${HEAD_NODE}'
start_ray_nsight_live_copier

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
      --temp-dir=\"\${RAY_TMPDIR}\" \\
      --plasma-directory=\"\${RAY_PLASMA_DIRECTORY}\" \\
      --object-store-memory=\"\${RAY_OBJECT_STORE_MEMORY}\" \\
      --num-gpus=${GPUS_PER_NODE} \\
      --num-cpus=${CPUS_PER_TASK}
else
  \"${RAY_BIN}\" start --block \\
    --head \\
    --node-ip-address=${HEAD_NODE_IP} \\
    --port=${RAY_PORT} \\
    --temp-dir=\"\${RAY_TMPDIR}\" \\
    --plasma-directory=\"\${RAY_PLASMA_DIRECTORY}\" \\
    --object-store-memory=\"\${RAY_OBJECT_STORE_MEMORY}\" \\
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
  --output="${TRACE_RUN_DIR}/slurm_ray_head_${HEAD_NODE}.out" \
  --error="${TRACE_RUN_DIR}/slurm_ray_head_${HEAD_NODE}.err" \
  bash -lc "${RAY_HEAD_CMD}" &

HEAD_RAY_PID=$!

echo "HEAD_RAY_PID=${HEAD_RAY_PID}"
echo "Sleeping ${RAY_HEAD_START_SLEEP_S}s to let Ray head start..."
sleep "${RAY_HEAD_START_SLEEP_S}"

# -----------------------------------------------------------------------------
# Start Ray workers
# -----------------------------------------------------------------------------

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
      declare -f configure_socket_ifnames
      declare -f network_debug_snapshot
      declare -f discover_arc_node_tmp_parent
      declare -f ray_tmp_link_for_node
      declare -f ray_tmp_real_dir_for_node
      declare -f ensure_ray_tmp_for_node
      declare -f copy_ray_nsight_locally_once
      declare -f copy_ray_nsight_locally_final
      declare -f start_ray_nsight_live_copier
)
source \"${VENV_DIR}/bin/activate\"

export HEAD_NODE='${HEAD_NODE}'
export WORKER_NODES='${WORKER_NODES}'
export HEAD_NODE_IP='${HEAD_NODE_IP}'
export HOST='${HOST}'
export PORT='${PORT}'
export RAY_PORT='${RAY_PORT}'
export RAY_ADDRESS='${RAY_ADDRESS}'
export RAY_CLIENT_PORT='${RAY_CLIENT_PORT}'

export TRACE_RUN_DIR='${TRACE_RUN_DIR}'
export SLURM_JOB_ID='${SLURM_JOB_ID:-unknown}'
export NSYS_COPY_DEBUG='${NSYS_COPY_DEBUG}'
export DEBUG_SLURM_SCRIPT='${DEBUG_SLURM_SCRIPT}'
export WORKER_NSYS_LIVE_COPY_INTERVAL='${WORKER_NSYS_LIVE_COPY_INTERVAL}'
export MIN_WORKER_NSYS_REP_BYTES='${MIN_WORKER_NSYS_REP_BYTES}'

export RAY_USAGE_STATS_ENABLED='${RAY_USAGE_STATS_ENABLED}'
export RAY_DEDUP_LOGS='${RAY_DEDUP_LOGS}'
export VLLM_WORKER_MULTIPROC_METHOD='${VLLM_WORKER_MULTIPROC_METHOD}'
export VLLM_LOGGING_LEVEL='${VLLM_LOGGING_LEVEL}'
export VLLM_LOG_STATS_INTERVAL='${VLLM_LOG_STATS_INTERVAL}'
export VLLM_ENGINE_READY_TIMEOUT_S='${VLLM_ENGINE_READY_TIMEOUT_S}'
export TORCH_DISTRIBUTED_DEBUG='${TORCH_DISTRIBUTED_DEBUG}'

export NCCL_IB_DISABLE='${NCCL_IB_DISABLE}'
export NCCL_NET='${NCCL_NET}'
export NCCL_IB_HCA='${NCCL_IB_HCA}'
export NCCL_SOCKET_FAMILY='${NCCL_SOCKET_FAMILY}'
export NCCL_DEBUG='${NCCL_DEBUG}'
export NCCL_DEBUG_SUBSYS='${NCCL_DEBUG_SUBSYS}'
export NCCL_DEBUG_FILE='${NCCL_DEBUG_FILE}'

export NSYS_ENABLE='${NSYS_ENABLE}'
export NSYS_PROFILE_RAY='${NSYS_PROFILE_RAY}'
export NSYS_TRACE='${NSYS_TRACE}'
export NSYS_DELAY='${NSYS_DELAY}'
export NSYS_DIR='${NSYS_DIR}'

export RAY_TMP_LINK_PARENT='${RAY_TMP_LINK_PARENT}'
export RAY_TMP_PREFIX='${RAY_TMP_PREFIX}'
export RAY_PLASMA_DIRECTORY='${RAY_PLASMA_DIRECTORY}'
export RAY_OBJECT_STORE_MEMORY='${RAY_OBJECT_STORE_MEMORY}'

unset GLOO_SOCKET_IFNAME
unset NCCL_SOCKET_IFNAME
configure_socket_ifnames \"${WORKER_IP}\"
network_debug_snapshot \"ray-worker-before-start\"

ensure_ray_tmp_for_node '${WORKER}'
start_ray_nsight_live_copier

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
      --temp-dir=\"\${RAY_TMPDIR}\" \\
      --plasma-directory=\"\${RAY_PLASMA_DIRECTORY}\" \\
      --object-store-memory=\"\${RAY_OBJECT_STORE_MEMORY}\" \\
      --num-gpus=${GPUS_PER_NODE} \\
      --num-cpus=${CPUS_PER_TASK}
else
  \"${RAY_BIN}\" start --block \\
    --address=${HEAD_NODE_IP}:${RAY_PORT} \\
    --node-ip-address=${WORKER_IP} \\
    --temp-dir=\"\${RAY_TMPDIR}\" \\
    --plasma-directory=\"\${RAY_PLASMA_DIRECTORY}\" \\
    --object-store-memory=\"\${RAY_OBJECT_STORE_MEMORY}\" \\
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
      --output="${TRACE_RUN_DIR}/slurm_ray_worker_${WORKER}.out" \
      --error="${TRACE_RUN_DIR}/slurm_ray_worker_${WORKER}.err" \
      bash -lc "${RAY_WORKER_CMD}" &

    WORKER_RAY_PIDS="${WORKER_RAY_PIDS} $!"
    echo "Worker Ray step pid: $! (WORKER_RAY_PIDS=${WORKER_RAY_PIDS})"
  done

  echo "Sleeping ${RAY_WORKER_START_SLEEP_S}s to let Ray workers start..."
  sleep "${RAY_WORKER_START_SLEEP_S}"
fi

# -----------------------------------------------------------------------------
# Ray TCP preflight + hard readiness gate
# -----------------------------------------------------------------------------

echo "=== Ray TCP preflight ==="
echo "Waiting for Ray head/GCS TCP port."

export HEAD_NODE_IP RAY_PORT RAY_CLIENT_PORT RAY_TCP_PREFLIGHT_TIMEOUT_S

python -u <<'PY'
import os
import sys
import time
import socket

head_ip = os.environ["HEAD_NODE_IP"]
ray_port = int(os.environ.get("RAY_PORT", "6378"))
ray_client_port = int(os.environ.get("RAY_CLIENT_PORT", "10001"))
timeout_s = int(os.environ.get("RAY_TCP_PREFLIGHT_TIMEOUT_S", "360"))

def wait_tcp(host, port, label, timeout_s):
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=5):
                print(f"{label} is reachable: {host}:{port}")
                return True
        except Exception as e:
            last_error = repr(e)
            print(f"{label} not ready yet at {host}:{port}: {last_error}")
            time.sleep(5)
    print(f"WARNING: {label} never became reachable at {host}:{port}; last_error={last_error}")
    return False

gcs_ok = wait_tcp(head_ip, ray_port, "Ray head/GCS port", timeout_s)
client_ok = wait_tcp(head_ip, ray_client_port, "Ray Client port", 30)

if not gcs_ok:
    print("ERROR: Ray head/GCS port is not reachable. vLLM would not be able to connect.", file=sys.stderr)
    sys.exit(1)

if not client_ok:
    print("WARNING: Ray Client port is not reachable, but continuing because vLLM uses RAY_ADDRESS.")

print("Ray TCP preflight passed.")
PY

EXPECTED_RAY_NODES="${NUM_NODES}"
EXPECTED_RAY_GPUS="$((GPUS_PER_NODE * NUM_NODES))"

wait_for_ray_cluster_ready \
  "${EXPECTED_RAY_NODES}" \
  "${EXPECTED_RAY_GPUS}" \
  "${RAY_CLUSTER_READY_TIMEOUT_S}" \
  "${RAY_CLUSTER_READY_POLL_S}"

echo "=== Ray process snapshot before vLLM ==="
ps -ef | egrep 'ray start|gcs_server|raylet|ray::|dashboard|client.server' | grep -v grep || true

# -----------------------------------------------------------------------------
# Start vLLM
# -----------------------------------------------------------------------------

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

echo "API_SERVER_OUTER_NSYS_WRAPPER=0"
if [ "${NSYS_ENABLE}" = "1" ] && [ "${NSYS_PROFILE_WORKERS}" = "1" ]; then
  nsight_copy_msg "worker Nsight enabled via --ray-workers-use-nsight"
  nsight_copy_msg "live sidecar copies from Ray per-node temp dirs"
  nsight_copy_msg "empty.nsys-rep and zero-byte reports are ignored"
fi

export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"
export VLLM_HOST_IP="${HEAD_NODE_IP}"

export RAY_TMPDIR
RAY_TMPDIR="$(ray_tmp_link_for_node "${HEAD_NODE}")"
export RAY_SPILL_DIR="${RAY_TMPDIR}/spill"
export TMPDIR="${RAY_TMPDIR}/py_tmp"
mkdir -p "${RAY_SPILL_DIR}" "${TMPDIR}" 2>/dev/null || true

echo "Launching vLLM with RAY_ADDRESS=${RAY_ADDRESS}"
echo "Launching vLLM with VLLM_HOST_IP=${VLLM_HOST_IP}"
echo "Launching vLLM with VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S}"
echo "Launching vLLM with GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"
echo "Launching vLLM with NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "API_SERVER_RAY_TMPDIR=${RAY_TMPDIR} -> $(readlink -f "${RAY_TMPDIR}" 2>/dev/null || echo MISSING)"
echo "API_SERVER_TMPDIR=${TMPDIR}"
echo "VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ID}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --distributed-executor-backend ray \
  --tensor-parallel-size "${TP}" \
  --pipeline-parallel-size "${PP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --enforce-eager \
  "${VLLM_TRACE_FLAGS[@]}" \
  --disable-custom-all-reduce &

SERVER_STEP_PID=$!

echo "Started vLLM server process (pid=${SERVER_STEP_PID}). Waiting for /health ..."

_health_wait_n=0
_health_deadline=$((SECONDS + HEALTH_TIMEOUT_S))

until curl -fsS "http://${HEAD_NODE_IP}:${PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
    echo "Server process exited before becoming ready." >&2
    wait "${SERVER_STEP_PID}" || true
    exit 1
  fi

  if [ "${SECONDS}" -ge "${_health_deadline}" ]; then
    echo "ERROR: timed out waiting for http://${HEAD_NODE_IP}:${PORT}/health after ${HEALTH_TIMEOUT_S}s" >&2
    echo "Recent Ray/vLLM processes:" >&2
    ps -ef | egrep 'vllm|api_server|EngineCore|ray::|worker_process|raylet' | grep -v grep >&2 || true
    echo "Recent Ray logs from per-node tmp:" >&2
    find /tmp/vray-${SLURM_JOB_ID}-*/session_*/logs -maxdepth 1 -type f \
      \( -name '*.err' -o -name '*.out' -o -name '*.log' \) \
      -print 2>/dev/null | head -100 >&2 || true
    exit 1
  fi

  _health_wait_n=$((_health_wait_n + 1))

  if [ "${DEBUG_SLURM_SCRIPT}" = "1" ] || [ "$((_health_wait_n % 12))" -eq 0 ]; then
    echo "Still waiting for http://${HEAD_NODE_IP}:${PORT}/health (attempt ${_health_wait_n}) ..."
    ps -ef | egrep 'vllm|api_server|EngineCore|ray::|worker_process' | grep -v grep || true
  fi

  sleep 5
done

unset _health_wait_n
unset _health_deadline

# -----------------------------------------------------------------------------
# Run benchmark
# -----------------------------------------------------------------------------

echo "Server is healthy. Running ${SERVE_SCRIPT} ..."
echo "SP=${SP} SD=${SD} NUM_PROMPTS=${NUM_PROMPTS} REQUEST_RATE=${REQUEST_RATE}"

set +e
HOST="${HEAD_NODE_IP}" \
PORT="${PORT}" \
MODEL_ID="${MODEL_ID}" \
SP="${SP}" \
SD="${SD}" \
NUM_PROMPTS="${NUM_PROMPTS}" \
REQUEST_RATE="${REQUEST_RATE}" \
BURSTINESS="${BURSTINESS}" \
SEED="${SEED}" \
ENDPOINT="${ENDPOINT}" \
AUTO_GENERATE_DATASET="${AUTO_GENERATE_DATASET}" \
DATASET_NAME="${DATASET_NAME}" \
DATASET_SPLIT="${DATASET_SPLIT}" \
DATASET_MAX_PER_BUCKET="${DATASET_MAX_PER_BUCKET}" \
DATASET_MAX_SOURCE_ROWS="${DATASET_MAX_SOURCE_ROWS}" \
DATASET_PATH="${DATASET_PATH}" \
SAVE_RESULT="${SAVE_RESULT}" \
SAVE_DETAILED="${SAVE_DETAILED}" \
RESULT_ROOT="${RESULT_ROOT}" \
RESULT_DIR="${RESULT_DIR}" \
RESULT_FILENAME="${RESULT_FILENAME}" \
HEAD_NODE_IP="${HEAD_NODE_IP}" \
GPUS_PER_NODE="${GPUS_PER_NODE}" \
CPUS_PER_TASK="${CPUS_PER_TASK}" \
RAY_PORT="${RAY_PORT}" \
bash "${SERVE_SCRIPT}" "${SLURM_JOB_ID}" "${HEAD_NODE}"
BENCH_STATUS=$?
set -e

echo "Benchmark exit status: ${BENCH_STATUS}"
if [ "${BENCH_STATUS}" != "0" ]; then
  echo "ERROR: benchmark failed." >&2
  exit "${BENCH_STATUS}"
fi

# -----------------------------------------------------------------------------
# Clean shutdown and trace collection
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

echo "Copying any live Nsight files before Ray shutdown..."
copy_ray_nsight_from_node "${HEAD_NODE}"
for WORKER in ${WORKER_NODES}; do
  copy_ray_nsight_from_node "${WORKER}"
done

echo "Stopping Ray cleanly so Nsight worker processes can exit and export reports..."

for WORKER in ${WORKER_NODES}; do
  stop_ray_cleanly_on_node "${WORKER}"
done
stop_ray_cleanly_on_node "${HEAD_NODE}"

echo "Copying Nsight files after clean Ray stop..."
copy_ray_nsight_from_node "${HEAD_NODE}"
for WORKER in ${WORKER_NODES}; do
  copy_ray_nsight_from_node "${WORKER}"
done

echo "Waiting for finalized worker Nsight reports after Ray stop..."
wait_for_real_worker_nsys_reports "${WORKER_NSYS_FINALIZE_WAIT_S}" "${WORKER_NSYS_FINALIZE_POLL_S}" || true

echo "Final Nsight copy after finalize wait..."
copy_ray_nsight_from_node "${HEAD_NODE}"
for WORKER in ${WORKER_NODES}; do
  copy_ray_nsight_from_node "${WORKER}"
done

echo "Terminating background Ray srun wrappers if still alive..."

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

nsight_copy_msg "=== final Nsight artifact summary ==="
nsight_copy_msg "API server trace dir, expected empty/no api-server nsys in this variant: ${NSYS_DIR}"
ls -la "${NSYS_DIR}/" 2>/dev/null || true

for NODE in ${HEAD_NODE} ${WORKER_NODES}; do
  nsight_summarize_dest "${NODE}"
done

echo "NCCL logs:"
find "${TRACE_RUN_DIR}/nccl_logs" -type f -printf "%p %s bytes\n" 2>/dev/null | sort || true

echo "Ray srun logs:"
find "${TRACE_RUN_DIR}" -maxdepth 1 -type f \
  \( -name 'slurm_ray_head_*.out' -o -name 'slurm_ray_head_*.err' -o -name 'slurm_ray_worker_*.out' -o -name 'slurm_ray_worker_*.err' \) \
  -printf "%p %s bytes\n" 2>/dev/null | sort || true

echo "Trace files:"
find "${TRACE_RUN_DIR}" -maxdepth 5 -type f -printf "%p %s bytes\n" 2>/dev/null | sort || true

echo "Done."