#!/usr/bin/env bash
# Usage: source pick-free-gpu-amd.sh <MIN_FREE_MEM_MB> <GPU_COUNT>
# Select AMD GPUs (ROCm) by free VRAM (computed from Total-Used) and low util.

MIN_FREE_MEM="${1:-10000}"      # MiB (default 10 GiB)
REQUESTED_GPU_COUNT="${2:-1}"   # number of GPUs to select
MAX_UTIL="${MAX_UTIL:-20}"      # (%)
GPU_LIMIT="${GPU_LIMIT:-8}"     # only consider GPU index < GPU_LIMIT (default 8 for your node)
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"
INTERVAL="${INTERVAL:-10}"

start_time=$(date +%s)

if ! command -v rocm-smi >/dev/null 2>&1; then
  echo "❌ rocm-smi not found. Are you on an AMD/ROCm machine?"
  return 1 2>/dev/null || exit 1
fi

# Return candidates as lines: "free_mib,util,idx"
get_candidates() {
  rocm-smi --showmeminfo vram --showuse 2>/dev/null | awk \
    -v min_mem="$MIN_FREE_MEM" \
    -v max_util="$MAX_UTIL" \
    -v gpu_limit="$GPU_LIMIT" '
    function mib_from_bytes(b) { return int(b / (1024*1024)) }

    # Util lines: "GPU[3] : GPU use (%): 0"
    /GPU\[[0-9]+\][[:space:]]*:[[:space:]]*GPU use[[:space:]]*\(%\):/ {
      if (match($0, /GPU\[([0-9]+)\]/, m)) idx=m[1]; else next
      if (idx+0 >= gpu_limit) next
      if (match($0, /GPU use[[:space:]]*\(%\):[[:space:]]*([0-9]+)/, u)) util=u[1]; else next
      gpu_util[idx]=util+0
      next
    }

    # Total bytes: "GPU[3] : VRAM Total Memory (B): 206141652992"
    /GPU\[[0-9]+\][[:space:]]*:[[:space:]]*VRAM Total Memory \(B\):/ {
      if (match($0, /GPU\[([0-9]+)\]/, m)) idx=m[1]; else next
      if (idx+0 >= gpu_limit) next
      if (match($0, /VRAM Total Memory \(B\):[[:space:]]*([0-9]+)/, t)) total=t[1]; else next
      vram_total[idx]=total+0
      next
    }

    # Used bytes: "GPU[3] : VRAM Total Used Memory (B): 303304704"
    /GPU\[[0-9]+\][[:space:]]*:[[:space:]]*VRAM Total Used Memory \(B\):/ {
      if (match($0, /GPU\[([0-9]+)\]/, m)) idx=m[1]; else next
      if (idx+0 >= gpu_limit) next
      if (match($0, /VRAM Total Used Memory \(B\):[[:space:]]*([0-9]+)/, t)) used=t[1]; else next
      vram_used[idx]=used+0
      next
    }

    END {
      for (i=0; i<gpu_limit; i++) {
        if ((i in vram_total) && (i in vram_used) && (i in gpu_util)) {
          free_bytes = vram_total[i] - vram_used[i]
          free_mib = mib_from_bytes(free_bytes)
          util = gpu_util[i] + 0
          if (free_mib >= min_mem && util <= max_util) {
            print free_mib "," util "," i
          }
        }
      }
    }'
}

while true; do
  now=$(date +%s)
  elapsed=$((now - start_time))

  if (( elapsed >= TIMEOUT_SECONDS )); then
    echo "❌ Timeout: No suitable GPU found within ${TIMEOUT_SECONDS}s (min_free=${MIN_FREE_MEM}MiB, max_util=${MAX_UTIL}%, gpu_limit=${GPU_LIMIT})"
    return 1 2>/dev/null || exit 1
  fi

  mapfile -t candidates < <(get_candidates)

  if [ "${#candidates[@]}" -gt 0 ]; then
    # Sort by free_mib desc, util asc, idx asc
    mapfile -t top_gpus < <(
      printf "%s\n" "${candidates[@]}" \
        | sort -t',' -k1,1nr -k2,2n -k3,3n \
        | head -n"${REQUESTED_GPU_COUNT}" \
        | awk -F',' '{print $3}'
    )

    if [ "${#top_gpus[@]}" -eq 1 ]; then
      export HIP_VISIBLE_DEVICES="${top_gpus[0]}"
      export CUDA_VISIBLE_DEVICES="${top_gpus[0]}"  # compatibility for some frameworks/scripts
      echo "✅ Selected GPU #${top_gpus[0]} (HIP_VISIBLE_DEVICES=${top_gpus[0]})"
    else
      chosen_gpus=$(IFS=','; echo "${top_gpus[*]}")
      export HIP_VISIBLE_DEVICES="${chosen_gpus}"
      export CUDA_VISIBLE_DEVICES="${chosen_gpus}"
      gpu_list=$(printf "#%s," "${top_gpus[@]}")
      echo "✅ Selected GPUs ${gpu_list%,} (HIP_VISIBLE_DEVICES=${chosen_gpus})"
    fi
    break
  fi

  # Not found yet; print one-line status so it doesn't look "hung"
  echo "⏳ No suitable GPU yet (elapsed=${elapsed}s). Waiting... (min_free=${MIN_FREE_MEM}MiB, max_util=${MAX_UTIL}%, gpu_limit=${GPU_LIMIT})"
  sleep "${INTERVAL}"
done
