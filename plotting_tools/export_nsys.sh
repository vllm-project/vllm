#!/usr/bin/env bash
# Export .nsys-rep (and optional .qdstrm) to JSONL for plotting_tools/analyze_job.py
#
# Usage:
#   bash plotting_tools/export_nsys.sh results/7702489
#   bash plotting_tools/export_nsys.sh results/7702489/ray_worker_nsight/htc-g059/*.nsys-rep
#
# On arc, load the same CUDA/Nsight module you use for profiling, e.g.:
#   module load CUDA/12.9.0
#   which nsys

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: export_nsys.sh <job-dir|rep-files...>

Writes one .jsonl next to each .nsys-rep (same basename).

Environment:
  NSYS_BIN          path to nsys (default: nsys on PATH)
  NSYS_EXPORT_TYPE  jsonlines | json  (auto-detect if unset)
  NSYS_FORCE        1 = --force-overwrite (default 1)

Examples:
  bash plotting_tools/export_nsys.sh results/7702489
  NSYS_BIN=/apps/.../nsys bash plotting_tools/export_nsys.sh worker.nsys-rep
EOF
}

if [ "$#" -lt 1 ]; then
  usage
  exit 1
fi

NSYS_BIN="${NSYS_BIN:-nsys}"
NSYS_FORCE="${NSYS_FORCE:-1}"

if ! command -v "${NSYS_BIN}" >/dev/null 2>&1; then
  echo "ERROR: nsys not found (${NSYS_BIN}). Load your CUDA/Nsight module on arc first." >&2
  echo "  module load CUDA/12.9.0   # or your site module" >&2
  echo "  which nsys" >&2
  exit 1
fi

detect_export_type() {
  if [ -n "${NSYS_EXPORT_TYPE:-}" ]; then
    echo "${NSYS_EXPORT_TYPE}"
    return
  fi
  if "${NSYS_BIN}" export --help 2>&1 | grep -q jsonlines; then
    echo "jsonlines"
  else
    echo "json"
  fi
}

EXPORT_TYPE="$(detect_export_type)"
echo "Using nsys: ${NSYS_BIN}"
echo "Export type: ${EXPORT_TYPE}"

collect_reps() {
  if [ "$#" -eq 1 ] && [ -d "$1" ]; then
    find "$1" -type f -name '*.nsys-rep' ! -name 'empty.nsys-rep' ! -name '*_empty.nsys-rep' -size +0
  else
    printf '%s\n' "$@"
  fi
}

FORCE_ARGS=()
if [ "${NSYS_FORCE}" = "1" ]; then
  FORCE_ARGS=(--force-overwrite=true)
fi

mapfile -t REPS < <(collect_reps "$@")
if [ "${#REPS[@]}" -eq 0 ]; then
  echo "No .nsys-rep files found." >&2
  exit 1
fi

for rep in "${REPS[@]}"; do
  out="${rep%.nsys-rep}.jsonl"
  echo "Exporting ${rep} -> ${out}"
  if [ "${EXPORT_TYPE}" = "jsonlines" ]; then
    "${NSYS_BIN}" export --type=jsonlines --output="${out}" "${FORCE_ARGS[@]}" "${rep}"
  else
    # Nsight 2024.x: legacy json export (one JSON object per line in practice).
    "${NSYS_BIN}" export --type=json --output="${out}" "${FORCE_ARGS[@]}" "${rep}"
  fi
done

echo "Done. ${#REPS[@]} file(s). Run analyze_job on the job dir or .jsonl paths."
