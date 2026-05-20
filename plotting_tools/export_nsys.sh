#!/usr/bin/env bash
# Export .nsys-rep files to Chrome-trace JSON for plotting_tools/analyze_job.py
#
# Run on ARC (login or post-job) where `nsys` is available:
#   bash plotting_tools/export_nsys.sh results/7651157
#
# Outputs land in <job-dir>/exported/

set -euo pipefail

JOB_DIR="${1:?Usage: export_nsys.sh <job-results-dir>}"
JOB_DIR="$(cd "$JOB_DIR" && pwd)"
OUT="${JOB_DIR}/exported"
mkdir -p "$OUT"

if ! command -v nsys >/dev/null 2>&1; then
  echo "ERROR: nsys not in PATH. Load the CUDA/Nsight module on ARC first." >&2
  exit 1
fi

export_one() {
  local rep="$1"
  local base
  base="$(basename "$rep" .nsys-rep)"
  local parent
  parent="$(basename "$(dirname "$rep")")"
  local stem="${parent}_${base}"

  if [[ ! -s "$rep" ]]; then
    echo "SKIP (empty): $rep"
    return 0
  fi

  echo "=== Exporting $rep ==="
  nsys export --type=json --force-export=true \
    -o "${OUT}/${stem}.json" "$rep"

  echo "=== NCCL stats CSV for $rep ==="
  nsys stats --report ncclsum --format csv \
    -o "${OUT}/${stem}_ncclsum.csv" "$rep" 2>/dev/null || true

  echo "=== CUDA API sum for $rep ==="
  nsys stats --report cudaapisum --format csv \
    -o "${OUT}/${stem}_cudaapisum.csv" "$rep" 2>/dev/null || true
}

while IFS= read -r -d '' rep; do
  export_one "$rep"
done < <(find "$JOB_DIR" -name '*.nsys-rep' -print0)

echo "Done. Chrome JSON under: $OUT"
echo "Next: python plotting_tools/analyze_job.py --job-dir $JOB_DIR"
