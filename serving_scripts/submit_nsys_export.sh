#!/usr/bin/env bash
# Submit nsys export as a Slurm job (CPU-only, no GPU needed).
#
# Usage:
#   1. Set JOB_DIR below to your job results directory
#   2. sbatch serving_scripts/submit_nsys_export.sh
#
# The script automatically finds ALL .nsys-rep files across all nodes
# in the job directory and exports each to .jsonl.
#
# You can also override JOB_DIR from the command line:
#   JOB_DIR=results/7717190 sbatch serving_scripts/submit_nsys_export.sh

#SBATCH --job-name=nsys-export
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:0
#SBATCH --output=results/nsys-export-%j.out
#SBATCH --error=results/nsys-export-%j.err
#SBATCH --mail-user=jason.miller@eng.ox.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --account=engs-glass

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# EDIT THIS: path to the job results directory (relative to repo root)
# ──────────────────────────────────────────────────────────────────────────────
JOB_DIR="${JOB_DIR:-results/7717190}"

# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

echo "=== nsys export job ==="
echo "Slurm Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo "Started:      $(date -Iseconds)"
echo "JOB_DIR:      ${JOB_DIR}"
echo ""

if [ ! -d "${JOB_DIR}" ]; then
    echo "ERROR: JOB_DIR does not exist: ${JOB_DIR}" >&2
    exit 1
fi

module load CUDA/12.9.0 2>/dev/null || module load CUDA 2>/dev/null || true

if ! command -v nsys >/dev/null 2>&1; then
    echo "ERROR: nsys not found after module load. Check available CUDA modules:" >&2
    module avail CUDA 2>&1 | head -20 >&2
    exit 1
fi

echo "nsys version: $(nsys --version 2>&1 | head -1)"

# Find all .nsys-rep files (skips empty/placeholder files)
mapfile -t REPS < <(find "${JOB_DIR}" -type f -name '*.nsys-rep' \
    ! -name 'empty.nsys-rep' ! -name '*_empty.nsys-rep' -size +0 | sort)

echo "Found ${#REPS[@]} .nsys-rep file(s):"
for f in "${REPS[@]}"; do
    echo "  $(du -h "$f" | cut -f1)  ${f}"
done
echo ""

if [ "${#REPS[@]}" -eq 0 ]; then
    echo "No .nsys-rep files found in ${JOB_DIR}" >&2
    exit 1
fi

# Detect export type
if nsys export --help 2>&1 | grep -q jsonlines; then
    EXPORT_TYPE="jsonlines"
else
    EXPORT_TYPE="json"
fi
echo "Export type: ${EXPORT_TYPE}"
echo ""

FAILED=0
for rep in "${REPS[@]}"; do
    out="${rep%.nsys-rep}.jsonl"
    echo "── Exporting: ${rep}"
    echo "   Output:    ${out}"
    if nsys export --type="${EXPORT_TYPE}" --output="${out}" --force-overwrite=true "${rep}"; then
        echo "   OK ($(du -h "$out" | cut -f1))"
    else
        echo "   FAILED (exit $?)" >&2
        ((FAILED++))
    fi
    echo ""
done

echo "=== Done: $(date -Iseconds) ==="
echo "Exported: $((${#REPS[@]} - FAILED))/${#REPS[@]} succeeded"
if [ "${FAILED}" -gt 0 ]; then
    echo "WARNING: ${FAILED} file(s) failed to export" >&2
    exit 1
fi
