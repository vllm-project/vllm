#!/usr/bin/env bash
# Client side: wait for the server, audit it, sweep concurrency, write results.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INCO_DIR="$(cd "${HERE}/.." && pwd)"
# shellcheck source=/dev/null
source "${HERE}/workload.env"

cd "${INCO_DIR}"
exec python -m bench.sweep \
  --url "${INCO_URL}" \
  --label "${INCO_LABEL}" \
  --model "${INCO_MODEL}" \
  --isl "${INCO_ISL}" \
  --osl "${INCO_OSL}" \
  --num-gpus "${INCO_NUM_GPUS}" \
  --random-seed "${INCO_RANDOM_SEED}" \
  --concurrency ${INCO_CONCURRENCIES//,/ } \
  --requests-per-concurrency "${INCO_REQUESTS_PER_CONCURRENCY}" \
  --min-requests "${INCO_MIN_REQUESTS}" \
  --max-requests "${INCO_MAX_REQUESTS}" \
  --warmup-requests "${INCO_WARMUP_REQUESTS}" \
  "$@"
