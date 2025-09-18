#!/usr/bin/env bash
set -euo pipefail

if [ "$USE_SCCACHE" = "1" ]; then
  echo "Installing sccache..."
  curl -L -o sccache.tar.gz "${SCCACHE_DOWNLOAD_URL}"
  tar -xzf sccache.tar.gz
  sudo mv sccache-*/sccache /usr/bin/sccache
  rm -rf sccache.tar.gz sccache-*

  if [ -n "${SCCACHE_ENDPOINT:-}" ]; then export SCCACHE_ENDPOINT; fi
  export SCCACHE_BUCKET="${SCCACHE_BUCKET_NAME}"
  export SCCACHE_REGION="${SCCACHE_REGION_NAME}"
  export SCCACHE_S3_NO_CREDENTIALS="${SCCACHE_S3_NO_CREDENTIALS}"
  export SCCACHE_IDLE_TIMEOUT=0
fi

# Always print stats if the binary exists
if command -v sccache &>/dev/null; then
  sccache --show-stats || true
fi
