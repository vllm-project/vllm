#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] start"

# Normalize CRLF in all repo shell scripts (best effort)
find /workspace/extras -type f -name '*.sh' -print0 2>/dev/null | while IFS= read -r -d '' f; do
  if grep -q $'\r' "$f" 2>/dev/null; then
    tmp="$f.tmp.$$"; tr -d '\r' < "$f" > "$tmp" 2>/dev/null || cp "$f" "$tmp"; mv "$tmp" "$f" || true
  fi
done || true

# Apply patches (idempotent)
if [[ -x /workspace/extras/patches/apply_patches.sh ]]; then
  bash /workspace/extras/patches/apply_patches.sh || true
fi

echo "[entrypoint] exec: $*"
exec "$@"
