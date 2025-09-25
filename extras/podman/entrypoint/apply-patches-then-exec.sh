#!/usr/bin/env bash#!/usr/bin/env bash

set -euo pipefailset -euo pipefail

# Normalize CRLF in repo scripts to avoid bash\r issues

find /workspace/extras -type f -name '*.sh' -print0 2>/dev/null | while IFS= read -r -d '' f; do# Apply repo patches if available; best-effort, normalization handled inside helper.

  if grep -q $'\r' "$f" 2>/dev/null; then tr -d '\r' < "$f" > "$f.tmp" && mv "$f.tmp" "$f" || true; fiif command -v apply-vllm-patches >/dev/null 2>&1; then

done  echo "[entrypoint] applying patches..."

chmod +x /workspace/extras/patches/apply_patches.sh 2>/dev/null || true  apply-vllm-patches || true

bash /workspace/extras/patches/apply_patches.sh || truefi

exec "$@"

# If first args are `bash -lc <path-to-script.sh>` (single token, no spaces), normalize CRLF then exec
if [[ "${1-}" == "bash" && "${2-}" == "-lc" ]]; then
  arg3="${3-}"
  # Only handle when it's a single token path ending in .sh with no spaces or shell operators
  if [[ -n "$arg3" && "$arg3" != *' '* && "$arg3" != *';'* && "$arg3" != *'&'* && "$arg3" != *'|'* && "$arg3" == *.sh ]]; then
    # Resolve to filesystem path if it exists
    if [[ -f "$arg3" ]]; then
      SRC_SCRIPT="$arg3"
      TMP_SCRIPT="$(mktemp /tmp/entry-XXXX.sh)"
      tr -d '\r' < "$SRC_SCRIPT" > "$TMP_SCRIPT" 2>/dev/null || cp "$SRC_SCRIPT" "$TMP_SCRIPT"
      chmod +x "$TMP_SCRIPT" 2>/dev/null || true
      exec bash -lc "$TMP_SCRIPT"
    fi
  fi
fi

# Preserve any PYTHONPATH/sitecustomize shim passed from caller
if [[ -n "${PYTHONPATH:-}" ]]; then export PYTHONPATH; fi
exec "$@"
