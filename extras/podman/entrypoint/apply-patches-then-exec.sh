#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] start"

# Normalize CRLF in podman helper scripts (best effort)
for dir in /workspace/extras/podman /workspace/extras/patches; do
  [[ -d "$dir" ]] || continue
  find "$dir" -type f -name '*.sh' -print0 2>/dev/null | while IFS= read -r -d '' f; do
    if grep -q $'\r' "$f" 2>/dev/null; then
      tmp="$f.tmp.$$"
      tr -d '\r' < "$f" > "$tmp" 2>/dev/null || cp "$f" "$tmp"
      mv "$tmp" "$f" || true
    fi
  done
done || true

if command -v git >/dev/null 2>&1; then
  git config --global --add safe.directory /workspace >/dev/null 2>&1 || true
fi

export PYTHON_PATCH_OVERLAY=${PYTHON_PATCH_OVERLAY:-1}

OVERLAY_HELPER=/workspace/extras/patches/apply_patches_overlay.sh
LEGACY_HELPER=/workspace/extras/patches/apply_patches.sh

if [[ -f "$OVERLAY_HELPER" ]]; then
  echo "[entrypoint] applying patches via overlay helper (overlay=$PYTHON_PATCH_OVERLAY)"
  bash "$OVERLAY_HELPER" || true
elif command -v apply-vllm-patches >/dev/null 2>&1; then
  echo "[entrypoint] applying patches via helper (overlay=$PYTHON_PATCH_OVERLAY)"
  apply-vllm-patches || true
elif [[ -f "$LEGACY_HELPER" ]]; then
  echo "[entrypoint] applying patches via workspace script"
  bash "$LEGACY_HELPER" || true
else
  echo "[entrypoint] no patch helper found" >&2
fi

if command -v git >/dev/null 2>&1; then
  dirty=$(git status --porcelain --untracked-files=no)
  if [[ -n "$dirty" ]]; then
    warn_limit=${PATCH_OVERLAY_WARN_LIMIT:-20}
    if [[ ! "$warn_limit" =~ ^[0-9]+$ ]]; then
      warn_limit=20
    fi
    dirty_count=$(printf '%s\n' "$dirty" | sed '/^$/d' | wc -l | tr -d ' ')
    echo "[entrypoint] WARNING: tracked files modified during patch application (${dirty_count})" >&2
    if (( warn_limit > 0 )); then
      printf '%s\n' "$dirty" | head -n "$warn_limit" >&2
      if (( dirty_count > warn_limit )); then
        echo "[entrypoint] ... suppressed $((dirty_count - warn_limit)) additional entries (set PATCH_OVERLAY_WARN_LIMIT to adjust)" >&2
      fi
    else
      echo "[entrypoint] diff output suppressed (PATCH_OVERLAY_WARN_LIMIT=$warn_limit)" >&2
    fi
  fi
fi

echo "[entrypoint] exec: $*"
if [[ -n "${PYTHONPATH:-}" ]]; then export PYTHONPATH; fi
exec "$@"
