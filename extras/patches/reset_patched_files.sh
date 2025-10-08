#!/usr/bin/env bash
set -euo pipefail

TRACK_FILE_DEFAULT="/opt/work/tmp/vllm_patched_files.txt"
PATCH_TRACK_FILE=${PATCH_TRACK_FILE:-$TRACK_FILE_DEFAULT}

if [[ ! -f "$PATCH_TRACK_FILE" ]]; then
  exit 0
fi

ROOT_DIR=${ROOT_DIR:-$(pwd)}
cd "$ROOT_DIR"

if ! command -v git >/dev/null 2>&1; then
  echo "[patches-reset] git not available; skipping revert" >&2
  exit 0
fi

mapfile -t TARGETS < <(grep -vE '^\s*$' "$PATCH_TRACK_FILE" | sort -u) || true
if [[ ${#TARGETS[@]} -eq 0 ]]; then
  rm -f "$PATCH_TRACK_FILE"
  exit 0
fi

echo "[patches-reset] Reverting ${#TARGETS[@]} file(s)"
if ! git checkout -- "${TARGETS[@]}" >/dev/null 2>&1; then
  echo "[patches-reset] git checkout failed" >&2
  exit 1
fi

git status --porcelain -- "${TARGETS[@]}" >/dev/null 2>&1 || true
rm -f "$PATCH_TRACK_FILE"
