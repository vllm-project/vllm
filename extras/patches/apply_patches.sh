#!/usr/bin/env bash
# overlay test
set -euo pipefail

# Normalize CRLF and re-exec if needed
if grep -q $'\r' "$0" 2>/dev/null; then
  TMP_SELF=$(mktemp /tmp/apply_patches_self.XXXXXX.sh)
  tr -d '\r' < "$0" > "$TMP_SELF" || cp "$0" "$TMP_SELF"
  chmod +x "$TMP_SELF" 2>/dev/null || true
  exec "$TMP_SELF" "$@"
fi

ROOT_DIR=${ROOT_DIR:-$(pwd)}
PATCH_DIR="$ROOT_DIR/extras/patches"
cd "$ROOT_DIR"

shopt -s nullglob
PATCHES=($PATCH_DIR/*.diff)
shopt -u nullglob

OVERLAY_MODE=${PYTHON_PATCH_OVERLAY:-0}

echo "[patches] ROOT_DIR=$ROOT_DIR"
echo "[patches] ${#PATCHES[@]} patch(es)"
echo "[patches] OVERLAY_MODE=$OVERLAY_MODE"
for p in "${PATCHES[@]}"; do echo "  - $(basename "$p")"; done

apply_one() {
  local p="$1"
  local base=$(basename "$p")
  if [ "$OVERLAY_MODE" = "1" ] && [ "$base" = "0001-cumem-alloc-env-fallback.diff" ]; then
    echo "[patches] Skipping ${base} (overlay mode)"
    return 0
  fi
  local tmp=$(mktemp /tmp/patch.XXXXXX.diff)
  tr -d '\r' < "$p" > "$tmp" 2>/dev/null || cp "$p" "$tmp"
  if git apply --check "$tmp" 2>/dev/null; then
    git apply "$tmp" || true
    return 0
  fi
  case "$base" in
    0001-cumem-alloc-env-fallback.diff)
      if [ "$OVERLAY_MODE" = "1" ]; then
        echo "[patches] Overlay mode active; skipping cumem fallback"
        return 0
      fi
      echo "[patches] Fallback editing cumem.py"
      python - <<'PY'
import io, os
path = os.path.join('vllm', 'device_allocator', 'cumem.py')
if not os.path.exists(path):
  raise SystemExit
src = io.open(path, 'r', encoding='utf-8').read()
if 'PYTORCH_ALLOC_CONF' in src:
  print('[patches] Already has PYTORCH_ALLOC_CONF')
  raise SystemExit
needle = 'conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")'
if needle in src:
  patched = src.replace(
      needle,
      'conf = os.environ.get("PYTORCH_ALLOC_CONF",\n       os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))')
  io.open(path, 'w', encoding='utf-8', newline='\n').write(patched)
  print('[patches] Added PYTORCH_ALLOC_CONF preference')
else:
  print('[patches] Pattern not found; skipping')
PY
      ;;
    *) echo "[patches] No fallback for ${base}" ;;
  esac
}

for p in "${PATCHES[@]}"; do
  echo "[patches] Applying $p"
  apply_one "$p"
done

if [ "$OVERLAY_MODE" != "1" ]; then
python - <<'PY'
import io, os, re
path = os.path.join('vllm', 'device_allocator', 'cumem.py')
if os.path.exists(path):
  src = io.open(path, 'r', encoding='utf-8').read()
  pattern = r'assert\s+"expandable_segments:True"[^\n]*\n(?:\s+\("Expandable segments[\s\S]*?updates\."\)\n)?'
  new_src = re.sub(pattern, '', src)
  if new_src != src:
    io.open(path, 'w', encoding='utf-8', newline='\n').write(new_src)
    print('[patches] Removed expandable_segments assert')
PY
fi

echo '[patches] Done.'

# End of patch script

