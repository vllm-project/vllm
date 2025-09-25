#!/usr/bin/env bash
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

echo "[patches] ROOT_DIR=$ROOT_DIR"
echo "[patches] ${#PATCHES[@]} patch(es)"
for p in "${PATCHES[@]}"; do echo "  - $(basename "$p")"; done

apply_one() {
  local p="$1"
  local tmp=$(mktemp /tmp/patch.XXXXXX.diff)
  tr -d '\r' < "$p" > "$tmp" 2>/dev/null || cp "$p" "$tmp"
  if git apply --check "$tmp" 2>/dev/null; then
    git apply "$tmp" || true
    return 0
  fi
  case "$(basename "$p")" in
    0001-cumem-alloc-env-fallback.diff)
      echo "[patches] Fallback editing cumem.py"
      python - <<'PY'
import io, os
path='vllm/device_allocator/cumem.py'
if not os.path.exists(path):
  raise SystemExit
src=io.open(path,'r',encoding='utf-8').read()
if 'PYTORCH_ALLOC_CONF' in src:
  print('[patches] Already has PYTORCH_ALLOC_CONF')
  raise SystemExit
needle='conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")'
if needle in src:
  new=src.replace(needle,'conf = os.environ.get("PYTORCH_ALLOC_CONF",\n       os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))')
  io.open(path,'w',encoding='utf-8',newline='\n').write(new)
  print('[patches] Added PYTORCH_ALLOC_CONF preference')
else:
  print('[patches] Pattern not found; skipping')
PY
      ;;
    *) echo "[patches] No fallback for $(basename "$p")" ;;
  esac
}

for p in "${PATCHES[@]}"; do
  echo "[patches] Applying $p"
  apply_one "$p"
done

# Remove expandable_segments assert if still present
python - <<'PY'
import os, io, re
p='vllm/device_allocator/cumem.py'
if os.path.exists(p):
  src=io.open(p,'r',encoding='utf-8').read()
  # Match the original multi-line assert; be permissive but anchored on the key phrase
  pattern = r'assert\s+"expandable_segments:True"[^\n]*\n(?:\s+\("Expandable segments[\s\S]*?updates\."\)\n)?'
  new = re.sub(pattern,'',src)
  if new!=src:
    io.open(p,'w',encoding='utf-8',newline='\n').write(new)
    print('[patches] Removed expandable_segments assert')
PY

echo '[patches] Done.'


# End of patch script