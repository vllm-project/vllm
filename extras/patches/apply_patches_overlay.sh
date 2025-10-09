#!/usr/bin/env bash
set -euo pipefail

# Normalize CRLF and re-exec if needed
if grep -q $'\r' "$0" 2>/dev/null; then
  tmp_self=$(mktemp /tmp/apply_patches_overlay_self.XXXXXX.sh)
  tr -d '\r' < "$0" > "$tmp_self" || cp "$0" "$tmp_self"
  chmod +x "$tmp_self" 2>/dev/null || true
  exec "$tmp_self" "$@"
fi

ROOT_DIR=${ROOT_DIR:-$(pwd)}
PATCH_DIR="${PATCH_DIR:-$ROOT_DIR/extras/patches}"
cd "$ROOT_DIR"

shopt -s nullglob
PATCHES=("$PATCH_DIR"/*.diff)
shopt -u nullglob

OVERLAY_MODE=${PYTHON_PATCH_OVERLAY:-0}
TRACK_FILE_DEFAULT="/opt/work/tmp/vllm_patched_files.txt"
PATCH_TRACK_FILE=${PATCH_TRACK_FILE:-$TRACK_FILE_DEFAULT}

if [[ "$OVERLAY_MODE" == "1" ]]; then
  mkdir -p "$(dirname "$PATCH_TRACK_FILE")" 2>/dev/null || true
  : > "$PATCH_TRACK_FILE"
fi

echo "[patches] ROOT_DIR=$ROOT_DIR"
echo "[patches] ${#PATCHES[@]} patch(es)"
echo "[patches] OVERLAY_MODE=$OVERLAY_MODE"
for p in "${PATCHES[@]}"; do
  echo "  - $(basename "$p")"
done

apply_one() {
  local patch_path="$1"
  local base=$(basename "$patch_path")

  if [[ "$OVERLAY_MODE" == "1" && "$base" == "0001-cumem-alloc-env-fallback.diff" ]]; then
    echo "[patches] Skipping ${base} (overlay mode)"
    return 0
  fi

  local tmp
  tmp=$(mktemp /tmp/patch.overlay.XXXXXX.diff)
  tr -d '\r' < "$patch_path" > "$tmp" 2>/dev/null || cp "$patch_path" "$tmp"

  local patch_targets=()
  while IFS= read -r line; do
    if [[ "$line" == "+++ b/"* ]]; then
      local file=${line#+++ b/}
      if [[ "$file" != "/dev/null" && -n "$file" ]]; then
        patch_targets+=("$file")
      fi
    fi
  done < "$tmp"

  if git apply --check "$tmp" 2>/dev/null; then
    git apply "$tmp" || true
    if [[ "$OVERLAY_MODE" == "1" && ${#patch_targets[@]} -gt 0 ]]; then
      printf '%s\n' "${patch_targets[@]}" >> "$PATCH_TRACK_FILE"
    fi
    rm -f "$tmp"
    return 0
  fi

  if git apply --reverse --check "$tmp" 2>/dev/null; then
    echo "[patches] ${base} already applied" >&2
    rm -f "$tmp"
    return 0
  fi

  rm -f "$tmp"

  if [[ "$OVERLAY_MODE" == "1" ]]; then
    echo "[patches] ERROR: failed to apply ${base} in overlay mode" >&2
    return 1
  fi

  case "$base" in
    0001-cumem-alloc-env-fallback.diff)
      if [[ "$OVERLAY_MODE" == "1" ]]; then
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
    *)
      echo "[patches] No fallback for ${base}"
      ;;
  esac
}

for p in "${PATCHES[@]}"; do
  echo "[patches] Applying $p"
  apply_one "$p"
done

if [[ "$OVERLAY_MODE" != "1" ]]; then
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

if [[ "$OVERLAY_MODE" == "1" && -f "$PATCH_TRACK_FILE" ]]; then
  sort -u "$PATCH_TRACK_FILE" -o "$PATCH_TRACK_FILE" 2>/dev/null || true
fi

if command -v git >/dev/null 2>&1; then
  if [[ "$OVERLAY_MODE" == "1" ]]; then
    if git status --porcelain --untracked-files=no | grep -q '.'; then
      echo "[patches] ERROR: overlay mode left tracked files dirty" >&2
      git status --short --untracked-files=no >&2 || true
      exit 1
    fi
  fi
fi

echo '[patches] Done.'
