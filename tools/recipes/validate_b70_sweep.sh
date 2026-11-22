#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Validate existing B70 Recipes sweep stages on real XPU hardware.
set -euo pipefail

SWEEP_DIR="${1:-sweep_b70}"
MODE="${2:-dry-run}"
if [[ ! -d "$SWEEP_DIR" ]]; then
  echo "Sweep directory does not exist: $SWEEP_DIR" >&2
  exit 2
fi
SWEEP_DIR="$(cd "$SWEEP_DIR" && pwd)"
[[ -f "$SWEEP_DIR/run_full_sweep.sh" ]] || {
  echo "Missing run_full_sweep.sh; generate --generate-full-sweep first" >&2
  exit 2
}
[[ -f "$SWEEP_DIR/parallel_layout_serve_params.json" ]] || {
  echo "Missing XPU layout candidates" >&2
  exit 2
}
if ! compgen -G '/dev/dri/renderD*' >/dev/null; then
  echo "WARNING: no /dev/dri/renderD* device visible; check XPU container mapping" >&2
fi
python3 - "$SWEEP_DIR" <<'PY'
import json
import pathlib
import sys
p = pathlib.Path(sys.argv[1])
candidates = json.loads((p / 'parallel_layout_serve_params.json').read_text())
if not candidates:
    raise SystemExit('No XPU layouts generated')
print(f'B70 sweep layout candidates ({len(candidates)}):')
for c in candidates:
    print(' ', c.get('_benchmark_name'), 'TP=', c.get('tensor_parallel_size'),
          'DP=', c.get('data_parallel_size'))
PY
case "$MODE" in
  dry-run)
    "$SWEEP_DIR/run_full_sweep.sh" --stages parallel-layout --dry-run
    echo "Dry run completed; rerun with: $0 '$SWEEP_DIR' run"
    ;;
  run)
    "$SWEEP_DIR/run_full_sweep.sh"
    python3 - "$SWEEP_DIR" <<'PY'
import json
import pathlib
import sys
p = pathlib.Path(sys.argv[1])
for name in ('parallel-layout-recommendation.json',
             'concurrency-recommendation.json', 'recommendation.json'):
    f = p / name
    print(f'{name}: {"present" if f.is_file() else "missing (check stage/fallback logs)"}')
if not (p / 'parallel-layout-config.yml').is_file():
    raise SystemExit('Parallel layout configuration missing after sweep')
print('Selected layout:', p / 'parallel-layout-config.yml')
PY
    ;;
  *) echo "Usage: $0 [sweep-directory] [dry-run|run]" >&2; exit 2 ;;
esac
