#!/bin/bash
set -euo pipefail

RUN_ID=${RUN_ID:-ci-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}-${GITHUB_SHA:-local}}
RESULT_ROOT=${RESULT_ROOT:-${GITHUB_WORKSPACE:-$PWD}/.benchmarks/ci/$RUN_ID}
GITHUB_SHA=${GITHUB_SHA:-$(git rev-parse HEAD)}
BASELINE_BRANCH=${PERFGATE_BASELINE_BRANCH:-benchmark-baselines}
BASELINE_FILE=${PERFGATE_BASELINE_SOURCE_FILE:-$RESULT_ROOT/submissions/$RUN_ID/run_leaderboard.json}
WORKTREE_DIR=${PERFGATE_BASELINE_WORKTREE:-${RUNNER_TEMP:-/tmp}/perfgate-baselines-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}}

if [[ ! -f "$BASELINE_FILE" ]]; then
  echo "Perfgate baseline source not found: $BASELINE_FILE" >&2
  exit 2
fi

"${PYTHON_BIN:-python}" - "$BASELINE_FILE" <<'PY'
import json
import math
import sys
from pathlib import Path

EXPECTED_SPEC_ID = "perfgate-ascend-qwen25-05b-910b3"
path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"Invalid perfgate baseline JSON: {path}: {exc}", file=sys.stderr)
    sys.exit(2)
metrics = payload.get("metrics")
if not isinstance(metrics, dict):
    print(f"Invalid perfgate baseline JSON: {path}: missing object key metrics", file=sys.stderr)
    sys.exit(2)
missing = []
for name in ("throughput_tps", "ttft_ms", "tbt_ms"):
    value = metrics.get(name)
    try:
        number = float(value)
    except (TypeError, ValueError):
        missing.append(name)
        continue
    if value is None or not math.isfinite(number):
        missing.append(name)
if missing:
    print(
        f"Invalid perfgate baseline JSON: {path}: missing/non-null finite metrics: {', '.join(missing)}",
        file=sys.stderr,
    )
    sys.exit(2)
same_spec = payload.get("same_spec")
if not isinstance(same_spec, dict):
    print(f"Invalid perfgate baseline JSON: {path}: missing object key same_spec", file=sys.stderr)
    sys.exit(2)
spec_id = str(same_spec.get("spec_id") or "").strip()
spec_hash = str(same_spec.get("resolved_spec_hash") or "").strip()
if spec_id != EXPECTED_SPEC_ID or not spec_hash:
    print(
        f"Invalid perfgate baseline JSON: {path}: expected same_spec.spec_id={EXPECTED_SPEC_ID!r} and non-empty resolved_spec_hash",
        file=sys.stderr,
    )
    sys.exit(2)
PY

rm -rf "$WORKTREE_DIR"
if git ls-remote --exit-code --heads origin "$BASELINE_BRANCH" >/dev/null 2>&1; then
  git fetch origin "$BASELINE_BRANCH:$BASELINE_BRANCH" || git fetch origin "$BASELINE_BRANCH"
  git worktree add "$WORKTREE_DIR" "origin/$BASELINE_BRANCH"
else
  git worktree add --detach "$WORKTREE_DIR" HEAD
  git -C "$WORKTREE_DIR" checkout --orphan "$BASELINE_BRANCH"
  git -C "$WORKTREE_DIR" rm -rf . >/dev/null 2>&1 || true
fi

mkdir -p "$WORKTREE_DIR/baselines/$GITHUB_SHA"
cp "$BASELINE_FILE" "$WORKTREE_DIR/baselines/$GITHUB_SHA/run_leaderboard.json"
cp "$BASELINE_FILE" "$WORKTREE_DIR/latest-main.json"
cat > "$WORKTREE_DIR/latest-main-pointer.json" <<EOF
{
  "commit": "$GITHUB_SHA",
  "run_id": "$RUN_ID",
  "path": "baselines/$GITHUB_SHA/run_leaderboard.json"
}
EOF

git -C "$WORKTREE_DIR" add baselines latest-main.json latest-main-pointer.json
if git -C "$WORKTREE_DIR" diff --cached --quiet; then
  echo "Perfgate baseline unchanged for $GITHUB_SHA"
else
  git -C "$WORKTREE_DIR" config user.name "vLLM-HUST Benchmark Bot"
  git -C "$WORKTREE_DIR" config user.email "benchmark-bot@vllm-hust.local"
  git -C "$WORKTREE_DIR" commit -m "chore(perfgate): store baseline for ${GITHUB_SHA:0:8}"
  git -C "$WORKTREE_DIR" push origin "HEAD:$BASELINE_BRANCH"
fi

git worktree remove "$WORKTREE_DIR" --force
