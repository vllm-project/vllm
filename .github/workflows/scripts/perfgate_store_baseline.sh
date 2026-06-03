#!/bin/bash
set -euo pipefail

RUN_ID=${RUN_ID:-ci-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}-${GITHUB_SHA:-local}}
RESULT_ROOT=${RESULT_ROOT:-${GITHUB_WORKSPACE:-$PWD}/.benchmarks/ci/$RUN_ID}
GITHUB_SHA=${GITHUB_SHA:-$(git rev-parse HEAD)}
RUN_ID=${RUN_ID:-manual}
BASELINE_BRANCH=${PERFGATE_BASELINE_BRANCH:-benchmark-baselines}
BASELINE_FILE=${PERFGATE_BASELINE_SOURCE_FILE:-$RESULT_ROOT/submissions/$RUN_ID/run_leaderboard.json}
WORKTREE_DIR=${PERFGATE_BASELINE_WORKTREE:-${RUNNER_TEMP:-/tmp}/perfgate-baselines-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}}

if [[ ! -f "$BASELINE_FILE" ]]; then
  echo "Perfgate baseline source not found: $BASELINE_FILE" >&2
  exit 2
fi

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
