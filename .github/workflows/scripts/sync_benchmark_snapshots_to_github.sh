#!/bin/bash
set -euo pipefail

BENCHMARK_REPO_DIR=${BENCHMARK_REPO_DIR:?BENCHMARK_REPO_DIR is required}
WEBSITE_REPO_DIR=${WEBSITE_REPO_DIR:?WEBSITE_REPO_DIR is required}
CURRENT_SUBMISSION_DIR=${CURRENT_SUBMISSION_DIR:?CURRENT_SUBMISSION_DIR is required}
VLLM_HUST_REPO_DIR=${VLLM_HUST_REPO_DIR:-${VLLM_HUST_REPO:-$BENCHMARK_REPO_DIR/../vllm-hust}}
PYTHON_BIN=${PYTHON_BIN:-python3}
SNAPSHOT_TARGET_BRANCH=${SNAPSHOT_TARGET_BRANCH:-main}
SNAPSHOT_OUTPUT_DIR=${SNAPSHOT_OUTPUT_DIR:-$BENCHMARK_REPO_DIR/leaderboard-data/snapshots}
LOCAL_SNAPSHOT_OUTPUT_DIR=${LOCAL_SNAPSHOT_OUTPUT_DIR:-}
SNAPSHOT_MAX_PUSH_ATTEMPTS=${SNAPSHOT_MAX_PUSH_ATTEMPTS:-4}
SNAPSHOT_PUSH_RETRY_SECONDS=${SNAPSHOT_PUSH_RETRY_SECONDS:-5}
SNAPSHOT_COMMIT_MESSAGE=${SNAPSHOT_COMMIT_MESSAGE:-chore(data): sync benchmark publication}
GIT_COMMITTER_NAME=${GIT_COMMITTER_NAME:-vLLM-HUST Benchmark Bot}
GIT_COMMITTER_EMAIL=${GIT_COMMITTER_EMAIL:-benchmark-bot@vllm-hust.local}
BENCHMARK_REPO_REMOTE=${BENCHMARK_REPO_REMOTE:-origin}
BENCHMARK_REPO_SLUG=${BENCHMARK_REPO_SLUG:-vLLM-HUST/vllm-hust-benchmark}
BENCHMARK_REPO_GH_TOKEN=${BENCHMARK_REPO_GH_TOKEN:-}
BENCHMARK_REPO_SSH_KEY=${BENCHMARK_REPO_SSH_KEY:-}

required_submission_files=(leaderboard_manifest.json run_leaderboard.json)
required_snapshot_files=(
  leaderboard_single.json
  leaderboard_multi.json
  leaderboard_compare.json
  last_updated.json
)

write_github_env() {
  local key=$1
  local value=$2
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    printf '%s=%s\n' "$key" "$value" >>"$GITHUB_ENV"
  fi
}

configure_push_remote() {
  local remote_url=

  # Prefer SSH key over GH token: the SSH key is provisioned specifically
  # for the benchmark repo, while the GH token may belong to a user
  # without write access to the target repository.
  if [[ -n "$BENCHMARK_REPO_SSH_KEY" ]]; then
    remote_url="git@github.com:${BENCHMARK_REPO_SLUG}.git"
    git -C "$BENCHMARK_REPO_DIR" remote set-url "$BENCHMARK_REPO_REMOTE" "$remote_url"
    return 0
  fi

  if [[ -n "$BENCHMARK_REPO_GH_TOKEN" ]]; then
    remote_url="https://x-access-token:${BENCHMARK_REPO_GH_TOKEN}@github.com/${BENCHMARK_REPO_SLUG}.git"
    git -C "$BENCHMARK_REPO_DIR" remote set-url "$BENCHMARK_REPO_REMOTE" "$remote_url"
    return 0
  fi

  if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
    echo "L3 benchmark repository publication is enabled, but no cross-repository write credential is available." >&2
    echo "Configure one of the following secrets on the vllm-hust workflow repository before enabling benchmark repo publish:" >&2
    echo "  - VLLM_ASCEND_HUST_BENCHMARK_SSH_KEY: SSH private key with write access to ${BENCHMARK_REPO_SLUG}" >&2
    echo "  - VLLM_HUST_BENCHMARK_GH_TOKEN: GitHub token with contents write access to ${BENCHMARK_REPO_SLUG}" >&2
    echo "Benchmark repo publish target: ${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}" >&2
    exit 2
  fi

  echo "No benchmark repo credential configured outside GitHub Actions; using existing ${BENCHMARK_REPO_REMOTE} remote."
}

for file_name in "${required_submission_files[@]}"; do
  if [[ ! -f "$CURRENT_SUBMISSION_DIR/$file_name" ]]; then
    echo "missing current submission file: $CURRENT_SUBMISSION_DIR/$file_name" >&2
    exit 2
  fi
done

if [[ ! -d "$BENCHMARK_REPO_DIR/.git" ]]; then
  echo "benchmark repository checkout not found: $BENCHMARK_REPO_DIR" >&2
  exit 2
fi

if [[ ! -f "$WEBSITE_REPO_DIR/scripts/aggregate_results.py" ]]; then
  echo "website aggregation script not found: $WEBSITE_REPO_DIR/scripts/aggregate_results.py" >&2
  exit 2
fi

if [[ ! -f "$VLLM_HUST_REPO_DIR/pyproject.toml" ]]; then
  echo "vllm-hust repository checkout not found: $VLLM_HUST_REPO_DIR" >&2
  exit 2
fi

if [[ "${GITHUB_ACTIONS:-}" != "true" && "${ALLOW_LOCAL_GIT_RESET:-0}" != "1" ]]; then
  echo "refusing to reset a local checkout outside GitHub Actions; set ALLOW_LOCAL_GIT_RESET=1 to override" >&2
  exit 2
fi

run_id=${RUN_ID:-$(basename "$CURRENT_SUBMISSION_DIR")}
target_submission_dir="$BENCHMARK_REPO_DIR/submissions/$run_id"
relative_submission_dir="submissions/$run_id"
relative_snapshot_dir="leaderboard-data/snapshots"

git -C "$BENCHMARK_REPO_DIR" config user.name "$GIT_COMMITTER_NAME"
git -C "$BENCHMARK_REPO_DIR" config user.email "$GIT_COMMITTER_EMAIL"
configure_push_remote

export VLLM_HUST_BENCHMARK_REPO="$BENCHMARK_REPO_DIR"
export VLLM_HUST_WEBSITE_REPO="$WEBSITE_REPO_DIR"
export VLLM_HUST_REPO="$VLLM_HUST_REPO_DIR"

prepare_publication_commit() {
  git -C "$BENCHMARK_REPO_DIR" fetch "$BENCHMARK_REPO_REMOTE" "$SNAPSHOT_TARGET_BRANCH"
  git -C "$BENCHMARK_REPO_DIR" checkout -B "$SNAPSHOT_TARGET_BRANCH" "$BENCHMARK_REPO_REMOTE/$SNAPSHOT_TARGET_BRANCH"

  mkdir -p "$target_submission_dir"
  for file_name in "${required_submission_files[@]}"; do
    cp "$CURRENT_SUBMISSION_DIR/$file_name" "$target_submission_dir/$file_name"
  done

  mkdir -p "$SNAPSHOT_OUTPUT_DIR"
  for file_name in "${required_snapshot_files[@]}"; do
    rm -f "$SNAPSHOT_OUTPUT_DIR/$file_name"
  done

  "$PYTHON_BIN" -m vllm_hust_benchmark.cli publish-website \
    --source-dir "$BENCHMARK_REPO_DIR/submissions" \
    --output-dir "$SNAPSHOT_OUTPUT_DIR" \
    --execute

  for file_name in "${required_snapshot_files[@]}"; do
    if [[ ! -f "$SNAPSHOT_OUTPUT_DIR/$file_name" ]]; then
      echo "missing generated snapshot file: $SNAPSHOT_OUTPUT_DIR/$file_name" >&2
      exit 2
    fi
  done

  if [[ -n "$LOCAL_SNAPSHOT_OUTPUT_DIR" ]]; then
    mkdir -p "$LOCAL_SNAPSHOT_OUTPUT_DIR"
    for file_name in "${required_snapshot_files[@]}"; do
      cp "$SNAPSHOT_OUTPUT_DIR/$file_name" "$LOCAL_SNAPSHOT_OUTPUT_DIR/$file_name"
    done
  fi

  git -C "$BENCHMARK_REPO_DIR" add "$relative_submission_dir" "$relative_snapshot_dir"
  if git -C "$BENCHMARK_REPO_DIR" diff --cached --quiet; then
    return 1
  fi

  git -C "$BENCHMARK_REPO_DIR" commit -m "$SNAPSHOT_COMMIT_MESSAGE"
}

verify_published_benchmark_repo_state() {
  local expected_commit=$1
  local verified_commit
  local file_name

  git -C "$BENCHMARK_REPO_DIR" fetch "$BENCHMARK_REPO_REMOTE" "$SNAPSHOT_TARGET_BRANCH"
  verified_commit=$(git -C "$BENCHMARK_REPO_DIR" rev-parse "$BENCHMARK_REPO_REMOTE/$SNAPSHOT_TARGET_BRANCH")
  if [[ "$verified_commit" != "$expected_commit" ]]; then
    echo "benchmark publication verification failed: expected $expected_commit, got $verified_commit" >&2
    return 1
  fi

  for file_name in "${required_submission_files[@]}"; do
    if ! git -C "$BENCHMARK_REPO_DIR" cat-file -e \
      "$verified_commit:$relative_submission_dir/$file_name"; then
      echo "benchmark publication verification failed: missing $relative_submission_dir/$file_name" >&2
      return 1
    fi
  done

  for file_name in "${required_snapshot_files[@]}"; do
    if ! git -C "$BENCHMARK_REPO_DIR" cat-file -e \
      "$verified_commit:$relative_snapshot_dir/$file_name"; then
      echo "benchmark publication verification failed: missing $relative_snapshot_dir/$file_name" >&2
      return 1
    fi
  done

  write_github_env GITHUB_SNAPSHOT_SYNC_VERIFICATION verified
  write_github_env GITHUB_SNAPSHOT_SYNC_VERIFIED_COMMIT "$verified_commit"
  echo "Verified benchmark publication at ${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}: $verified_commit"
}

for attempt in $(seq 1 "$SNAPSHOT_MAX_PUSH_ATTEMPTS"); do
  if ! prepare_publication_commit; then
    echo "Benchmark publication already includes submission $run_id"
    echo "Benchmark repo target: ${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}"
    echo "Submission path: $relative_submission_dir"
    echo "Snapshot path: $relative_snapshot_dir"
    write_github_env GITHUB_SNAPSHOT_SYNC_STATUS unchanged
    write_github_env GITHUB_SNAPSHOT_SYNC_REPO "$BENCHMARK_REPO_SLUG"
    write_github_env GITHUB_SNAPSHOT_SYNC_BRANCH "$SNAPSHOT_TARGET_BRANCH"
    write_github_env GITHUB_SNAPSHOT_SYNC_SUBMISSION_PATH "$relative_submission_dir"
    write_github_env GITHUB_SNAPSHOT_SYNC_SNAPSHOT_PATH "$relative_snapshot_dir"
    verify_published_benchmark_repo_state "$(git -C "$BENCHMARK_REPO_DIR" rev-parse "$BENCHMARK_REPO_REMOTE/$SNAPSHOT_TARGET_BRANCH")"
    exit 0
  fi

  snapshot_commit=$(git -C "$BENCHMARK_REPO_DIR" rev-parse HEAD)
  if git -C "$BENCHMARK_REPO_DIR" push "$BENCHMARK_REPO_REMOTE" "HEAD:$SNAPSHOT_TARGET_BRANCH"; then
    verify_published_benchmark_repo_state "$snapshot_commit"
    echo "Pushed benchmark publication to ${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}: $snapshot_commit"
    echo "Submission path: $relative_submission_dir"
    echo "Snapshot path: $relative_snapshot_dir"
    write_github_env GITHUB_SNAPSHOT_SYNC_STATUS pushed
    write_github_env GITHUB_SNAPSHOT_SYNC_COMMIT "$snapshot_commit"
    write_github_env GITHUB_SNAPSHOT_SYNC_REPO "$BENCHMARK_REPO_SLUG"
    write_github_env GITHUB_SNAPSHOT_SYNC_BRANCH "$SNAPSHOT_TARGET_BRANCH"
    write_github_env GITHUB_SNAPSHOT_SYNC_SUBMISSION_PATH "$relative_submission_dir"
    write_github_env GITHUB_SNAPSHOT_SYNC_SNAPSHOT_PATH "$relative_snapshot_dir"
    exit 0
  fi

  if [[ "$attempt" -lt "$SNAPSHOT_MAX_PUSH_ATTEMPTS" ]]; then
    echo "benchmark publication push failed; retrying with fresh ${BENCHMARK_REPO_REMOTE}/${SNAPSHOT_TARGET_BRANCH} in ${SNAPSHOT_PUSH_RETRY_SECONDS}s (attempt $attempt/$SNAPSHOT_MAX_PUSH_ATTEMPTS)" >&2
    sleep "$SNAPSHOT_PUSH_RETRY_SECONDS"
  fi
done

echo "failed to push benchmark publication after $SNAPSHOT_MAX_PUSH_ATTEMPTS attempts" >&2
echo "Benchmark repo target: ${BENCHMARK_REPO_SLUG}@${SNAPSHOT_TARGET_BRANCH}" >&2
echo "Submission path: $relative_submission_dir" >&2
echo "Snapshot path: $relative_snapshot_dir" >&2
exit 1
