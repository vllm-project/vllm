#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
BASE_BRANCH="${BASE_BRANCH:-main}"

cd "$REPO_ROOT"

resolve_upstream_remote() {
  if git remote get-url "$UPSTREAM_REMOTE" >/dev/null 2>&1; then
    printf '%s\n' "$UPSTREAM_REMOTE"
    return 0
  fi

  if git remote get-url origin >/dev/null 2>&1; then
    local origin_url
    origin_url="$(git remote get-url origin)"
    if printf '%s' "$origin_url" | grep -Eq 'vllm-project/vllm(\.git)?$'; then
      printf 'origin\n'
      return 0
    fi
  fi

  return 1
}

print_status() {
  local upstream_remote
  upstream_remote="$(resolve_upstream_remote || true)"

  local latest_stable
  latest_stable="$(git tag -l 'v[0-9]*.[0-9]*.[0-9]*' | grep -Ev 'rc|post|dev' | sort -V | tail -n 1)"
  echo
  echo "Contributor workspace status"
  echo "----------------------------"
  echo "Repo:            ${REPO_ROOT}"
  echo "Branch:          $(git branch --show-current)"
  echo
  echo "Remotes:"
  git remote -v
  echo

  if [ -n "$upstream_remote" ]; then
    if git rev-parse --verify --quiet "refs/remotes/$upstream_remote/$BASE_BRANCH" >/dev/null; then
      echo "Tracking base:"
      local local_ahead upstream_ahead
      read -r local_ahead upstream_ahead < <(git rev-list --left-right --count "${BASE_BRANCH}...$upstream_remote/$BASE_BRANCH")
      printf "  %s: upstream=%s\n" "$BASE_BRANCH" "$upstream_remote/$BASE_BRANCH"
      printf "  local commits not in upstream / upstream commits not in local: %s\n" "$local_ahead/$upstream_ahead"
    else
      echo "Remote branch not cached locally yet: $upstream_remote/$BASE_BRANCH (run: scripts/contributor-workspace.sh sync-main)"
    fi
  else
    echo "No upstream project remote found (expected upstream project remote)."
    echo "Run: git remote add upstream https://github.com/vllm-project/vllm.git"
    echo "If your fork is in origin and upstream points to your fork, you can set UPSTREAM_REMOTE=origin."
  fi

  echo
  if [ -n "$latest_stable" ]; then
    echo "Latest stable tag (non-rc/post): $latest_stable"
    echo "  To create a fresh baseline branch:"
    echo "  git switch -c contrib/stable-base \"$latest_stable\""
  else
    echo "No stable version tag found by pattern: vN.N.N"
  fi
  echo
}

sync_main() {
  local current_branch
  current_branch="$(git symbolic-ref --short -q HEAD || true)"
  local upstream_remote
  upstream_remote="$(resolve_upstream_remote || true)"

  if [ -z "$upstream_remote" ]; then
    echo "Missing upstream remote. Add it first:"
    echo "git remote add $UPSTREAM_REMOTE https://github.com/vllm-project/vllm.git"
    exit 1
  fi

  if [ -n "$(git status --porcelain)" ]; then
    echo "Workspace has uncommitted changes. Commit/stash or reset before sync."
    git status --short
    exit 1
  fi

  if [ "$upstream_remote" != "$UPSTREAM_REMOTE" ]; then
    echo "Using upstream remote '$upstream_remote' (set UPSTREAM_REMOTE explicitly to override)."
    UPSTREAM_REMOTE="$upstream_remote"
  fi

  git fetch "$UPSTREAM_REMOTE" --prune --tags

  if git rev-parse --verify --quiet "refs/remotes/$UPSTREAM_REMOTE/$BASE_BRANCH" >/dev/null; then
    git switch "$BASE_BRANCH"
    git pull --ff-only "$UPSTREAM_REMOTE" "$BASE_BRANCH"
  else
    echo "Creating local ${BASE_BRANCH} from $UPSTREAM_REMOTE/${BASE_BRANCH}"
    git switch --track -c "$BASE_BRANCH" "$UPSTREAM_REMOTE/$BASE_BRANCH"
  fi

  if [ -n "$current_branch" ] && [ "$current_branch" != "$BASE_BRANCH" ]; then
    git switch "$current_branch"
    echo
    echo "Updated $BASE_BRANCH from $UPSTREAM_REMOTE/$BASE_BRANCH"
    echo "Returned to your working branch: $current_branch"
  else
    echo
    echo "Updated $BASE_BRANCH from $UPSTREAM_REMOTE/$BASE_BRANCH"
    echo "Working branch remains checked out: $BASE_BRANCH"
  fi
}

show_help() {
  cat <<'EOF'
Usage:
  scripts/contributor-workspace.sh status
  scripts/contributor-workspace.sh sync-main

Environment:
  UPSTREAM_REMOTE  Remote that points to https://github.com/vllm-project/vllm (default: upstream)
  BASE_BRANCH     Base branch to track and sync (default: main)
EOF
}

case "${1:-status}" in
  status)
    print_status
    ;;
  sync-main)
    sync_main
    ;;
  -h|--help|help)
    show_help
    ;;
  *)
    echo "Unknown mode: ${1:-}"
    show_help
    exit 1
    ;;
esac
