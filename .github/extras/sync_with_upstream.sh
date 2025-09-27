#!/usr/bin/env bash
# Sync this fork with upstream while preserving the extras/ directory.
#
# Usage:
#   ./github/extras/sync_with_upstream.sh [upstream_remote] [branch]
# Defaults: upstream remote "upstream", branch "main".
set -euo pipefail

REMOTE=${1:-${UPSTREAM_REMOTE:-upstream}}
BRANCH=${2:-${UPSTREAM_BRANCH:-main}}
EXTRAS_DIR=${EXTRAS_DIR:-extras}

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[sync-extras] This script must be run inside a git repository." >&2
  exit 1
fi

if ! git rev-parse --verify "${REMOTE}/${BRANCH}" >/dev/null 2>&1; then
  echo "[sync-extras] Fetching ${REMOTE}/${BRANCH}"; git fetch "$REMOTE" "$BRANCH" || true
fi

echo "[sync-extras] Updating from ${REMOTE}/${BRANCH}"
TMPDIR=""
if [ -d "$EXTRAS_DIR" ]; then
  TMPDIR=$(mktemp -d -t extras-backup-XXXXXX)
  echo "[sync-extras] Backing up ${EXTRAS_DIR}/ to ${TMPDIR}"
  cp -a "$EXTRAS_DIR/." "$TMPDIR"/
fi

set +e
MERGE_OUTPUT=$(git merge --ff-only "${REMOTE}/${BRANCH}" 2>&1)
MERGE_STATUS=$?
set -e
if [ $MERGE_STATUS -ne 0 ]; then
  echo "$MERGE_OUTPUT"
  echo "[sync-extras] --ff-only merge failed; attempting a regular merge"
  git merge "${REMOTE}/${BRANCH}"
fi

if [ -n "$TMPDIR" ] && [ -d "$TMPDIR" ]; then
  echo "[sync-extras] Restoring ${EXTRAS_DIR}/ from backup"
  mkdir -p "$EXTRAS_DIR"
  cp -a "$TMPDIR/." "$EXTRAS_DIR"/
  rm -rf "$TMPDIR"
  git add "$EXTRAS_DIR"
  if ! git diff --cached --quiet; then
    git commit -m "Restore ${EXTRAS_DIR} after upstream sync" || true
  fi
fi

echo "[sync-extras] Done. You may now push with: git push origin $(git rev-parse --abbrev-ref HEAD)"
