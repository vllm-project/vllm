#!/usr/bin/env bash
# Sync this fork with upstream while preserving fork-specific paths like extras/.
#
# Usage:
#   ./extras/tools/fork-sync/sync_with_upstream.sh [upstream_remote] [branch]
# Defaults: upstream remote "upstream", branch "main".
# Override protected paths via PROTECTED_PATHS="path1 path2".
set -euo pipefail

REMOTE=${1:-${UPSTREAM_REMOTE:-upstream}}
BRANCH=${2:-${UPSTREAM_BRANCH:-main}}
DEFAULT_PROTECTED=("extras" ".github/workflows/fork-sync.yml")
if [ -n "${PROTECTED_PATHS:-}" ]; then
  # Allow caller to override via space-separated list
  read -r -a PROTECTED <<< "$PROTECTED_PATHS"
else
  PROTECTED=(${DEFAULT_PROTECTED[@]})
fi

if [ ${#PROTECTED[@]} -eq 0 ]; then
  echo "[sync-extras] No protected paths configured; falling back to 'extras'"
  PROTECTED=("extras")
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[sync-extras] This script must be run inside a git repository." >&2
  exit 1
fi

if [ -d .git/rebase-merge ] || [ -d .git/rebase-apply ]; then
  echo "[sync-extras] Detected an in-progress rebase; aborting. Finish the rebase (git rebase --continue|--abort) and re-run." >&2
  exit 1
fi

if git rev-parse --verify MERGE_HEAD >/dev/null 2>&1; then
  echo "[sync-extras] Detected an in-progress merge; aborting. Resolve or cancel the merge before re-running." >&2
  exit 1
fi

echo "[sync-extras] Fetching ${REMOTE}/${BRANCH}"
git fetch "$REMOTE" "$BRANCH" || true

echo "[sync-extras] Updating from ${REMOTE}/${BRANCH}"
TMPDIR=""
BACKED_UP=()
for path in "${PROTECTED[@]}"; do
  if [ -e "$path" ]; then
    if [ -z "$TMPDIR" ]; then
      TMPDIR=$(mktemp -d -t extras-backup-XXXXXX)
    fi
    dest="$TMPDIR/$path"
    mkdir -p "$(dirname "$dest")"
    if [ -d "$path" ]; then
      mkdir -p "$dest"
      cp -a "$path/." "$dest/"
      echo "[sync-extras] Backed up directory $path -> $dest"
    else
      cp -a "$path" "$dest"
      echo "[sync-extras] Backed up file $path -> $dest"
    fi
    BACKED_UP+=("$path")
  fi
done

set +e
MERGE_OUTPUT=$(git merge --ff-only "${REMOTE}/${BRANCH}" 2>&1)
MERGE_STATUS=$?
set -e
if [ $MERGE_STATUS -ne 0 ]; then
  echo "$MERGE_OUTPUT"
  echo "[sync-extras] --ff-only merge failed; attempting a regular merge without editor"
  git merge --no-edit "${REMOTE}/${BRANCH}"
fi

if [ -n "$TMPDIR" ] && [ -d "$TMPDIR" ]; then
  for path in "${BACKED_UP[@]}"; do
    src="$TMPDIR/$path"
    if [ -d "$src" ]; then
      mkdir -p "$path"
      cp -a "$src/." "$path/"
    else
      mkdir -p "$(dirname "$path")"
      cp -a "$src" "$path"
    fi
    git add "$path"
    echo "[sync-extras] Restored $path"
  done
  rm -rf "$TMPDIR"
  if ! git diff --cached --quiet; then
    git commit -m "Restore protected paths after upstream sync" || true
  fi
fi

echo "[sync-extras] Done. You may now push with: git push origin $(git rev-parse --abbrev-ref HEAD)"
