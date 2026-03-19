#!/usr/bin/env bash
# Check whether a PR is eligible to run CI.
# Passes if the PR has the 'ready' label OR the author has >=4 merged PRs.
#
# Required environment variables:
#   PR_NUMBER  – pull request number
#   AUTHOR     – PR author login
#   REPO       – owner/repo (e.g. vllm-project/vllm)
#   GH_TOKEN   – GitHub token with read access

set -euo pipefail

: "${PR_NUMBER:?must be set}"
: "${AUTHOR:?must be set}"
: "${REPO:?must be set}"

# Check for 'ready' label
HAS_READY_LABEL=$(gh pr view "$PR_NUMBER" --repo "$REPO" --json labels --jq '.labels[].name' | grep -c '^ready$' || true)

# Count author's merged PRs
MERGED_COUNT=$(gh pr list --repo "$REPO" --author "$AUTHOR" --state merged --limit 4 --json number --jq 'length')

if [[ "$HAS_READY_LABEL" -ge 1 || "$MERGED_COUNT" -ge 4 ]]; then
  echo "Check passed: ready label=$HAS_READY_LABEL, merged PRs=${MERGED_COUNT}+"
else
  echo "::error::PR must have the 'ready' label or the author must have at least 4 merged PRs (found $MERGED_COUNT)."
  exit 1
fi
