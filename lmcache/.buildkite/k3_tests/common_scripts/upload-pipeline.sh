#!/usr/bin/env bash
# Wraps `buildkite-agent pipeline upload` with a path-based skip check.
#
# Usage (called from each test's buildkite-pipeline.yml upload step):
#   command: bash .buildkite/k3_tests/common_scripts/upload-pipeline.sh \
#       .buildkite/k3_tests/<test-name>/pipeline.yml
#
# If every changed file in this build is trivial (markdown, LICENSE, .github,
# etc.) and none touch .buildkite/, this script:
#   - Annotates the build with a "skipped" note
#   - Exits 0 without uploading any further steps → the build is green
# Otherwise it execs `buildkite-agent pipeline upload <pipeline.yml>`, adding
# the real test steps to the build.
#
# Add a "force-ci" label to the PR on GitHub to bypass the check.

set -euo pipefail

PIPELINE_FILE="${1:?Usage: upload-pipeline.sh <path/to/pipeline.yml>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# shellcheck source=path-filter.sh
source "${SCRIPT_DIR}/path-filter.sh"

if should_skip_ci; then
    echo "+++ :fast_forward: Skipping CI — only trivial files changed"
    if command -v buildkite-agent >/dev/null 2>&1; then
        buildkite-agent annotate \
            --style success \
            --context "path-filter-skip" \
            "Skipped: only trivial files (docs, license, etc.) changed. Add a \`force-ci\` label to the PR to force a full run." \
            || true
    fi
    exit 0
fi

echo "--- :pipeline: Uploading ${PIPELINE_FILE}"
exec buildkite-agent pipeline upload "${PIPELINE_FILE}"
