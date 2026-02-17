#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Check if vllm[audio] can be installed under Ray's dependency constraints.
# This is an informational check — failure means a PR introduces dependencies
# that would prevent Ray from installing vllm in its locked environment.
#
# See: https://github.com/vllm-project/vllm/issues/33599

set -o pipefail

RAY_LOCK_BASE_URL="https://raw.githubusercontent.com/ray-project/ray/master/python/deplocks/llm"
RAY_LOCK_FILES=(
    "ray_py311_cu128.lock"
    "rayllm_test_py311_cu128.lock"
)

cd /vllm-workspace/

# Locate the pre-built wheel
WHEEL=$(find /vllm-workspace/dist/ -name '*.whl' 2>/dev/null | head -1)
if [ -z "$WHEEL" ]; then
    echo "No pre-built wheel found in /vllm-workspace/dist/, building one..."
    pip wheel --no-deps -w /tmp/vllm-wheel .
    WHEEL=$(find /tmp/vllm-wheel -name '*.whl' | head -1)
fi

if [ -z "$WHEEL" ]; then
    echo "ERROR: Could not find or build a vllm wheel"
    exit 1
fi

OVERALL_EXIT=0
FAILED_LOCKS=()

for LOCK_NAME in "${RAY_LOCK_FILES[@]}"; do
    LOCK_URL="${RAY_LOCK_BASE_URL}/${LOCK_NAME}"
    LOCK_FILE="/tmp/${LOCK_NAME}"
    CONSTRAINTS_FILE="/tmp/${LOCK_NAME%.lock}_constraints.txt"

    echo ""
    echo "============================================================"
    echo ">>> Checking against: ${LOCK_NAME}"
    echo "============================================================"

    echo ">>> Fetching Ray lock file from ${LOCK_URL}"
    curl -fsSL -o "$LOCK_FILE" "$LOCK_URL"

    # The lock file contains --hash= entries which trigger pip's --require-hashes
    # mode for all packages, including the local wheel. Strip hashes and comments
    # to produce a clean constraints file with only package==version pins.
    # Also remove any vllm pin — the lock file may pin the currently-released
    # vllm version, which would conflict with the local wheel we're testing.
    sed -E '/^\s*--hash=/d; /^\s*#/d; s/ \\$//' "$LOCK_FILE" \
        | sed '/^$/d' \
        | grep -v '^vllm==' \
        > "$CONSTRAINTS_FILE"

    echo ">>> Constraints file (first 20 lines):"
    head -20 "$CONSTRAINTS_FILE"

    echo ">>> Testing: pip install --dry-run '${WHEEL}[audio]' -c ${CONSTRAINTS_FILE}"
    set +e
    pip install --dry-run "${WHEEL}[audio]" -c "$CONSTRAINTS_FILE" 2>&1
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        echo ">>> PASS: vllm[audio] is compatible with ${LOCK_NAME}"
    else
        echo ">>> FAIL: vllm[audio] conflicts with ${LOCK_NAME}"
        OVERALL_EXIT=1
        FAILED_LOCKS+=("$LOCK_NAME")
    fi
done

echo ""
echo "=========================================="
if [ $OVERALL_EXIT -eq 0 ]; then
    echo "SUCCESS: vllm[audio] is compatible with all Ray lock files."
    echo "=========================================="
    exit 0
fi

echo "WARNING: This PR introduces dependencies that conflict with Ray's lock files."
echo "Failing lock files: ${FAILED_LOCKS[*]}"
echo "Ray installs vllm via: pip install 'vllm[audio]'"
echo "Lock file base: ${RAY_LOCK_BASE_URL}/"
echo "See: https://github.com/vllm-project/vllm/issues/33599"
echo "=========================================="

FAILED_LIST=""
for f in "${FAILED_LOCKS[@]}"; do
    FAILED_LIST="${FAILED_LIST}\n- [${f}](${RAY_LOCK_BASE_URL}/${f})"
done

# if the agent binary is not found, skip annotations
if [ -f /usr/bin/buildkite-agent ]; then
    buildkite-agent annotate --style 'warning' --context 'ray-compat' << EOF
### :warning: Ray Dependency Compatibility Warning
This PR introduces dependencies that conflict with Ray's pinned environment.
Ray installs vllm via \`pip install 'vllm[audio]'\` with constraints from its lock files.

**Failing lock files:**
$(echo -e "$FAILED_LIST")

Please check the **Ray Dependency Compatibility Check** step logs for details.
See [issue #33599](https://github.com/vllm-project/vllm/issues/33599) for context.
EOF
fi

# Notify Slack if webhook is configured
if [ -n "$RAY_COMPAT_SLACK_WEBHOOK_URL" ]; then
    curl -s -X POST "$RAY_COMPAT_SLACK_WEBHOOK_URL" \
        -H 'Content-type: application/json' \
        -d "{
            \"text\": \":warning: Ray Dependency Compatibility Check Failed\",
            \"blocks\": [
                {
                    \"type\": \"section\",
                    \"text\": {
                        \"type\": \"mrkdwn\",
                        \"text\": \"*:warning: Ray Dependency Compatibility Check Failed*\nPR #${BUILDKITE_PULL_REQUEST:-N/A} on branch \`${BUILDKITE_BRANCH:-unknown}\` introduces dependencies that conflict with Ray's lock file(s): ${FAILED_LOCKS[*]}\n<${BUILDKITE_BUILD_URL:-#}|View Build>\"
                    }
                }
            ]
        }"
fi

exit $OVERALL_EXIT
