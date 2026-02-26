#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Check if vllm can be installed under Ray's dependency constraints.
# This is an informational check — failure means a PR introduces dependencies
# that would prevent Ray from installing vllm in its locked environment.
#
# See: https://github.com/vllm-project/vllm/issues/33599

set -eo pipefail

RAY_LOCK_BASE_URL="https://raw.githubusercontent.com/ray-project/ray/master/python/deplocks/llm"
RAY_LOCK_FILES=(
    "ray_py311_cu128.lock"
    "rayllm_test_py311_cu128.lock"
)

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
    sed -E '/^\s*--hash=/d; /^\s*#/d; /^\s*--(index-url|extra-index-url)/d; s/ \\$//' "$LOCK_FILE" \
        | sed '/^$/d' \
        | grep -v '^vllm==' \
        > "$CONSTRAINTS_FILE"

    echo ">>> Constraints file (first 20 lines):"
    head -20 "$CONSTRAINTS_FILE"

    echo ">>> Checking installed vllm deps against ${LOCK_NAME} constraints"
    set +e
    python3 - "$CONSTRAINTS_FILE" <<'PYEOF'
"""Check if the installed vllm dependencies are satisfiable
under the version pins in a Ray lock-file-derived constraints file.

Reads the installed vllm metadata (including the [audio] extra) and
checks every dependency against the constraints.  Exits 0 if all
constraints are satisfiable, 1 otherwise.
"""
import importlib.metadata
import re
import sys
from packaging.requirements import Requirement
from packaging.version import Version

constraints_file = sys.argv[1]

# Parse constraints file into {package_name: pinned_version}
pins: dict[str, str] = {}
with open(constraints_file) as f:
    for line in f:
        line = line.strip()
        # Skip environment markers (e.g. "cffi==1.17.1 ; platform...")
        # — we only care about the name==version part.
        m = re.match(r'^([A-Za-z0-9_.-]+)==([^\s;]+)', line)
        if m:
            pins[m.group(1).lower().replace("-", "_")] = m.group(2)

# Gather all unconditional vllm requirements (no extras needed).
raw_reqs = importlib.metadata.requires("vllm") or []

reqs = []
for r in raw_reqs:
    req = Requirement(r)
    if req.marker is None:
        reqs.append(req)

conflicts = []
for req in reqs:
    name = req.name.lower().replace("-", "_")
    if name not in pins:
        continue
    pinned = Version(pins[name])
    if not req.specifier.contains(pinned):
        conflicts.append(
            f"  {req.name}: vllm requires {req.specifier}, "
            f"but Ray pins {pins[name]}"
        )

if conflicts:
    print("Conflicts found:")
    for c in conflicts:
        print(c)
    sys.exit(1)
else:
    print("No conflicts found.")
    sys.exit(0)
PYEOF
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        echo ">>> PASS: vllm is compatible with ${LOCK_NAME}"
    else
        echo ">>> FAIL: vllm conflicts with ${LOCK_NAME}"
        OVERALL_EXIT=1
        FAILED_LOCKS+=("$LOCK_NAME")
    fi
done

echo ""
echo "=========================================="
if [ $OVERALL_EXIT -eq 0 ]; then
    echo "SUCCESS: vllm is compatible with all Ray lock files."
    echo "=========================================="
    exit 0
fi

echo "WARNING: This PR introduces dependencies that conflict with Ray's lock files."
echo "Failing lock files: ${FAILED_LOCKS[*]}"
echo "Ray installs vllm via: pip install 'vllm'"
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
Ray installs vllm via \`pip install 'vllm'\` with constraints from its lock files.

**Failing lock files:**
$(echo -e "$FAILED_LIST")

Please check the **Ray Dependency Compatibility Check** step logs for details.
See [issue #33599](https://github.com/vllm-project/vllm/issues/33599) for context.
EOF
fi

# Notify Slack if webhook is configured.
if [ -n "$RAY_COMPAT_SLACK_WEBHOOK_URL" ]; then
    # Single quotes are intentional: the f-string expressions are Python, not shell.
    # shellcheck disable=SC2016
    PAYLOAD=$(python3 -c '
import json, os, sys
failed = sys.argv[1]
pr = os.getenv("BUILDKITE_PULL_REQUEST", "N/A")
branch = os.getenv("BUILDKITE_BRANCH", "unknown")
url = os.getenv("BUILDKITE_BUILD_URL", "#")
data = {
    "text": ":warning: Ray Dependency Compatibility Check Failed",
    "blocks": [{
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": (
                "*:warning: Ray Dependency Compatibility Check Failed*\n"
                f"PR #{pr} on branch `{branch}` introduces dependencies "
                f"that conflict with Ray'\''s lock file(s): {failed}\n"
                f"<{url}|View Build>"
            ),
        },
    }],
}
print(json.dumps(data))
' "${FAILED_LOCKS[*]}")

    curl -s -X POST "$RAY_COMPAT_SLACK_WEBHOOK_URL" \
        -H 'Content-type: application/json' \
        -d "$PAYLOAD"
fi

exit $OVERALL_EXIT
