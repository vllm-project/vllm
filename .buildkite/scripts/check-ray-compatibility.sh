#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Check if Ray LLM can generate lock files that are compatible with this
# version of vllm. Downloads Ray's requirement files and runs a full
# dependency resolution with the installed vllm's constraints to see if
# a valid lock file can be produced.
#
# See: https://github.com/vllm-project/vllm/issues/33599

set -eo pipefail

RAY_BASE_URL="https://raw.githubusercontent.com/ray-project/ray/master/python"

WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

# Fetch all Ray requirement files used in the LLM depset pipeline
echo ">>> Fetching Ray requirement files"
RAY_FILES=(
    "requirements.txt"
    "requirements/cloud-requirements.txt"
    "requirements/base-test-requirements.txt"
    "requirements/llm/llm-requirements.txt"
    "requirements/llm/llm-test-requirements.txt"
)
for FILE in "${RAY_FILES[@]}"; do
    LOCAL_PATH="${WORK_DIR}/$(basename "$FILE")"
    echo "    ${FILE}"
    curl -fsSL -o "$LOCAL_PATH" "${RAY_BASE_URL}/${FILE}"
done

# Extract installed vllm deps
echo ">>> Extracting installed vllm dependency constraints"
python3 - "${WORK_DIR}/vllm-constraints.txt" <<'PYEOF'
"""Write out the installed vllm's dependencies as pip constraint lines.

Ray uses vllm[audio], so audio-extra deps are included with their extra
markers stripped. The resolver cannot evaluate extra markers for a
package that is not itself being resolved from an index, so we activate
them manually here.
"""
import importlib.metadata
import re
import sys

out_path = sys.argv[1]
raw_reqs = importlib.metadata.requires("vllm") or []

# Ray uses vllm[audio] – activate that extra.
ACTIVE_EXTRAS = {"audio"}
EXTRA_RE = re.compile(r"""extra\s*==\s*['"]([^'"]+)['"]""")

lines = []
for r in raw_reqs:
    if ";" not in r:
        # Unconditional dep — always include.
        lines.append(r.strip())
        continue

    req_part, _, marker_part = r.partition(";")
    marker_part = marker_part.strip()

    extra_matches = EXTRA_RE.findall(marker_part)
    if not extra_matches:
        # Non-extra marker (python_version, etc.) — keep as-is.
        lines.append(r.strip())
        continue

    if not ACTIVE_EXTRAS.intersection(extra_matches):
        continue  # Skip inactive extras (tensorizer, bench, …).

    # Strip the extra== conditions but keep any remaining markers
    # (e.g. python_version).
    cleaned = EXTRA_RE.sub("", marker_part)
    cleaned = re.sub(r"\band\b\s*\band\b", "and", cleaned)
    cleaned = re.sub(r"^\s*and\s+|\s+and\s*$", "", cleaned).strip()

    if cleaned:
        lines.append(f"{req_part.strip()} ; {cleaned}")
    else:
        lines.append(req_part.strip())

with open(out_path, "w") as f:
    for line in lines:
        f.write(line + "\n")

print(f"Wrote {len(lines)} constraints to {out_path}")
PYEOF

echo ">>> Installed vllm deps (first 20 lines):"
head -20 "${WORK_DIR}/vllm-constraints.txt"

# Remove Ray's vllm pin — the installed vllm's transitive deps
# (written above) replace it in the resolution. vllm itself cannot
# be resolved from PyPI for in-development versions, so we test
# whether Ray's requirements can coexist with vllm's dependency
# constraints instead.
sed -i '/^vllm/d' "${WORK_DIR}/llm-requirements.txt"

# Install uv if needed
if ! command -v uv &>/dev/null; then
    echo ">>> Installing uv"
    pip install uv -q
fi

# Resolve: given vllm's constraints, can Ray compile a lock file?
#
# vllm's dependency constraints are the fixed side — Ray is flexible and
# can regenerate its lock files. We pass vllm's constraints via -c so
# the resolver treats them as non-negotiable bounds, then check whether
# Ray's own requirements can still be satisfied within those bounds.
echo ""
echo "============================================================"
echo ">>> Resolving: Can Ray generate compatible lock files?"
echo "============================================================"

set +e
uv pip compile \
    "${WORK_DIR}/requirements.txt" \
    "${WORK_DIR}/cloud-requirements.txt" \
    "${WORK_DIR}/base-test-requirements.txt" \
    "${WORK_DIR}/llm-requirements.txt" \
    "${WORK_DIR}/llm-test-requirements.txt" \
    -c "${WORK_DIR}/vllm-constraints.txt" \
    --python-version 3.12 \
    --python-platform x86_64-manylinux_2_31 \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match \
    --unsafe-package setuptools \
    --unsafe-package ray \
    --no-header \
    -o "${WORK_DIR}/resolved.txt" \
    2>&1
EXIT_CODE=$?
set -e

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Ray can generate lock files compatible with this vllm."
    echo ""
    echo "Key resolved versions:"
    grep -E '^(protobuf|torch|numpy|transformers)==' \
        "${WORK_DIR}/resolved.txt" | sort || true
    echo "=========================================="
    exit 0
fi

echo "FAILURE: Ray cannot generate lock files compatible with this vllm."
echo "This means a fundamental dependency conflict exists that Ray"
echo "cannot resolve by regenerating its lock files."
echo "See: https://github.com/vllm-project/vllm/issues/33599"
echo "=========================================="

# Buildkite annotation
if [ -f /usr/bin/buildkite-agent ]; then
    buildkite-agent annotate --style 'warning' --context 'ray-compat' << EOF
### :warning: Ray Dependency Compatibility Warning
This PR introduces dependencies that **cannot** be resolved with Ray's requirements.
Ray would not be able to regenerate its lock files to accommodate this vllm version.

Please check the **Ray Dependency Compatibility Check** step logs for details.
See [issue #33599](https://github.com/vllm-project/vllm/issues/33599) for context.
EOF
fi

# Notify Slack if webhook is configured.
if [ -n "$RAY_COMPAT_SLACK_WEBHOOK_URL" ]; then
    echo ">>> Sending Slack notification"
    # Single quotes are intentional: the f-string expressions are Python, not shell.
    # shellcheck disable=SC2016
    PAYLOAD=$(python3 -c '
import json, os, sys
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
                f"that cannot be resolved with Ray'\''s requirements.\n"
                f"<{url}|View Build>"
            ),
        },
    }],
}
print(json.dumps(data))
')

    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$RAY_COMPAT_SLACK_WEBHOOK_URL" \
        -H 'Content-type: application/json' \
        -d "$PAYLOAD")
    echo "    Slack webhook response: $HTTP_CODE"
else
    echo ">>> Skipping Slack notification (RAY_COMPAT_SLACK_WEBHOOK_URL not set)"
fi

exit 1
