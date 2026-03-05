#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Skip PR builds unless the PR has the "documentation" or "ready" label.
# Used by Read the Docs (see .readthedocs.yaml).

if [[ "$READTHEDOCS_VERSION_TYPE" != "external" ]]; then
  exit 0
fi

PR_URL="https://api.github.com/repos/vllm-project/vllm/pulls/${READTHEDOCS_VERSION}"
CURL_ARGS=(-s -o /tmp/pr_response.json -w "%{http_code}")
if [[ -n "$GITHUB_TOKEN" ]]; then
  CURL_ARGS+=(-H "Authorization: token ${GITHUB_TOKEN}")
fi
HTTP_CODE=$(curl "${CURL_ARGS[@]}" "$PR_URL")

if [[ "$HTTP_CODE" -ne 200 ]]; then
  echo "GitHub API returned HTTP ${HTTP_CODE}, proceeding with build."
elif grep -qE '"name": *"(documentation|ready)"' /tmp/pr_response.json; then
  echo "Found required label, proceeding with build."
else
  echo "PR #${READTHEDOCS_VERSION} lacks 'documentation' or 'ready' label, skipping build."
  exit 183
fi
