#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Build the macOS arm64 CPU wheel on a GitHub-hosted Apple Silicon runner.
#
# vLLM's Buildkite fleet has no macOS agents, and macOS wheels cannot be
# cross-compiled from Linux. This dispatches the macos-wheel.yml GitHub Actions
# workflow (which runs on macos-latest), waits for it, and downloads the wheel
# into artifacts/dist/ so the normal upload-nightly-wheels.sh path can take it
# from here -- the macOS wheel flows through the release pipeline like any other.
#
# Requires the `gh` CLI and GH_TOKEN with actions:read+write on the repo.

set -euo pipefail

REPO="${MACOS_WHEEL_REPO:-vllm-project/vllm}"
WORKFLOW="${MACOS_WHEEL_WORKFLOW:-macos-wheel.yml}"
SHA="${BUILDKITE_COMMIT:?BUILDKITE_COMMIT is not set}"
# Unique handle, set as the workflow run-name, so we can find the run we just
# dispatched (gh workflow run does not return a run id).
BK_ID="${BUILDKITE_BUILD_ID:-manual-${SHA}}"
RUN_NAME="macos-wheel ${BK_ID}"

echo "Dispatching ${WORKFLOW} on ${REPO} for ${SHA} (bk_id=${BK_ID})"
gh workflow run "${WORKFLOW}" --repo "${REPO}" -f sha="${SHA}" -f bk_id="${BK_ID}"

# Dispatch is async; poll for the run with our unique run-name to appear.
RUN_ID=""
for _ in $(seq 1 30); do
  RUN_ID="$(gh run list --repo "${REPO}" --workflow "${WORKFLOW}" \
    --json databaseId,name \
    -q "[.[] | select(.name == \"${RUN_NAME}\")][0].databaseId")"
  if [[ -n "${RUN_ID}" && "${RUN_ID}" != "null" ]]; then
    break
  fi
  sleep 10
done
if [[ -z "${RUN_ID}" || "${RUN_ID}" == "null" ]]; then
  echo "ERROR: could not locate dispatched run named '${RUN_NAME}'" >&2
  exit 1
fi
echo "Dispatched run ${RUN_ID}: https://github.com/${REPO}/actions/runs/${RUN_ID}"

# Block until the run finishes; non-zero exit if it failed.
gh run watch "${RUN_ID}" --repo "${REPO}" --exit-status

# Place the wheel where upload-nightly-wheels.sh expects it (artifacts/dist/*.whl).
mkdir -p artifacts/dist
gh run download "${RUN_ID}" --repo "${REPO}" -n macos-wheel -D artifacts/dist
echo "Downloaded wheel(s):"
ls -l artifacts/dist/*.whl
