#!/bin/bash
# Report this build's per-job results to the PyTorch Cross-Repo CI Relay (CRCR).
#
# Only runs in the torch-nightly lane on main. CRCR's nightly path is a
# self-report: unlike PR-triggered callbacks there is no upstream dispatch to
# correlate with, so the relay accepts a single "completed" callback per job and
# forwards it to HUD (hud.pytorch.org/crcr).
#
# Reporting is best-effort. A relay outage, an expired mapping or a missing
# token must never fail the nightly build, so every failure path here exits 0
# after logging. The step is also marked soft_fail in the pipeline.

set -uo pipefail

# --- Gating -----------------------------------------------------------------
# Matches the image_build.sh convention: the lane is selected inside the script
# rather than in pipeline YAML.
if [[ "${TORCH_NIGHTLY:-0}" != "1" ]]; then
    echo "TORCH_NIGHTLY != 1 -- not the nightly lane, nothing to report"
    exit 0
fi

# Defence in depth. The relay decides what a Buildkite pipeline may claim via
# ci_providers.yml, and a pipeline that builds fork PRs should constrain
# build_branch there. Refusing to even mint a token off main means a fork PR
# cannot report as vllm-project/vllm even if that mapping is ever relaxed.
if [[ "${BUILDKITE_BRANCH:-}" != "main" ]]; then
    echo "branch '${BUILDKITE_BRANCH:-}' is not main -- refusing to report"
    exit 0
fi

CALLBACK_URL="${CRCR_CALLBACK_URL:-}"
if [[ -z "${CALLBACK_URL}" ]]; then
    echo "CRCR_CALLBACK_URL unset -- skipping report"
    exit 0
fi

# read_builds only. Needed because a job cannot see its siblings' outcomes:
# the agent exposes only its own step, so the job list comes from the REST API.
BK_TOKEN="${BUILDKITE_API_TOKEN:-}"
if [[ -z "${BK_TOKEN}" ]]; then
    echo "BUILDKITE_API_TOKEN unset -- skipping report"
    exit 0
fi

AUDIENCE="pytorch-cross-repo-ci-relay"
OIDC_TOKEN="$(buildkite-agent oidc request-token --audience "${AUDIENCE}" 2>/dev/null)"
if [[ -z "${OIDC_TOKEN}" ]]; then
    echo "could not mint a Buildkite OIDC token -- skipping report"
    exit 0
fi

BUILD_JSON="$(mktemp)"
trap 'rm -f "${BUILD_JSON}"' EXIT
http_code="$(curl -sS -w '%{http_code}' -o "${BUILD_JSON}" \
    -H "Authorization: Bearer ${BK_TOKEN}" \
    "https://api.buildkite.com/v2/organizations/${BUILDKITE_ORGANIZATION_SLUG}/pipelines/${BUILDKITE_PIPELINE_SLUG}/builds/${BUILDKITE_BUILD_NUMBER}")"
if [[ "${http_code}" != "200" ]]; then
    echo "buildkite API returned ${http_code} -- skipping report"
    exit 0
fi

# One callback per job, matching HUD's per-job crcr_workflow_job schema.
# delivery_id is synthetic: the nightly path has no upstream dispatch to borrow
# one from, so build+job is used as the idempotency key.
python3 .buildkite/scripts/crcr_report.py \
    --build-json "${BUILD_JSON}" \
    --callback-url "${CALLBACK_URL}" \
    --oidc-token "${OIDC_TOKEN}" \
  || echo "crcr report failed -- continuing (reporting is best-effort)"

exit 0
