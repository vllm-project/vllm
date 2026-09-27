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
TOKEN_SECRET_KEY="${CRCR_BUILDKITE_TOKEN_SECRET_KEY:-CRCR_BUILDKITE_API_TOKEN}"
BK_TOKEN="${BUILDKITE_API_TOKEN:-}"
if [[ -z "${BK_TOKEN}" ]]; then
    # Not in the job environment, so read it from a Buildkite secret.
    #
    # Report why a lookup failed. Swallowing stderr made a missing secret, a
    # denied policy and an unusable agent indistinguishable, all surfacing as the
    # same "no token" line. Only stderr is echoed -- stdout is the secret.
    if ! command -v buildkite-agent >/dev/null 2>&1; then
        echo "buildkite-agent is not on PATH; cannot read secret '${TOKEN_SECRET_KEY}'"
    else
        secret_err="$(mktemp)"
        # Requires agent >= 3.107.0: the Docker plugin does not mount the Job API
        # socket needed for redaction. Capture the value without logging it.
        if BK_TOKEN="$(buildkite-agent secret get --skip-redaction "${TOKEN_SECRET_KEY}" 2>"${secret_err}")"; then
            if [[ -z "${BK_TOKEN}" ]]; then
                echo "secret '${TOKEN_SECRET_KEY}' resolved but is empty"
            fi
        else
            BK_TOKEN=""
            echo "buildkite-agent secret get '${TOKEN_SECRET_KEY}' failed" \
                "(agent $(buildkite-agent --version 2>&1 | head -1)):"
            sed 's/^/    /' "${secret_err}"
        fi
        rm -f "${secret_err}"
    fi
fi
if [[ -z "${BK_TOKEN}" ]]; then
    echo "no Buildkite API token (env BUILDKITE_API_TOKEN or secret" \
        "'${TOKEN_SECRET_KEY}') -- skipping report"
    exit 0
fi

AUDIENCE="pytorch-cross-repo-ci-relay"
# OIDC redaction also requires the unavailable Job API socket.
OIDC_TOKEN="$(buildkite-agent oidc request-token --skip-redaction --audience "${AUDIENCE}" 2>/dev/null)"
if [[ -z "${OIDC_TOKEN}" ]]; then
    echo "could not mint a Buildkite OIDC token -- skipping report"
    exit 0
fi

BUILD_JSON="$(mktemp)"
trap 'rm -f "${BUILD_JSON}"' EXIT
BUILD_URL="https://api.buildkite.com/v2/organizations/${BUILDKITE_ORGANIZATION_SLUG}/pipelines/${BUILDKITE_PIPELINE_SLUG}/builds/${BUILDKITE_BUILD_NUMBER}"

fetch_build() {
    curl -sS -w '%{http_code}' -o "${BUILD_JSON}" \
        -H "Authorization: Bearer ${BK_TOKEN}" "${BUILD_URL}"
}

# Count jobs that have not reached a terminal state, ignoring this job: the
# report is itself part of the build and cannot wait for itself to finish.
outstanding_jobs() {
    python3 - "${BUILD_JSON}" "${BUILDKITE_JOB_ID:-}" <<'PY'
import json, sys
TERMINAL = {
    "passed", "failed", "blocked", "canceled", "skipped", "not_run",
    "broken", "timed_out", "waiting_failed", "finished", "expired",
}
build = json.load(open(sys.argv[1]))
self_id = sys.argv[2]
print(sum(
    1 for j in build.get("jobs", [])
    if j.get("type") == "script"
    and j.get("id") != self_id
    and j.get("state") not in TERMINAL
))
PY
}

# Wait for the rest of the build. The report is a snapshot of the Buildkite
# API's view of this build, so taking it early does not merely omit jobs -- it
# actively reports them as clean. Build 90830 reported with 14 of 347 jobs
# finished and none of the eventual 152 hard failures visible, and HUD recorded
# that nightly as green.
#
# Polling rather than a full `depends_on` barrier: Buildkite groups carry no
# key, so a barrier means naming all ~250 step keys, which silently rots when a
# step is added and fails the pipeline upload when one is removed or excluded
# from this lane (AMD, retired A100). The pipeline does gate this step on the
# long pole so it is scheduled near the end of the build -- that keeps the agent
# from idling for hours, but it is only a hint, and this loop is what actually
# guarantees the snapshot is complete.
POLL_INTERVAL_S="${CRCR_POLL_INTERVAL_S:-60}"
# Under the step's own timeout, so a stuck build still gets a partial report
# rather than the job being killed with nothing sent.
WAIT_DEADLINE=$(( $(date +%s) + ${CRCR_MAX_WAIT_S:-46800} ))
while :; do
    http_code="$(fetch_build)"
    if [[ "${http_code}" != "200" ]]; then
        echo "buildkite API returned ${http_code} -- skipping report"
        exit 0
    fi
    remaining="$(outstanding_jobs)" || remaining=""
    if [[ -z "${remaining}" ]]; then
        echo "could not parse build JSON -- reporting what we have"
        break
    fi
    if (( remaining == 0 )); then
        echo "all other jobs have finished -- reporting"
        break
    fi
    if (( $(date +%s) >= WAIT_DEADLINE )); then
        echo "still ${remaining} job(s) running at the wait deadline --" \
            "reporting a partial view"
        break
    fi
    echo "waiting for ${remaining} job(s) to finish..."
    sleep "${POLL_INTERVAL_S}"
done

# One callback per job, matching HUD's per-job crcr_workflow_job schema.
# delivery_id is synthetic: the nightly path has no upstream dispatch to borrow
# one from, so build+job is used as the idempotency key.
# Resolved from this script's location: the pipeline runs it from
# /vllm-workspace/tests, so a repo-relative path would not resolve.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/crcr_report.py" \
    --build-json "${BUILD_JSON}" \
    --callback-url "${CALLBACK_URL}" \
    --oidc-token "${OIDC_TOKEN}" \
  || echo "crcr report failed -- continuing (reporting is best-effort)"

exit 0
