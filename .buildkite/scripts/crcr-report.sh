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
# Pre-flight only; the token minted here is discarded. Checking now means an
# unusable OIDC setup costs seconds rather than being discovered after hours of
# polling. The token that is actually sent is minted after the wait, below.
# OIDC redaction also requires the unavailable Job API socket.
if [[ -z "$(buildkite-agent oidc request-token --skip-redaction --audience "${AUDIENCE}" 2>/dev/null)" ]]; then
    echo "could not mint a Buildkite OIDC token -- skipping report"
    exit 0
fi

BUILD_JSON="$(mktemp)"
BUILD_JSON_STAGING="$(mktemp)"
trap 'rm -f "${BUILD_JSON}" "${BUILD_JSON_STAGING}"' EXIT
BUILD_URL="https://api.buildkite.com/v2/organizations/${BUILDKITE_ORGANIZATION_SLUG}/pipelines/${BUILDKITE_PIPELINE_SLUG}/builds/${BUILDKITE_BUILD_NUMBER}"

# Staged so a failed request cannot clobber the last good snapshot: curl -o
# writes the error body too, and the deadline path reports from whatever
# snapshot it has.
fetch_build() {
    local code
    code="$(curl -sS -w '%{http_code}' -o "${BUILD_JSON_STAGING}" \
        -H "Authorization: Bearer ${BK_TOKEN}" "${BUILD_URL}")"
    if [[ "${code}" == "200" ]]; then
        mv -f "${BUILD_JSON_STAGING}" "${BUILD_JSON}"
        BUILD_JSON_STAGING="$(mktemp)"
    fi
    echo "${code}"
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
# Polling rather than a `depends_on` barrier: Buildkite groups carry no key, so
# a barrier means naming all ~250 step keys, which silently rots when a step is
# added and fails the pipeline upload when one is removed or excluded from this
# lane (AMD, retired A100). Gating on a single step was tried and is worse: a
# dependency that fails or is cancelled takes the report with it despite
# allow_dependency_failure (builds 91191 and 91461 both lost their report that
# way), so this step depends only on the image it runs in.
# 15 min: the wait now spans the whole build, so a 60s poll made ~420 requests
# against an API shared with the rest of CI to learn something that changes on
# the order of minutes. The cost is up to one interval of extra latency after
# the last job lands.
POLL_INTERVAL_S="${CRCR_POLL_INTERVAL_S:-900}"

# The deadline is measured from when the *build* started, not from when this
# script did, so "report no later than N hours into the build" holds however
# late this step is scheduled. Anchoring it to script start made the bound
# meaningless whenever the step was itself delayed.
#
# Resolved after the first fetch, from the build's own created_at.
MAX_BUILD_AGE_S="${CRCR_MAX_BUILD_AGE_S:-14400}"
WAIT_DEADLINE=""
# Used until created_at is known. Without it, an API outage on the very first
# request leaves the deadline unresolved and the loop spins to the step timeout.
FALLBACK_DEADLINE=$(( $(date +%s) + MAX_BUILD_AGE_S ))

build_deadline() {
    python3 - "${BUILD_JSON}" "${MAX_BUILD_AGE_S}" <<'PY'
import datetime, json, sys
build = json.load(open(sys.argv[1]))
created = build.get("created_at")
if not created:
    raise SystemExit(1)
started = datetime.datetime.fromisoformat(created.replace("Z", "+00:00"))
print(int(started.timestamp()) + int(sys.argv[2]))
PY
}

while :; do
    http_code="$(fetch_build)"
    if [[ "${http_code}" != "200" ]]; then
        # A transient 429/5xx must not cost the whole report: treating one
        # failure as fatal would drop a nightly for a blip. Keep polling on the
        # last good snapshot; only give up if the deadline passes having never
        # fetched one.
        echo "buildkite API returned ${http_code}"
        if [[ ! -s "${BUILD_JSON}" ]]; then
            if (( $(date +%s) >= ${WAIT_DEADLINE:-$FALLBACK_DEADLINE} )); then
                echo "no build data was ever fetched -- nothing to report"
                exit 0
            fi
            sleep "${POLL_INTERVAL_S}"
            continue
        fi
        if (( $(date +%s) >= ${WAIT_DEADLINE:-$FALLBACK_DEADLINE} )); then
            echo "deadline reached while the API is unavailable --" \
                "reporting from the last good snapshot"
            break
        fi
        sleep "${POLL_INTERVAL_S}"
        continue
    fi
    if [[ -z "${WAIT_DEADLINE}" ]]; then
        WAIT_DEADLINE="$(build_deadline)" || WAIT_DEADLINE=""
        if [[ -z "${WAIT_DEADLINE}" ]]; then
            # No created_at to anchor to; fall back to script start so the loop
            # still has a bound rather than running until the step times out.
            WAIT_DEADLINE=$(( $(date +%s) + MAX_BUILD_AGE_S ))
            echo "build created_at unavailable -- deadline measured from now"
        fi
        echo "reporting deadline: $(date -u -d "@${WAIT_DEADLINE}" '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null \
            || date -u -r "${WAIT_DEADLINE}" '+%Y-%m-%dT%H:%M:%SZ')" \
            "(build age limit ${MAX_BUILD_AGE_S}s)"
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
        echo "still ${remaining} job(s) running at the build-age deadline --" \
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

# Minted here, not before the poll loop: a Buildkite OIDC token defaults to a
# five-minute lifetime and the loop above can run to the build-age deadline.
# Build 91827 waited 6h48m and then took a 401 on all 360 callbacks from a
# token that had expired hours earlier.
OIDC_TOKEN="$(buildkite-agent oidc request-token --skip-redaction --audience "${AUDIENCE}" 2>/dev/null)"
if [[ -z "${OIDC_TOKEN}" ]]; then
    echo "could not mint a Buildkite OIDC token after the wait -- skipping report"
    exit 0
fi

python3 "${SCRIPT_DIR}/crcr_report.py" \
    --build-json "${BUILD_JSON}" \
    --callback-url "${CALLBACK_URL}" \
    --oidc-token "${OIDC_TOKEN}" \
  || echo "crcr report failed -- continuing (reporting is best-effort)"

exit 0
