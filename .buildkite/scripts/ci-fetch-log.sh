#!/bin/bash
# Fetch vLLM Buildkite CI logs (public; no login required).
#
# Usage:
#   ci-fetch-log.sh [--soft|--all] --pr [<PR>]  failed jobs in the PR's latest
#                                               build (current branch if omitted)
#   ci-fetch-log.sh [--soft|--all] <build_url>  failed jobs in that build
#   ci-fetch-log.sh <job_url> [output]          one job; both #<job_uuid> and
#                                               ?sid=<id> URL forms work
#   ci-fetch-log.sh <build> <job_uuid> [output]
#
# --soft also fetches soft-failed jobs; --all fetches every finished job.
# Saves each log as ci-<build>-<job-name>.log (ANSI/timestamps stripped) and
# prints "<file>\t<job name>" per job. [output] is single-job only; "-"
# streams to stdout. Existing files are kept; CI_FETCH_LOG_FORCE=1 refetches.

set -euo pipefail

ORG="vllm"
PIPELINE="ci"
UA="vllm-ci-fetch-log"
UUID_RE='[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'

usage() {
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

die() {
    echo "$1" >&2
    exit 1
}

BUILD="" JOB="" SID="" OUT=""
SCOPE="failed"

while :; do
    case "${1:-}" in
    --soft) SCOPE="soft" ;;
    --all) SCOPE="all" ;;
    *) break ;;
    esac
    shift
done

case "${1:-}" in
--pr)
    PR="${2:-}"
    # gh pr checks exits non-zero when checks are failing; that is the
    # expected case here.
    URL=$(gh pr checks ${PR:+"$PR"} --repo vllm-project/vllm 2>/dev/null |
        grep -oE "https://buildkite.com/${ORG}/${PIPELINE}/builds/[0-9]+" |
        sort -t/ -k7 -n | tail -1 || true)
    [ -n "$URL" ] || die "No Buildkite build found via: gh pr checks ${PR:-<current branch>}"
    BUILD="${URL##*/}"
    ;;
https://*)
    BUILD=$(echo "$1" | sed -nE 's#.*/builds/([0-9]+).*#\1#p')
    JOB=$(echo "$1" | grep -oE "#${UUID_RE}" | head -n 1 | cut -c2- || true)
    SID=$(echo "$1" | grep -oE "[?&]sid=${UUID_RE}" | head -n 1 | sed 's/.*sid=//' || true)
    OUT="${2:-}"
    [ -n "$BUILD" ] || die "Could not parse build number from: $1"
    ;;
[0-9]*)
    [ $# -ge 2 ] || usage
    BUILD="$1"
    JOB="$2"
    OUT="${3:-}"
    ;;
*)
    usage
    ;;
esac

COOKIES=$(mktemp)
JOBS_TSV=$(mktemp)
trap 'rm -f "$COOKIES" "$JOBS_TSV"' EXIT

# Buildkite issues a session cookie on first hit; later requests need it.
curl -fsSL -c "$COOKIES" -A "$UA" \
    "https://buildkite.com/${ORG}/${PIPELINE}/builds/${BUILD}" -o /dev/null

# The build's job list (id, step uuid, state, name) is served as JSON from
# the user-facing /data/jobs endpoint. Flatten it to TSV for easy filtering:
#   job_id  step_uuid  failed  soft_failed  finished  slug  name
curl -fsSL -b "$COOKIES" -A "$UA" \
    "https://buildkite.com/${ORG}/${PIPELINE}/builds/${BUILD}/data/jobs" |
    python3 -c '
import json, re, sys

data = json.load(sys.stdin)
if data.get("has_next_page"):
    print("warning: job list is paginated; some jobs not shown", file=sys.stderr)
for r in data["records"]:
    if r.get("type") != "script":
        continue
    name = (r.get("name") or "").replace("\t", " ").replace("\n", " ")
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60]
    print("\t".join([
        r["id"],
        r.get("step_uuid") or "",
        str(r.get("passed") is False),
        str(bool(r.get("soft_failed"))),
        str(bool(r.get("finished_at"))),
        slug,
        name,
    ]))
' >"$JOBS_TSV" || die "Could not list jobs for build ${BUILD}"

if [ -n "$SID" ] && [ -z "$JOB" ]; then
    # The ?sid= in builds/<N>/list URLs is the *step* uuid, not the job uuid.
    JOB=$(awk -F'\t' -v s="$SID" '$1 == s || $2 == s {print $1; exit}' "$JOBS_TSV")
    [ -n "$JOB" ] || die "No job matching sid=${SID} in build ${BUILD}"
fi

fetch_job() { # <job_uuid> <output_file>
    curl -fsSL -b "$COOKIES" -A "$UA" \
        "https://buildkite.com/organizations/${ORG}/pipelines/${PIPELINE}/builds/${BUILD}/jobs/$1/download" \
        -o "$2"
    bash "$(dirname "$0")/ci-clean-log.sh" "$2"
}

if [ -n "$JOB" ]; then
    # Single-job mode.
    NAME=$(awk -F'\t' -v j="$JOB" '$1 == j {print $7; exit}' "$JOBS_TSV")
    SLUG=$(awk -F'\t' -v j="$JOB" '$1 == j {print $6; exit}' "$JOBS_TSV")
    [ -n "$OUT" ] || OUT="ci-${BUILD}-${SLUG:-${JOB:0:13}}.log"
    if [ "$OUT" = "-" ]; then
        TMP=$(mktemp)
        fetch_job "$JOB" "$TMP"
        cat "$TMP"
        rm -f "$TMP"
        exit 0
    fi
    if [ -e "$OUT" ] && [ -z "${CI_FETCH_LOG_FORCE:-}" ]; then
        die "Refusing to overwrite existing ${OUT} (set CI_FETCH_LOG_FORCE=1 or pass an output path)."
    fi
    fetch_job "$JOB" "$OUT"
    printf '%s\t%s\n' "$OUT" "${NAME:-$JOB}"
    exit 0
fi

# Build-wide mode: fetch finished jobs matching $SCOPE.
[ -z "$OUT" ] || die "[output_file] is only valid when fetching a single job."

case "$SCOPE" in
failed) FILTER='$3 == "True" && $4 == "False" && $5 == "True"' ;;
soft) FILTER='$3 == "True" && $5 == "True"' ;;
all) FILTER='$5 == "True"' ;;
esac

if [ "$SCOPE" = "failed" ]; then
    SOFT=$(awk -F'\t' '$3 == "True" && $4 == "True"' "$JOBS_TSV" | wc -l)
    [ "$SOFT" -eq 0 ] || echo "Skipping ${SOFT} soft-failed job(s); use --soft to include them." >&2
fi

FOUND=0
EMITTED=" "
while IFS=$'\t' read -r job_id _ _ _ _ slug name; do
    FOUND=$((FOUND + 1))
    out="ci-${BUILD}-${slug:-${job_id:0:13}}.log"
    # Retries share a name with the original job; disambiguate by uuid.
    case "$EMITTED" in
    *" $out "*) out="ci-${BUILD}-${slug:-job}-${job_id:0:13}.log" ;;
    esac
    EMITTED="${EMITTED}${out} "
    if [ -e "$out" ] && [ -z "${CI_FETCH_LOG_FORCE:-}" ]; then
        echo "Keeping existing ${out} (set CI_FETCH_LOG_FORCE=1 to refetch)." >&2
    elif ! fetch_job "$job_id" "$out"; then
        echo "Failed to download log for job ${job_id} (${name})." >&2
        continue
    fi
    printf '%s\t%s\n' "$out" "$name"
done < <(awk -F'\t' "$FILTER" "$JOBS_TSV")

if [ "$FOUND" -eq 0 ]; then
    echo "No matching jobs in build ${BUILD} (scope: ${SCOPE})." >&2
fi
