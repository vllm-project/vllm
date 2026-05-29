#!/bin/bash
# Usage: ./ci-fetch-log.sh <buildkite_job_url> [output_file]
#        ./ci-fetch-log.sh <build_number> <job_uuid> [output_file]
#
# Downloads the raw log for a Buildkite job from the public, unauthenticated
# /organizations/<org>/pipelines/<pipeline>/builds/<n>/jobs/<uuid>/download
# endpoint, then strips ANSI/timestamps via ci-clean-log.sh.
#
# Find <build_number> and <job_uuid> via:
#   gh pr checks <PR> --repo vllm-project/vllm
# Each failing row's URL is .../builds/<build_number>#<job_uuid>.
#
# Default output path: ci-<build>-<uuid_first_13_chars>.log (e.g.
# ci-68478-019e6b07-daae.log). Jobs in the same build share the UUID's
# first 8 chars, so the second segment is needed for uniqueness when
# fetching multiple jobs in parallel. The script refuses to overwrite an
# existing output file; pass an explicit path or set CI_FETCH_LOG_FORCE=1
# to override.

set -euo pipefail

ORG="vllm"
PIPELINE="ci"

usage() {
    echo "Usage: $0 <buildkite_job_url> [output_file]"
    echo "       $0 <build_number> <job_uuid> [output_file]"
    exit 1
}

if [ $# -lt 1 ]; then usage; fi

if [[ "$1" == https://* ]]; then
    BUILD=$(echo "$1" | sed -nE 's#.*/builds/([0-9]+).*#\1#p')
    JOB=$(echo "$1" | grep -oE '[0-9a-f]{8}-[0-9a-f-]+' | head -n 1)
    OUT="${2:-}"
else
    if [ $# -lt 2 ]; then usage; fi
    BUILD="$1"
    JOB="$2"
    OUT="${3:-}"
fi

if [ -z "$BUILD" ] || [ -z "$JOB" ]; then
    echo "Could not parse build number or job UUID from: $1" >&2
    usage
fi

# Jobs in the same build share the UUID's first segment, so include the
# second segment (chars 9-13, e.g. "019e6b07-daae") to keep default filenames
# unique when fetching multiple jobs from one build in parallel.
if [ -z "$OUT" ]; then
    OUT="ci-${BUILD}-${JOB:0:13}.log"
fi

if [ -e "$OUT" ] && [ -z "${CI_FETCH_LOG_FORCE:-}" ]; then
    echo "Refusing to overwrite existing $OUT (set CI_FETCH_LOG_FORCE=1 or pass an explicit output path)." >&2
    exit 1
fi

COOKIES=$(mktemp)
trap 'rm -f "$COOKIES"' EXIT

# Buildkite issues a session cookie on first hit; subsequent /download needs it.
curl -fsSL -c "$COOKIES" -A "vllm-ci-fetch-log" \
    "https://buildkite.com/${ORG}/${PIPELINE}/builds/${BUILD}" -o /dev/null

curl -fsSL -b "$COOKIES" -A "vllm-ci-fetch-log" \
    "https://buildkite.com/organizations/${ORG}/pipelines/${PIPELINE}/builds/${BUILD}/jobs/${JOB}/download" \
    -o "$OUT"

bash "$(dirname "$0")/ci-clean-log.sh" "$OUT"

echo "$OUT"
