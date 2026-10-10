#!/bin/bash
# Install the AITER nightly wheel over this build's ci_base.
#
# Run by the aiter-nightly-amd step of ci-infra's test-template-amd.j2 when
# AITER_NIGHTLY=1, after ensure-ci-base-amd, which publishes the stock ci_base
# this step builds on. In order:
#   1. Select the newest nightly wheel that fits the stock ci_base
#      (select_aiter_nightly_wheel.py, beside this script, run inside that image)
#   2. Build a new image: pip install that wheel over the stock ci_base
#   3. Push the new image as this build's ci_base-build tag, replacing the stock
#      one, so later steps and native jobs use it; record its digest
#
# Buildkite meta-data, a key-value store shared by every job in this build:
#   reads   rocm-ci-base-image       the stock ci_base, set by ensure-ci-base-amd
#   r/w     aiter-nightly-wheel-url  the wheel step 1 chose; a retry reads it
#                                    back and installs that same wheel
#   writes  aiter-nightly-ci-base    the new image as tag@digest; image-build-amd
#                                    builds vLLM on it
#
# Exit status:
#   10  no wheel fits this ci_base
#   11  the wheel does not install
#   Anything else is unexpected and passed through as is.
#   Buildkite retries this step once on exit status 1 or 2, which it treats as
#   transient Docker/registry failures (amd_infra_retry in the template).
#   10 and 11 sit outside that range: they are real AITER failures, and a retry
#   would only fail the same way.

set -euo pipefail

SELECT_WHEEL="$(dirname "${BASH_SOURCE[0]}")/select_aiter_nightly_wheel.py"
BUILD_TAG="rocm/vllm-dev:ci_base-build-${BUILDKITE_BUILD_ID:?BUILDKITE_BUILD_ID is required}"

# Scratch space for the build context and captured errors. The trap deletes it
# however the script exits: success, fail, or set -e.
scratch="$(mktemp -d)"
trap 'rm -rf "${scratch}"' EXIT
errors="${scratch}/errors"

fail() {
    local status="$1" title="$2" detail="$3"
    echo "+++ :x: AITER nightly: ${title}"
    echo "${detail}" >&2
    # shellcheck disable=SC2016 # the backticks are a literal markdown fence
    printf ':x: **AITER nightly: %s**\n\n```\n%s\n```\n' "${title}" "${detail}" \
        | buildkite-agent annotate --style error --context aiter-nightly
    exit "${status}"
}

# Set by ensure-ci-base-amd; this step's depends_on guarantees it is there.
ci_base="$(buildkite-agent meta-data get rocm-ci-base-image)"
# Set below on the first attempt. Meta-data is per build, so a retry reuses
# that wheel, not a newer nightly.
pinned="$(buildkite-agent meta-data get aiter-nightly-wheel-url --default "")"

echo "--- :docker: Pulling ${ci_base}"
docker pull "${ci_base}"

echo "--- :mag: Selecting the AITER nightly for ${ci_base}"
set +e
# `-` reads select_aiter_nightly_wheel.py from stdin. -P stops an `aiter`
# directory in the working directory shadowing the installed package.
out="$(docker run --rm -i --entrypoint python3 "${ci_base}" \
    -P - ${pinned:+--wheel-url "${pinned}"} \
    < "${SELECT_WHEEL}" 2> "${errors}")"
status=$?
set -e
cat "${errors}" >&2
case "${status}" in
    0) ;;
    10) fail 10 "no wheel for this image" "${out}" ;;
    *) fail "${status}" "could not select a wheel" "$(tail -n 20 "${errors}")" ;;
esac
# The first line is the wheel's URL; the rest, the annotation naming it.
wheel_url="$(head -n 1 <<< "${out}")"
note="$(tail -n +2 <<< "${out}")"
buildkite-agent meta-data set aiter-nightly-wheel-url "${wheel_url}"

cat > "${scratch}/Dockerfile" <<'EOF'
ARG CI_BASE_IMAGE
FROM ${CI_BASE_IMAGE}
ARG AITER_WHEEL_URL
RUN python3 -m pip install --no-cache-dir "$AITER_WHEEL_URL"
EOF

echo "--- :docker: Installing ${wheel_url##*/}"
# --builder default: build into local Docker, where docker push below finds it.
if ! docker buildx build \
    --builder default \
    --provenance=false \
    --progress plain \
    --build-arg "CI_BASE_IMAGE=${ci_base}" \
    --build-arg "AITER_WHEEL_URL=${wheel_url}" \
    -t "${BUILD_TAG}" \
    "${scratch}"; then
    fail 11 "the wheel does not install" \
        "Installing ${wheel_url##*/} over ${ci_base} failed; see this step's log."
fi

docker push "${BUILD_TAG}"
# Handed off as tag@digest: vLLM's build-test-image.sh, which image-build-amd
# runs on this image, rejects a ci_base that is not digest-pinned.
digest="$(docker buildx imagetools inspect "${BUILD_TAG}" | awk '$1 == "Digest:" { print $2; exit }')"
if [[ ! "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "Could not resolve the pushed digest of ${BUILD_TAG}" >&2
    exit 1
fi
# image-build-amd points the ci_base handoff at this.
buildkite-agent meta-data set aiter-nightly-ci-base "${BUILD_TAG}@${digest}"
# The annotation starts with :warning: when the wheel is stale.
style=info
if [[ "${note}" == :warning:* ]]; then
    style=warning
fi
echo "${note}" | buildkite-agent annotate --style "${style}" --context aiter-nightly
echo "--- :white_check_mark: ${BUILD_TAG}@${digest}"
