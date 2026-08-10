#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Append the Docker image tag(s) an image-build step pushed to a Buildkite
# annotation, so the built image tags show up on the build page instead of
# being buried in the job logs.
#
# Usage: annotate-image-build.sh <image_tag> [<image_tag> ...]
set -euo pipefail

# buildkite-agent only exists on Buildkite agents; no-op elsewhere so the
# image build scripts stay runnable locally.
if ! command -v buildkite-agent >/dev/null 2>&1; then
    echo "buildkite-agent not found; skipping image tag annotation"
    exit 0
fi

label="${BUILDKITE_LABEL:-Image build}"
content=""
for image in "$@"; do
    [[ -n "$image" ]] || continue
    content+="- **${label}**: \`${image}\`"$'\n'
done

if [[ -z "$content" ]]; then
    echo "No image tags provided; nothing to annotate"
    exit 0
fi

# Best-effort: a flaky annotation must never fail an otherwise successful
# (and expensive) image build.
if ! printf '%s' "$content" | \
    buildkite-agent annotate --append --style 'info' --context 'docker-images'; then
    echo "warning: failed to annotate build with image tags"
fi
