#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

urls=$(sed -n 's/^triton @ \([^ ;]*\).*/\1/p' requirements/cpu.txt)
if [[ -z "$urls" ]]; then
    echo "No Triton CPU wheels found in requirements/cpu.txt" >&2
    exit 1
fi

work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT

while IFS= read -r url; do
    digest="${url##*#sha256=}"
    if [[ ! "$digest" =~ ^[0-9a-f]{64}$ ]]; then
        echo "Triton CPU wheel must have a SHA256 pin: $url" >&2
        exit 1
    fi
    filename="${url%%#*}"
    filename="${filename##*/}"
    curl --fail --location --retry 3 --output "$work_dir/$filename" "$url"
    printf '%s  %s\n' "$digest" "$work_dir/$filename" | sha256sum --check -
done <<< "$urls"

# Verify every supported Python wheel before publishing any of them.
aws s3 cp --recursive "$work_dir/" "s3://vllm-wheels/${BUILDKITE_COMMIT:?}/"
