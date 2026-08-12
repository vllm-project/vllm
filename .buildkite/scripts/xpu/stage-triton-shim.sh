#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

readonly BUCKET="vllm-wheels"
readonly WHEEL_FILENAME="triton-3.7.2+xpu-py3-none-any.whl"
readonly COMMIT="${BUILDKITE_COMMIT:-}"
readonly DRY_RUN="${DRY_RUN:-0}"
readonly SOURCE="s3://${BUCKET}/xpu/${WHEEL_FILENAME}"
readonly DESTINATION="s3://${BUCKET}/${COMMIT}/${WHEEL_FILENAME}"

if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1" >&2
    exit 2
fi
if [[ ! "$COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "BUILDKITE_COMMIT must be a full lowercase commit hash" >&2
    exit 2
fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Would copy $SOURCE to $DESTINATION"
else
    aws s3 cp "$SOURCE" "$DESTINATION"
    echo "Staged XPU Triton shim for https://wheels.vllm.ai/$COMMIT/xpu/"
fi