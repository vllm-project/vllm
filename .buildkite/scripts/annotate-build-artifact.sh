#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Append a build artifact line to the Buildkite annotation.
# Usage: annotate-build-artifact.sh <label> <value> <context>
set -e
echo "- **${1}**: \`${2}\`" | \
  buildkite-agent annotate --append --style 'info' \
    --context "${3:?context is required}"
