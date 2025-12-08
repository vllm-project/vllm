#!/bin/bash
set -e

# Standard-supervisor from github.com/aws/model-hosting-container-standards provides features like:
# - Process supervision and automatic restart
# - ENV variable to CLI argument conversion (VLLM_ARG_* -> --arg)
# - Graceful shutdown handling
#
# Control standard-supervisor usage with VLLM_USE_STANDARD_SUPERVISOR:
# - "false" (default): Direct execution without standard-supervisor
# - "true": Enable all standard-supervisor features
# - Future: Could support selective feature flags like "env-conversion-only"

VLLM_USE_STANDARD_SUPERVISOR="${VLLM_USE_STANDARD_SUPERVISOR:-false}"

if [ "$VLLM_USE_STANDARD_SUPERVISOR" = "true" ]; then
    # Use standard-supervisor launcher with all features
    exec standard-supervisor vllm serve "$@"
else
    # Direct execution without standard-supervisor
    exec vllm serve "$@"
fi
