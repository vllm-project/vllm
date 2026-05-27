#!/bin/bash
# Thin shim — delegates to the Python implementation.
# Keeps the same interface so Buildkite pipeline YAML needs no changes:
#
#   VLLM_TEST_COMMANDS='...' bash run-amd-test.sh   (preferred)
#   bash run-amd-test.sh "commands here"             (legacy)
#
exec python3 "$(dirname "$0")/run-amd-test.py" "$@"
