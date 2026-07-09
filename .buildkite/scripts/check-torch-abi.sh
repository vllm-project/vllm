#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Verify vLLM's compiled extensions comply with the PyTorch stable ABI, except
# for libraries explicitly listed in check-torch-abi.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo ">>> Auditing vLLM extension modules for PyTorch stable ABI compliance"
python3 "${REPO_ROOT}/.buildkite/check-torch-abi.py" "$@"
