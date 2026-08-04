#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Quick-start script for the lmc_external_l2_adapter
# example plugin.
#
# Usage:
#   cd examples/lmc_external_l2_adapter
#   bash scripts/install_and_test.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Step 1: Install the plugin in editable mode ==="
pip install -e "$PROJECT_DIR" --quiet

echo ""
echo "=== Step 2: Verify the module is importable ==="
python -c "
from lmc_external_l2_adapter import InMemoryL2Adapter
print('  OK: InMemoryL2Adapter imported successfully')
print('  Class:', InMemoryL2Adapter)
"

echo ""
echo "=== Step 3: Verify LMCache recognises 'external' type ==="
python -c "
from lmcache.v1.distributed.l2_adapters.config import (
    get_registered_l2_adapter_types,
)
types = get_registered_l2_adapter_types()
assert 'plugin' in types, f'plugin not in {types}'
print('  OK: registered adapter types:', types)
"

echo ""
echo "=== Step 4: Run the plugin unit tests ==="
python -m pytest "$PROJECT_DIR/tests/" -v

echo ""
echo "=== All checks passed! ==="
echo ""
echo "To use this adapter with LMCache, pass:"
echo '  --l2-adapter '"'"'{"type":"plugin","module_path":"lmc_external_l2_adapter","class_name":"InMemoryL2Adapter","adapter_params":{"max_size_gb":1.0,"mock_bandwidth_gb":20.0}}'"'"''
