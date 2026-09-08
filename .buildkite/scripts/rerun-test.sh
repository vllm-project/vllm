#!/bin/bash

# Usage: ./rerun_test.sh path/to/test.py::test_name

# Check if argument is given
if [ $# -lt 1 ]; then
    echo "Usage: $0 path/to/test.py::test_name"
    echo "Example: $0 tests/v1/engine/test_engine_core_client.py::test_kv_cache_events[True-tcp]"
    exit 1
fi

TEST=$1
COUNT=1

while pytest -sv "$TEST"; do
    COUNT=$((COUNT + 1))
    echo "RUN NUMBER ${COUNT}"
done
