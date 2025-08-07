#!/bin/bash
# Usage: ./ci_clean_log.sh ci.log
# This script strips timestamps and color codes from CI log files.

# Check if argument is given
if [ $# -lt 1 ]; then
    echo "Usage: $0 ci.log"
    exit 1
fi

INPUT_FILE="$1"

# Strip timestamps
sed -i 's/^\[[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}T[0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}Z\] //' "$INPUT_FILE"

# Strip colorization
sed -i -r 's/\x1B\[[0-9;]*[mK]//g' "$INPUT_FILE"
