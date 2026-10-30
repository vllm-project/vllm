#!/bin/bash
# Usage: ./ci_clean_log.sh ci.log
# This script strips timestamps and color codes from CI log files.

# Check if argument is given
if [ $# -lt 1 ]; then
    echo "Usage: $0 ci.log"
    exit 1
fi

INPUT_FILE="$1"

# GNU sed uses "-i [-r]" while BSD/macOS sed needs "-i '' [-E]"; detect once.
if sed --version >/dev/null 2>&1; then
    sedi() { sed -i "$@"; }
    SED_ERE=-r
else
    sedi() { sed -i '' "$@"; }
    SED_ERE=-E
fi

# Strip timestamps
sedi 's/^\[[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}T[0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}Z\] //' "$INPUT_FILE"

# Strip Buildkite inline timestamp markers (ESC _bk;t=<ms> BEL)
sedi 's/\x1B_bk;t=[0-9]*\x07//g' "$INPUT_FILE"

# Strip colorization
sedi "$SED_ERE" 's/\x1B\[[0-9;]*[mK]//g' "$INPUT_FILE"
