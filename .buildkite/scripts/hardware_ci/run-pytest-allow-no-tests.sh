#!/bin/bash

set -uo pipefail

"$@"
exit_code=$?

if [ "$exit_code" -eq 0 ] || [ "$exit_code" -eq 5 ]; then
    exit 0
fi

exit "$exit_code"
