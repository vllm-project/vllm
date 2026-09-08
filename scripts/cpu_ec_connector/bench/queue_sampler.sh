#!/bin/bash
# Sample each instance's queue depth from its Prometheus endpoint.
#
# Usage: queue_sampler.sh <interval_s> <out_csv> <name>=<port> [<name>=<port> ...]
#
# Emits "epoch,name,metric,value" so a run can be sliced by wall-clock time
# afterwards. Queue depth is what distinguishes "the system is working" from
# "the system is a queue with a benchmark attached", which is exactly the
# confusion that invalidated the rate=inf measurements.
set -u
INTERVAL=$1
OUT=$2
shift 2

while true; do
    NOW=$(date +%s.%N)
    for pair in "$@"; do
        NAME=${pair%%=*}
        PORT=${pair##*=}
        curl -s --max-time 2 "http://127.0.0.1:${PORT}/metrics" 2>/dev/null \
            | grep -E '^vllm:num_requests_(running|waiting)[ {]' \
            | awk -v t="$NOW" -v n="$NAME" \
                '{split($1, f, "{"); print t "," n "," f[1] "," $NF}' \
            >> "$OUT"
    done
    sleep "$INTERVAL"
done
