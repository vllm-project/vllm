#!/bin/bash
RANK=0 python3 test_lookup_buffer.py &
PID0=$!
RANK=1 python3 test_lookup_buffer.py &
PID1=$!

wait $PID0
wait $PID1
