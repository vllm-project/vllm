#!/bin/bash

RANK=0 python3 test_send_recv.py &
PID0=$!
RANK=1 python3 test_send_recv.py &
PID1=$!

wait $PID0
wait $PID1
