#!/bin/bash
RANK=0 python3 test_send_recv.py &
RANK=1 python3 test_send_recv.py &