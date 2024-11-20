#!/bin/bash
RANK=0 python test_send_recv.py &
RANK=1 python test_send_recv.py &