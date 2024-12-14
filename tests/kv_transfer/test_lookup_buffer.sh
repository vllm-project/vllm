#!/bin/bash
RANK=0 python test_lookup_buffer.py &
RANK=1 python test_lookup_buffer.py &