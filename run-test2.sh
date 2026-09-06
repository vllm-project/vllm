#!/bin/bash
python -m pytest tests/v1/worker/test_gpu_model_runner.py -k config_for_kv_group 2>&1 | grep -B5 -A25 'FAILED\|ERROR at' | head -60
