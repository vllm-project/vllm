#!/bin/bash
set -e
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q -r requirements/common.txt prometheus_client pytest tblib
python -c "import pytest; print('pytest ok')"
python -m pytest tests/v1/worker/test_gpu_model_runner.py -k config_for_kv_group -q 2>&1 | tail -5
