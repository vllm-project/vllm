#!/bin/bash
pip install -q torch --index-url https://download.pytorch.org/whl/cpu 2>/dev/null
pip install -q -r requirements/common.txt prometheus_client pytest tblib 2>/dev/null
python -m pytest tests/v1/worker/test_gpu_model_runner.py::test_config_for_kv_group_uses_draft_heads_for_eagle_groups 2>&1 | tail -45
