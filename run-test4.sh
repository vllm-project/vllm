#!/bin/bash
python -m pytest tests/v1/worker/test_gpu_model_runner.py::test_config_for_kv_group_uses_draft_heads_for_eagle_groups 2>&1 | grep -B30 'RuntimeError' | head -45
