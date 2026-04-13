# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vllm.v1.executor.ray_env_utils."""

import os
from unittest.mock import patch

from vllm.v1.executor.ray_env_utils import get_driver_env_vars

WORKER_VARS: set[str] = {
    "CUDA_VISIBLE_DEVICES",
    "LOCAL_RANK",
}


class TestDefaultPropagation:
    """All env vars are propagated unless explicitly excluded."""

    @patch.dict(os.environ, {"NCCL_DEBUG": "INFO"}, clear=False)
    def test_nccl_prefix(self):
        assert get_driver_env_vars(WORKER_VARS)["NCCL_DEBUG"] == "INFO"

    @patch.dict(os.environ, {"HF_TOKEN": "secret"}, clear=False)
    def test_hf_token(self):
        assert "HF_TOKEN" in get_driver_env_vars(WORKER_VARS)

    @patch.dict(os.environ, {"LMCACHE_LOCAL_CPU": "True"}, clear=False)
    def test_lmcache_prefix(self):
        assert "LMCACHE_LOCAL_CPU" in get_driver_env_vars(WORKER_VARS)

    @patch.dict(os.environ, {"PYTHONHASHSEED": "42"}, clear=False)
    def test_pythonhashseed(self):
        assert get_driver_env_vars(WORKER_VARS)["PYTHONHASHSEED"] == "42"

    @patch.dict(os.environ, {"MYLIB_FOO": "bar"}, clear=False)
    def test_arbitrary_var_propagated(self):
        assert get_driver_env_vars(WORKER_VARS)["MYLIB_FOO"] == "bar"


class TestExclusion:
    @patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=False)
    def test_worker_specific_excluded(self):
        assert "CUDA_VISIBLE_DEVICES" not in get_driver_env_vars(WORKER_VARS)

    @patch.dict(os.environ, {"LMCACHE_LOCAL_CPU": "True"}, clear=False)
    @patch(
        "vllm.v1.executor.ray_env_utils.RAY_NON_CARRY_OVER_ENV_VARS",
        {"LMCACHE_LOCAL_CPU"},
    )
    def test_non_carry_over_blacklist(self):
        assert "LMCACHE_LOCAL_CPU" not in get_driver_env_vars(WORKER_VARS)
