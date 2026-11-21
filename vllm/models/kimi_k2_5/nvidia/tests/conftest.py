# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest


@pytest.fixture
def default_vllm_config():
    """Set a default VllmConfig for tests that directly test CustomOps or pathways
    that use get_current_vllm_config() outside of a full engine context.

    Local copy of the top-level ``tests/conftest.py`` fixture so these kernel
    tests stay self-contained under the ``vllm`` package tree.
    """
    from vllm.config import VllmConfig, set_current_vllm_config

    config = VllmConfig()
    with set_current_vllm_config(config):
        yield config
