# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import Mock

import pytest

from vllm.config import FaultToleranceConfig, ParallelConfig


@pytest.fixture
def mock_parallel_config():
    """Create mock ParallelConfig object"""
    config = Mock(spec=ParallelConfig)  # 加上 spec=ParallelConfig 更规范

    # ParallelConfig 字段
    config.data_parallel_index = 0
    config.data_parallel_size = 2
    config.data_parallel_size_local = 2
    config.local_engines_only = False

    # 你重构后的容错配置
    config.fault_tolerance_config = FaultToleranceConfig(
        enable_fault_tolerance=True, engine_recovery_timeout_sec=10
    )
    return config
