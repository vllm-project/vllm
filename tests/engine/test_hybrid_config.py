# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import PropertyMock, patch

from vllm.engine.arg_utils import EngineArgs


def test_hybrid_batch_size_override():
    """
    Verify that max_num_batched_tokens is safely scaled down
    when a hybrid architecture (GDN/Mamba) is detected and relying on defaults.
    """
    # 1. User starts the engine WITHOUT explicitly setting the batch size.
    # It will attempt to default to 8192.
    args = EngineArgs(
        model="facebook/opt-125m",
    )

    # 2. Trick the engine into thinking this is a hybrid model.
    with patch(
        "vllm.config.ModelConfig.is_hybrid", new_callable=PropertyMock
    ) as mock_is_hybrid:
        mock_is_hybrid.return_value = True

        config = args.create_engine_config()

        # 3. Assert that the hybrid logic intercepted the default and capped it to 2048.
        assert config.scheduler_config.max_num_batched_tokens == 2048
