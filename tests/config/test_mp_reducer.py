# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from unittest.mock import patch

from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM


def test_mp_reducer():
    """
    Test that _reduce_config reducer is registered when AsyncLLM is instantiated
    without transformers_modules. This is a regression test for
    https://github.com/vllm-project/vllm/pull/18640.
    """

    # Ensure transformers_modules is not in sys.modules
    if "transformers_modules" in sys.modules:
        del sys.modules["transformers_modules"]

    with patch("multiprocessing.reducer.register") as mock_register:
        engine_args = AsyncEngineArgs(
            model="facebook/opt-125m",
            max_model_len=32,
            gpu_memory_utilization=0.1,
            disable_log_stats=True,
        )

        async_llm = AsyncLLM.from_engine_args(
            engine_args,
            start_engine_loop=False,
        )

        assert (
            mock_register.called
        ), "multiprocessing.reducer.register should have been called"

        vllm_config_registered = False
        for call_args in mock_register.call_args_list:
            # Verify that a reducer for VllmConfig was registered
            if len(call_args[0]) >= 2 and call_args[0][0] == VllmConfig:
                vllm_config_registered = True

                reducer_func = call_args[0][1]
                assert callable(reducer_func), "Reducer function should be callable"
                break

        assert (
            vllm_config_registered
        ), "VllmConfig should have been registered to multiprocessing.reducer"

        async_llm.shutdown()
