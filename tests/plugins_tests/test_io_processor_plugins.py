# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest

from vllm import AsyncEngineArgs
from vllm.config import VllmConfig
from vllm.plugins.io_processors import get_io_processor
from vllm.v1.engine.async_llm import AsyncLLM


def test_loading_missing_plugin():
    vllm_config = VllmConfig()
    with pytest.raises(ValueError):
        get_io_processor(vllm_config, "plugin")


@pytest.mark.asyncio
def test_loading_engine_with_wrong_plugin():

    os.environ['VLLM_USE_V1'] = '1'

    engine_args = AsyncEngineArgs(
        model="christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM",
        enforce_eager=True,
        skip_tokenizer_init=True,
        io_processor_plugin="plugin")

    with pytest.raises(ValueError):
        AsyncLLM.from_engine_args(engine_args)
