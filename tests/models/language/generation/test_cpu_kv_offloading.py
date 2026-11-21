# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform

pytestmark = pytest.mark.cpu_model

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def test_cpu_rejects_native_kv_offloading() -> None:
    # Before the fix, this was accepted and startup failed only after the
    # weights were loaded, when the offloading worker was created.
    engine_args = EngineArgs(MODEL, max_model_len=512, kv_offloading_size=0.5)
    with pytest.raises(NotImplementedError, match="--kv-offloading-size"):
        engine_args.create_engine_config()
