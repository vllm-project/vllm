# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import LLM
from vllm.model_executor.offloader import (
    NoopOffloader,
    UVAOffloader,
    get_offloader,
    set_offloader,
)

from ..utils import compare_two_settings


@pytest.mark.parametrize("disable_pin_memory", [False, True])
@pytest.mark.parametrize("disable_uva", [False, True])
def test_cpu_offload(disable_pin_memory, disable_uva):
    env_vars = {
        "VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY": str(int(disable_pin_memory)),
        "VLLM_WEIGHT_OFFLOADING_DISABLE_UVA": str(int(disable_uva)),
    }

    args = ["--cpu-offload-gb", "1"]

    # cuda graph only works with UVA offloading
    if disable_uva:
        args.append("--enforce-eager")

    compare_two_settings(
        model="hmellor/tiny-random-LlamaForCausalLM",
        arg1=[],
        arg2=args,
        env1=None,
        env2=env_vars,
    )


@pytest.mark.parametrize("use_v2_model_runner", ["0", "1"])
def test_cpu_offload_is_applied(monkeypatch, use_v2_model_runner):
    """--cpu-offload-gb must actually offload weights on both model runners.

    The output comparison above cannot catch an offloader that is never
    installed, since not offloading produces identical outputs.
    """
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", use_v2_model_runner)

    # The offloader is a process-global, so reset it to the default first.
    set_offloader(NoopOffloader())

    LLM(
        model="hmellor/tiny-random-LlamaForCausalLM",
        cpu_offload_gb=1,
        max_model_len=128,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
    )

    offloader = get_offloader()
    assert isinstance(offloader, UVAOffloader)
    assert offloader.cpu_offload_bytes > 0
