# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The dummy batch of an idle DP rank must not drop the loaded adapters."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.config.lora import LoRAConfig
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

MAX_LORAS = 2


def _make_runner(flag_during_forward: list[bool]) -> GPUModelRunner:
    """A runner stripped down to what _dummy_run() touches."""
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.eplb = SimpleNamespace(step=lambda **kwargs: None)
    runner.max_num_reqs = 4
    runner.decode_query_len = 1
    runner.max_num_tokens = 64
    runner.speculator = None
    runner.kv_connector = SimpleNamespace(set_disabled=lambda disabled: None)
    runner.is_first_pp_rank = True
    runner.is_last_pp_rank = False
    runner.lora_config = LoRAConfig(max_loras=MAX_LORAS, max_lora_rank=8)
    runner.lora_manager = MagicMock()
    runner._idle_dp_dummy_run = False
    runner.execute_model = lambda *args, **kwargs: flag_during_forward.append(
        runner._idle_dp_dummy_run
    )
    return runner


@pytest.mark.parametrize("idle_dp_rank", [False, True])
def test_dummy_run_keeps_adapters_on_an_idle_dp_rank(idle_dp_rank: bool):
    flag_during_forward: list[bool] = []
    runner = _make_runner(flag_during_forward)

    runner._dummy_run(4, uniform_decode=True, idle_dp_rank=idle_dp_rank)

    # The flag is what the cudagraph dispatch reads to pick the no-LoRA case,
    # and it must not leak past the dummy run.
    assert flag_during_forward == [idle_dp_rank]
    assert runner._idle_dp_dummy_run is False

    calls = runner.lora_manager.set_active_adapters.call_args_list
    assert len(calls) == 1
    lora_requests, lora_mapping = calls[0].args
    if idle_dp_rank:
        # No dummy adapter is built for the batch, and - the point of the
        # branch - the adapters this rank has loaded survive the step.
        assert lora_requests == set()
        assert set(lora_mapping.index_mapping) == {0}
        assert not runner.lora_manager.remove_all_adapters.called
    else:
        # Warmup and capture still exercise every slot and clean up after.
        assert len(lora_requests) == MAX_LORAS
        assert runner.lora_manager.remove_all_adapters.called
