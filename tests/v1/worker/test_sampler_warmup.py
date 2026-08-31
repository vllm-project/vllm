# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.worker.gpu.warmup import warmup_kernels


def _make_runner(num_speculative_steps: int) -> SimpleNamespace:
    cache_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    return SimpleNamespace(
        num_speculative_steps=num_speculative_steps,
        decode_query_len=num_speculative_steps + 1,
        is_pooling_model=False,
        is_encoder_decoder=False,
        is_last_pp_rank=True,
        max_num_reqs=2,
        max_model_len=1024,
        model_config=SimpleNamespace(get_vocab_size=lambda: 64),
        model_state=SimpleNamespace(max_encoder_len=0),
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=128),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[KVCacheGroupSpec(["layer"], cache_spec)],
            num_blocks=1024,
        ),
        vllm_config=SimpleNamespace(num_lookahead_tokens=0, is_mm_encoder_only=False),
        kv_block_zeroer=None,
        kv_connector=SimpleNamespace(set_disabled=lambda disabled: None),
    )


@pytest.mark.parametrize("num_speculative_steps", [0, 2])
def test_warmup_kernels_runs_sampler_configs_separately(num_speculative_steps: int):
    runner = _make_runner(num_speculative_steps)
    sampling_temperatures: list[float] = []

    def execute_model(scheduler_output):
        sampling_temperatures.extend(
            req.sampling_params.temperature
            for req in scheduler_output.scheduled_new_reqs
        )

    def sample_tokens(_grammar_output):
        return None

    with patch.object(torch.accelerator, "synchronize", return_value=None):
        warmup_kernels(runner, execute_model, sample_tokens)

    assert sampling_temperatures == [0.9, 0.9, 0.0, 0.0]
