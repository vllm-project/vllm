# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.warmup import kernel_warmup as warmup


def _replayssm_mixer() -> MambaMixer2:
    mixer = MambaMixer2.__new__(MambaMixer2)
    torch.nn.Module.__init__(mixer)
    mixer.use_replayssm = True
    mixer.replayssm_buffer_len = 16
    mixer.kv_cache = (
        torch.full((4, 2), 3.0),
        torch.full((4, 2), 3.0),
        *(torch.full((4, 2, 17), 3.0) for _ in range(3)),
    )
    mixer._replayssm_ring_start = torch.full((4,), 3, dtype=torch.int32)
    mixer._replayssm_prev_num_accepted = torch.full((4,), 3, dtype=torch.int32)
    return mixer


@pytest.mark.parametrize("use_v2_model_runner", [False, True])
def test_replayssm_autotune_decode_kwargs(use_v2_model_runner):
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(use_replayssm=True),
            mamba_config=SimpleNamespace(backend=MambaBackendEnum.FLASHINFER),
            use_v2_model_runner=use_v2_model_runner,
        ),
        uniform_decode_query_len=6,
        decode_query_len=6,
        max_num_tokens=100,
        scheduler_config=SimpleNamespace(max_num_seqs=32),
        kv_cache_config=SimpleNamespace(num_blocks=17),
    )
    prefill_kwargs = {
        "num_tokens": 128,
        "skip_eplb": True,
        "is_profile": True,
        "randomize_inputs": True,
    }

    result = warmup._replayssm_autotune_kwargs(runner, prefill_kwargs)

    expected_kwargs = {
        **prefill_kwargs,
        "num_tokens": 96,
        "uniform_decode": True,
    }
    if use_v2_model_runner:
        expected_kwargs["valid_dummy_state_slots"] = True
    else:
        expected_kwargs.update(
            allow_microbatching=False,
            force_attention=True,
            profile_seq_lens=7,
        )
    assert result == (16, expected_kwargs)


def test_replayssm_autotune_decode_kwargs_clamps_to_state_capacity():
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(use_replayssm=True),
            mamba_config=SimpleNamespace(backend=MambaBackendEnum.FLASHINFER),
            use_v2_model_runner=False,
        ),
        uniform_decode_query_len=1,
        max_num_tokens=128,
        scheduler_config=SimpleNamespace(max_num_seqs=64),
        kv_cache_config=SimpleNamespace(num_blocks=5),
    )

    result = warmup._replayssm_autotune_kwargs(runner, {})

    assert result is not None
    assert result[0] == 4
    assert result[1]["num_tokens"] == 4


def test_replayssm_autotune_slots_restore_state_and_trackers():
    mixer = _replayssm_mixer()

    block_ids = np.arange(10, 14, dtype=np.int32).reshape(4, 1)
    original_block_ids = block_ids.copy()
    block_table = SimpleNamespace(block_table=SimpleNamespace(np=block_ids))
    multi_group_block_table = SimpleNamespace(
        block_tables=[block_table], commit_block_table=Mock()
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(use_v2_model_runner=False),
        input_batch=SimpleNamespace(block_table=multi_group_block_table),
        get_model=lambda: SimpleNamespace(modules=lambda: (mixer,)),
    )

    with warmup._temporary_replayssm_autotune_state(runner, 2):
        assert block_ids[:2, 0].tolist() == [1, 2]
        for tensor in (
            *mixer.kv_cache,
            mixer._replayssm_ring_start,
            mixer._replayssm_prev_num_accepted,
        ):
            tensor[1:3].fill_(9)

    assert np.array_equal(block_ids, original_block_ids)
    assert multi_group_block_table.commit_block_table.call_args_list == [
        ((2,), {}),
        ((2,), {}),
    ]
    for tensor in (
        *mixer.kv_cache,
        mixer._replayssm_ring_start,
        mixer._replayssm_prev_num_accepted,
    ):
        assert torch.count_nonzero(tensor[1:3]) == 0
        assert torch.all(tensor[0] == 3)
        assert torch.all(tensor[3] == 3)
