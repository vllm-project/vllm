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


@pytest.mark.parametrize(
    ("backend", "use_replayssm", "expected"),
    [
        (MambaBackendEnum.FLASHINFER, True, True),
        (MambaBackendEnum.TRITON, True, False),
        (MambaBackendEnum.FLASHINFER, False, False),
    ],
)
def test_replayssm_autotune_decode_kwargs(backend, use_replayssm, expected):
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(use_replayssm=use_replayssm),
            mamba_config=SimpleNamespace(backend=backend),
        ),
        uniform_decode_query_len=6,
        max_num_tokens=100,
        scheduler_config=SimpleNamespace(max_num_seqs=32),
    )
    prefill_kwargs = {
        "num_tokens": 128,
        "skip_eplb": True,
        "is_profile": True,
        "randomize_inputs": True,
    }

    result = warmup._flashinfer_replayssm_autotune_kwargs(runner, prefill_kwargs)

    if not expected:
        assert result is None
        return
    assert result == (
        16,
        {
            **prefill_kwargs,
            "num_tokens": 96,
            "uniform_decode": True,
            "allow_microbatching": False,
            "force_attention": True,
            "profile_seq_lens": 7,
        },
    )


def test_replayssm_autotune_slots_restore_state_and_trackers():
    mixer = MambaMixer2.__new__(MambaMixer2)
    torch.nn.Module.__init__(mixer)
    mixer.use_replayssm = True
    mixer.kv_cache = (
        torch.full((4, 2), 3.0),
        torch.full((4, 2), 3.0),
    )
    mixer._replayssm_ring_start = torch.full((4,), 3, dtype=torch.int32)
    mixer._replayssm_prev_num_accepted = torch.full((4,), 3, dtype=torch.int32)

    block_ids = np.arange(10, 14, dtype=np.int32).reshape(4, 1)
    original_block_ids = block_ids.copy()
    block_table = SimpleNamespace(block_table=SimpleNamespace(np=block_ids))
    multi_group_block_table = SimpleNamespace(
        block_tables=[block_table], commit_block_table=Mock()
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(block_table=multi_group_block_table),
        get_model=lambda: SimpleNamespace(modules=lambda: (mixer,)),
    )

    with warmup._temporary_replayssm_autotune_slots(runner, 2):
        assert block_ids[:2, 0].tolist() == [1, 2]
        for tensor in (
            *mixer.kv_cache,
            mixer._replayssm_ring_start,
            mixer._replayssm_prev_num_accepted,
        ):
            tensor[1:3].fill_(9)

    assert np.array_equal(block_ids, original_block_ids)
    multi_group_block_table.commit_block_table.assert_called_once_with(2)
    for tensor in (
        *mixer.kv_cache,
        mixer._replayssm_ring_start,
        mixer._replayssm_prev_num_accepted,
    ):
        assert torch.count_nonzero(tensor[1:3]) == 0
        assert torch.all(tensor[0] == 3)
        assert torch.all(tensor[3] == 3)
