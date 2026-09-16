# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.warmup import replayssm_warmup as warmup
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not has_flashinfer(),
    reason="FlashInfer ReplaySSM warmup tests require CUDA and FlashInfer",
)

PREFILL_KWARGS = {
    "num_tokens": 128,
    "skip_eplb": True,
    "is_profile": True,
    "randomize_inputs": True,
}


def _autotune_runner(
    *,
    use_v2_model_runner: bool = False,
    query_len: int = 6,
    max_num_seqs: int = 32,
    num_blocks: int = 17,
    max_num_tokens: int = 100,
    use_replayssm: bool = True,
    backend: MambaBackendEnum = MambaBackendEnum.FLASHINFER,
) -> SimpleNamespace:
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(use_replayssm=use_replayssm),
            mamba_config=SimpleNamespace(backend=backend),
            use_v2_model_runner=use_v2_model_runner,
        ),
        uniform_decode_query_len=query_len,
        decode_query_len=query_len,
        max_num_tokens=max_num_tokens,
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        kv_cache_config=SimpleNamespace(num_blocks=num_blocks),
    )


@pytest.mark.parametrize(
    ("runner_kwargs", "expected_num_reqs"),
    [
        # max_num_seqs (32) vs max_num_tokens // query_len (16) vs blocks-1 (16).
        (dict(query_len=6, use_v2_model_runner=False), 16),
        (dict(query_len=6, use_v2_model_runner=True), 16),
        # num_blocks - 1 is the binding constraint.
        (dict(query_len=1, max_num_tokens=128, max_num_seqs=64, num_blocks=5), 4),
    ],
    ids=["v1", "v2", "clamped_to_state_capacity"],
)
def test_replayssm_autotune_decode_kwargs(runner_kwargs, expected_num_reqs):
    query_len = runner_kwargs["query_len"]
    with patch.object(
        warmup, "flashinfer_replayssm_autotune_supported", return_value=True
    ):
        result = warmup._replayssm_autotune_kwargs(_autotune_runner(**runner_kwargs))

    expected_kwargs = {
        **PREFILL_KWARGS,
        "num_tokens": expected_num_reqs * query_len,
        "uniform_decode": True,
    }
    if runner_kwargs.get("use_v2_model_runner"):
        expected_kwargs["valid_dummy_state_slots"] = True
    else:
        expected_kwargs.update(
            allow_microbatching=False,
            force_attention=True,
            profile_seq_lens=query_len + 1,
        )
    assert result == (expected_num_reqs, expected_kwargs)


@pytest.mark.parametrize(
    ("runner_kwargs", "flashinfer_supported"),
    [
        (dict(use_replayssm=False), True),
        (dict(backend=MambaBackendEnum.TRITON), True),
        ({}, False),
    ],
    ids=["replayssm_disabled", "non_flashinfer_backend", "kernel_unavailable"],
)
def test_replayssm_autotune_kwargs_skipped(runner_kwargs, flashinfer_supported):
    with patch.object(
        warmup,
        "flashinfer_replayssm_autotune_supported",
        return_value=flashinfer_supported,
    ):
        result = warmup._replayssm_autotune_kwargs(_autotune_runner(**runner_kwargs))
    assert result is None


def test_replayssm_autotune_slots_restore_state_and_trackers():
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
    tracked = (
        *mixer.kv_cache,
        mixer._replayssm_ring_start,
        mixer._replayssm_prev_num_accepted,
    )

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
        for tensor in tracked:
            tensor[1:3].fill_(9)

    assert np.array_equal(block_ids, original_block_ids)
    assert multi_group_block_table.commit_block_table.call_args_list == [
        ((2,), {}),
        ((2,), {}),
    ]
    for tensor in tracked:
        assert torch.count_nonzero(tensor[1:3]) == 0
        assert torch.all(tensor[0] == 3)
        assert torch.all(tensor[3] == 3)
