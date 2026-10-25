# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from vllm.config import CacheConfig
from vllm.config.mamba import MambaBackendEnum
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.warmup import replayssm_warmup as warmup
from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheLayout, MambaSpec
from vllm.v1.worker.utils import allocate_kv_cache

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
            use_flashinfer_replayssm=(
                use_replayssm and backend == MambaBackendEnum.FLASHINFER
            ),
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
        result = warmup._replayssm_autotune_kwargs(
            _autotune_runner(**runner_kwargs), PREFILL_KWARGS
        )

    assert result is not None
    max_num_reqs, decode_kwargs = result
    assert max_num_reqs == expected_num_reqs
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
    assert decode_kwargs == expected_kwargs


@pytest.mark.parametrize(
    ("runner_kwargs", "flashinfer_supported"),
    [
        (dict(use_replayssm=False), True),
        (dict(backend=MambaBackendEnum.TRITON), True),
        ({}, False),
        (dict(num_blocks=1), True),
    ],
    ids=[
        "replayssm_disabled",
        "non_flashinfer_backend",
        "kernel_unavailable",
        "zero_non_padding_slots",
    ],
)
def test_replayssm_autotune_kwargs_skipped(runner_kwargs, flashinfer_supported):
    with patch.object(
        warmup,
        "flashinfer_replayssm_autotune_supported",
        return_value=flashinfer_supported,
    ):
        result = warmup._replayssm_autotune_kwargs(
            _autotune_runner(**runner_kwargs), PREFILL_KWARGS
        )
    assert result is None


@pytest.mark.parametrize(
    ("use_v2", "layout"),
    [(False, KVCacheLayout.LBNHC), (True, KVCacheLayout.BLHNC)],
    ids=["v1", "v2"],
)
def test_replayssm_autotune_slots_restore_state_and_trackers(use_v2, layout):
    spec = MambaSpec(
        block_size=16,
        shapes=((2,), (2,), (2, 11), (2, 11), (2, 11)),
        dtypes=(torch.float32,) * 5,
        page_size_padded=384,
        use_flashinfer_replayssm=True,
    )
    config = SimpleNamespace(
        cache_config=CacheConfig(),
        attention_config=SimpleNamespace(hisparse_config=None),
    )
    config.cache_config.kv_cache_layout = layout.name
    kv_config = get_kv_cache_config_from_groups(
        config,
        [
            KVCacheGroupSpec(["layers.0.mixer", "layers.1.mixer"], spec),
            KVCacheGroupSpec(["layers.2.mixer"], spec),
        ],
        5 * (2 * spec.page_size_bytes + 16),
    )
    pages = allocate_kv_cache(kv_config, torch.device("cpu"), layout)
    mixers = []
    states: list[torch.Tensor] = []
    trackers: list[torch.Tensor] = []
    for page in pages.values():
        page.fill_(3)
        mixer = MambaMixer2.__new__(MambaMixer2)
        torch.nn.Module.__init__(mixer)
        mixer.use_replayssm = True
        mixer.use_flashinfer_replayssm = True
        mixer.replayssm_buffer_len = 16
        mixer.get_state_shape = lambda: spec.shapes
        mixer.get_state_dtype = lambda: spec.dtypes
        mixer.bind_kv_cache(page)
        for tensor in mixer.kv_cache:
            tensor.fill_(3)
        mixer._replayssm_ring_start = torch.full((5,), 3, dtype=torch.int32)
        mixer._replayssm_prev_num_accepted = torch.full((5,), 3, dtype=torch.int32)
        states.extend(mixer.kv_cache)
        trackers.extend(
            (mixer._replayssm_ring_start, mixer._replayssm_prev_num_accepted)
        )
        mixers.append(mixer)
    assert mixers[0].kv_cache[0].data_ptr() == mixers[2].kv_cache[0].data_ptr()
    block_ids = np.arange(10, 14, dtype=np.int32).reshape(4, 1)
    original_block_ids = block_ids.copy()
    block_table = SimpleNamespace(block_table=SimpleNamespace(np=block_ids))
    multi_group_block_table = SimpleNamespace(
        block_tables=[block_table], commit_block_table=Mock()
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(use_v2_model_runner=use_v2),
        input_batch=SimpleNamespace(block_table=multi_group_block_table),
        block_tables=SimpleNamespace(get_dummy_block_tables=Mock()),
        get_model=lambda: SimpleNamespace(modules=lambda: mixers),
    )

    with (
        pytest.raises(RuntimeError, match="warmup failed"),
        warmup._temporary_replayssm_autotune_state(runner, 2),
    ):
        if not use_v2:
            assert block_ids[:2, 0].tolist() == [1, 2]
        for tensor in (*states, *trackers):
            tensor[1:3].fill_(9)
        raise RuntimeError("warmup failed")

    assert np.array_equal(block_ids, original_block_ids)
    if use_v2:
        runner.block_tables.get_dummy_block_tables.assert_called_once_with(2)
    else:
        assert multi_group_block_table.commit_block_table.call_args_list == [
            ((2,), {}),
            ((2,), {}),
        ]
    for tensor in states:
        assert torch.count_nonzero(tensor[1:3]) == 0
        assert torch.all(tensor[0] == 3)
        assert torch.all(tensor[3:] == 3)
    for tensor in trackers:
        assert torch.count_nonzero(tensor) == 0
    for page in pages.values():
        assert torch.all(page[..., spec.state_content_size_bytes :] == 3)
