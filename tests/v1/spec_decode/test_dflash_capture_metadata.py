# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash CUDA-graph capture metadata (#61337).

FlexAttention's capture path requires ``seq_lens_cpu_upper_bound`` to size the
recorded graph's static buffers, and the DFlash query-capture path built its
metadata without it, so capture died on the backend's assertion before any
draft weight was touched. These tests pin the field through
``_prepare_dflash_inputs_to_capture`` without needing a GPU.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash import cudagraph as dflash_cudagraph


def _fake_block_tables() -> SimpleNamespace:
    return SimpleNamespace(
        cp_size=1,
        get_dummy_block_tables=lambda num_reqs: (
            torch.zeros(num_reqs, 4, dtype=torch.int32),
        ),
        get_dummy_slot_mappings=lambda num_tokens: torch.full(
            (num_tokens,), -1, dtype=torch.int32
        ),
    )


def _fake_kv_cache_config() -> SimpleNamespace:
    return SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["draft.0"])],
    )


def _capture_kwargs(num_reqs: int, num_tokens: int) -> dict:
    input_buffers = InputBuffers(
        max_num_reqs=8, max_num_tokens=64, device=torch.device("cpu")
    )
    recorded: dict = {}

    def fake_build_attn_metadata(**kwargs):
        recorded.update(kwargs)
        return {"fake": True}

    with patch.object(
        dflash_cudagraph, "build_attn_metadata", fake_build_attn_metadata
    ):
        dflash_cudagraph._prepare_dflash_inputs_to_capture(
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            input_buffers=input_buffers,
            block_tables=_fake_block_tables(),
            attn_groups=[],
            kv_cache_config=_fake_kv_cache_config(),
            max_model_len=128,
            skip_attn=False,
            causal=True,
        )
    return recorded


def test_capture_metadata_carries_seq_lens_cpu_upper_bound() -> None:
    recorded = _capture_kwargs(num_reqs=2, num_tokens=8)
    upper = recorded["seq_lens_cpu_upper_bound"]
    assert upper is not None
    # The dummy batch splits 8 tokens evenly, and a capture-time request has
    # no computed context, so its own token count is the upper bound.
    assert upper.tolist() == [4, 4]


def test_capture_metadata_upper_bound_matches_uneven_split() -> None:
    # 9 tokens over 2 requests: the last request holds the remainder, and the
    # upper bound must track the per-request split rather than the average.
    recorded = _capture_kwargs(num_reqs=2, num_tokens=9)
    assert recorded["seq_lens_cpu_upper_bound"].tolist() == [4, 5]


def test_capture_still_marks_cudagraph_capture() -> None:
    recorded = _capture_kwargs(num_reqs=1, num_tokens=4)
    assert recorded["for_cudagraph_capture"] is True
    assert recorded["causal"] is True
