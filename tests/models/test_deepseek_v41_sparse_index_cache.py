# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import AbstractContextManager

import torch

from vllm.forward_context import ForwardContext, override_forward_context
from vllm.models.deepseek_v41.cross_layer import (
    DeepseekV41CrossLayerRuntime,
    get_cross_layer_runtime,
)


def _forward_context() -> AbstractContextManager[None]:
    return override_forward_context(
        ForwardContext(
            no_compile_layers={},
            attn_metadata={},
            slot_mapping={},
        )
    )


def test_cross_layer_runtime_reuses_matching_sources():
    runtime = DeepseekV41CrossLayerRuntime()
    decode_key = (2, 1)
    decode_value = (torch.zeros(4, 8), torch.ones(4))

    assert runtime.reuse_decode_topk(decode_key, is_index_source=True) is None
    runtime.cache_decode_topk(decode_key, decode_value)
    assert runtime.reuse_decode_topk(decode_key, is_index_source=False) is decode_value
    ragged_value = (*decode_value, torch.zeros(4))
    runtime.cache_decode_topk(("rocm", 1), ragged_value)
    assert runtime.reuse_decode_topk(("rocm", 1), is_index_source=False) is ragged_value
    assert runtime.reuse_decode_topk((3, 1), is_index_source=False) is None
    assert runtime.reuse_decode_topk(decode_key, is_index_source=True) is None
    assert not runtime.decode_topk

    prefill_key = (2, 2)
    prefill_value = (torch.zeros(8, 640), torch.ones(8))

    assert runtime.reuse_prefill_indices(prefill_key, is_index_source=True) is None
    runtime.cache_prefill_indices(prefill_key, prefill_value)
    assert (
        runtime.reuse_prefill_indices(prefill_key, is_index_source=False)
        is prefill_value
    )
    assert runtime.reuse_prefill_indices((3, 2), is_index_source=False) is None
    assert runtime.reuse_prefill_indices(prefill_key, is_index_source=True) is None
    assert not runtime.prefill_indices


def test_cross_layer_runtime_is_scoped_to_forward_context():
    key = (2, 1)
    value = (torch.zeros(4, 8), torch.ones(4))

    with _forward_context():
        first = get_cross_layer_runtime()
        assert get_cross_layer_runtime() is first
        first.cache_decode_topk(key, value)

    with _forward_context():
        second = get_cross_layer_runtime()
        assert second is not first
        assert second.reuse_decode_topk(key, is_index_source=False) is None
