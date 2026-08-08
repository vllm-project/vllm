# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer paged-KV buffer sizing under chunked local attention."""

import unittest.mock

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("FlashInfer backend requires a CUDA platform.", allow_module_level=True)

from tests.v1.attention.utils import (  # noqa: E402
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.attention.chunked_local_attention import (  # noqa: E402
    create_chunked_local_attention_backend,
)
from vllm.v1.attention.backends.flashinfer import FlashInferBackend  # noqa: E402
from vllm.v1.attention.backends.utils import (  # noqa: E402
    PerLayerParameters,
    make_local_attention_virtual_batches,
)
from vllm.v1.kv_cache_interface import (  # noqa: E402
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
)

ATTN_CHUNK_SIZE = 256
BLOCK_SIZE = 16
# Longer than ATTN_CHUNK_SIZE, so the request is split into several virtual
# batches; with max_num_seqs=1 that is what overflowed the buffers.
QUERY_LEN = 1000
MAX_NUM_SEQS = 1


def _mock_get_per_layer_parameters(vllm_config, layer_names, impl_cls):
    head_size = vllm_config.model_config.get_head_size()
    return {
        name: PerLayerParameters(
            window_left=-1,
            logits_soft_cap=0.0,
            sm_scale=1.0 / (head_size**0.5),
        )
        for name in layer_names
    }


def _build_builder(vllm_config, kv_cache_spec):
    backend = create_chunked_local_attention_backend(FlashInferBackend, ATTN_CHUNK_SIZE)
    with (
        set_current_vllm_config(vllm_config),
        unittest.mock.patch(
            "vllm.v1.attention.backends.flashinfer.get_per_layer_parameters",
            _mock_get_per_layer_parameters,
        ),
    ):
        # Buffer sizing happens in __init__ and is device-independent, so the
        # test stays on CPU and needs no particular GPU.
        return backend.get_builder_cls()(
            kv_cache_spec, ["layer.0"], vllm_config, torch.device("cpu")
        )


def _make_specs(vllm_config):
    """Both specs a chunked-local layer can reach a builder with.

    With the hybrid KV cache manager enabled (the default on CUDA) the builder
    sees a `ChunkedLocalAttentionSpec`. When it is disabled, the spec is promoted
    to a `FullAttentionSpec` that keeps `attention_chunk_size` set.
    """
    common = dict(
        block_size=BLOCK_SIZE,
        num_kv_heads=vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        ),
        head_size=vllm_config.model_config.get_head_size(),
        dtype=vllm_config.model_config.dtype,
    )
    return {
        "chunked_local": ChunkedLocalAttentionSpec(
            attention_chunk_size=ATTN_CHUNK_SIZE, **common
        ),
        "promoted_full": FullAttentionSpec(
            attention_chunk_size=ATTN_CHUNK_SIZE, **common
        ),
    }


@pytest.mark.parametrize("spec_name", ["chunked_local", "promoted_full"])
def test_paged_kv_buffers_fit_local_attention_virtual_batches(spec_name: str):
    """Regression test for https://github.com/vllm-project/vllm/issues/49980.

    `make_local_attention_virtual_batches` reports a `num_reqs` equal to the
    virtual batch count, which is decoupled from `max_num_seqs`. Sizing the
    paged-KV buffers from `max_num_seqs` made the cumsum in
    `_compute_flashinfer_kv_metadata` raise "provided out is the wrong size for
    the accumulation" whenever a prefill exceeded `attention_chunk_size`.
    """
    vllm_config = create_vllm_config(
        max_model_len=2048,
        block_size=BLOCK_SIZE,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=2048,
    )
    builder = _build_builder(vllm_config, _make_specs(vllm_config)[spec_name])

    common_attn_metadata = create_common_attn_metadata(
        BatchSpec(query_lens=[QUERY_LEN], seq_lens=[QUERY_LEN]),
        BLOCK_SIZE,
        torch.device("cpu"),
    )
    local_metadata, _ = make_local_attention_virtual_batches(
        ATTN_CHUNK_SIZE, common_attn_metadata, BLOCK_SIZE
    )
    num_reqs = local_metadata.num_reqs
    # Guard against the test passing vacuously if the split ever stops
    # inflating the request count.
    assert num_reqs > MAX_NUM_SEQS

    # The exact operation that raised before the fix.
    seq_lens_np = local_metadata.seq_lens.numpy()
    num_blocks_np = (seq_lens_np + BLOCK_SIZE - 1) // BLOCK_SIZE
    np.cumsum(
        num_blocks_np,
        dtype=np.int32,
        out=builder.paged_kv_indptr.np[1 : num_reqs + 1],
    )

    assert builder.paged_kv_last_page_len.np.shape[0] >= num_reqs
    num_actual_pages = int(builder.paged_kv_indptr.np[num_reqs])
    assert builder.paged_kv_indices.np.shape[0] >= num_actual_pages


def test_chunked_local_sizing_never_shrinks_full_attention_capacity():
    """`attention_chunk_size` on a merged spec must not shrink the allocation.

    When the hybrid KV cache manager is disabled, every Llama-4 layer becomes a
    `FullAttentionSpec` and they merge into one KV cache group whose spec keeps
    `attention_chunk_size`. The global attention layers form their own attention
    group but share that spec, and they attend over the whole sequence, so
    `paged_kv_indices` must still cover `max_num_seqs * max_num_pages_per_req`.
    """
    max_num_seqs = 4
    max_model_len = 8192
    vllm_config = create_vllm_config(
        max_model_len=max_model_len,
        block_size=BLOCK_SIZE,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_model_len,
    )
    kv_cache_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        ),
        head_size=vllm_config.model_config.get_head_size(),
        dtype=vllm_config.model_config.dtype,
        attention_chunk_size=ATTN_CHUNK_SIZE,
    )
    builder = _build_builder(vllm_config, kv_cache_spec)

    full_attention_pages = max_num_seqs * -(-max_model_len // BLOCK_SIZE)
    assert builder.paged_kv_indices.np.shape[0] >= full_attention_pages


def test_full_attention_buffer_sizing_is_unchanged():
    """A spec without `attention_chunk_size` must keep the original sizing."""
    vllm_config = create_vllm_config(
        max_model_len=2048, block_size=BLOCK_SIZE, max_num_seqs=8
    )
    kv_cache_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        ),
        head_size=vllm_config.model_config.get_head_size(),
        dtype=vllm_config.model_config.dtype,
    )
    backend = create_chunked_local_attention_backend(FlashInferBackend, ATTN_CHUNK_SIZE)
    with (
        set_current_vllm_config(vllm_config),
        unittest.mock.patch(
            "vllm.v1.attention.backends.flashinfer.get_per_layer_parameters",
            _mock_get_per_layer_parameters,
        ),
    ):
        builder = backend.get_builder_cls()(
            kv_cache_spec, ["layer.0"], vllm_config, torch.device("cpu")
        )

    max_num_pages_per_req = -(-2048 // BLOCK_SIZE)
    assert builder.paged_kv_indptr.np.shape[0] == 8 + 1
    assert builder.paged_kv_last_page_len.np.shape[0] == 8
    assert builder.paged_kv_indices.np.shape[0] == 8 * max_num_pages_per_req
