# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import FullAttentionSpec


pytestmark = pytest.mark.skip_global_cleanup


def _make_minimal_triton_vllm_config() -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda parallel_config: 8,
            get_num_kv_heads=lambda parallel_config: 8,
            get_head_size=lambda: 64,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE,
            cudagraph_capture_sizes=[],
        ),
        speculative_config=None,
    )


def test_triton_cascade_build_uses_cached_seq_lens_cpu():
    device = torch.device("cpu")
    vllm_config = _make_minimal_triton_vllm_config()
    kv_cache_spec = FullAttentionSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=8,
        head_size=64,
        dtype=torch.float16,
        sliding_window=None,
    )
    builder = TritonAttentionMetadataBuilder(
        kv_cache_spec,
        ["layer0"],
        vllm_config,
        device,
    )
    common_attn_metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=[32, 40], query_lens=[4, 4]),
        block_size=kv_cache_spec.block_size,
        device=device,
    )

    original_cpu = torch.Tensor.cpu
    seq_lens_data_ptr = common_attn_metadata.seq_lens.data_ptr()

    def tracking_cpu(self: torch.Tensor, *args, **kwargs):
        if self.data_ptr() == seq_lens_data_ptr:
            raise AssertionError("cascade build should use cached seq_lens_cpu")
        return original_cpu(self, *args, **kwargs)

    with patch.object(torch.Tensor, "cpu", autospec=True, side_effect=tracking_cpu):
        attn_metadata = builder.build(
            common_prefix_len=8,
            common_attn_metadata=common_attn_metadata,
        )

    assert attn_metadata.use_cascade
    torch.testing.assert_close(
        attn_metadata.suffix_kv_lens.cpu(),
        common_attn_metadata.seq_lens_cpu - 8,
    )