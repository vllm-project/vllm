# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import FullAttentionSpec


def _builder(use_non_causal: bool, num_speculative_tokens: int | None):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda _: 8,
            get_num_kv_heads=lambda _: 1,
            # A drafter builder sees the target's model config (e.g. MLA).
            get_head_size=lambda: 576,
            rswa_window=None,
        ),
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(max_num_seqs=16),
        speculative_config=(
            SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
            if num_speculative_tokens is not None
            else None
        ),
        attention_config=SimpleNamespace(use_non_causal=use_non_causal),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
            static_forward_context={},
        ),
    )
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=64, dtype=torch.bfloat16
    )
    return TritonAttentionMetadataBuilder(
        spec, ["layer.0"], config, torch.device("cpu")
    )


@pytest.mark.parametrize(
    ("use_non_causal", "num_speculative_tokens", "num_tokens", "draft_rows"),
    [
        (False, None, 128, False),
        (False, 3, 128, False),
        (True, None, 128, False),
        (True, 3, 64, True),
    ],
)
def test_3d_scratch_holds_draft_tokens_only_for_non_causal_drafts(
    use_non_causal: bool,
    num_speculative_tokens: int | None,
    num_tokens: int,
    draft_rows: bool,
):
    builder = _builder(use_non_causal, num_speculative_tokens)
    segments = 128 if draft_rows and current_platform.is_rocm() else 16

    assert builder.softmax_segm_output.shape == (num_tokens, 8, segments, 64)
    assert builder.softmax_segm_max.shape == (num_tokens, 8, segments)
    assert builder.softmax_segm_expsum.shape == (num_tokens, 8, segments)
