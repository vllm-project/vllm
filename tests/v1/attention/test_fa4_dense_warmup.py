# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FA4's dense JIT warmup on SM90 at head size 512 compiles the SplitKV combine
kernel that a split-key call of the layer then runs."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(90),
    reason="FA4's SplitKV warmup covers head size 512 on SM90",
)


def _config():
    """The parts of a VllmConfig the dense FA4 warmup reads."""
    return SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            rswa_window=None,
            hf_text_config=SimpleNamespace(attn_logit_softcapping=None),
        ),
    )


@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 1), (16, 2)])
def test_warmup_compiles_the_combine_of_split_calls(num_heads, num_kv_heads):
    """From an empty combine cache, the warmup's compiles cover the combine of
    a split-key decode over a long context, which keys on the query heads."""
    from vllm.v1.attention.backends.fa_utils import is_fa_version_supported
    from vllm.v1.attention.backends.flash_attn import _FA4_DENSE_ATTENTION_KERNEL
    from vllm.v1.kv_cache_interface import AttentionSpec
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    if not is_fa_version_supported(4):
        pytest.skip("FA4 is not available")
    from vllm.vllm_flash_attn.cute import interface
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd_combine

    head_size, page_size, num_keys = 512, 64, 8192
    spec = AttentionSpec(
        block_size=page_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.bfloat16,
    )
    cache = _flash_attn_fwd_combine.compile_cache  # type: ignore[attr-defined]
    saved = dict(cache.cache)
    try:
        cache.cache.clear()
        _FA4_DENSE_ATTENTION_KERNEL.warmup(
            vllm_config=_config(),
            kv_cache_spec=spec,
            num_qo_heads=num_heads,
            window_size=(-1, -1),
        )
        compiled = set(cache.cache)
        assert compiled

        torch.manual_seed(0)
        num_blocks = num_keys // page_size
        k = torch.randn(
            num_blocks, page_size, num_kv_heads, head_size, dtype=torch.bfloat16
        ).cuda()
        v = torch.randn_like(k)
        q = torch.randn(1, num_heads, head_size, dtype=torch.bfloat16).cuda()
        args = dict(
            q=q,
            k=k,
            v=v,
            max_seqlen_q=1,
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32).cuda(),
            max_seqlen_k=num_keys,
            seqused_k=torch.tensor([num_keys], dtype=torch.int32).cuda(),
            block_table=torch.arange(num_blocks, dtype=torch.int32).cuda()[None],
            softmax_scale=head_size**-0.5,
            causal=True,
            fa_version=4,
        )
        # A decode over 8K keys splits the keys (the kernel's own count), so it
        # runs the combine, which reads its cache through the module's name.
        combines: list[int] = []

        def counted(*a, **kw):
            combines.append(1)
            return _flash_attn_fwd_combine(*a, **kw)

        counted.compile_cache = cache  # type: ignore[attr-defined]
        with patch.object(interface, "_flash_attn_fwd_combine", counted):
            split = flash_attn_varlen_func(**args, num_splits=0)
        new = set(cache.cache) - compiled
    finally:
        cache.cache.clear()
        cache.cache.update(saved)
    assert combines == [1]
    assert new == set()
    unsplit = flash_attn_varlen_func(**args, num_splits=1)
    torch.testing.assert_close(split, unsplit, atol=1e-2, rtol=1e-2)
