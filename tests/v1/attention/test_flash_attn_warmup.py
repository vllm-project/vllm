# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from itertools import product
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends import flash_attn
from vllm.v1.kv_cache_interface import FullAttentionSpec


@pytest.fixture
def config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        model_config=SimpleNamespace(
            rswa_window=None,
            hf_text_config=SimpleNamespace(),
            dtype=torch.bfloat16,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=512),
    )


@pytest.mark.skipif(
    not flash_attn.current_platform.is_cuda(), reason="requires native FA4"
)
@pytest.mark.parametrize("max_num_seqs", [256, 512])
@pytest.mark.parametrize("num_qo_heads", [1, 8])
@pytest.mark.parametrize("page_size, num_pages", [(128, 64), (16384, 1)])
def test_dense_warmup_populates_native_forward_cache(
    monkeypatch, config, max_num_seqs, num_qo_heads, page_size, num_pages
):
    """Each warmup key compiles one distinct kernel that served shapes reuse."""
    capability = flash_attn.current_platform.get_device_capability()
    if capability is None or capability.major not in (10, 11):
        pytest.skip("requires Blackwell FA4")
    from vllm.vllm_flash_attn.cute import interface
    from vllm.vllm_flash_attn.cute.cache_utils import JITCache

    compile_from_specs = flash_attn.compile_flash_attn_varlen_func_from_specs
    assert compile_from_specs is not None
    forward_cache = JITCache()
    monkeypatch.setattr(interface._flash_attn_fwd, "compile_cache", forward_cache)
    config.scheduler_config.max_num_seqs = max_num_seqs
    spec = FullAttentionSpec(
        block_size=page_size, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
    )
    kernel = flash_attn.FA4DenseAttentionKernel()
    keys = kernel.get_warmup_keys(
        vllm_config=config, kv_cache_spec=spec, num_qo_heads=num_qo_heads
    )
    kernel.compile_many(keys)
    warm_kernels = set(forward_cache.cache)
    assert len(warm_kernels) == len(keys)

    for batch, query_len, splits in product(
        (1, 16, max_num_seqs), (1, 17, 33, 129, 162, 257, 1611), (1, 2)
    ):
        compile_from_specs(
            q_shape=(batch * query_len, num_qo_heads, 128),
            k_shape=(256, page_size, 1, 128),
            v_shape=(256, page_size, 1, 128),
            q_dtype=torch.bfloat16,
            cu_seqlens_q_shape=(batch + 1,),
            seqused_k_shape=(batch,),
            page_table_shape=(batch, num_pages),
            max_seqlen_q=query_len,
            max_seqlen_k=8192,
            causal=True,
            num_splits=splits,
            fa_version=4,
            return_softmax_lse=False,
        )
        assert set(forward_cache.cache) == warm_kernels, (batch, query_len, splits)
