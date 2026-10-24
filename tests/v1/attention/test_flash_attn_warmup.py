# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from itertools import product
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms.interface import DeviceCapability
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


@pytest.mark.parametrize("num_qo_heads", [1, 8])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("page_size", [128, 1536, 16384])
def test_dense_warmup_covers_prefill_schedulers(
    monkeypatch, config, num_qo_heads, head_dim, page_size
):
    """Short prefills and SplitKV must not wait for the first user request."""
    monkeypatch.setattr(
        flash_attn.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(10, 3),
    )
    spec = FullAttentionSpec(
        block_size=page_size, num_kv_heads=1, head_size=head_dim, dtype=torch.bfloat16
    )
    compiled = []
    monkeypatch.setattr(
        flash_attn,
        "compile_flash_attn_varlen_func_from_specs",
        lambda **kwargs: compiled.append(kwargs),
        raising=False,
    )
    kernel = flash_attn.FA4DenseAttentionKernel()
    kernel.warmup(vllm_config=config, kv_cache_spec=spec, num_qo_heads=num_qo_heads)

    classes = set()
    for args in compiled:
        assert args["page_table_shape"][1] * page_size >= args["max_seqlen_k"]
        packed_q = args["max_seqlen_q"] * num_qo_heads
        stage = 2 if packed_q > 128 else 1
        split = args["num_splits"] > 1
        persistent = packed_q <= stage * 128 and not split
        classes.add((stage, persistent, split, args["causal"]))
    assert classes == {
        (stage, persistent, split, causal)
        for stage, persistent, split in [
            (1, True, False),
            (2, True, False),
            (2, False, False),
            (1, False, True),
            (2, False, True),
        ]
        for causal in (True, False)
    }
    assert {args["cu_seqlens_q_shape"][0] - 1 for args in compiled} == {2, 512}


@pytest.mark.parametrize("head_dim, major, count", [(256, 10, 4), (512, 9, 5)])
def test_dense_warmup_preserves_specialized_kernels(
    monkeypatch, config, head_dim, major, count
):
    monkeypatch.setattr(
        flash_attn.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(major, 0),
    )
    spec = FullAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=head_dim, dtype=torch.bfloat16
    )
    keys = flash_attn.FA4DenseAttentionKernel().get_warmup_keys(
        vllm_config=config, kv_cache_spec=spec, num_qo_heads=8
    )
    assert len(keys) == count
    assert all(key.causal for key in keys)


@pytest.mark.skipif(
    not flash_attn.current_platform.is_cuda(), reason="requires native FA4"
)
@pytest.mark.parametrize("num_qo_heads", [1, 8])
@pytest.mark.parametrize("page_size, num_pages", [(128, 64), (16384, 1)])
def test_dense_warmup_populates_native_forward_cache(
    monkeypatch, config, num_qo_heads, page_size, num_pages
):
    """Exercise native scheduler dispatch, not just the representative shapes."""
    capability = flash_attn.current_platform.get_device_capability()
    if capability is None or capability.major not in (10, 11):
        pytest.skip("requires Blackwell FA4")
    from vllm.vllm_flash_attn.cute import interface
    from vllm.vllm_flash_attn.cute.cache_utils import JITCache

    compile_from_specs = flash_attn.compile_flash_attn_varlen_func_from_specs
    assert compile_from_specs is not None
    monkeypatch.setattr(interface._flash_attn_fwd, "compile_cache", JITCache())
    spec = FullAttentionSpec(
        block_size=page_size, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
    )
    flash_attn.FA4DenseAttentionKernel().warmup(
        vllm_config=config, kv_cache_spec=spec, num_qo_heads=num_qo_heads
    )

    def unexpected_compile(*args, **kwargs):
        pytest.fail("FA4 forward compilation after dense warmup")

    monkeypatch.setattr(interface.cute, "compile", unexpected_compile)
    for batch, query_len, splits, causal in product(
        (1, 16, 512), (1, 17, 33, 129, 162, 257, 1611), (1, 2), (True, False)
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
            causal=causal,
            num_splits=splits,
            fa_version=4,
            return_softmax_lse=False,
        )
