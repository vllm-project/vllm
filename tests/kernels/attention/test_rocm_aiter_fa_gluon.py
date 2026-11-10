# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_AITER_FA_GLUON: which layers it decodes on gluon, when the ROCm selector
picks it, and how it wires the null block and the kernel call."""

import importlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)
pytest.importorskip("aiter.ops.triton.gluon.pa_decode_gluon")


def _reload_envs():
    import vllm.envs as envs

    return importlib.reload(envs)


@pytest.fixture
def aiter_env(monkeypatch):
    from vllm._aiter_ops import rocm_aiter_ops

    def apply(shuffle: bool) -> None:
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MHA", "1")
        monkeypatch.setenv("VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT", "1" if shuffle else "0")
        _reload_envs()
        rocm_aiter_ops.refresh_env_variables()

    yield apply
    monkeypatch.undo()
    _reload_envs()
    rocm_aiter_ops.refresh_env_variables()


@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads", "head_size", "kv_cache_dtype", "shuffle", "expected"),
    [
        (24, 4, 256, "auto", True, True),
        (24, 4, 256, "auto", False, False),
        (24, 4, 256, "fp8", True, False),
        (16, 2, 256, "auto", True, False),
        (32, 8, 128, "auto", True, False),
    ],
    ids=["qwen38", "plain-kv", "fp8-kv", "gqa-8", "head-128"],
)
def test_pa_gluon_supports(
    num_heads, num_kv_heads, head_size, kv_cache_dtype, shuffle, expected, aiter_env
):
    from vllm.platforms.rocm import on_gfx950
    from vllm.v1.attention.backends.rocm_aiter_fa_gluon import pa_gluon_supports

    aiter_env(shuffle)
    assert pa_gluon_supports(num_heads, num_kv_heads, head_size, kv_cache_dtype) is (
        expected and on_gfx950()
    )


@pytest.mark.parametrize(
    ("heads", "kv_cache_dtype", "shuffle", "parallel", "spec", "gluon"),
    [
        ((24, 4), "auto", True, (1, False), False, True),
        ((24, 4), "auto", False, (1, False), False, False),
        ((16, 2), "auto", True, (1, False), False, False),
        ((24, 4), "fp8", True, (1, False), False, False),
        ((24, 4), "auto", True, (2, False), False, False),
        ((24, 4), "auto", True, (1, True), False, False),
        ((24, 4), "auto", True, (1, False), True, False),
    ],
    ids=["qwen38", "plain-kv", "gqa-8", "fp8-kv", "tp2", "ep", "spec"],
)
def test_selector_picks_gluon_for_supported_layers(
    heads, kv_cache_dtype, shuffle, parallel, spec, gluon, aiter_env, monkeypatch
):
    from vllm.platforms.rocm import RocmPlatform, on_gfx950
    from vllm.v1.attention.backends import rocm_aiter_fa_gluon
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import AttentionSelectorConfig

    if not on_gfx950():
        pytest.skip("gfx950 only")
    aiter_env(shuffle)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda _: heads[0],
            get_num_kv_heads=lambda _: heads[1],
        ),
        parallel_config=SimpleNamespace(
            world_size=parallel[0], enable_expert_parallel=parallel[1]
        ),
        speculative_config=SimpleNamespace() if spec else None,
    )
    monkeypatch.setattr(
        rocm_aiter_fa_gluon, "get_current_vllm_config_or_none", lambda: config
    )
    selected = RocmPlatform.get_attn_backend_cls(
        None,
        AttentionSelectorConfig(
            head_size=256,
            dtype=torch.bfloat16,
            kv_cache_dtype=kv_cache_dtype,
            block_size=None,
        ),
    )
    gluon_path = AttentionBackendEnum.ROCM_AITER_FA_GLUON.get_path()
    assert (selected == gluon_path) is gluon


def test_build_zeroes_null_block(monkeypatch):
    from vllm.v1.attention.backends.rocm_aiter_fa import (
        AiterFlashAttentionMetadataBuilder,
    )
    from vllm.v1.attention.backends.rocm_aiter_fa_gluon import (
        AiterFlashAttentionGluonMetadataBuilder,
    )

    monkeypatch.setattr(
        AiterFlashAttentionMetadataBuilder,
        "build",
        lambda *args, **kwargs: SimpleNamespace(num_decodes=1),
    )
    kv_cache = torch.full((4, 2, 16, 8), float("nan"))
    builder = object.__new__(AiterFlashAttentionGluonMetadataBuilder)
    layer: Any = SimpleNamespace(kv_cache=kv_cache)
    builder._gluon_layers = [layer]
    builder.build(0, None)
    assert torch.count_nonzero(kv_cache[0]) == 0
    assert kv_cache[1:].isnan().all()


@pytest.mark.parametrize("kv_row_size", [False, True])
def test_decode_passes_shuffle_views_to_gluon(monkeypatch, kv_row_size):
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.backends import rocm_aiter_fa_gluon
    from vllm.v1.attention.backends.rocm_aiter_fa_gluon import (
        AiterFlashAttentionGluonImpl,
    )

    kernel_args, split_args = {}, {}

    def recommended_splits(num_seqs, heads, **kwargs):
        split_args.update(kwargs)
        return 2

    monkeypatch.setattr(
        rocm_aiter_fa_gluon, "get_recommended_splits", recommended_splits
    )
    monkeypatch.setattr(
        rocm_aiter_ops, "pa_gluon_splits_support_kv_row_size", lambda: kv_row_size
    )
    monkeypatch.setattr(
        rocm_aiter_fa_gluon,
        "pa_decode_gluon",
        lambda **kwargs: kernel_args.update(kwargs),
    )
    monkeypatch.setattr(rocm_aiter_ops, "_SHUFFLE_KV_CACHE_ENABLED", True)
    impl = object.__new__(AiterFlashAttentionGluonImpl)
    impl.use_gluon_decode, impl.scale = True, 256**-0.5
    impl.num_kv_heads, impl.head_size = 4, 256
    kv_cache = torch.zeros(3, 2, 16, 4 * 256, dtype=torch.bfloat16)
    key_cache, value_cache = impl._split_kv_cache(kv_cache)
    assert not key_cache.is_contiguous()
    query = torch.zeros(2, 24, 256, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        seq_lens=torch.tensor([5, 17]), block_table=torch.zeros(2, 2)
    )
    impl._decode_shuffle_layout(
        None, query, key_cache, value_cache, torch.empty_like(query), metadata, 2, 2
    )
    assert kernel_args["key_cache"].shape == (3, 4, 32, 16, 8)
    assert kernel_args["value_cache"].shape == (3, 4, 2, 256, 8)
    assert kernel_args["key_cache"].data_ptr() == key_cache.data_ptr()
    assert kernel_args["value_cache"].data_ptr() == value_cache.data_ptr()
    assert kernel_args["exp_sums"].shape == (2, 4, 2, 6)
    assert kernel_args["block_tables"].dtype == torch.int32
    expected = {"head_size": 256, "kv_cache_dtype": torch.bfloat16}
    assert split_args == (expected if kv_row_size else {})
