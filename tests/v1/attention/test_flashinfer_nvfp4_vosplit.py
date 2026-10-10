# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VO-split geometry and wrapper contract; no GPU kernels are launched."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("FlashInfer requires a CUDA platform.", allow_module_level=True)

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends import flashinfer as fi
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode


@pytest.mark.parametrize(
    "bits,strategy,accepted",
    [
        (4, "tensor", True),
        (4, "attn_head", False),
        (8, "tensor", True),
        (8, "attn_head", True),
        (6, "tensor", False),
    ],
)
def test_calibrated_kv_scheme(bits, strategy, accepted):
    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors as ct,
    )

    scheme = dict(type="float", num_bits=bits, strategy=strategy, symmetric=True)
    if accepted:
        ct.CompressedTensorsKVCacheMethod.validate_kv_cache_scheme(scheme)
    else:
        with pytest.raises(NotImplementedError):
            ct.CompressedTensorsKVCacheMethod.validate_kv_cache_scheme(scheme)


@pytest.mark.parametrize(
    "dtype,pinned,expected",
    [
        ("nvfp4", None, "TRITON_ATTN"),
        ("nvfp4", "FLASHINFER", "FLASHINFER"),
        ("auto", None, "TRITON_ATTN"),
        ("fp8", None, "TRITON_ATTN"),
        ("nvfp4", "TRITON_ATTN", "TRITON_ATTN"),
    ],
)
def test_gemma4_route_preserves_other_dtypes_and_user_backend(
    monkeypatch, dtype, pinned, expected
):
    from vllm.model_executor.models.config import Gemma4Config
    from vllm.v1.attention.backends import fa_utils
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda x: x == 120
    )
    monkeypatch.setattr(fa_utils, "is_fa_version_supported", lambda x: False)

    class Arch:
        total_num_hidden_layers = 2

        def __getitem__(self, i):
            return SimpleNamespace(head_size=[256, 512][i])

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            is_mm_prefix_lm=False,
            model_arch_config=Arch(),
            hf_text_config=SimpleNamespace(layer_types=["sliding", "full"]),
        ),
        cache_config=SimpleNamespace(cache_dtype=dtype),
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum[pinned] if pinned else None,
            flash_attn_version=None,
        ),
    )
    Gemma4Config.verify_and_update_config(config)
    assert config.attention_config.backend == AttentionBackendEnum[expected]


def test_gemma4_multimodal_scope_is_rejected_early(monkeypatch):
    from vllm.model_executor.models.config import Gemma4Config
    from vllm.v1.attention.backends import fa_utils
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda x: x == 120
    )
    monkeypatch.setattr(fa_utils, "is_fa_version_supported", lambda x: False)

    class Arch:
        total_num_hidden_layers = 2

        def __getitem__(self, i):
            return SimpleNamespace(head_size=[256, 512][i])

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            is_mm_prefix_lm=True,
            model_arch_config=Arch(),
            hf_text_config=SimpleNamespace(layer_types=["sliding", "full"]),
        ),
        cache_config=SimpleNamespace(cache_dtype="nvfp4"),
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum.FLASHINFER, flash_attn_version=None
        ),
    )
    with pytest.raises(ValueError, match="language-model-only"):
        Gemma4Config.verify_and_update_config(config)


@pytest.mark.parametrize(
    "head_size,is_nvfp4,split",
    [(128, True, 1), (256, True, 1), (512, True, 2), (512, False, 1), (768, False, 1)],
)
def test_vo_split_factor(head_size, is_nvfp4, split):
    assert fi._vo_split_factor(head_size, is_nvfp4) == split


def test_vo_split_keeps_k_and_slices_v_with_original_strides():
    impl = object.__new__(fi.FlashInferImpl)
    impl.head_size = 512
    impl.vo_split = 2
    query = torch.zeros(3, 2, 512)
    k = torch.zeros(2, 1, 16, 256, dtype=torch.uint8)
    v = torch.arange(2 * 16 * 256).to(torch.uint8).reshape(2, 1, 16, 256)
    k_sf = torch.zeros(2, 1, 16, 32)
    v_sf = torch.arange(2 * 16 * 32).reshape(2, 1, 16, 32)
    out = torch.empty_like(query)
    calls: list[int] = []

    class Wrapper:
        def run(self, q, kv, **kwargs):
            i = len(calls)
            assert q is query
            assert kv[0] is k
            assert kwargs["kv_cache_sf"][0] is k_sf
            torch.testing.assert_close(kv[1], v[..., i * 128 : (i + 1) * 128])
            torch.testing.assert_close(
                kwargs["kv_cache_sf"][1], v_sf[..., i * 16 : (i + 1) * 16]
            )
            assert kv[1].stride() == v.stride()
            assert kwargs["kv_cache_sf"][1].stride() == v_sf.stride()
            assert kwargs["out"].is_contiguous()
            assert (kwargs["q_scale"], kwargs["k_scale"], kwargs["v_scale"]) == (
                1.0,
                0.3,
                0.7,
            )
            kwargs["out"].fill_(i + 1)
            calls.append(i)

    impl._run_vo_split_prefill(
        Wrapper(), query, (k, v), (k_sf, v_sf), out, k_scale=0.3, v_scale=0.7
    )
    assert calls == [0, 1]
    assert torch.all(out[..., :256] == 1)
    assert torch.all(out[..., 256:] == 2)


@pytest.mark.parametrize(
    "fa2,quant,expected",
    [
        (True, KVQuantMode.NVFP4, AttentionCGSupport.NEVER),
        (False, KVQuantMode.NVFP4, AttentionCGSupport.UNIFORM_BATCH),
        (True, KVQuantMode.FP8_PER_TENSOR, AttentionCGSupport.UNIFORM_BATCH),
        (True, KVQuantMode.NONE, AttentionCGSupport.UNIFORM_BATCH),
    ],
)
def test_vo_split_graph_safety(monkeypatch, fa2, quant, expected):
    monkeypatch.setattr(fi, "_nvfp4_kv_on_fa2", lambda: fa2)
    monkeypatch.setattr(fi, "can_use_trtllm_attention", lambda **kw: True)
    monkeypatch.setattr(fi.current_platform, "is_device_capability", lambda x: False)
    monkeypatch.setattr(
        fi.current_platform, "is_device_capability_family", lambda x: False
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        model_config=SimpleNamespace(get_num_attention_heads=lambda pc: 2),
        attention_config=SimpleNamespace(use_non_causal=False),
    )
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        kv_quant_mode=quant,
    )
    assert fi.FlashInferMetadataBuilder.get_cudagraph_support(config, spec) == expected
