# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm.kernels  # noqa: F401
from tests.ir.ir_test_utils import assert_close, clone_args, supported_providers
from vllm import ir
from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.platforms import current_platform

rope_native = ir.ops.rotary_embedding.impls["native"].impl_fn
rope_query_only_native = ir.ops.rotary_embedding_query_only.impls["native"].impl_fn

IS_GPGPU_DEVICE = current_platform.is_cuda_alike() or current_platform.is_xpu()
DEVICE = current_platform.device_type

HEAD_SIZES = [64, 128, 256]
N_TOKENS = [7, 83]


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _inputs(n_tokens, head_size, dtype, is_neox, rotary_dim=None):
    return ir.ops.rotary_embedding.generate_inputs(
        num_tokens=n_tokens,
        num_heads=8,
        num_kv_heads=2,
        head_size=head_size,
        rotary_dim=rotary_dim or head_size,
        dtype=dtype,
        is_neox=is_neox,
        device=current_platform.device_type,
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="vllm_c rotary kernel is CUDA/ROCm only",
)
def test_rotary_embedding_registration():
    expected = {
        "native": True,
        "vllm_c": True,
        "aiter": is_aiter_found_and_supported(),
    }
    actual = {
        provider: impl.supported
        for provider, impl in ir.ops.rotary_embedding.impls.items()
    }
    assert actual == expected
    assert {
        provider: impl.supported
        for provider, impl in ir.ops.rotary_embedding_query_only.impls.items()
    } == {"native": True, "vllm_c": True}


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="AITER rotary is only available on ROCm",
)
def test_aiter_partial_rotary_falls_back():
    """The AITER kernel only rotates full heads; partial rotary must not use it."""
    impl = ir.ops.rotary_embedding.impls["aiter"]
    if not impl.supported:
        pytest.skip("AITER rotary is not supported on this platform")

    args = _inputs(7, 128, torch.bfloat16, True, rotary_dim=64)
    assert not impl.supports_args(*args)

    expected = rope_native(*clone_args(args))
    with ir.ops.rotary_embedding.set_priority(["aiter", "vllm_c", "native"]):
        assert ir.ops.rotary_embedding.dispatch(*args).provider == "vllm_c"
        actual = ir.ops.rotary_embedding(*clone_args(args))
    assert_close(ir.ops.rotary_embedding, actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n_tokens", N_TOKENS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.skipif(
    not IS_GPGPU_DEVICE,
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
class TestRotaryEmbedding:
    @pytest.mark.parametrize("rotary_ratio", [1.0, 0.5])
    def test_native_matches_forward_static(
        self, dtype, n_tokens, head_size, is_neox, rotary_ratio
    ):
        rotary_dim = int(head_size * rotary_ratio)
        positions, query, key, hs, cos_sin_cache, neox = _inputs(
            n_tokens, head_size, dtype, is_neox, rotary_dim
        )

        q_ir, k_ir = rope_native(
            positions, query.clone(), key.clone(), hs, cos_sin_cache, neox
        )
        q_ref, k_ref = RotaryEmbedding.forward_static(
            positions, query.clone(), key.clone(), hs, rotary_dim, cos_sin_cache, neox
        )

        torch.testing.assert_close(q_ir, q_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(k_ir, k_ref, rtol=1e-3, atol=1e-3)

        q_only = rope_query_only_native(
            positions, query.clone(), hs, cos_sin_cache, neox
        )
        torch.testing.assert_close(q_only, q_ir, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.rotary_embedding))
    def test_impls_match_native(self, dtype, n_tokens, head_size, is_neox, provider):
        impl = ir.ops.rotary_embedding.impls[provider]
        args = _inputs(n_tokens, head_size, dtype, is_neox)
        if not impl.supports_args(*args):
            pytest.skip(f"{provider} does not support args")

        ref_q, ref_k = rope_native(*clone_args(args))
        out_q, out_k = impl.impl_fn(*clone_args(args))
        assert_close(ir.ops.rotary_embedding, out_q, ref_q)
        assert_close(ir.ops.rotary_embedding, out_k, ref_k)

        with ir.ops.rotary_embedding.set_priority([provider, "native"]):
            disp_q, disp_k = ir.ops.rotary_embedding(*clone_args(args))
        direct_q, direct_k = impl.impl_fn(*clone_args(args))
        torch.testing.assert_close(disp_q, direct_q, rtol=0.0, atol=0.0)
        torch.testing.assert_close(disp_k, direct_k, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("rotary_ratio", [1.0, 0.5])
    @pytest.mark.parametrize("provider", supported_providers(ir.ops.rotary_embedding))
    def test_functional_overload_matches_native(
        self, dtype, n_tokens, head_size, is_neox, provider, rotary_ratio
    ):
        """The functional path (used by Inductor lowering) leaves inputs intact."""
        impl = ir.ops.rotary_embedding.impls[provider]
        args = _inputs(
            n_tokens, head_size, dtype, is_neox, int(head_size * rotary_ratio)
        )
        if not impl.supports_args(*args):
            pytest.skip(f"{provider} does not support args")
        # Strided q/k views, as produced by splitting a fused qkv tensor.
        positions, query, key, hs, cos_sin_cache, neox = args
        qkv = torch.cat([query, key, key], dim=-1)
        q_view, k_view, _ = qkv.split(
            [query.shape[-1], key.shape[-1], key.shape[-1]], -1
        )
        qkv_before = qkv.clone()

        ref_q, ref_k = rope_native(*clone_args(args))
        out_q, out_k = impl.func_impl_fn(
            positions, q_view, k_view, hs, cos_sin_cache, neox
        )
        assert_close(ir.ops.rotary_embedding, out_q, ref_q)
        assert_close(ir.ops.rotary_embedding, out_k, ref_k)
        torch.testing.assert_close(qkv, qkv_before, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize(
        "provider", supported_providers(ir.ops.rotary_embedding_query_only)
    )
    def test_query_only_impls_match_native(
        self, dtype, n_tokens, head_size, is_neox, provider
    ):
        positions, query, _, hs, cos_sin_cache, neox = _inputs(
            n_tokens, head_size, dtype, is_neox
        )
        args = (positions, query, hs, cos_sin_cache, neox)
        ref = rope_query_only_native(*clone_args(args))
        out = ir.ops.rotary_embedding_query_only.impls[provider].impl_fn(
            *clone_args(args)
        )
        assert_close(ir.ops.rotary_embedding_query_only, out, ref)

    def test_default_overload_not_inplace(self, dtype, n_tokens, head_size, is_neox):
        provider = "vllm_c"
        impl = ir.ops.rotary_embedding.impls[provider]
        if not impl.supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        positions, query, key, hs, cos_sin_cache, neox = _inputs(
            n_tokens, head_size, dtype, is_neox
        )
        q_default, k_default = query.clone(), key.clone()
        q_ptr, k_ptr = q_default.data_ptr(), k_default.data_ptr()

        with ir.ops.rotary_embedding.set_priority([provider, "native"]):
            out_q, out_k = ir.ops.rotary_embedding(
                positions, q_default, k_default, hs, cos_sin_cache, neox
            )

        assert out_q.data_ptr() != q_ptr
        assert out_k.data_ptr() != k_ptr
        torch.testing.assert_close(q_default, query, rtol=0.0, atol=0.0)
        torch.testing.assert_close(k_default, key, rtol=0.0, atol=0.0)

        q_inplace, k_inplace = query.clone(), key.clone()
        with ir.ops.rotary_embedding.set_priority([provider, "native"]):
            ip_q, ip_k = ir.ops.rotary_embedding.maybe_inplace(
                positions, q_inplace, k_inplace, hs, cos_sin_cache, neox
            )
        assert ip_q.data_ptr() == q_inplace.data_ptr()
        assert ip_k.data_ptr() == k_inplace.data_ptr()
        torch.testing.assert_close(ip_q, out_q, rtol=0.0, atol=0.0)
        torch.testing.assert_close(ip_k, out_k, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.rotary_embedding))
    def test_torch_opcheck(self, dtype, n_tokens, head_size, is_neox, provider):
        if not ir.ops.rotary_embedding.impls[provider].supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        args = _inputs(n_tokens, head_size, dtype, is_neox)
        with ir.ops.rotary_embedding.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.rotary_embedding.default, args)


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA/ROCm only")
@pytest.mark.parametrize("key_present", [True, False])
def test_forward_cuda_is_inplace_with_out_of_place_provider(
    key_present, default_vllm_config
):
    """forward_cuda mutates q/k even when dispatch picks a non-inplace impl."""
    rope = RotaryEmbedding(128, 128, 4096, 10000, True, torch.bfloat16).to(
        current_platform.device_type
    )
    positions, query, key, *_ = _inputs(7, 128, torch.bfloat16, True)
    expected = rope.forward_native(
        positions, query.clone(), key.clone() if key_present else None
    )
    key_arg = key.clone() if key_present else None
    query_arg = query.clone()
    with (
        ir.ops.rotary_embedding.set_priority(["native"]),
        ir.ops.rotary_embedding_query_only.set_priority(["native"]),
    ):
        out_q, out_k = rope.forward_cuda(positions, query_arg, key_arg)
    assert out_q is query_arg and out_k is key_arg
    torch.testing.assert_close(query_arg, expected[0])
    if key_present:
        torch.testing.assert_close(key_arg, expected[1])


@pytest.mark.skipif(not IS_GPGPU_DEVICE, reason="CUDA, ROCm and XPU only")
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("provider", supported_providers(ir.ops.rotary_embedding))
def test_functional_overload_per_head_strides(provider, is_neox):
    """Per-head q/k with different head strides (DeepSeek indexer layout)."""
    impl = ir.ops.rotary_embedding.impls[provider]
    n_tokens, n_heads, head_dim, rope_dim = 7, 4, 128, 64
    q_full = torch.randn(
        n_tokens, n_heads, head_dim, dtype=torch.bfloat16, device=DEVICE
    )
    k_full = torch.randn(n_tokens, head_dim, dtype=torch.bfloat16, device=DEVICE)
    q_pe = q_full[..., :rope_dim]
    k_pe = k_full[:, :rope_dim].unsqueeze(1)
    _, _, _, _, cos_sin_cache, _ = _inputs(n_tokens, rope_dim, torch.bfloat16, True)
    positions = torch.randint(0, 4096, (n_tokens,), device=DEVICE)
    args = (positions, q_pe, k_pe, rope_dim, cos_sin_cache, is_neox)
    if not impl.supports_args(*args):
        pytest.skip(f"{provider} does not support args")
    q_before, k_before = q_full.clone(), k_full.clone()

    ref_q, ref_k = rope_native(*clone_args(args))
    out_q, out_k = impl.func_impl_fn(*args)
    assert out_q.shape == q_pe.shape and out_k.shape == k_pe.shape
    assert_close(ir.ops.rotary_embedding, out_q, ref_q)
    assert_close(ir.ops.rotary_embedding, out_k, ref_k)
    torch.testing.assert_close(q_full, q_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(k_full, k_before, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not IS_GPGPU_DEVICE, reason="CUDA, ROCm and XPU only")
@pytest.mark.parametrize("rotary_ratio", [1.0, 0.5])
@pytest.mark.parametrize(
    "provider", supported_providers(ir.ops.rotary_embedding_query_only)
)
def test_query_only_functional_overload(provider, rotary_ratio):
    impl = ir.ops.rotary_embedding_query_only.impls[provider]
    head_size = 128
    positions, query, key, hs, cos_sin_cache, neox = _inputs(
        13, head_size, torch.bfloat16, True, int(head_size * rotary_ratio)
    )
    qkv = torch.cat([query, key], dim=-1)
    q_view = qkv[:, : query.shape[-1]]
    qkv_before = qkv.clone()

    ref = rope_query_only_native(positions, query.clone(), hs, cos_sin_cache, neox)
    out = impl.func_impl_fn(positions, q_view, hs, cos_sin_cache, neox)
    assert_close(ir.ops.rotary_embedding_query_only, out, ref)
    torch.testing.assert_close(qkv, qkv_before, rtol=0.0, atol=0.0)
