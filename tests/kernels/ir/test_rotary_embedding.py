# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

# Registers all provider implementations
import vllm.kernels  # noqa: F401
from tests.ir.ir_test_utils import assert_close, clone_args, supported_providers
from vllm import ir
from vllm.platforms import current_platform

rotary_embedding_native = ir.ops.rotary_embedding.impls["native"].impl_fn
rotary_embedding_query_only_native = ir.ops.rotary_embedding_query_only.impls[
    "native"
].impl_fn

NUM_TOKENS = [1, 8, 32]
HEAD_SIZES = [64, 128]
DTYPES = [torch.float16, torch.bfloat16]


# ---------------------------------------------------------------------------
# Registration checks
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
    reason="Custom kernels only available on CUDA, ROCm and XPU",
)
def test_rotary_embedding_registration():
    from vllm.kernels.aiter_ops import AITER_TRITON_ROTARY_SUPPORTED

    expected = {
        "native": True,
        "vllm_c": current_platform.is_cuda_alike(),
        "aiter": AITER_TRITON_ROTARY_SUPPORTED,
        "xpu_kernels": current_platform.is_xpu(),
    }
    actual = {
        provider: impl.supported
        for provider, impl in ir.ops.rotary_embedding.impls.items()
    }
    assert actual == expected


def test_rotary_embedding_query_only_has_only_native():
    """rotary_embedding_query_only currently has no custom provider implementations."""
    assert set(ir.ops.rotary_embedding_query_only.impls.keys()) == {"native"}


# ---------------------------------------------------------------------------
# Main test class — matches TestRMSNorm / TestFusedAddRMSNorm structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("n_tokens", NUM_TOKENS)
@pytest.mark.skipif(
    not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
    reason="Custom kernels only available on CUDA, ROCm and XPU",
)
class TestRotaryEmbedding:
    @classmethod
    def setup_class(cls, **kwargs):
        torch.set_default_device(current_platform.device_type)

    def test_native_semantics(self, n_tokens, head_size, dtype, is_neox_style):
        # Use small fixed sizes for fast deterministic semantic checks,
        # matching the pattern in test_layernorm.py::TestRMSNorm::test_native_semantics.
        args = ir.ops.rotary_embedding.generate_inputs(
            num_tokens=4, head_size=head_size, dtype=dtype, is_neox_style=is_neox_style
        )
        positions, query, key = args[0], args[1], args[2]
        head_size_arg, rotary_dim = args[3], args[4]
        cos_sin_cache = args[5]

        q_out, k_out = rotary_embedding_native(*args)

        # shape / dtype / device
        assert q_out.shape == query.shape
        assert k_out.shape == key.shape
        assert q_out.dtype == dtype
        assert q_out.device == query.device

        # zero-angle identity: cos=1, sin=0 → output equals input
        identity_cache = torch.cat(
            [
                torch.ones(1, rotary_dim // 2, dtype=dtype),
                torch.zeros(1, rotary_dim // 2, dtype=dtype),
            ],
            dim=-1,
        )
        pos_zero = torch.zeros(4, dtype=torch.int64)
        q_id = torch.randn(4, 8 * head_size_arg, dtype=dtype)
        k_id = torch.randn(4, 2 * head_size_arg, dtype=dtype)
        q_id_out, k_id_out = rotary_embedding_native(
            pos_zero,
            q_id,
            k_id,
            head_size_arg,
            rotary_dim,
            identity_cache,
            is_neox_style,
        )
        torch.testing.assert_close(q_id_out, q_id, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(k_id_out, k_id, rtol=1e-5, atol=1e-5)

        # pass-through region unchanged for partial rotary (rotary_dim = head_size // 2)
        partial_rd = head_size_arg // 2
        partial_cache = torch.randn(1, partial_rd, dtype=dtype)
        q_p = torch.randn(4, 8 * head_size_arg, dtype=dtype)
        k_p = torch.randn(4, 2 * head_size_arg, dtype=dtype)
        q_p_out, _ = rotary_embedding_native(
            pos_zero, q_p, k_p, head_size_arg, partial_rd, partial_cache, is_neox_style
        )
        q_pass_in = q_p.view(4, 8, head_size_arg)[..., partial_rd:]
        q_pass_out = q_p_out.view(4, 8, head_size_arg)[..., partial_rd:]
        torch.testing.assert_close(q_pass_out, q_pass_in, rtol=0.0, atol=0.0)

        # rotary_embedding_query_only must return the same Q as rotary_embedding
        q_full, _ = rotary_embedding_native(*clone_args(args))
        q_qonly = rotary_embedding_query_only_native(
            positions,
            query.clone(),
            head_size_arg,
            rotary_dim,
            cos_sin_cache,
            is_neox_style,
        )
        torch.testing.assert_close(q_qonly, q_full, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.rotary_embedding))
    def test_impls(self, n_tokens, head_size, dtype, is_neox_style, provider):
        impl = ir.ops.rotary_embedding.impls[provider]
        args = ir.ops.rotary_embedding.generate_inputs(
            num_tokens=n_tokens,
            head_size=head_size,
            dtype=dtype,
            is_neox_style=is_neox_style,
        )

        if not impl.supports_args(*args):
            pytest.skip(f"{provider} does not support args")

        ref_q, ref_k = rotary_embedding_native(*clone_args(args))
        out_q, out_k = impl.impl_fn(*clone_args(args))
        assert_close(ir.ops.rotary_embedding, out_q, ref_q)
        assert_close(ir.ops.rotary_embedding, out_k, ref_k)

        # dispatched call must match direct call
        with ir.ops.rotary_embedding.set_priority([provider, "native"]):
            q_dispatched, k_dispatched = ir.ops.rotary_embedding(*clone_args(args))
        q_direct, k_direct = impl.impl_fn(*clone_args(args))
        torch.testing.assert_close(q_dispatched, q_direct, rtol=0.0, atol=0.0)
        torch.testing.assert_close(k_dispatched, k_direct, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("provider", ["vllm_c"])
    def test_inplace_semantics(
        self, n_tokens, head_size, dtype, is_neox_style, provider
    ):
        """Test that inplace impls reuse inputs for maybe_inplace but not default."""
        impl = ir.ops.rotary_embedding.impls[provider]
        if not impl.supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        args = ir.ops.rotary_embedding.generate_inputs(
            num_tokens=n_tokens,
            head_size=head_size,
            dtype=dtype,
            is_neox_style=is_neox_style,
        )
        query, key = args[1], args[2]
        q_ptr, k_ptr = query.data_ptr(), key.data_ptr()
        query_copy, key_copy = query.clone(), key.clone()

        # default overload — must NOT be in-place even with an inplace impl
        with ir.ops.rotary_embedding.set_priority([provider, "native"]):
            q_default, k_default = ir.ops.rotary_embedding(*clone_args(args))

        assert q_default.data_ptr() != q_ptr
        assert k_default.data_ptr() != k_ptr
        torch.testing.assert_close(query, query_copy, rtol=0.0, atol=0.0)
        torch.testing.assert_close(key, key_copy, rtol=0.0, atol=0.0)

        # maybe_inplace overload — must reuse input buffers.
        # Capture data_ptr AFTER cloning, since maybe_inplace reuses the
        # clone's memory, not the original's (same pattern as test_layernorm.py).
        args_inplace = clone_args(args)
        q_inplace_ptr = args_inplace[1].data_ptr()
        k_inplace_ptr = args_inplace[2].data_ptr()

        with ir.ops.rotary_embedding.set_priority([provider, "native"]):
            q_inplace, k_inplace = ir.ops.rotary_embedding.maybe_inplace(*args_inplace)

        assert q_inplace.data_ptr() == q_inplace_ptr
        assert k_inplace.data_ptr() == k_inplace_ptr

        # both overloads must produce identical results
        torch.testing.assert_close(q_default, q_inplace, atol=0.0, rtol=0.0)
        torch.testing.assert_close(k_default, k_inplace, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("provider", ["vllm_c", "aiter", "xpu_kernels", "native"])
    def test_torch_opcheck(self, n_tokens, head_size, dtype, is_neox_style, provider):
        if not ir.ops.rotary_embedding.impls[provider].supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        args = ir.ops.rotary_embedding.generate_inputs(
            num_tokens=n_tokens,
            head_size=head_size,
            dtype=dtype,
            is_neox_style=is_neox_style,
        )

        with ir.ops.rotary_embedding.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.rotary_embedding.default, args)

            # Inplace impls return aliases of inputs, which opcheck forbids.
            # We accept this for maybe_inplace (same caveat as fused_add_rms_norm).
            if not ir.ops.rotary_embedding.impls[provider].inplace:
                torch.library.opcheck(
                    torch.ops.vllm_ir.rotary_embedding.maybe_inplace, args
                )


# ---------------------------------------------------------------------------
# aiter-specific: reject partial rotary_dim
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="aiter only available on ROCm",
)
def test_aiter_rejects_partial_rotary_dim():
    from vllm.kernels.aiter_ops import AITER_TRITON_ROTARY_SUPPORTED

    if not AITER_TRITON_ROTARY_SUPPORTED:
        pytest.skip("AITER Triton rotary not enabled")

    torch.set_default_device(current_platform.device_type)
    impl = ir.ops.rotary_embedding.impls["aiter"]
    args = ir.ops.rotary_embedding.generate_inputs(
        num_tokens=8, head_size=128, rotary_dim=64, dtype=torch.float16
    )
    with pytest.raises(AssertionError, match="rotary_dim=head_size"):
        impl.impl_fn(*args)
