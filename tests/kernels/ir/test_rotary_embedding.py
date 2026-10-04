# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm.kernels  # noqa: F401
from tests.ir.ir_test_utils import assert_close, clone_args, supported_providers
from vllm import ir
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.platforms import current_platform

rope_native = ir.ops.rotary_embedding.impls["native"].impl_fn

IS_GPGPU_DEVICE = current_platform.is_cuda_alike() or current_platform.is_xpu()

HEAD_SIZES = [64, 128, 256]
N_TOKENS = [7, 83]


def _inputs(n_tokens, head_size, dtype, is_neox):
    return ir.ops.rotary_embedding.generate_inputs(
        num_tokens=n_tokens,
        num_heads=8,
        num_kv_heads=2,
        head_size=head_size,
        rotary_dim=head_size,
        dtype=dtype,
        is_neox=is_neox,
        device=current_platform.device_type,
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="vllm_c rotary kernel is CUDA/ROCm only",
)
def test_rotary_embedding_registration():
    expected = {"native": True, "vllm_c": current_platform.is_cuda_alike()}
    actual = {
        provider: impl.supported
        for provider, impl in ir.ops.rotary_embedding.impls.items()
    }
    assert actual == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n_tokens", N_TOKENS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.skipif(
    not IS_GPGPU_DEVICE,
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
class TestRotaryEmbedding:
    def test_native_matches_forward_static(self, dtype, n_tokens, head_size, is_neox):
        positions, query, key, hs, rotary_dim, cos_sin_cache, neox = _inputs(
            n_tokens, head_size, dtype, is_neox
        )

        q_ir, k_ir = rope_native(
            positions, query.clone(), key.clone(), hs, rotary_dim, cos_sin_cache, neox
        )
        q_ref, k_ref = RotaryEmbedding.forward_static(
            positions, query.clone(), key.clone(), hs, rotary_dim, cos_sin_cache, neox
        )

        torch.testing.assert_close(q_ir, q_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(k_ir, k_ref, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize(
        "provider", supported_providers(ir.ops.rotary_embedding)
    )
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

    def test_default_overload_not_inplace(self, dtype, n_tokens, head_size, is_neox):
        provider = "vllm_c"
        impl = ir.ops.rotary_embedding.impls[provider]
        if not impl.supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        positions, query, key, hs, rotary_dim, cos_sin_cache, neox = _inputs(
            n_tokens, head_size, dtype, is_neox
        )
        q_default, k_default = query.clone(), key.clone()
        q_ptr, k_ptr = q_default.data_ptr(), k_default.data_ptr()

        with ir.ops.rotary_embedding.set_priority([provider, "native"]):
            out_q, out_k = ir.ops.rotary_embedding(
                positions, q_default, k_default, hs, rotary_dim, cos_sin_cache, neox
            )

        assert out_q.data_ptr() != q_ptr
        assert out_k.data_ptr() != k_ptr
        torch.testing.assert_close(q_default, query, rtol=0.0, atol=0.0)
        torch.testing.assert_close(k_default, key, rtol=0.0, atol=0.0)

        q_inplace, k_inplace = query.clone(), key.clone()
        with ir.ops.rotary_embedding.set_priority([provider, "native"]):
            ip_q, ip_k = ir.ops.rotary_embedding.maybe_inplace(
                positions, q_inplace, k_inplace, hs, rotary_dim, cos_sin_cache, neox
            )
        assert ip_q.data_ptr() == q_inplace.data_ptr()
        assert ip_k.data_ptr() == k_inplace.data_ptr()
        torch.testing.assert_close(ip_q, out_q, rtol=0.0, atol=0.0)
        torch.testing.assert_close(ip_k, out_k, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize(
        "provider", supported_providers(ir.ops.rotary_embedding)
    )
    def test_torch_opcheck(self, dtype, n_tokens, head_size, is_neox, provider):
        if not ir.ops.rotary_embedding.impls[provider].supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        args = _inputs(n_tokens, head_size, dtype, is_neox)
        with ir.ops.rotary_embedding.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.rotary_embedding.default, args)
