# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

# This registers op implementations
import vllm.kernels  # noqa: F401
from tests.ir.ir_test_utils import supported_providers
from vllm import ir
from vllm.ir.ops.quant import get_tma_aligned_size
from vllm.platforms import current_platform

IS_CUDA = current_platform.is_cuda()
IS_ROCM = current_platform.is_rocm()
IS_CUDA_ALIKE = current_platform.is_cuda_alike()
IS_XPU = current_platform.is_xpu()
IS_GPGPU = IS_CUDA_ALIKE or IS_XPU

FP8_DTYPE = current_platform.fp8_dtype()

static_quant_fp8_native = ir.ops.static_quant_fp8.impls["native"].impl_fn
static_group_quant_fp8_native = ir.ops.static_group_quant_fp8.impls["native"].impl_fn
dynamic_quant_fp8_native = ir.ops.dynamic_quant_fp8.impls["native"].impl_fn
dynamic_group_quant_fp8_native = ir.ops.dynamic_group_quant_fp8.impls["native"].impl_fn

_SKIP_UNSUPPORTED = pytest.mark.skipif(
    not IS_GPGPU,
    reason="Currently only kernels on CUDA, ROCm and XPU",
)


@_SKIP_UNSUPPORTED
class TestRegistration:
    def test_static_quant_fp8(self):
        expected = {
            "native": True,
            "vllm_c": IS_CUDA_ALIKE or IS_XPU,
            "aiter": IS_ROCM,
        }
        actual = {
            provider: impl.supported
            for provider, impl in ir.ops.static_quant_fp8.impls.items()
        }
        assert actual == expected

    def test_static_group_quant_fp8(self):
        expected = {
            "native": True,
            "vllm_c": IS_CUDA_ALIKE or IS_XPU,
        }
        actual = {
            provider: impl.supported
            for provider, impl in ir.ops.static_group_quant_fp8.impls.items()
        }
        assert actual == expected

    def test_dynamic_quant_fp8(self):
        expected = {
            "native": True,
            "vllm_c": IS_CUDA_ALIKE or IS_XPU,
            "aiter": IS_ROCM,
        }
        actual = {
            provider: impl.supported
            for provider, impl in ir.ops.dynamic_quant_fp8.impls.items()
        }
        assert actual == expected

    def test_dynamic_group_quant_fp8(self):
        expected = {
            "native": True,
            "vllm_c": IS_CUDA,
            "aiter": IS_ROCM,
            "triton": IS_CUDA_ALIKE,
            "xpu_kernels": IS_XPU,
        }
        actual = {
            provider: impl.supported
            for provider, impl in ir.ops.dynamic_group_quant_fp8.impls.items()
        }
        assert actual == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_tokens", [1, 8, 12])
@pytest.mark.parametrize("hidden_size", [128, 256])
@_SKIP_UNSUPPORTED
class TestStaticQuantFP8:
    @classmethod
    def setup_class(cls, **kwargs):
        torch.set_default_device(current_platform.device_type)

    @pytest.mark.parametrize("num_token_padding", [None, 16])
    def test_native_semantics(self, dtype, n_tokens, hidden_size, num_token_padding):
        x = torch.randn(n_tokens, hidden_size, dtype=dtype)
        scale = torch.full((1,), 0.5, dtype=torch.float32)
        expected_tokens = (
            max(num_token_padding, n_tokens) if num_token_padding else n_tokens
        )

        out = static_quant_fp8_native(x, scale, FP8_DTYPE, num_token_padding)
        assert out.shape[0] == expected_tokens
        assert out.shape[1:] == x.shape[1:]
        assert out.dtype == FP8_DTYPE
        assert out.device == x.device
        assert out.is_contiguous()

        # Scale invariance: (2x) / (2s) == x / s (valid rows only)
        out2 = static_quant_fp8_native(x * 2, scale * 2, FP8_DTYPE, num_token_padding)
        torch.testing.assert_close(
            out.to(torch.float32)[:n_tokens],
            out2.to(torch.float32)[:n_tokens],
            atol=0.0,
            rtol=0.0,
        )

        # Per-token scale produces correct shape
        scale_pt = torch.full((n_tokens, 1), 0.5, dtype=torch.float32)
        out3 = static_quant_fp8_native(x, scale_pt, FP8_DTYPE, num_token_padding)
        assert out3.shape[0] == expected_tokens
        assert out3.shape[1:] == x.shape[1:]
        assert out3.dtype == FP8_DTYPE

    @pytest.mark.parametrize("num_token_padding", [None, 16])
    @pytest.mark.parametrize("provider", supported_providers(ir.ops.static_quant_fp8))
    def test_impls(self, dtype, n_tokens, hidden_size, num_token_padding, provider):
        impl = ir.ops.static_quant_fp8.impls[provider]

        x = torch.randn(n_tokens, hidden_size, dtype=dtype)
        scale = torch.full((1,), 0.5, dtype=torch.float32)
        args = (x, scale, FP8_DTYPE, num_token_padding)

        if not impl.supports_args(*args):
            pytest.skip(f"{provider} does not support these args")

        out_impl = impl.impl_fn(*args)
        out_native = static_quant_fp8_native(*args)

        torch.testing.assert_close(
            out_impl.to(torch.float32)[:n_tokens],
            out_native.to(torch.float32)[:n_tokens],
            rtol=0.0,
            atol=0.0,
        )

        # Dispatched call must match direct impl exactly (padding rows are
        # uninitialized, so only compare actual token rows)
        with ir.ops.static_quant_fp8.set_priority([provider, "native"]):
            out_dispatch = ir.ops.static_quant_fp8(*args)
        torch.testing.assert_close(
            out_dispatch.to(torch.float32)[:n_tokens],
            out_impl.to(torch.float32)[:n_tokens],
            atol=0.0,
            rtol=0.0,
        )

        # Different inputs must produce different outputs
        args_diff = (x + 1, scale, FP8_DTYPE, num_token_padding)
        out_impl_diff = impl.impl_fn(*args_diff)
        assert not torch.all(
            out_impl.to(torch.float32)[:n_tokens]
            == out_impl_diff.to(torch.float32)[:n_tokens]
        )

    @pytest.mark.parametrize(
        "provider", supported_providers(ir.ops.static_quant_fp8) + ["native"]
    )
    def test_torch_opcheck(self, dtype, n_tokens, hidden_size, provider):
        x = torch.randn(n_tokens, hidden_size, dtype=dtype)
        scale = torch.full((1,), 0.5, dtype=torch.float32)
        args = (x, scale, FP8_DTYPE)
        if not ir.ops.static_quant_fp8.impls[provider].supports_args(*args):
            pytest.skip(f"{provider} does not support these args")

        with ir.ops.static_quant_fp8.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.static_quant_fp8, args)


@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_tokens", [1, 8, 12])
@pytest.mark.parametrize("hidden_size", [128, 256])
@_SKIP_UNSUPPORTED
class TestStaticGroupQuantFP8:
    @classmethod
    def setup_class(cls, **kwargs):
        torch.set_default_device(current_platform.device_type)

    @pytest.mark.parametrize("num_token_padding", [None, 16])
    def test_native_semantics(
        self, dtype, n_tokens, hidden_size, group_size, num_token_padding
    ):
        x = torch.randn(n_tokens, hidden_size, dtype=dtype)
        scale = torch.full(
            (n_tokens, hidden_size // group_size), 0.5, dtype=torch.float32
        )
        expected_tokens = (
            max(num_token_padding, n_tokens) if num_token_padding else n_tokens
        )

        out = static_group_quant_fp8_native(x, scale, FP8_DTYPE, num_token_padding)
        assert out.shape[0] == expected_tokens
        assert out.shape[1:] == x.shape[1:]
        assert out.dtype == FP8_DTYPE
        assert out.device == x.device
        assert out.is_contiguous()

        # Scale invariance: (2x) / (2s) == x / s (valid rows only)
        out2 = static_group_quant_fp8_native(
            x * 2, scale * 2, FP8_DTYPE, num_token_padding
        )
        torch.testing.assert_close(
            out.to(torch.float32)[:n_tokens],
            out2.to(torch.float32)[:n_tokens],
            atol=0.0,
            rtol=0.0,
        )

        # When group_size == hidden_size, one group per token == per-token static quant
        if group_size == hidden_size:
            out_per_token = static_quant_fp8_native(x, scale, FP8_DTYPE)
            torch.testing.assert_close(
                out.to(torch.float32)[:n_tokens],
                out_per_token.to(torch.float32),
                atol=0.0,
                rtol=0.0,
            )

    @pytest.mark.parametrize("num_token_padding", [None, 16])
    @pytest.mark.parametrize(
        "provider", supported_providers(ir.ops.static_group_quant_fp8)
    )
    def test_impls(
        self, dtype, n_tokens, hidden_size, group_size, num_token_padding, provider
    ):
        impl = ir.ops.static_group_quant_fp8.impls[provider]

        x = torch.randn(n_tokens, hidden_size, dtype=dtype)
        scale = torch.full(
            (n_tokens, hidden_size // group_size), 0.5, dtype=torch.float32
        )
        args = (x, scale, FP8_DTYPE, num_token_padding)

        if not impl.supports_args(*args):
            pytest.skip(f"{provider} does not support these args")

        out_impl = impl.impl_fn(*args)
        out_native = static_group_quant_fp8_native(*args)

        torch.testing.assert_close(
            out_impl.to(torch.float32)[:n_tokens],
            out_native.to(torch.float32)[:n_tokens],
            atol=0.0,
            rtol=0.0,
        )

        # Dispatched call must match direct impl exactly (padding rows are
        # uninitialized, so only compare actual token rows)
        with ir.ops.static_group_quant_fp8.set_priority([provider, "native"]):
            out_dispatch = ir.ops.static_group_quant_fp8(*args)
        torch.testing.assert_close(
            out_dispatch.to(torch.float32)[:n_tokens],
            out_impl.to(torch.float32)[:n_tokens],
            atol=0.0,
            rtol=0.0,
        )

        # Different inputs must produce different outputs
        args_diff = (x + 1, scale, FP8_DTYPE, num_token_padding)
        out_impl_diff = impl.impl_fn(*args_diff)
        assert not torch.all(
            out_impl.to(torch.float32)[:n_tokens]
            == out_impl_diff.to(torch.float32)[:n_tokens]
        )

        # When group_size == hidden_size, one group per token == per-token static quant
        if group_size == hidden_size:
            out_per_token = static_quant_fp8_native(x, scale, FP8_DTYPE)
            torch.testing.assert_close(
                out_impl.to(torch.float32)[:n_tokens],
                out_per_token.to(torch.float32),
                atol=0.0,
                rtol=0.0,
            )

    @pytest.mark.parametrize(
        "provider", supported_providers(ir.ops.static_group_quant_fp8) + ["native"]
    )
    def test_torch_opcheck(self, dtype, n_tokens, hidden_size, group_size, provider):
        x = torch.randn(n_tokens, hidden_size, dtype=dtype)
        scale = torch.full(
            (n_tokens, hidden_size // group_size), 0.5, dtype=torch.float32
        )
        args = (x, scale, FP8_DTYPE)
        if not ir.ops.static_group_quant_fp8.impls[provider].supports_args(*args):
            pytest.skip(f"{provider} does not support these args")

        with ir.ops.static_group_quant_fp8.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.static_group_quant_fp8, args)


@pytest.mark.parametrize("per_token", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_tokens", [1, 8, 12])
@pytest.mark.parametrize("hidden_size", [128, 256])
@_SKIP_UNSUPPORTED
class TestDynamicQuantFP8:
    @classmethod
    def setup_class(cls, **kwargs):
        torch.set_default_device(current_platform.device_type)

    @pytest.mark.parametrize("num_token_padding", [None, 16])
    def test_native_semantics(
        self, dtype, n_tokens, hidden_size, per_token, num_token_padding
    ):
        x = torch.randn(n_tokens, hidden_size, dtype=dtype)
        expected_tokens = (
            max(num_token_padding, n_tokens) if num_token_padding else n_tokens
        )

        out, scale = dynamic_quant_fp8_native(
            x, per_token, FP8_DTYPE, num_token_padding=num_token_padding
        )

        assert out.shape[0] == expected_tokens
        assert out.shape[1:] == x.shape[1:]
        assert out.dtype == FP8_DTYPE
        assert out.device == x.device
        assert out.is_contiguous()
        assert scale.dtype == torch.float32
        assert (scale > 0).all()

        if per_token:
            assert scale.shape == (n_tokens, 1)
        else:
            assert scale.shape == (1,)

        # Dequantized output should approximate original (valid rows only)
        x_deq = out.to(torch.float32)[:n_tokens] * scale[:n_tokens]
        torch.testing.assert_close(x_deq, x.to(torch.float32), rtol=0.15, atol=0.01)

        # static_quant with the dynamic scale should reproduce dynamic
        # quant output exactly
        x_q_static = static_quant_fp8_native(x, scale, FP8_DTYPE)
        torch.testing.assert_close(
            out.to(torch.float32)[:n_tokens],
            x_q_static.to(torch.float32),
            atol=0.0,
            rtol=0.0,
        )

    @pytest.mark.parametrize("num_token_padding", [None, 16])
    @pytest.mark.parametrize("provider", supported_providers(ir.ops.dynamic_quant_fp8))
    def test_impls(
        self, dtype, n_tokens, hidden_size, per_token, num_token_padding, provider
    ):
        impl = ir.ops.dynamic_quant_fp8.impls[provider]

        x = torch.randn(n_tokens, hidden_size, dtype=dtype)
        args = (x, per_token, FP8_DTYPE, None, num_token_padding)

        if not impl.supports_args(*args):
            pytest.skip(f"{provider} does not support these args")

        out_impl, scale_impl = impl.impl_fn(*args)

        assert (scale_impl[:n_tokens] > 0).all()
        if per_token:
            assert scale_impl.shape[0] >= n_tokens and scale_impl.shape[1:] == (1,)
        else:
            assert scale_impl.shape == (1,)

        # Verify the impl correctly quantizes: dequantized output should approximate
        # the original input regardless of internal precision differences across impls.
        x_deq_impl = out_impl.to(torch.float32)[:n_tokens] * scale_impl[:n_tokens]
        torch.testing.assert_close(
            x_deq_impl, x.to(torch.float32), rtol=0.15, atol=0.01
        )

        # Dispatched call must match direct impl exactly (padding rows are
        # uninitialized, so only compare actual token rows)
        with ir.ops.dynamic_quant_fp8.set_priority([provider, "native"]):
            out_dispatch, scale_dispatch = ir.ops.dynamic_quant_fp8(*args)
        torch.testing.assert_close(
            out_dispatch.to(torch.float32)[:n_tokens],
            out_impl.to(torch.float32)[:n_tokens],
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            scale_dispatch[:n_tokens], scale_impl[:n_tokens], atol=0.0, rtol=0.0
        )

        # Different inputs must produce different outputs
        args_diff = (x + 1, per_token, FP8_DTYPE, None, num_token_padding)
        out_impl_diff, _ = impl.impl_fn(*args_diff)
        assert not torch.all(
            out_impl.to(torch.float32)[:n_tokens]
            == out_impl_diff.to(torch.float32)[:n_tokens]
        )

        # static_quant with the impl's dynamic scale should reproduce the impl's output
        x_q_static = static_quant_fp8_native(x, scale_impl[:n_tokens], FP8_DTYPE)
        torch.testing.assert_close(
            out_impl.to(torch.float32)[:n_tokens],
            x_q_static.to(torch.float32),
            atol=0.0,
            rtol=1e-1,
        )

    @pytest.mark.parametrize(
        "provider", supported_providers(ir.ops.dynamic_quant_fp8) + ["native"]
    )
    def test_torch_opcheck(self, dtype, n_tokens, hidden_size, per_token, provider):
        x = torch.randn(n_tokens, hidden_size, dtype=dtype)
        args = (x, per_token, FP8_DTYPE)
        if not ir.ops.dynamic_quant_fp8.impls[provider].supports_args(*args):
            pytest.skip(f"{provider} does not support these args")

        with ir.ops.dynamic_quant_fp8.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.dynamic_quant_fp8, args)


@pytest.mark.parametrize("use_ue8m0", [False, True] if IS_CUDA or IS_XPU else [False])
@pytest.mark.parametrize("scale_alignment", [1, 4])
@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_tokens", [1, 8, 12])
@pytest.mark.parametrize("hidden_size", [128, 256])
@_SKIP_UNSUPPORTED
class TestDynamicGroupQuantFP8:
    @classmethod
    def setup_class(cls, **kwargs):
        torch.set_default_device(current_platform.device_type)

    def test_native_semantics(
        self,
        dtype,
        n_tokens,
        hidden_size,
        group_size,
        column_major,
        scale_alignment,
        use_ue8m0,
    ):
        x = torch.randn(n_tokens, hidden_size, dtype=dtype)
        n_groups = hidden_size // group_size

        x_q, x_s = dynamic_group_quant_fp8_native(
            x, [group_size], column_major, use_ue8m0, FP8_DTYPE, scale_alignment
        )

        assert x_q.shape == x.shape
        assert x_q.dtype == FP8_DTYPE
        assert x_q.device == x.device
        assert x_q.is_contiguous()
        assert x_s.shape == (n_tokens, n_groups)
        assert x_s.dtype == torch.float32
        assert (x_s.contiguous() > 0).all()

        if not column_major:
            assert x_s.stride() == (n_groups, 1)
        elif scale_alignment == 1:
            assert x_s.stride() == (1, n_tokens)
        else:
            tma_m = get_tma_aligned_size(n_tokens, scale_alignment)
            assert x_s.stride() == (1, tma_m)

        # Dequantized output should approximate original
        x_deq = x_q.to(torch.float32) * x_s.contiguous().repeat_interleave(
            group_size, dim=-1
        )
        torch.testing.assert_close(x_deq, x.to(torch.float32), rtol=0.15, atol=0.01)

        # static_group_quant with the dynamic scale should reproduce dynamic
        # group quant output. dynamic_group_quant uses true division
        # (x/scale) while static_group_quant uses multiply-by-reciprocal
        # (x * (1/scale)) which might cause some discrepancies and is why
        # we need to use rtol=0.15.
        x_q_static = static_group_quant_fp8_native(x, x_s.contiguous(), FP8_DTYPE)
        torch.testing.assert_close(
            x_q.to(torch.float32),
            x_q_static.to(torch.float32),
            atol=0.0,
            rtol=0.15,
        )

    @pytest.mark.parametrize(
        "provider", supported_providers(ir.ops.dynamic_group_quant_fp8)
    )
    def test_impls(
        self,
        dtype,
        n_tokens,
        hidden_size,
        group_size,
        column_major,
        scale_alignment,
        use_ue8m0,
        provider,
    ):
        impl = ir.ops.dynamic_group_quant_fp8.impls[provider]

        x = torch.randn(n_tokens, hidden_size, dtype=dtype)
        n_groups = hidden_size // group_size
        args = (x, [group_size], column_major, use_ue8m0, FP8_DTYPE, scale_alignment)

        if not impl.supports_args(*args):
            pytest.skip(f"{provider} does not support these args")

        x_q_impl, x_s_impl = impl.impl_fn(*args)

        assert x_s_impl.shape == (n_tokens, n_groups)
        assert (x_s_impl.contiguous() > 0).all()

        if not column_major:
            assert x_s_impl.stride() == (n_groups, 1)
        elif scale_alignment == 1:
            assert x_s_impl.stride() == (1, n_tokens)
        else:
            tma_m = get_tma_aligned_size(n_tokens, scale_alignment)
            assert x_s_impl.stride() == (1, tma_m)

        # Dequantized output should approximate the original input
        # regardless of the precision differences across impls.
        x_deq_impl = x_q_impl.to(
            torch.float32
        ) * x_s_impl.contiguous().repeat_interleave(group_size, dim=-1)
        torch.testing.assert_close(
            x_deq_impl, x.to(torch.float32), rtol=0.15, atol=0.01
        )

        # Dispatched call must match direct impl exactly
        with ir.ops.dynamic_group_quant_fp8.set_priority([provider, "native"]):
            x_q_dispatch, x_s_dispatch = ir.ops.dynamic_group_quant_fp8(*args)
        torch.testing.assert_close(
            x_q_dispatch.to(torch.float32),
            x_q_impl.to(torch.float32),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            x_s_dispatch.contiguous(),
            x_s_impl.contiguous(),
            atol=0.0,
            rtol=0.0,
        )

        # Different inputs must produce different outputs
        x_q_impl_diff, _ = impl.impl_fn(
            x + 1, [group_size], column_major, use_ue8m0, FP8_DTYPE, scale_alignment
        )
        assert not torch.all(
            x_q_impl.to(torch.float32) == x_q_impl_diff.to(torch.float32)
        )

        # static_group_quant with the impl's dynamic scale should reproduce
        # the impl's output
        x_q_static = static_group_quant_fp8_native(x, x_s_impl.contiguous(), FP8_DTYPE)
        torch.testing.assert_close(
            x_q_impl.to(torch.float32),
            x_q_static.to(torch.float32),
            atol=0.0,
            rtol=1e-1,
        )

    @pytest.mark.parametrize(
        "provider", supported_providers(ir.ops.dynamic_group_quant_fp8) + ["native"]
    )
    def test_torch_opcheck(
        self,
        dtype,
        n_tokens,
        hidden_size,
        group_size,
        column_major,
        scale_alignment,
        use_ue8m0,
        provider,
    ):
        x = torch.randn(n_tokens, hidden_size, dtype=dtype)
        args = (x, [group_size], column_major, use_ue8m0, FP8_DTYPE, scale_alignment)

        if not ir.ops.dynamic_group_quant_fp8.impls[provider].supports_args(*args):
            pytest.skip(f"{provider} does not support these args")

        with ir.ops.dynamic_group_quant_fp8.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.dynamic_group_quant_fp8, args)


@pytest.mark.skipif(not IS_ROCM, reason="aiter is only supported on ROCm")
def test_aiter_static_quant_fp8_rejects_unsupported_args():
    torch.set_default_device(current_platform.device_type)
    impl = ir.ops.static_quant_fp8.impls["aiter"]
    x = torch.randn(8, 2048, dtype=torch.float16)
    scale = torch.full((1,), 0.5, dtype=torch.float32)
    assert not impl.supports_args(x.to(torch.float32), scale, FP8_DTYPE, None), (
        "aiter should reject float32"
    )
    assert not impl.supports_args(x, scale, FP8_DTYPE, 16), (
        "aiter should reject num_token_padding"
    )


@pytest.mark.skipif(not IS_ROCM, reason="aiter is only supported on ROCm")
def test_aiter_dynamic_quant_fp8_rejects_unsupported_args():
    torch.set_default_device(current_platform.device_type)
    impl = ir.ops.dynamic_quant_fp8.impls["aiter"]
    x = torch.randn(8, 2048, dtype=torch.float16)
    assert not impl.supports_args(x.to(torch.float32), True, FP8_DTYPE, None, None), (
        "aiter should reject float32"
    )
    assert not impl.supports_args(x, True, FP8_DTYPE, None, 16), (
        "aiter should reject num_token_padding"
    )
    assert not impl.supports_args(x, True, FP8_DTYPE, torch.tensor(1.0), None), (
        "aiter should reject scale_ub"
    )


@pytest.mark.skipif(not IS_ROCM, reason="aiter is only supported on ROCm")
def test_aiter_dynamic_group_quant_fp8_rejects_group_size_64():
    torch.set_default_device(current_platform.device_type)
    impl = ir.ops.dynamic_group_quant_fp8.impls["aiter"]
    x = torch.randn(8, 2048, dtype=torch.float16)
    assert not impl.supports_args(x, [64], False, False, FP8_DTYPE, 1), (
        "aiter should reject group_size=64"
    )
    assert not impl.supports_args(
        x.to(torch.float32), [128], False, False, FP8_DTYPE, 1
    ), "aiter should reject float32"
    assert not impl.supports_args(
        x.t().contiguous().t(), [128], False, False, FP8_DTYPE, 1
    ), "aiter should reject non-contiguous"
    assert not impl.supports_args(x, [128], True, False, FP8_DTYPE, 1), (
        "aiter should reject column_major"
    )
    assert not impl.supports_args(x, [128], False, True, FP8_DTYPE, 1), (
        "aiter should reject use_ue8m0"
    )


@pytest.mark.skipif(
    not IS_CUDA_ALIKE, reason="triton is only supported on CUDA-like platforms"
)
def test_triton_dynamic_group_quant_fp8_rejects_unsupported_args():
    torch.set_default_device(current_platform.device_type)
    impl = ir.ops.dynamic_group_quant_fp8.impls["triton"]
    x = torch.randn(8, 2048, dtype=torch.float16)
    assert not impl.supports_args(x, [128], False, False, torch.float8_e4m3fnuz, 1), (
        "triton should reject fnuz dtype"
    )
    assert not impl.supports_args(x, [128], False, False, FP8_DTYPE, 4), (
        "triton should reject row-major with scale_alignment=4"
    )
    assert not impl.supports_args(
        x.t().contiguous().t(), [128], False, False, FP8_DTYPE, 1
    ), "triton should reject non-contiguous last dim"


@pytest.mark.skipif(
    not IS_GPGPU, reason="vllm_c is only supported on CUDA-like and XPU"
)
def test_vllm_c_static_quant_fp8_rejects_unsupported_args():
    torch.set_default_device(current_platform.device_type)
    impl = ir.ops.static_quant_fp8.impls["vllm_c"]
    x_3d = torch.randn(2, 8, 2048, dtype=torch.float16)
    scale = torch.full((1,), 0.5, dtype=torch.float32)
    assert not impl.supports_args(x_3d, scale, FP8_DTYPE, None), (
        "vllm_c should reject non-2D input"
    )


@pytest.mark.skipif(
    not IS_GPGPU, reason="vllm_c is only supported on CUDA-like and XPU"
)
def test_vllm_c_static_group_quant_fp8_rejects_unsupported_args():
    torch.set_default_device(current_platform.device_type)
    impl = ir.ops.static_group_quant_fp8.impls["vllm_c"]
    x = torch.randn(8, 2048, dtype=torch.float16)
    scale_2d = torch.full((8, 2048 // 128), 0.5, dtype=torch.float32)
    scale_1d = torch.full((2048 // 128,), 0.5, dtype=torch.float32)
    x_3d = torch.randn(2, 8, 2048, dtype=torch.float16)
    assert not impl.supports_args(x, scale_1d, FP8_DTYPE, None), (
        "vllm_c should reject 1D scale"
    )
    assert not impl.supports_args(x_3d, scale_2d, FP8_DTYPE, None), (
        "vllm_c should reject non-2D input"
    )


@pytest.mark.skipif(
    not IS_GPGPU, reason="vllm_c is only supported on CUDA-like and XPU"
)
def test_vllm_c_dynamic_quant_fp8_rejects_unsupported_args():
    torch.set_default_device(current_platform.device_type)
    impl = ir.ops.dynamic_quant_fp8.impls["vllm_c"]
    x = torch.randn(8, 2048, dtype=torch.float16)
    x_3d = torch.randn(2, 8, 2048, dtype=torch.float16)
    scale_ub = torch.tensor(1.0)
    assert not impl.supports_args(x, False, FP8_DTYPE, scale_ub, None), (
        "vllm_c should reject per_tensor with scale_ub"
    )
    assert not impl.supports_args(x_3d, True, FP8_DTYPE, None, None), (
        "vllm_c should reject non-2D input"
    )


@pytest.mark.skipif(
    not IS_CUDA, reason="vllm_c dynamic_group_quant_fp8 is only supported on CUDA"
)
def test_vllm_c_dynamic_group_quant_fp8_rejects_unsupported_args():
    torch.set_default_device(current_platform.device_type)
    impl = ir.ops.dynamic_group_quant_fp8.impls["vllm_c"]
    x = torch.randn(8, 2048, dtype=torch.float16)
    assert not impl.supports_args(
        x.t().contiguous().t(), [128], False, False, FP8_DTYPE, 1
    ), "vllm_c should reject non-contiguous"
    assert not impl.supports_args(x, [128], False, False, FP8_DTYPE, 2), (
        "vllm_c should reject scale_alignment=2"
    )
    assert not impl.supports_args(x, [96], False, False, FP8_DTYPE, 1), (
        "vllm_c should reject hidden not divisible by group_size"
    )


@pytest.mark.skipif(not IS_XPU, reason="xpu_kernels is only supported on XPU")
def test_xpu_kernels_dynamic_group_quant_fp8_rejects_unsupported_args():
    torch.set_default_device(current_platform.device_type)
    impl = ir.ops.dynamic_group_quant_fp8.impls["xpu_kernels"]
    x = torch.randn(8, 2048, dtype=torch.float16)
    assert not impl.supports_args(
        x.t().contiguous().t(), [128], False, False, FP8_DTYPE, 1
    ), "xpu_kernels should reject non-contiguous"
    assert not impl.supports_args(x, [128], False, False, FP8_DTYPE, 4), (
        "xpu_kernels should reject scale_alignment=4"
    )
    assert not impl.supports_args(x, [96], False, False, FP8_DTYPE, 1), (
        "xpu_kernels should reject hidden not divisible by group_size"
    )
