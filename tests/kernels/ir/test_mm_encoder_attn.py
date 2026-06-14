# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm.kernels  # noqa: F401
from tests.ir.ir_test_utils import assert_close, clone_args, supported_providers
from vllm import ir
from vllm.platforms import current_platform

mm_encoder_attn_native = ir.ops.mm_encoder_attn.impls["native"].impl_fn


def ref_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    query, key, value = (x.transpose(1, 2) for x in (query, key, value))
    attn_weights = scale * torch.matmul(query, key.transpose(2, 3))
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.matmul(attn_weights, value).transpose(1, 2)
    return out


@pytest.mark.skipif(
    not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
def test_mm_encoder_attn_registration():
    expected = {
        "native": True,
        "flash_attn": current_platform.is_cuda_alike() or current_platform.is_xpu(),
        "triton": current_platform.is_cuda_alike() or current_platform.is_xpu(),
    }

    actual = {
        provider: impl.supported
        for provider, impl in ir.ops.mm_encoder_attn.impls.items()
    }

    assert actual == expected


BATCH_SIZES = [1, 4]
SEQ_LENS = [1, 16]
NUM_HEADS = [1, 16]
NUM_KV_HEADS = [1]
HEAD_SIZES = [64, 80]
DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES + [torch.float32])
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.skipif(
    not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
class TestMMEncoderAttnNative:
    @classmethod
    def setup_class(cls, **kwargs):
        torch.set_default_device(current_platform.device_type)

    def test_native_semantics(self, dtype, batch_size, seq_len, num_heads, head_size):
        scale = 1.0 / (head_size**0.5)
        q, k, v, s = ir.ops.mm_encoder_attn.generate_inputs(
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_size=head_size,
            dtype=dtype,
            scale=scale,
        )
        out = mm_encoder_attn_native(q, k, v, s)

        assert out.shape == q.shape
        assert out.dtype == q.dtype
        assert out.device == q.device

        ref_out = ref_attention(q, k, v, scale)
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.skipif(
    not current_platform.is_cuda_alike() and not current_platform.is_xpu(),
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
class TestMMEncoderAttn:
    @classmethod
    def setup_class(cls, **kwargs):
        torch.set_default_device(current_platform.device_type)

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.mm_encoder_attn))
    def test_impls(
        self, dtype, batch_size, seq_len, num_heads, num_kv_heads, head_size, provider
    ):
        impl = ir.ops.mm_encoder_attn.impls[provider]
        scale = 1.0 / (head_size**0.5)
        q, k, v, s = ir.ops.mm_encoder_attn.generate_inputs(
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            scale=scale,
        )
        args = (q, k, v, s)

        if not impl.supports_args(*args):
            pytest.skip(f"{provider} does not support args")

        ref_output = mm_encoder_attn_native(*clone_args(args))
        output = impl.impl_fn(*clone_args(args))
        assert_close(ir.ops.mm_encoder_attn, output, ref_output)

        with ir.ops.mm_encoder_attn.set_priority([provider, "native"]):
            out_dispatched = ir.ops.mm_encoder_attn(*args)
        out_direct = impl.impl_fn(*args)
        torch.testing.assert_close(out_dispatched, out_direct, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("provider", supported_providers(ir.ops.mm_encoder_attn))
    def test_varlen(
        self, dtype, batch_size, seq_len, num_heads, num_kv_heads, head_size, provider
    ):
        impl = ir.ops.mm_encoder_attn.impls[provider]
        scale = 1.0 / (head_size**0.5)
        var_seq_lens = [seq_len, seq_len + 1]
        total_len = sum(var_seq_lens)
        q = torch.randn(1, total_len, num_heads, head_size, dtype=dtype)
        k = torch.randn(1, total_len, num_kv_heads, head_size, dtype=dtype)
        v = torch.randn(1, total_len, num_kv_heads, head_size, dtype=dtype)
        cu_seqlens = torch.tensor([0, var_seq_lens[0], total_len], dtype=torch.int32)
        max_seqlen = max(var_seq_lens)
        args = (q, k, v, scale, cu_seqlens, max_seqlen)

        if not impl.supports_args(*args):
            pytest.skip(f"{provider} does not support args")

        ref_output = mm_encoder_attn_native(*clone_args(args))
        output = impl.impl_fn(*clone_args(args))
        assert_close(ir.ops.mm_encoder_attn, output, ref_output)

    @pytest.mark.parametrize(
        "provider", supported_providers(ir.ops.mm_encoder_attn) + ["native"]
    )
    def test_torch_opcheck(
        self, dtype, batch_size, seq_len, num_heads, num_kv_heads, head_size, provider
    ):
        if not ir.ops.mm_encoder_attn.impls[provider].supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        args = ir.ops.mm_encoder_attn.generate_inputs(
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
        )

        with ir.ops.mm_encoder_attn.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.mm_encoder_attn, args)
