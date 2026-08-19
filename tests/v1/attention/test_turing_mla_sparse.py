import pytest
import torch

from vllm.models.deepseek_v4.turing.sparse import triton_mla_sparse_interface


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7,
    reason="SM75 only",
)
def test_turing_mla_sparse_fp16_shape_and_variance():
    torch.manual_seed(0)
    num_tokens, nq, dim_qk, d_v, topk = 4, 64, 640, 512, 64
    num_kv_tokens = 8
    q = torch.randn(num_tokens, nq, dim_qk, dtype=torch.float16, device="cuda") * 0.02
    kv = (
        torch.randn(num_kv_tokens, 1, dim_qk, dtype=torch.float16, device="cuda") * 0.02
    )
    indices = torch.arange(topk, dtype=torch.int64, device="cuda") % num_kv_tokens
    indices = indices.reshape(1, 1, -1).expand(num_tokens, 1, topk)
    out, max_logits, lse = triton_mla_sparse_interface(
        q, kv, indices, sm_scale=dim_qk**-0.5, d_v=d_v, block_dpe=128
    )
    assert out.shape == (num_tokens, nq, d_v)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
    assert torch.isfinite(max_logits).all()
    assert torch.isfinite(lse).all()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7,
    reason="SM75 only",
)
def test_turing_mla_attention_backend():
    from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLABackend
    from vllm.models.deepseek_v4.turing.attention import (
        DeepseekV4TuringSparseBackend,
        TuringMLAAttention,
    )
    from vllm.platforms.interface import DeviceCapability

    assert TuringMLAAttention.backend_cls is DeepseekV4TuringSparseBackend
    assert issubclass(DeepseekV4TuringSparseBackend, DeepseekV4FlashMLABackend)
    assert DeepseekV4TuringSparseBackend.supports_compute_capability(
        DeviceCapability(7, 5)
    )
    assert not DeepseekV4TuringSparseBackend.supports_compute_capability(
        DeviceCapability(9, 0)
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7,
    reason="SM75 only",
)
def test_turing_mla_sparse_fp16_padded_indices():
    torch.manual_seed(0)
    num_tokens, nq, dim_qk, d_v, topk = 4, 16, 640, 512, 64
    num_kv_tokens = 8
    q = torch.randn(num_tokens, nq, dim_qk, dtype=torch.float16, device="cuda") * 0.02
    kv = (
        torch.randn(num_kv_tokens, 1, dim_qk, dtype=torch.float16, device="cuda") * 0.02
    )
    indices = torch.full((num_tokens, 1, topk), -1, dtype=torch.int64, device="cuda")
    indices[:, :, :16] = torch.arange(16, dtype=torch.int64, device="cuda") % (
        num_kv_tokens
    )
    out, max_logits, lse = triton_mla_sparse_interface(
        q, kv, indices, sm_scale=dim_qk**-0.5, d_v=d_v, block_dpe=128
    )
    assert torch.isfinite(out).all()
    assert torch.isfinite(max_logits).all()
    assert torch.isfinite(lse).all()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7,
    reason="SM75 only",
)
def test_turing_mla_sparse_fp16_model_geometry():
    # Real DeepSeek-V4-Flash MLA head dims: 512 = 448 nope + 64 rope, with the
    # full-dim dot product (block_dpe=0) as used by decode/prefill. This is the
    # geometry that must compile on SM75.
    torch.manual_seed(0)
    num_tokens, nq, dim_qk, d_v, topk = 4, 64, 512, 512, 64
    num_kv_tokens = 8
    q = torch.randn(num_tokens, nq, dim_qk, dtype=torch.float16, device="cuda") * 0.02
    kv = (
        torch.randn(num_kv_tokens, 1, dim_qk, dtype=torch.float16, device="cuda") * 0.02
    )
    indices = torch.arange(topk, dtype=torch.int64, device="cuda") % num_kv_tokens
    indices = indices.reshape(1, 1, -1).expand(num_tokens, 1, topk)
    out, max_logits, lse = triton_mla_sparse_interface(
        q, kv, indices, sm_scale=dim_qk**-0.5, d_v=d_v, block_dpe=0
    )
    assert out.shape == (num_tokens, nq, d_v)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
    assert torch.isfinite(max_logits).all()
    assert torch.isfinite(lse).all()
