# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from tests.kernels.quantization.nvfp4_utils import dequant_nvfp4_kv_cache
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import fp8_k_nvfp4_v_cache_split_views
from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend,
    FlashInferDecodeKernel,
    FlashInferMetadataBuilder,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="FP8-K/NVFP4-V TRTLLM-gen attention requires SM100",
)


def _to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    scale = float(x.abs().amax().item() / torch.finfo(torch.float8_e4m3fn).max)
    return (x / scale).to(torch.float8_e4m3fn), scale


def test_fp8_k_nvfp4_v_query_dtype_is_e4m3() -> None:
    builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
    builder.vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(disable_flashinfer_q_quantization=False)
    )
    builder.model_config = SimpleNamespace(dtype=torch.bfloat16)
    builder.cache_dtype = "fp8_k_nvfp4_v"

    assert builder.get_q_data_type(is_prefill=True) == torch.float8_e4m3fn
    assert builder.get_q_data_type(is_prefill=False) == torch.float8_e4m3fn


def test_fp8_k_nvfp4_v_rejects_sm107() -> None:
    kwargs = dict(
        head_size=128,
        dtype=torch.bfloat16,
        kv_cache_dtype="fp8_k_nvfp4_v",
        block_size=64,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
    )
    assert (
        FlashInferBackend.supports_combination(
            **kwargs, device_capability=DeviceCapability(10, 3)
        )
        is None
    )
    reason = FlashInferBackend.supports_combination(
        **kwargs, device_capability=DeviceCapability(10, 7)
    )
    assert reason == "fp8_k_nvfp4_v is not supported on SM107"


def test_fp8_k_nvfp4_v_routes_spec_decode_to_context() -> None:
    builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
    builder.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=3,
            parallel_drafting=False,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    builder.flashinfer_trtllm_api_decode_kernel = FlashInferDecodeKernel.TRTLLM_GEN
    builder.is_kvcache_fp8_k_nvfp4_v = True
    builder.use_dedicated_xqa = False

    builder._init_reorder_batch_threshold(
        1,
        supports_spec_as_decode=builder._supports_spec_as_decode(),
    )

    assert builder.reorder_batch_threshold == 1

    builder.is_kvcache_fp8_k_nvfp4_v = False
    builder._init_reorder_batch_threshold(
        1,
        supports_spec_as_decode=builder._supports_spec_as_decode(),
    )

    assert builder.reorder_batch_threshold == 4


@torch.inference_mode()
def test_fp8_k_nvfp4_v_store_rejects_fp16() -> None:
    page_size, num_heads, head_size = 64, 1, 128
    total_dim = head_size + head_size // 2 + head_size // 16
    packed_cache = torch.empty(
        1,
        page_size,
        num_heads,
        total_dim,
        dtype=torch.uint8,
        device="cuda",
    )
    key = torch.randn(
        1, num_heads, head_size, dtype=torch.float16, device="cuda"
    )
    value = torch.randn_like(key)
    slot_mapping = torch.zeros(1, dtype=torch.int64, device="cuda")
    scale = torch.ones(1, dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError):
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            packed_cache,
            packed_cache,
            slot_mapping,
            "fp8_k_nvfp4_v",
            scale,
            scale,
        )


@torch.inference_mode()
def test_fp8_k_nvfp4_v_store_and_native_decode() -> None:
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    torch.manual_seed(0)
    num_pages, page_size = 4, 64
    num_kv_heads, group_size = 4, 4
    num_qo_heads, head_size = num_kv_heads * group_size, 128
    total_dim = head_size + head_size // 2 + head_size // 16

    packed_cache = torch.empty(
        num_pages,
        num_kv_heads,
        page_size,
        total_dim,
        dtype=torch.uint8,
        device="cuda",
    )
    key = torch.randn(
        num_pages * page_size,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    value = torch.randn_like(key)
    slot_mapping = torch.arange(key.shape[0], dtype=torch.int64, device="cuda")
    k_scale = torch.tensor(0.5, dtype=torch.float32, device="cuda")
    v_scale = torch.tensor(0.75, dtype=torch.float32, device="cuda")

    packed_cache_nhd = packed_cache.permute(0, 2, 1, 3)
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key,
        value,
        packed_cache_nhd,
        packed_cache_nhd,
        slot_mapping,
        "fp8_k_nvfp4_v",
        k_scale,
        v_scale,
    )
    k_cache, v_cache, v_block_scales = fp8_k_nvfp4_v_cache_split_views(
        packed_cache, head_size
    )

    key_qdq = k_cache.bfloat16() * k_scale
    value_qdq = dequant_nvfp4_kv_cache(
        v_cache, v_block_scales, float(v_scale.item()), head_size, page_size
    )
    torch.testing.assert_close(
        key_qdq.permute(0, 2, 1, 3).reshape_as(key),
        key,
        atol=0.05,
        rtol=0.05,
    )

    batch_size, pages_per_seq = 2, 2
    query = torch.randn(
        batch_size,
        num_qo_heads,
        head_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    query_fp8, q_scale = _to_fp8(query)
    block_tables = torch.arange(
        batch_size * pages_per_seq, dtype=torch.int32, device="cuda"
    ).reshape(batch_size, pages_per_seq)
    seq_lens = torch.tensor([96, 128], dtype=torch.int32, device="cuda")
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    output = trtllm_batch_decode_with_kv_cache(
        query_fp8,
        (k_cache, v_cache),
        workspace,
        block_tables,
        seq_lens,
        int(seq_lens.max().item()),
        bmm1_scale=q_scale * float(k_scale.item()) / math.sqrt(head_size),
        bmm2_scale=float(v_scale.item()),
        out_dtype=torch.bfloat16,
        backend="trtllm-gen",
        kv_layout="HND",
        kv_cache_sf=(None, v_block_scales),
    )

    output_ref = []
    query_qdq = query_fp8.float() * q_scale
    for batch_idx, seq_len in enumerate(seq_lens.tolist()):
        page_ids = block_tables[batch_idx]
        key_seq = (
            key_qdq[page_ids]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, head_size)[:, :seq_len]
            .repeat_interleave(group_size, dim=0)
            .float()
        )
        value_seq = (
            value_qdq[page_ids]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, head_size)[:, :seq_len]
            .repeat_interleave(group_size, dim=0)
        )
        logits = torch.einsum("hd,hnd->hn", query_qdq[batch_idx], key_seq)
        probs = torch.softmax(logits / math.sqrt(head_size), dim=-1)
        output_ref.append(torch.einsum("hn,hnd->hd", probs, value_seq))

    cosine = torch.nn.functional.cosine_similarity(
        output.float().reshape(-1), torch.stack(output_ref).reshape(-1), dim=0
    )
    assert cosine.item() > 0.99
