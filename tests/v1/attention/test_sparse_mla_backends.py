# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the sparse MLA backends and utilities."""

import math
from collections import deque
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.attention.test_mla_backends import (
    BATCH_SPECS,
    BatchSpec,
    MockSparseMLAAttentionLayer,
    create_and_prepopulate_kv_cache,
)
from tests.v1.attention.utils import (
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from vllm import _custom_ops as ops
from vllm.config import (
    CUDAGraphMode,
    HiSparseConfig,
    SpeculativeConfig,
    set_current_vllm_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
    HiSparseConnectorWorker,
)
from vllm.model_executor.layers.attention import mla_attention
from vllm.model_executor.layers.attention.mla_attention import _use_masked_mha
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    GLOBAL_TOPK_MASK_MAX_BYTES,
    SparseMLAPrefillMetadata,
    _masked_mha_workspace_fits,
    _topk_mask_shape,
    _use_dense_mha_prefill,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.platforms import current_platform

# TODO: Integrate ROCMAiterMLASparseBackend for ROCm.
# The ROCm sparse MLA backend (rocm_aiter_mla_sparse.py) has a compatible
# forward_mqa interface but needs validation on ROCm hardware.
if not current_platform.is_cuda():
    pytest.skip(
        "Sparse MLA backend tests currently only support CUDA. "
        "ROCm support requires integrating ROCMAiterMLASparseBackend.",
        allow_module_level=True,
    )

import vllm.v1.attention.backends.mla.flashinfer_mla_sparse as flashinfer_sparse_mod
from vllm.model_executor.layers.attention.mla_attention import (
    _canonicalize_sparse_mla_kv_cache_dtype,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import current_stream
from vllm.v1.attention.backends.mla import index_group as index_group_module
from vllm.v1.attention.backends.mla.flashattn_mla_sparse import (
    FlashAttnMLASparseImpl,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseImpl,
    FlashInferMLASparseMetadataBuilder,
    FlashInferMLASparseTRTLLMBackend,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
    FlashInferMLASparseSM120Impl,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseImpl,
    FlashMLASparseMetadata,
    FlashMLASparseMetadataBuilder,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.mla.index_group import (
    HiSparseMLAIndexGroup,
    SparseMLAIndexGroupBuilder,
)
from vllm.v1.attention.backends.mla.indexer import split_indexer_prefill_chunks
from vllm.v1.attention.backends.mla.prefill import get_mla_prefill_backend
from vllm.v1.attention.backends.utils import (
    split_decodes_and_prefills,
    split_prefill_chunks,
)
from vllm.v1.attention.ops import flashmla
from vllm.v1.hisparse import runtime as hisparse_runtime
from vllm.v1.hisparse import runtime as hisparse_runtime_module
from vllm.v1.hisparse.runtime import (
    HiSparseCacheHandle,
    HiSparseRuntime,
    ResolvedHiSparseConfig,
    _has_hisparse_ops,
    build_hisparse_prefill_staging_plan,
    hisparse_prefill_staging_remap,
)
from vllm.v1.hisparse.types import SparseKVRowMirror

SPARSE_BACKEND_BATCH_SPECS = {
    name: BATCH_SPECS[name]
    for name in [
        "mixed_small",
        "mixed_medium",
        "small_prefill",
        "medium_prefill",
        "single_prefill",
    ]
}

SPARSE_BACKEND_BATCH_SPECS["large_q_prefill"] = BatchSpec(
    seq_lens=[1024] * 2, query_lens=[256] * 2
)
SPARSE_BACKEND_BATCH_SPECS["large_q_pure_prefill"] = BatchSpec(
    seq_lens=[256] * 2, query_lens=[256] * 2
)

DEVICE_TYPE = current_platform.device_type


def test_nope_flashinfer_sparse_mla_uses_model_scale(monkeypatch):
    """Weight absorption must not change the model's attention temperature."""
    model_scale = 256**-0.5
    kv_lora_rank = 512
    topk = torch.zeros((1, 1), dtype=torch.int32)
    metadata = SimpleNamespace(
        req_id_per_token=torch.zeros(1, dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        block_size=1,
    )
    recorded_scale = None

    impl = object.__new__(FlashInferMLASparseImpl)
    impl.scale = model_scale
    impl.qk_nope_head_dim = 256
    impl.kv_lora_rank = kv_lora_rank
    impl.qk_rope_head_dim = 0
    impl.kv_cache_dtype = "auto"
    impl.topk_indices_buffer = topk
    impl.dcp_world_size = 1
    impl._workspace_buffer = torch.empty(1)
    impl.bmm1_scale = None
    impl.bmm2_scale = None
    impl.is_nope_mla = True
    impl.need_to_return_lse_for_decode = False
    monkeypatch.setattr(
        flashinfer_sparse_mod,
        "triton_convert_req_index_to_global_index",
        lambda *args, **kwargs: (topk, torch.ones(1, dtype=torch.int32)),
    )

    import flashinfer.decode

    def fake_flashinfer(**kwargs):
        nonlocal recorded_scale
        recorded_scale = kwargs["bmm1_scale"]
        return torch.zeros((1, 1, 1, kv_lora_rank))

    monkeypatch.setattr(
        flashinfer.decode,
        "trtllm_batch_decode_with_kv_cache_mla",
        fake_flashinfer,
    )
    impl.forward_mqa(
        torch.zeros(1, 1, kv_lora_rank),
        torch.zeros(1, 1, kv_lora_rank),
        metadata,
        SimpleNamespace(),
    )

    assert recorded_scale == model_scale
    assert recorded_scale != kv_lora_rank**-0.5


def test_hisparse_routes_prefill_to_sparse_mqa():
    layer = SimpleNamespace(hisparse_cache=object())

    assert not mla_attention.MLAAttention._use_sparse_mha(layer, SimpleNamespace())


def test_hisparse_metadata_keeps_short_prefill_indexer_enabled():
    config = SimpleNamespace(
        attention_config=SimpleNamespace(
            hisparse_config=object(), sparse_mla_force_mqa=False
        )
    )

    assert not _use_dense_mha_prefill(config, prefill_max_seq_len=512, topk_tokens=2048)


def test_unified_mla_prepares_batch_without_slot_mapping(monkeypatch):
    layer = SimpleNamespace(
        prepare_kv_cache_update=MagicMock(),
        update_kv_cache=MagicMock(),
    )
    metadata = object()
    monkeypatch.setattr(
        mla_attention,
        "get_attention_context",
        lambda _: (metadata, layer, torch.empty(0), None),
    )
    tensor = torch.empty(0, device=DEVICE_TYPE)

    mla_attention.unified_mla_kv_cache_update(tensor, tensor, "layer", "auto", tensor)

    layer.prepare_kv_cache_update.assert_called_once_with(metadata)
    layer.update_kv_cache.assert_not_called()


def _float_to_e8m0_truncate(f: float) -> float:
    """Simulate SM100's float -> e8m0 -> bf16 scale conversion.
    e8m0 format only stores the exponent (power of 2).
    cudaRoundZero truncates toward zero, meaning we round down to the
    nearest power of 2.
    """
    if f <= 0:
        return 0.0
    # e8m0 = floor(log2(f)), then 2^(e8m0)
    # This is equivalent to truncating to the nearest power of 2 below f
    exp = math.floor(math.log2(f))
    return 2.0**exp


def _dequantize_fp8_ds_mla_entry(
    cache_slice: torch.Tensor,
    kv_lora_rank: int,
    rope_dim: int,
    dtype: torch.dtype,
    simulate_sm100_e8m0_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize a single fp8_ds_mla cache entry back to latent + rope.

    Args:
        simulate_sm100_e8m0_scales: If True, simulate the SM100 kernel's
            float -> e8m0 -> bf16 scale conversion path.
    """

    # The first kv_lora_rank bytes store FP8 latent values with one scale per
    # 128 element tile written as float32 right after the latent payload.
    scales = cache_slice.view(torch.float32)[kv_lora_rank // 4 : kv_lora_rank // 4 + 4]
    latent = torch.empty(kv_lora_rank, dtype=torch.float16, device=cache_slice.device)
    for tile_idx in range(4):
        tile_start = tile_idx * 128
        tile_end = tile_start + 128
        scale_val = float(scales[tile_idx].item())
        if simulate_sm100_e8m0_scales:
            # Simulate the lossy float -> e8m0 -> bf16 conversion
            scale_val = _float_to_e8m0_truncate(scale_val)
        ops.convert_fp8(
            latent[tile_start:tile_end],
            cache_slice[tile_start:tile_end],
            scale_val,
            kv_dtype="fp8",
        )
    latent = latent.to(dtype)

    rope_offset = kv_lora_rank // 2 + 8
    rope_vals = cache_slice.view(dtype)[rope_offset : rope_offset + rope_dim]
    return latent, rope_vals.clone()


def _quantize_dequantize_fp8_ds_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    block_size: int,
    scale: torch.Tensor,
    simulate_sm100_e8m0_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round-trip kv_c/k_pe though the fp8_ds_mla cache layout.

    Args:
        simulate_sm100_e8m0_scales: If True, simulate the SM100 kernel's
            float -> e8m0 -> bf16 scale conversion in dequantization.
    """

    if kv_c.numel() == 0:
        return kv_c.clone(), k_pe.clone()

    kv_lora_rank = kv_c.shape[-1]
    rope_dim = k_pe.shape[-1]
    num_tokens = kv_c.shape[0]
    num_blocks = max(1, math.ceil(num_tokens / block_size))
    entry_size = kv_lora_rank + 4 * 4 + 2 * rope_dim

    tmp_cache = torch.zeros(
        num_blocks, block_size, entry_size, dtype=torch.uint8, device=kv_c.device
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=kv_c.device)

    ops.concat_and_cache_mla(
        kv_c, k_pe, tmp_cache, slot_mapping, kv_cache_dtype="fp8_ds_mla", scale=scale
    )

    dequant_kv_c = torch.empty_like(kv_c)
    dequant_k_pe = torch.empty_like(k_pe)

    for token_idx in range(num_tokens):
        slot = slot_mapping[token_idx].item()
        block_idx = slot // block_size
        block_offset = slot % block_size
        cache_slice = tmp_cache[block_idx, block_offset]
        latent, rope_vals = _dequantize_fp8_ds_mla_entry(
            cache_slice,
            kv_lora_rank,
            rope_dim,
            kv_c.dtype,
            simulate_sm100_e8m0_scales=simulate_sm100_e8m0_scales,
        )
        dequant_kv_c[token_idx] = latent
        dequant_k_pe[token_idx] = rope_vals

    return dequant_kv_c, dequant_k_pe


_E2M1_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _quantize_dequantize_nvfp4_ds_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    block_size: int,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round-trip kv_c/k_pe through the nvfp4_ds_mla cache layout.

    Layout (per token, uint8): [e2m1-packed NoPE | unscaled e4m3 RoPE |
    per-16 e4m3 NoPE SFs]. The SF bytes are stored permuted (an 8x4 -> 4x8
    transpose: the scale for element block s lives at byte 8 * (s & 3) +
    (s >> 2)) so that the 8 scales one FlashMLA dequant thread needs are
    contiguous. Dequant: x = value * float(sf), which is exact in bf16, so
    this python dequant matches the kernels bit-for-bit.
    """
    if kv_c.numel() == 0:
        return kv_c.clone(), k_pe.clone()

    kv_lora_rank = kv_c.shape[-1]
    rope_dim = k_pe.shape[-1]
    num_tokens = kv_c.shape[0]
    num_blocks = max(1, math.ceil(num_tokens / block_size))
    entry_size = (
        math.ceil((kv_lora_rank // 2 + rope_dim + kv_lora_rank // 16) / 16) * 16
    )
    sf_nope_off = kv_lora_rank // 2 + rope_dim

    tmp_cache = torch.zeros(
        num_blocks, block_size, entry_size, dtype=torch.uint8, device=kv_c.device
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=kv_c.device)
    ops.concat_and_cache_mla(
        kv_c, k_pe, tmp_cache, slot_mapping, kv_cache_dtype=kv_cache_dtype, scale=scale
    )

    tokens = tmp_cache.view(-1, entry_size)[:num_tokens]
    table = torch.tensor(
        _E2M1_TABLE + [-v for v in _E2M1_TABLE],
        dtype=torch.float32,
        device=kv_c.device,
    )

    def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
        lo = (packed & 0xF).long()
        hi = (packed >> 4).long()
        return table[torch.stack([lo, hi], dim=-1).flatten(-2)]

    nope_vals = unpack_e2m1(tokens[:, : kv_lora_rank // 2])
    num_nope_sf = kv_lora_rank // 16
    nope_sf = (
        # undo the on-wire SF permutation -> element-block order
        tokens[:, sf_nope_off : sf_nope_off + num_nope_sf]
        .unflatten(-1, (4, num_nope_sf // 4))
        .transpose(-1, -2)
        .flatten(-2)
        .view(torch.float8_e4m3fn)
        .float()
    )
    dequant_kv_c = (
        (nope_vals.unflatten(-1, (-1, 16)) * nope_sf.unsqueeze(-1)).flatten(-2)
    ).to(kv_c.dtype)

    # RoPE: plain e4m3, no scale factor
    rope_raw = tokens[:, kv_lora_rank // 2 : sf_nope_off]
    dequant_k_pe = rope_raw.view(torch.float8_e4m3fn).to(k_pe.dtype)

    return dequant_kv_c, dequant_k_pe


@pytest.mark.parametrize(
    "backend_cls",
    [FlashMLASparseBackend, FlashInferMLASparseTRTLLMBackend],
    ids=["FlashMLA", "FlashInferTRTLLM"],
)
@pytest.mark.parametrize("batch_name", list(SPARSE_BACKEND_BATCH_SPECS.keys()))
@pytest.mark.parametrize(
    "kv_cache_dtype",
    ["auto", "fp8", "fp8_ds_mla", "nvfp4_ds_mla"],
)
@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4])
@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize(("q_scale", "k_scale"), [(1.0, 1.0), (2.0, 3.0)])
def test_sparse_backend_decode_correctness(
    default_vllm_config,
    dist_init,
    backend_cls,
    batch_name,
    kv_cache_dtype,
    tensor_parallel_size,
    block_size,
    workspace_init,
    q_scale: float,
    k_scale: float,
):
    if kv_cache_dtype not in backend_cls.supported_kv_cache_dtypes:
        pytest.skip(f"{backend_cls.get_name()} does not support {kv_cache_dtype}")

    if (
        backend_cls == FlashMLASparseBackend
        and kv_cache_dtype.startswith("fp8")
        and kv_cache_dtype != "fp8_ds_mla"
    ):
        pytest.skip(
            "FlashMLA Sparse Attention backend fp8 only supports "
            "fp8_ds_mla kv-cache dtype"
        )

    if kv_cache_dtype == "nvfp4_ds_mla":
        # Unlike "fp8" (an alias canonicalized to "fp8_ds_mla" above), the
        # NVFP4 DS-MLA dtype must reach the backend unchanged.
        assert (
            _canonicalize_sparse_mla_kv_cache_dtype(backend_cls, kv_cache_dtype)
            == kv_cache_dtype
        )
        device_capability = current_platform.get_device_capability()
        if device_capability is None or device_capability.major != 10:
            pytest.skip("The NVFP4 DS-MLA kv-cache dtype requires SM 10.x")

    supported_block_sizes = backend_cls.get_supported_kernel_block_sizes()
    if block_size not in supported_block_sizes:
        pytest.skip(
            f"{backend_cls.get_name()} does not support block_size={block_size}"
        )

    if backend_cls == FlashMLASparseBackend:
        ok, reason = flashmla.is_flashmla_sparse_supported()
        if not ok:
            pytest.skip(reason)
    elif backend_cls == FlashInferMLASparseTRTLLMBackend:
        device_capability = current_platform.get_device_capability()
        if device_capability is None or not backend_cls.supports_compute_capability(
            device_capability
        ):
            pytest.skip("FlashInferMLASparseTRTLLMBackend requires SM 10.x capability")

    batch_spec = SPARSE_BACKEND_BATCH_SPECS[batch_name]
    use_fp8_ds_mla_quantization = kv_cache_dtype == "fp8_ds_mla"
    use_nvfp4_ds_mla_quantization = kv_cache_dtype == "nvfp4_ds_mla"

    device = torch.device(DEVICE_TYPE)
    dtype = torch.bfloat16

    # Model hyper-parameters (kept intentionally small for the unit test)
    total_num_heads = 128
    # Compute per-rank heads for simulated TP
    num_heads = max(1, total_num_heads // tensor_parallel_size)

    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    head_size = kv_lora_rank + qk_rope_head_dim
    topk_tokens = 128

    max_seqlen = max(batch_spec.seq_lens)
    total_cache_tokens = sum(batch_spec.seq_lens)

    # Note: We use TP=1 to avoid multi-GPU requirements in CI.
    # The test simulates head partitioning via mocked methods below.
    vllm_config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-V2-Lite-Chat",
        tensor_parallel_size=1,
        max_model_len=max_seqlen,
        num_gpu_blocks=max(2048, cdiv(total_cache_tokens, block_size) + 1),
        block_size=block_size,
        hf_config_override={
            "index_topk": topk_tokens,
            "attn_module_list_cfg": [{"topk_tokens": topk_tokens}],
        },
    )
    model_config = vllm_config.model_config
    model_config.hf_text_config = SimpleNamespace(
        q_lora_rank=None,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        model_type="deepseek_v2",
    )
    model_config.dtype = dtype
    model_config.get_num_attention_heads = MethodType(
        lambda self, parallel_config: num_heads,
        model_config,
    )
    model_config.get_num_kv_heads = MethodType(
        lambda self, parallel_config: 1, model_config
    )
    model_config.get_head_size = MethodType(lambda self: head_size, model_config)
    model_config.get_sliding_window = MethodType(lambda self: None, model_config)

    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)

    torch.manual_seed(0)

    scale = 1.0 / math.sqrt(head_size)

    # Shared MLA projection weights to keep reference and backend in sync
    W_UK = torch.rand(
        kv_lora_rank, num_heads, qk_nope_head_dim, dtype=dtype, device=device
    )
    W_UV = torch.rand(kv_lora_rank, num_heads, v_head_dim, dtype=dtype, device=device)

    # Build synthetic decode-only workload
    seq_lens = batch_spec.seq_lens
    query_lens = batch_spec.query_lens

    # Pre-compute positions and sparse indices for all tokens.
    # We need these BEFORE computing the reference to use sparse attention masks.
    total_query_tokens = sum(query_lens)
    positions = []
    for i in range(batch_spec.batch_size):
        s_len = seq_lens[i]
        q_len = query_lens[i]
        ctx_len = s_len - q_len
        for q_idx in range(q_len):
            positions.append(ctx_len + q_idx)

    # Create sparse indices with UNIQUE per-token offsets to catch bugs where
    # the kernel uses wrong indices for some tokens (e.g., due to incorrect
    # tensor shapes like [1, num_tokens, ...] instead of [num_tokens, 1, ...]).
    # Also include -1 masked indices to verify the kernel handles them correctly.
    sparse_indices = torch.empty(
        total_query_tokens, topk_tokens, dtype=torch.int32, device=device
    )
    for tok_idx in range(total_query_tokens):
        max_valid_idx = positions[tok_idx]
        offset = tok_idx * 7  # Prime number for varied offsets
        # Use only half the topk indices as valid, mask the rest with -1
        # This tests that the kernel correctly ignores -1 indices
        num_valid = min(topk_tokens // 2, max_valid_idx + 1)
        if num_valid > 0:
            valid_range = torch.arange(num_valid, device=device, dtype=torch.int32)
            tok_indices = (valid_range + offset) % (max_valid_idx + 1)
            # Pad with -1 for the remaining positions
            tok_indices = torch.cat(
                [
                    tok_indices,
                    torch.full(
                        (topk_tokens - num_valid,), -1, device=device, dtype=torch.int32
                    ),
                ]
            )
        else:
            tok_indices = torch.full(
                (topk_tokens,), -1, device=device, dtype=torch.int32
            )
            tok_indices[0] = 0  # At least one valid index
        sparse_indices[tok_idx] = tok_indices

    all_q_vllm, all_kv_c_vllm, all_k_pe_vllm = [], [], []
    kv_c_contexts, k_pe_contexts = [], []
    reference_outputs = []

    kv_cache_scale = torch.tensor(k_scale, dtype=torch.float32, device=device)
    global_token_idx = 0

    for i in range(batch_spec.batch_size):
        s_len = seq_lens[i]
        q_len = query_lens[i]
        ctx_len = s_len - q_len

        q_c = torch.rand(
            q_len,
            num_heads,
            qk_nope_head_dim + qk_rope_head_dim,
            dtype=dtype,
            device=device,
        )
        kv_c_full = torch.rand(s_len, kv_lora_rank, dtype=dtype, device=device)
        k_pe_full = torch.rand(s_len, 1, qk_rope_head_dim, dtype=dtype, device=device)

        if use_fp8_ds_mla_quantization:
            is_sm100 = torch.cuda.get_device_capability()[0] >= 10
            kv_c_full, k_pe_squeezed = _quantize_dequantize_fp8_ds_mla(
                kv_c_full,
                k_pe_full.squeeze(1),
                block_size=block_size,
                scale=kv_cache_scale,
                simulate_sm100_e8m0_scales=is_sm100,
            )
            k_pe_full = k_pe_squeezed.unsqueeze(1)
        elif use_nvfp4_ds_mla_quantization:
            kv_c_full, k_pe_squeezed = _quantize_dequantize_nvfp4_ds_mla(
                kv_c_full,
                k_pe_full.squeeze(1),
                block_size=block_size,
                kv_cache_dtype=kv_cache_dtype,
                scale=kv_cache_scale,
            )
            k_pe_full = k_pe_squeezed.unsqueeze(1)

        q_nope, q_pe = q_c.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
        ql_nope = torch.einsum("qnh,lnh->qnl", q_nope, W_UK)
        q_mqa = torch.cat([ql_nope, q_pe], dim=-1)

        k_mqa = torch.cat([kv_c_full, k_pe_full.squeeze(1)], dim=-1)
        v_mqa = kv_c_full

        # Compute sparse SDPA reference per query token using its sparse indices
        for q_idx in range(q_len):
            tok_sparse_idx = sparse_indices[global_token_idx]
            valid_mask = tok_sparse_idx >= 0
            valid_indices = tok_sparse_idx[valid_mask].long()

            q_tok = q_mqa[q_idx : q_idx + 1]  # [1, num_heads, head_dim]
            k_sparse = k_mqa[valid_indices]  # [num_valid, head_dim]
            v_sparse = v_mqa[valid_indices]  # [num_valid, kv_lora_rank]

            k_sparse = k_sparse.unsqueeze(1).expand(-1, num_heads, -1)
            v_sparse = v_sparse.unsqueeze(1).expand(-1, num_heads, -1)

            # SDPA: [1, num_heads, 1, head_dim] x [1, num_heads, num_valid, head_dim]
            q_sdpa_in = q_tok.unsqueeze(0).transpose(1, 2)
            k_sdpa_in = k_sparse.unsqueeze(0).transpose(1, 2)
            v_sdpa_in = v_sparse.unsqueeze(0).transpose(1, 2)

            sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa_in, k_sdpa_in, v_sdpa_in, scale=scale
            )
            sdpa_out = sdpa_out.transpose(1, 2).squeeze(
                0
            )  # [1, num_heads, kv_lora_rank]

            sdpa_out = torch.einsum("qnl,lnv->qnv", sdpa_out, W_UV)
            reference_outputs.append(sdpa_out.flatten(start_dim=-2))

            global_token_idx += 1

        all_q_vllm.append(q_c)
        all_kv_c_vllm.append(kv_c_full[ctx_len:])
        all_k_pe_vllm.append(k_pe_full[ctx_len:])
        kv_c_contexts.append(kv_c_full[: ctx_len + 1])
        k_pe_contexts.append(k_pe_full[: ctx_len + 1])

    query_vllm = torch.cat(all_q_vllm, dim=0)
    kv_c_vllm = torch.cat(all_kv_c_vllm, dim=0)
    k_pe_vllm = torch.cat(all_k_pe_vllm, dim=0)
    sdpa_reference = torch.cat(reference_outputs, dim=0)

    vllm_config.cache_config.cache_dtype = kv_cache_dtype
    vllm_config.model_config.hf_config.index_topk = topk_tokens

    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        vllm_config.cache_config.block_size,
        device,
        arange_block_indices=True,
    )

    kv_cache = create_and_prepopulate_kv_cache(
        kv_c_contexts=kv_c_contexts,
        k_pe_contexts=k_pe_contexts,
        block_size=vllm_config.cache_config.block_size,
        head_size=head_size,
        dtype=dtype,
        device=device,
        num_blocks=vllm_config.cache_config.num_gpu_blocks,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=False,
        kv_cache_dtype=kv_cache_dtype,
        scale=kv_cache_scale,
    )

    # The sparse builder clones the layer's dense-MHA prefill backend from
    # static_forward_context; register a mock layer carrying one.
    prefill_backend = get_mla_prefill_backend(vllm_config)(
        num_heads=num_heads,
        scale=scale,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        vllm_config=vllm_config,
    )
    vllm_config.compilation_config.static_forward_context["placeholder"] = (
        SimpleNamespace(prefill_backend=prefill_backend)
    )

    builder_cls = backend_cls.get_builder_cls()
    builder = builder_cls(kv_cache_spec, ["placeholder"], vllm_config, device)
    metadata = builder.build(
        common_prefix_len=0, common_attn_metadata=common_attn_metadata
    )

    # Use the pre-computed sparse_indices for the mock indexer
    mock_indexer = SimpleNamespace(topk_indices_buffer=sparse_indices)

    kv_b_proj_weight = torch.cat([W_UK, W_UV], dim=-1)
    kv_b_proj_weight = kv_b_proj_weight.view(
        kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)
    )

    mock_kv_b_proj = ColumnParallelLinear(
        input_size=kv_lora_rank,
        output_size=num_heads * (qk_nope_head_dim + v_head_dim),
        bias=False,
    ).to(device=device, dtype=dtype)
    mock_kv_b_proj.weight = torch.nn.Parameter(kv_b_proj_weight.T.contiguous())

    impl_cls = backend_cls.get_impl_cls()
    with set_current_vllm_config(vllm_config):
        impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=vllm_config.cache_config.cache_dtype,
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            q_lora_rank=None,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_head_dim=qk_nope_head_dim + qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_b_proj=mock_kv_b_proj,
            indexer=mock_indexer,
        )

        impl.process_weights_after_loading(dtype)

        # Create mock sparse MLA layer with weight matrices
        mock_layer = MockSparseMLAAttentionLayer(
            impl=impl,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            device=device,
            W_UK=W_UK,
            W_UV=W_UV,
            q_scale=q_scale,
            k_scale=k_scale,
        )

    out_buffer = torch.empty(
        metadata.num_actual_tokens, num_heads * v_head_dim, dtype=dtype, device=device
    )

    with torch.inference_mode():
        backend_output = mock_layer.forward_impl(
            query_vllm,
            kv_c_vllm,
            k_pe_vllm,
            kv_cache,
            metadata,
            out_buffer,
        )

    assert backend_output.shape == sdpa_reference.shape
    assert backend_output.dtype == sdpa_reference.dtype
    assert torch.isfinite(backend_output).all()

    # FP8 quantization introduces some error, but should be within reasonable bounds
    # BF16 (auto) should be very accurate, FP8 allows slightly more tolerance
    if kv_cache_dtype.startswith("fp8"):
        torch.testing.assert_close(
            backend_output, sdpa_reference, rtol=0.065, atol=0.05
        )
    else:
        torch.testing.assert_close(backend_output, sdpa_reference, rtol=0.01, atol=0.01)


def _triton_convert_reference_impl(
    req_ids: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    block_size: int,
    num_topk_tokens: int,
    HAS_PREFILL_WORKSPACE: bool = False,
    prefill_workspace_request_ids: torch.Tensor | None = None,
    prefill_workspace_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference implementation for triton_convert_req_index_to_global_index."""
    num_tokens = req_ids.shape[0]
    max_blocks_per_req = block_table.shape[1]
    result = torch.empty(
        num_tokens, num_topk_tokens, dtype=torch.int32, device=req_ids.device
    )

    for token_id in range(num_tokens):
        req_id = req_ids[token_id].item()

        # Determine if this token uses workspace or paged cache
        use_prefill_workspace = False
        workspace_start = 0
        if HAS_PREFILL_WORKSPACE and prefill_workspace_request_ids is not None:
            assert prefill_workspace_starts is not None
            prefill_req_id = prefill_workspace_request_ids[token_id].item()
            if prefill_req_id >= 0:
                use_prefill_workspace = True
                workspace_start = prefill_workspace_starts[prefill_req_id].item()

        for idx_id in range(num_topk_tokens):
            token_idx = token_indices[token_id, idx_id].item()

            if token_idx == -1:
                result[token_id, idx_id] = -1
            elif use_prefill_workspace:
                # Prefill + using prefill workspace: map to workspace offset
                result[token_id, idx_id] = workspace_start + token_idx
            else:
                # Decode: map to paged cache
                block_id = token_idx // block_size
                if block_id >= max_blocks_per_req:
                    result[token_id, idx_id] = -1
                else:
                    block_num = block_table[req_id, block_id].item()
                    offset = token_idx % block_size
                    result[token_id, idx_id] = block_num * block_size + offset

    return result


@pytest.mark.parametrize("block_size", [16, 64, 128])
@pytest.mark.parametrize("num_topk_tokens", [128, 256, 512])
@pytest.mark.skipif(
    torch.cuda.get_device_capability() < (9, 0),
    reason="FlashMLASparseBackend requires CUDA 9.0 or higher",
)
def test_triton_convert_req_index_to_global_index_decode_only(
    block_size, num_topk_tokens
):
    device = torch.device(DEVICE_TYPE)
    num_tokens = 8
    num_requests = 4
    max_blocks_per_req = 10

    req_id = torch.randint(
        0, num_requests, (num_tokens,), dtype=torch.int32, device=device
    )
    block_table = torch.randint(
        0, 100, (num_requests, max_blocks_per_req), dtype=torch.int32, device=device
    )

    token_indices = torch.randint(
        0,
        block_size * max_blocks_per_req,
        (num_tokens, num_topk_tokens),
        dtype=torch.int32,
        device=device,
    )

    # Set some to -1 to test masking
    token_indices[0, :10] = -1
    token_indices[3, 50:60] = -1

    # Set some to out of bounds
    token_indices[2, 100:110] = max_blocks_per_req * block_size
    token_indices[6, 150:160] = max_blocks_per_req * block_size

    result = triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk_tokens,
    )

    reference_result = _triton_convert_reference_impl(
        req_id,
        block_table,
        token_indices,
        block_size,
        num_topk_tokens,
    )

    torch.testing.assert_close(result, reference_result, rtol=0, atol=0)


@pytest.mark.parametrize("block_size", [16])
@pytest.mark.skipif(
    torch.cuda.get_device_capability() < (9, 0),
    reason="FlashMLASparseBackend requires CUDA 9.0 or higher",
)
def test_triton_convert_req_index_to_global_index_with_prefill_workspace(block_size):
    device = torch.device(DEVICE_TYPE)
    num_requests = 4
    max_blocks_per_req = 8
    num_topk_tokens = 128

    # First 6 tokens are decode (reqs 0, 1), last 6 are prefill (reqs 2, 3)
    req_id = torch.tensor(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=torch.int32, device=device
    )
    prefill_workspace_request_ids = torch.tensor(
        [-1, -1, -1, -1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=torch.int32, device=device
    )

    # Workspace starts for the 2 prefill reqs: req 2 starts at 0, req 3 starts at 100
    prefill_workspace_starts = torch.tensor([0, 100], dtype=torch.int32, device=device)

    block_table = torch.randint(
        0, 50, (num_requests, max_blocks_per_req), dtype=torch.int32, device=device
    )
    token_indices = torch.randint(
        0,
        block_size * max_blocks_per_req,
        (req_id.shape[0], num_topk_tokens),
        dtype=torch.int32,
        device=device,
    )

    # Set some to -1 to test masking
    token_indices[0, :10] = -1
    token_indices[3, 50:60] = -1

    # Set some to out of bounds
    token_indices[2, 100:110] = max_blocks_per_req * block_size
    token_indices[6, 150:160] = max_blocks_per_req * block_size

    result = triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk_tokens,
        HAS_PREFILL_WORKSPACE=True,
        prefill_workspace_request_ids=prefill_workspace_request_ids,
        prefill_workspace_starts=prefill_workspace_starts,
    )

    reference_result = _triton_convert_reference_impl(
        req_id,
        block_table,
        token_indices,
        block_size,
        num_topk_tokens,
        HAS_PREFILL_WORKSPACE=True,
        prefill_workspace_request_ids=prefill_workspace_request_ids,
        prefill_workspace_starts=prefill_workspace_starts,
    )

    torch.testing.assert_close(result, reference_result, rtol=0, atol=0)


@pytest.mark.skipif(
    torch.cuda.get_device_capability() < (9, 0),
    reason="FlashMLASparseBackend requires CUDA 9.0 or higher",
)
def test_triton_convert_rejects_req_id_longer_than_token_indices():
    """Guard against the #47327 regression: the kernel grid is sized by
    req_id but the output is allocated like token_indices, so a full-batch
    req_id combined with an MQA-subset token_indices wrote past the end of
    the output buffer. The wrapper must reject the length mismatch instead
    of corrupting memory."""
    device = torch.device(DEVICE_TYPE)
    num_topk_tokens = 128
    block_size = 64
    block_table = torch.arange(40, dtype=torch.int32, device=device).view(4, 10)

    # Full batch: 2 decode tokens + 10 prefill tokens
    req_id_full = torch.tensor(
        [0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], dtype=torch.int32, device=device
    )
    num_mqa_tokens = 2
    token_indices = torch.randint(
        0,
        block_size * 10,
        (num_mqa_tokens, num_topk_tokens),
        dtype=torch.int32,
        device=device,
    )

    with pytest.raises(AssertionError, match="must cover the same tokens"):
        triton_convert_req_index_to_global_index(
            req_id_full,
            block_table,
            token_indices,
            BLOCK_SIZE=block_size,
            NUM_TOPK_TOKENS=num_topk_tokens,
        )

    # The sliced call is the intended usage and must match the reference.
    result = triton_convert_req_index_to_global_index(
        req_id_full[:num_mqa_tokens],
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk_tokens,
    )
    reference = _triton_convert_reference_impl(
        req_id_full[:num_mqa_tokens],
        block_table,
        token_indices,
        block_size,
        num_topk_tokens,
    )
    torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.skipif(
    torch.cuda.get_device_capability() < (9, 0),
    reason="FlashMLASparseBackend requires CUDA 9.0 or higher",
)
def test_flashmla_forward_bf16_kv_slices_req_id_to_mqa_tokens():
    """Guard against the #47327 regression: when the dense-MHA prefill split
    is active, forward_mqa only receives the leading decode tokens, but
    _forward_bf16_kv passed the full-batch req_id_per_token to the index
    conversion, making it write past the end of its output buffer. The call
    site must slice req_id_per_token to the MQA tokens."""
    device = torch.device(DEVICE_TYPE)
    num_topk_tokens = 128
    block_size = 64
    num_batch_tokens = 12
    num_mqa_tokens = 2

    attn_metadata = SimpleNamespace(
        req_id_per_token=torch.tensor(
            [0, 1] + [2] * 5 + [3] * 5, dtype=torch.int32, device=device
        ),
        block_table=torch.arange(40, dtype=torch.int32, device=device).view(4, 10),
        block_size=block_size,
        num_decode_tokens=num_mqa_tokens,
        num_actual_tokens=num_mqa_tokens,
    )
    assert attn_metadata.req_id_per_token.shape[0] == num_batch_tokens

    q = torch.zeros(num_mqa_tokens, 4, 576, dtype=torch.bfloat16, device=device)
    kv_cache = torch.zeros(40, block_size, 576, dtype=torch.bfloat16, device=device)
    topk_indices = torch.randint(
        0,
        block_size * 10,
        (num_mqa_tokens, num_topk_tokens),
        dtype=torch.int32,
        device=device,
    )

    captured = {}

    def _stub_kernel(q, kv, indices, lengths, actual_num_heads):
        captured["indices"] = indices
        captured["actual_num_heads"] = actual_num_heads
        return (
            torch.zeros(q.shape[0], q.shape[1], 512, dtype=q.dtype, device=q.device),
            None,
        )

    def _convert_topk(indices, metadata, **kwargs):
        return triton_convert_req_index_to_global_index(
            metadata.req_id_per_token[: indices.shape[0]],
            metadata.block_table,
            indices,
            BLOCK_SIZE=metadata.block_size,
            BLOCK_STRIDE_ROWS=kwargs["block_stride_rows"],
            NUM_TOPK_TOKENS=indices.shape[1],
            return_valid_counts=kwargs["return_valid_counts"],
        )

    stub_impl = SimpleNamespace(
        _bf16_flash_mla_kernel=_stub_kernel,
        _convert_logical_to_physical_topk=_convert_topk,
        index_group=None,
        index_group_index=0,
    )

    out, _ = FlashMLASparseImpl._forward_bf16_kv(
        stub_impl, q, kv_cache, topk_indices, attn_metadata, q.shape[1]
    )

    assert out.shape[0] == num_mqa_tokens
    assert captured["indices"].shape[0] == num_mqa_tokens
    assert captured["actual_num_heads"] == q.shape[1]
    reference = _triton_convert_reference_impl(
        attn_metadata.req_id_per_token[:num_mqa_tokens],
        attn_metadata.block_table,
        topk_indices,
        block_size,
        num_topk_tokens,
    )
    torch.testing.assert_close(captured["indices"], reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    "seq_lens,max_buf,expected",
    [
        # Basic split: totals per chunk ≤ max_buf
        (torch.tensor([2, 3, 4, 2]), 5, [(0, 2), (2, 3), (3, 4)]),
        # Exact fits should split between items when adding the next would overflow
        (torch.tensor([5, 5, 5]), 5, [(0, 1), (1, 2), (2, 3)]),
        # All requests fit in a single chunk
        (torch.tensor([1, 1, 1]), 10, [(0, 3)]),
        # Large buffer
        (torch.tensor([4, 4, 4]), 100, [(0, 3)]),
    ],
)
def test_split_prefill_chunks(seq_lens, max_buf, expected):
    out = split_prefill_chunks(seq_lens, max_buf)
    assert out == expected


@pytest.mark.parametrize(
    ("max_query_len", "expected"),
    [(32768, True), (33024, False)],
)
def test_masked_mha_workspace_fits_single_request_boundary(max_query_len, expected):
    """A 32K prefill needs the default workspace exactly; shrinking it would
    push a supported request onto MQA."""
    assert (
        _masked_mha_workspace_fits(
            batch_size=1,
            max_query_len=max_query_len,
            max_context_chunk_seq_len=0,
            workspace_numel=GLOBAL_TOPK_MASK_MAX_BYTES // torch.int32.itemsize,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("backend_name", "tensor_parallel_size", "query_len"),
    [
        ("FLASHMLA_SPARSE", 4, 48 * 1024),
        ("FLASHMLA_SPARSE", 8, 112 * 1024),
        ("FLASHINFER_MLA_SPARSE", 4, 36 * 1024),
        ("FLASHINFER_MLA_SPARSE", 8, 64 * 1024),
    ],
)
def test_masked_mha_workspace_guards_long_routing_policy(
    backend_name, tensor_parallel_size, query_len
):
    assert _use_masked_mha(
        backend_name=backend_name,
        tensor_parallel_size=tensor_parallel_size,
        qk_head_dim=256,
        v_head_dim=256,
        query_len=query_len,
        seq_len=query_len,
        has_context=False,
    )
    assert not _masked_mha_workspace_fits(
        batch_size=1,
        max_query_len=query_len,
        max_context_chunk_seq_len=0,
        workspace_numel=GLOBAL_TOPK_MASK_MAX_BYTES // torch.int32.itemsize,
    )


def test_masked_mha_workspace_fits_accounts_for_batch_and_context():
    """Request count and context chunk length are independent multipliers."""
    base = dict(batch_size=2, max_query_len=2048, max_context_chunk_seq_len=2048)
    exact = math.prod(_topk_mask_shape(2, 2048, 2048))

    assert _masked_mha_workspace_fits(**base, workspace_numel=exact)
    assert not _masked_mha_workspace_fits(
        **{**base, "batch_size": 3}, workspace_numel=exact
    )
    assert not _masked_mha_workspace_fits(
        **{**base, "max_context_chunk_seq_len": 4096}, workspace_numel=exact
    )


PREFILL_BATCH_SPECS = {
    "short_dense_mha": BatchSpec(seq_lens=[64, 128], query_lens=[64, 128]),
    "short_context_dense_mha": BatchSpec(seq_lens=[128, 160], query_lens=[64, 32]),
    "masked_mha": BatchSpec(seq_lens=[256], query_lens=[256]),
    "masked_mha_chunked_context": BatchSpec(seq_lens=[448, 384], query_lens=[256, 256]),
}


@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 10,
    reason="Sparse MLA forward_mha requires FA4 (SM100+)",
)
@pytest.mark.parametrize("batch_name", list(PREFILL_BATCH_SPECS.keys()))
@pytest.mark.parametrize("kv_cache_dtype", ["auto"])
@pytest.mark.parametrize(
    ("num_heads", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"),
    [
        pytest.param(128, 128, 64, 128, id="deepseek_hd192_v128"),
        pytest.param(64, 192, 64, 256, id="glm5_hd256_v256"),
    ],
)
def test_sparse_backend_prefill_correctness(
    default_vllm_config,
    dist_init,
    batch_name,
    kv_cache_dtype,
    num_heads,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    workspace_init,
):
    """Test dense and masked MHA across supported sparse MLA dimensions."""
    backend_cls = FlashMLASparseBackend
    batch_spec = PREFILL_BATCH_SPECS[batch_name]

    device = torch.device("cuda")
    dtype = torch.bfloat16
    block_size = 64

    kv_lora_rank = 512
    head_size = kv_lora_rank + qk_rope_head_dim
    masked_mha = batch_name.startswith("masked_mha")
    topk_tokens = 200 if masked_mha else 512

    max_seqlen = max(batch_spec.seq_lens)
    total_cache_tokens = sum(batch_spec.seq_lens)

    vllm_config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-V2-Lite-Chat",
        tensor_parallel_size=1,
        max_model_len=max_seqlen,
        num_gpu_blocks=max(2048, cdiv(total_cache_tokens, block_size) + 1),
        block_size=block_size,
        hf_config_override={
            "index_topk": topk_tokens,
            "attn_module_list_cfg": [{"topk_tokens": topk_tokens}],
        },
    )
    model_config = vllm_config.model_config
    model_config.hf_text_config = SimpleNamespace(
        q_lora_rank=None,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        model_type="deepseek_v2",
    )
    model_config.dtype = dtype
    model_config.model_arch_config.total_num_attention_heads = num_heads
    model_config.get_num_attention_heads = MethodType(
        lambda self, parallel_config: num_heads, model_config
    )
    model_config.get_num_kv_heads = MethodType(
        lambda self, parallel_config: 1, model_config
    )
    model_config.get_head_size = MethodType(lambda self: head_size, model_config)
    model_config.get_sliding_window = MethodType(lambda self: None, model_config)

    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)

    torch.manual_seed(42)

    W_UK = torch.rand(
        kv_lora_rank, num_heads, qk_nope_head_dim, dtype=dtype, device=device
    )
    W_UV = torch.rand(kv_lora_rank, num_heads, v_head_dim, dtype=dtype, device=device)

    seq_lens = batch_spec.seq_lens
    query_lens = batch_spec.query_lens

    # Compute dense reference outputs.
    total_query_tokens = sum(query_lens)
    sparse_indices = torch.full(
        (total_query_tokens, topk_tokens), -1, dtype=torch.int32, device=device
    )

    all_q, all_kv_c_new, all_k_pe_new = [], [], []
    kv_c_contexts, k_pe_contexts = [], []
    reference_outputs = []
    global_token_idx = 0

    for i in range(batch_spec.batch_size):
        s_len = seq_lens[i]
        q_len = query_lens[i]
        ctx_len = s_len - q_len

        q_mha = torch.rand(
            q_len,
            num_heads,
            qk_nope_head_dim + qk_rope_head_dim,
            dtype=dtype,
            device=device,
        )
        kv_c_full = torch.rand(s_len, kv_lora_rank, dtype=dtype, device=device)
        k_pe_full = torch.rand(s_len, 1, qk_rope_head_dim, dtype=dtype, device=device)

        # Decompress all KV for reference
        kv_b_weight = torch.cat([W_UK, W_UV], dim=-1).view(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)
        )
        kv_decompressed = (kv_c_full @ kv_b_weight).view(
            s_len, num_heads, qk_nope_head_dim + v_head_dim
        )
        k_nope_all, v_all = kv_decompressed.split(
            [qk_nope_head_dim, v_head_dim], dim=-1
        )
        k_pe_expanded = k_pe_full.expand(-1, num_heads, -1)
        k_all = torch.cat([k_nope_all, k_pe_expanded], dim=-1)

        for j in range(q_len):
            attend_end = ctx_len + j + 1
            q_tok = q_mha[j : j + 1]  # (1, H, D_qk)
            if masked_mha:
                actual_topk = min(topk_tokens, attend_end)
                attend_indices = torch.randperm(attend_end, device=device)[:actual_topk]
                sparse_indices[global_token_idx, :actual_topk] = attend_indices
                k_attend = k_all[attend_indices]
                v_attend = v_all[attend_indices]
            else:
                k_attend = k_all[:attend_end]  # (N, H, D_qk)
                v_attend = v_all[:attend_end]  # (N, H, D_v)

            q_sdpa = q_tok.unsqueeze(0).transpose(1, 2).float()
            k_sdpa = k_attend.unsqueeze(0).transpose(1, 2).float()
            v_sdpa = v_attend.unsqueeze(0).transpose(1, 2).float()

            out = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, scale=scale
            )
            out = out.transpose(1, 2).squeeze(0)  # (1, H, D_v)
            reference_outputs.append(out.to(dtype).flatten(start_dim=-2))
            global_token_idx += 1

        all_q.append(q_mha)
        all_kv_c_new.append(kv_c_full[ctx_len:])
        all_k_pe_new.append(k_pe_full[ctx_len:])
        kv_c_contexts.append(kv_c_full)
        k_pe_contexts.append(k_pe_full)

    query_cat = torch.cat(all_q, dim=0)
    kv_c_cat = torch.cat(all_kv_c_new, dim=0)
    k_pe_cat = torch.cat(all_k_pe_new, dim=0)
    ref_output = torch.cat(reference_outputs, dim=0)

    vllm_config.cache_config.cache_dtype = kv_cache_dtype
    vllm_config.model_config.hf_config.index_topk = topk_tokens

    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        vllm_config.cache_config.block_size,
        device,
        arange_block_indices=True,
    )

    kv_cache = create_and_prepopulate_kv_cache(
        kv_c_contexts=kv_c_contexts,
        k_pe_contexts=k_pe_contexts,
        block_size=block_size,
        head_size=head_size,
        dtype=dtype,
        device=device,
        num_blocks=vllm_config.cache_config.num_gpu_blocks,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=False,
        kv_cache_dtype=kv_cache_dtype,
    )

    # The sparse builder clones the layer's dense-MHA prefill backend from
    # static_forward_context; register a mock layer carrying one.
    prefill_backend = get_mla_prefill_backend(vllm_config)(
        num_heads=num_heads,
        scale=scale,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        vllm_config=vllm_config,
    )
    vllm_config.compilation_config.static_forward_context["placeholder"] = (
        SimpleNamespace(prefill_backend=prefill_backend)
    )

    builder_cls = backend_cls.get_builder_cls()
    builder = builder_cls(kv_cache_spec, ["placeholder"], vllm_config, device)
    if batch_name == "masked_mha_chunked_context":
        builder.chunked_prefill_workspace_size = block_size * batch_spec.batch_size
        builder.chunked_prefill_workspace = torch.empty(
            (builder.chunked_prefill_workspace_size, head_size),
            dtype=dtype,
            device=device,
        )
    # Drive the queries through the dense-MHA prefill path directly (the routing
    # threshold would otherwise classify these short queries as MQA decodes).
    builder.reorder_batch_threshold = 1
    metadata = builder.build(
        common_prefix_len=0, common_attn_metadata=common_attn_metadata
    )

    mock_indexer = SimpleNamespace(topk_indices_buffer=sparse_indices)

    kv_b_proj_weight = torch.cat([W_UK, W_UV], dim=-1).view(
        kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)
    )

    mock_kv_b_proj = ColumnParallelLinear(
        input_size=kv_lora_rank,
        output_size=num_heads * (qk_nope_head_dim + v_head_dim),
        bias=False,
    ).to(device=device, dtype=dtype)
    mock_kv_b_proj.weight = torch.nn.Parameter(kv_b_proj_weight.T.contiguous())

    impl_cls = backend_cls.get_impl_cls()
    with set_current_vllm_config(vllm_config):
        impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            q_lora_rank=None,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_head_dim=qk_nope_head_dim + qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_b_proj=mock_kv_b_proj,
            indexer=mock_indexer,
        )
        impl.process_weights_after_loading(dtype)

    out_buffer = torch.empty(
        total_query_tokens, num_heads * v_head_dim, dtype=dtype, device=device
    )

    with torch.inference_mode():
        impl.forward_mha(
            q=query_cat,
            kv_c_normed=kv_c_cat,
            k_pe=k_pe_cat,
            # Impls see the bind-time-squeezed [B, N, C] cache; mirror bind_kv_cache.
            kv_c_and_k_pe_cache=kv_cache.squeeze(1),
            attn_metadata=metadata,
            k_scale=torch.tensor(1.0, device=device),
            output=out_buffer,
        )

    assert out_buffer.shape == ref_output.shape
    assert torch.isfinite(out_buffer).all(), "Non-finite values in output"
    torch.testing.assert_close(out_buffer, ref_output, rtol=0.01, atol=0.01)


@pytest.mark.parametrize(
    "seq_lens,query_lens,workspace_size,max_logits_bytes,expected",
    [
        (
            torch.tensor([0]),
            torch.tensor([0]),
            100,
            1000,
            [],
        ),
        # Logits constraint triggers split (M*N exceeds budget)
        # req0: M=10, N=100 -> 1000 elems (4000 bytes) - fits in 5000
        # req1: adding M=10, N=100 -> new_M=20, new_N=200 -> 4000 elems > 1250
        (
            torch.tensor([100, 100, 100]),
            torch.tensor([10, 10, 10]),
            1000,  # workspace allows all
            5000,  # 1250 float32 elems -> forces split
            [
                (slice(0, 1), slice(0, 10)),
                (slice(1, 2), slice(0, 10)),
                (slice(2, 3), slice(0, 10)),
            ],
        ),
        # Both constraints satisfied - all fit in one chunk
        (
            torch.tensor([10, 10, 10]),
            torch.tensor([5, 5, 5]),
            100,
            10000,  # 2500 elems, M*N = 15*30 = 450 < 2500
            [(slice(0, 3), slice(0, 15))],
        ),
        # Workspace constraint triggers first
        (
            torch.tensor([50, 50, 50]),
            torch.tensor([1, 1, 1]),
            50,  # workspace only fits one at a time
            1000000,  # logits budget is huge
            [
                (slice(0, 1), slice(0, 1)),
                (slice(1, 2), slice(0, 1)),
                (slice(2, 3), slice(0, 1)),
            ],
        ),
        # Greedy filling: first two fit, third doesn't
        # req0: M=5, N=10 -> 50 elems
        # req0+1: M=10, N=20 -> 200 elems <= 250
        # req0+1+2: M=15, N=30 -> 450 elems > 250
        (
            torch.tensor([10, 10, 10]),
            torch.tensor([5, 5, 5]),
            100,
            1000,  # 250 elems
            [(slice(0, 2), slice(0, 10)), (slice(2, 3), slice(0, 5))],
        ),
    ],
)
def test_split_indexer_prefill_chunks(
    seq_lens, query_lens, workspace_size, max_logits_bytes, expected
):
    out = split_indexer_prefill_chunks(
        seq_lens,
        query_lens,
        workspace_size,
        max_logits_bytes,
    )
    assert out == expected


def test_split_indexer_prefill_chunks_single_request_overflow():
    """Test that single request exceeding budget is sub-chunked on query dim."""
    seq_lens = torch.tensor([1000, 50])
    query_lens = torch.tensor([100, 5])

    out = split_indexer_prefill_chunks(seq_lens, query_lens, 2000, 1000)
    # max_logits_elems = 250, N=1000 -> max_q = 1 -> 100 query sub-chunks
    expected = [(slice(0, 1), slice(i, i + 1)) for i in range(100)]
    # req1: M=5, N=50 -> 250 elems fits budget
    expected.append((slice(1, 2), slice(0, 5)))
    assert out == expected


# 384 is not a power of two, so it counts via the tiled atomic accumulation
# rather than the single-tile path 128 takes.
@pytest.mark.parametrize("num_topk_tokens", [128, 384])
def test_triton_convert_returns_valid_counts(num_topk_tokens: int):
    """Test that return_valid_counts correctly counts non-negative indices."""
    device = torch.device(DEVICE_TYPE)
    num_tokens = 8
    num_requests = 2
    max_blocks_per_req = 10
    block_size = 64

    req_id = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device=device)
    block_table = torch.arange(
        num_requests * max_blocks_per_req, dtype=torch.int32, device=device
    ).view(num_requests, max_blocks_per_req)

    # Create token indices with varying numbers of valid entries: half the row,
    # a quarter of it, the whole row, then a single valid entry -- twice over.
    token_indices = torch.full(
        (num_tokens, num_topk_tokens), -1, dtype=torch.int32, device=device
    )
    valid_counts_per_token = [
        num_topk_tokens // 2,
        num_topk_tokens // 4,
        num_topk_tokens,
        1,
    ] * 2
    expected_valid = []
    for i in range(num_tokens):
        num_valid = valid_counts_per_token[i]
        token_indices[i, :num_valid] = torch.arange(
            num_valid, dtype=torch.int32, device=device
        ) % (block_size * max_blocks_per_req)
        expected_valid.append(num_valid)

    expected_valid_tensor = torch.tensor(
        expected_valid, dtype=torch.int32, device=device
    )

    # Test with return_valid_counts=True
    result, valid_counts = triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk_tokens,
        return_valid_counts=True,
    )

    torch.testing.assert_close(valid_counts, expected_valid_tensor, rtol=0, atol=0)

    # Test that return_valid_counts=False returns only the indices
    result_only = triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk_tokens,
        return_valid_counts=False,
    )
    assert isinstance(result_only, torch.Tensor)
    for row, num_valid in enumerate(expected_valid):
        compact_valid = result[row, :num_valid].sort().values
        original_valid = result_only[row][result_only[row] >= 0].sort().values
        torch.testing.assert_close(compact_valid, original_valid, rtol=0, atol=0)
        assert torch.all(result[row, num_valid:] == -1)


def test_sparse_mla_index_group_converts_decode_indices_once(monkeypatch):
    device = torch.device(DEVICE_TYPE)
    logical = torch.full((2, 128), -1, dtype=torch.int32, device=device)
    logical[:, :4] = torch.tensor(
        [[0, 2, 5, -1], [1, 3, 7, -1]], dtype=torch.int32, device=device
    )
    builder = SparseMLAIndexGroupBuilder(logical)
    group, leader_index = builder.register_layer(True)
    follower_group, follower_index = builder.register_layer(False)
    assert follower_group is group

    block_table = torch.tensor([[2, 3], [4, 5]], dtype=torch.int32, device=device)
    metadata = SimpleNamespace(
        block_table=block_table,
        block_size=4,
        req_id_per_token=torch.tensor([0, 1], dtype=torch.int32, device=device),
        num_decode_tokens=2,
        num_actual_tokens=2,
    )
    calls = []
    convert = index_group_module.triton_convert_req_index_to_global_index

    def convert_spy(*args, **kwargs):
        calls.append((args, kwargs))
        return convert(*args, **kwargs)

    monkeypatch.setattr(
        index_group_module,
        "triton_convert_req_index_to_global_index",
        convert_spy,
    )
    group.set_logical_topk_ready(leader_index)
    leader_result = group.convert_logical_to_physical_topk(
        leader_index,
        logical,
        metadata,
        block_stride_rows=None,
        return_valid_counts=True,
    )
    follower_result = group.convert_logical_to_physical_topk(
        follower_index,
        logical,
        metadata,
        block_stride_rows=None,
        return_valid_counts=True,
    )
    torch.accelerator.synchronize()

    assert len(calls) == 1
    leader_indices, leader_counts = leader_result
    follower_indices, follower_counts = follower_result
    torch.testing.assert_close(follower_indices, leader_indices)
    torch.testing.assert_close(follower_counts, leader_counts)


def test_sparse_mla_index_group_falls_back_for_prefill_sized_batch(monkeypatch):
    device = torch.device(DEVICE_TYPE)
    logical = torch.full((4, 128), -1, dtype=torch.int32, device=device)
    logical[:, 0] = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)
    builder = SparseMLAIndexGroupBuilder(logical, max_decode_rows=2)
    group, leader_index = builder.register_layer(True)
    follower_group, follower_index = builder.register_layer(False)
    assert follower_group is group

    metadata = SimpleNamespace(
        block_table=torch.tensor(
            [[2], [3], [4], [5]], dtype=torch.int32, device=device
        ),
        block_size=4,
        req_id_per_token=torch.arange(4, dtype=torch.int32, device=device),
    )
    calls = 0
    convert = index_group_module.triton_convert_req_index_to_global_index

    def convert_spy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return convert(*args, **kwargs)

    monkeypatch.setattr(
        index_group_module,
        "triton_convert_req_index_to_global_index",
        convert_spy,
    )
    leader_result = group.convert_logical_to_physical_topk(
        leader_index,
        logical,
        metadata,
        block_stride_rows=None,
        return_valid_counts=False,
    )
    follower_result = group.convert_logical_to_physical_topk(
        follower_index,
        logical,
        metadata,
        block_stride_rows=None,
        return_valid_counts=False,
    )
    torch.accelerator.synchronize()

    assert calls == 2
    torch.testing.assert_close(follower_result, leader_result)


def test_sparse_mla_index_groups_own_distinct_physical_buffers():
    logical = torch.empty((8, 128), dtype=torch.int32, device=DEVICE_TYPE)
    builder = SparseMLAIndexGroupBuilder(logical, max_decode_rows=2)

    first, _ = builder.register_layer(True)
    follower, _ = builder.register_layer(False)
    second, _ = builder.register_layer(True)

    assert follower is first
    assert second is not first
    assert first.physical_topk_indices.shape == (2, 128)
    assert second.physical_topk_indices.shape == (2, 128)
    assert first.physical_topk_indices.data_ptr() != (
        second.physical_topk_indices.data_ptr()
    )


# HiSparse is host-resident-only and kernel-only: runtime construction
# raises without the compiled CUDA ops.
requires_hisparse_ops = pytest.mark.skipif(
    not _has_hisparse_ops(),
    reason="HiSparse CUDA ops not compiled",
)


def _make_hisparse_runtime(
    *,
    top_k: int = 4,
    device_buffer_size: int = 5,
    max_num_reqs: int = 2,
    row_width: int = 8,
    block_size: int = 64,
    max_swap_rows: int | None = None,
    index_group: hisparse_runtime.HiSparseIndexGroup | None = None,
) -> HiSparseRuntime:
    runtime = HiSparseRuntime(
        config=ResolvedHiSparseConfig(
            top_k=top_k,
            device_buffer_size=device_buffer_size,
        ),
        max_num_reqs=max_num_reqs,
        row_width=row_width,
        kv_dtype=torch.float32,
        device=DEVICE_TYPE,
        max_swap_rows=max_swap_rows,
        index_group=index_group,
    )
    blocks_per_request = cdiv(runtime.region_stride, block_size)
    # Leave one extra physical block for resident-tier tests. HMA group layouts
    # alias this backing, but distinct group allocations use distinct block IDs.
    num_blocks = max_num_reqs * blocks_per_request + 1
    raw = torch.zeros(
        num_blocks * block_size * row_width,
        dtype=torch.float32,
        device=DEVICE_TYPE,
    ).view(torch.int8)
    block_table = torch.arange(
        max_num_reqs * blocks_per_request,
        dtype=torch.int32,
        device=DEVICE_TYPE,
    ).view(max_num_reqs, blocks_per_request)
    runtime.bind_hot_cache(
        raw,
        byte_offset=0,
        block_stride=block_size * row_width * torch.float32.itemsize,
        num_blocks=num_blocks,
        block_size=block_size,
        block_table=block_table,
    )
    runtime.request_state_indices = torch.arange(
        max_num_reqs, dtype=torch.int32, device=runtime.device
    )
    return runtime


def _make_hisparse_cache_handle(
    *,
    top_k: int = 4,
    device_buffer_size: int = 5,
    max_num_reqs: int = 2,
    row_width: int = 8,
    block_size: int = 64,
    max_swap_rows: int | None = None,
) -> HiSparseCacheHandle:
    runtime = _make_hisparse_runtime(
        top_k=top_k,
        device_buffer_size=device_buffer_size,
        max_num_reqs=max_num_reqs,
        row_width=row_width,
        block_size=block_size,
        max_swap_rows=max_swap_rows,
    )
    return HiSparseCacheHandle(runtime)


def _make_hisparse_index_group(
    cache_handle: HiSparseCacheHandle, kv_cache_dtype: str = "auto"
) -> HiSparseMLAIndexGroup:
    group = object.__new__(HiSparseMLAIndexGroup)
    group.caches = [cache_handle]
    return group


@requires_hisparse_ops
def test_hisparse_uses_graph_stable_request_state_mapping():
    device = torch.device(DEVICE_TYPE)
    block_size, row_width = 64, 8
    runtime = _make_hisparse_runtime(
        top_k=1,
        device_buffer_size=2,
        max_num_reqs=2,
        row_width=row_width,
        block_size=block_size,
    )
    runtime.bind_source_cache(
        torch.arange(block_size * row_width, dtype=torch.float32)
        .view(1, block_size, row_width)
        .pin_memory()
    )
    runtime.request_state_indices = torch.tensor([1], dtype=torch.int32, device=device)

    cache_handle = HiSparseCacheHandle(runtime)
    cache_handle.all_context_pages_resident = False
    runtime.begin_forward()
    cache_handle.swap_in(
        req_id_per_token=torch.tensor([0], dtype=torch.int32, device=device),
        block_table=torch.tensor([[0]], dtype=torch.int32, device=device),
        logical_topk_indices=torch.tensor([[0]], dtype=torch.int32, device=device),
        block_size=block_size,
    )
    torch.accelerator.synchronize()

    assert (runtime.index_group.device_global_indices[0] == -1).all()
    assert (runtime.index_group.device_global_indices[1] == 0).any()


@requires_hisparse_ops
def test_hisparse_maps_speculative_rows_through_request_state():
    """Multiple verification rows for one request share its persistent state."""
    device = torch.device(DEVICE_TYPE)
    block_size, row_width = 64, 8
    runtime = _make_hisparse_runtime(
        top_k=1,
        device_buffer_size=2,
        max_num_reqs=1,
        row_width=row_width,
        block_size=block_size,
        max_swap_rows=4,
    )
    runtime.bind_source_cache(
        torch.zeros(1, block_size, row_width, dtype=torch.float32).pin_memory()
    )

    cache_handle = HiSparseCacheHandle(runtime)
    resident_table = torch.tensor([[1]], dtype=torch.int32, device=device)
    cache_handle.bind_cache(
        runtime.hot.cache.view(torch.int8),
        byte_offset=0,
        block_stride=block_size * row_width * torch.float32.itemsize,
        num_blocks=2,
        block_size=block_size,
        block_table=resident_table,
        slot_mapping=torch.arange(4, dtype=torch.int64, device=device),
    )
    cache_handle.all_context_pages_resident = True
    runtime.begin_forward()
    physical = cache_handle.swap_in(
        req_id_per_token=torch.zeros(4, dtype=torch.int32, device=device),
        block_table=torch.zeros((1, 1), dtype=torch.int32, device=device),
        logical_topk_indices=torch.zeros((4, 1), dtype=torch.int32, device=device),
        block_size=block_size,
    )
    torch.accelerator.synchronize()

    assert physical.tolist() == [[block_size]] * 4


@requires_hisparse_ops
def test_hisparse_resident_rows_bypass_hot_lru():
    device = torch.device(DEVICE_TYPE)
    block_size, row_width = 64, 8
    cache_handle = _make_hisparse_cache_handle(
        top_k=4,
        device_buffer_size=5,
        max_num_reqs=1,
        row_width=row_width,
        block_size=block_size,
    )
    raw = cache_handle.runtime.hot.cache.view(torch.int8)
    resident_table = torch.tensor([[1]], dtype=torch.int32, device=device)
    resident_slots = torch.tensor([block_size], dtype=torch.int64, device=device)
    cache_handle.bind_cache(
        raw,
        byte_offset=0,
        block_stride=block_size * row_width * torch.float32.itemsize,
        num_blocks=2,
        block_size=block_size,
        block_table=resident_table,
        slot_mapping=resident_slots,
    )

    host = torch.randn(1, block_size, row_width).pin_memory()
    cache_handle.runtime.bind_source_cache(host)
    resident_rows = host[0].to(device).add(7)
    cache_handle.runtime.hot.cache[1].copy_(resident_rows)
    source_table = torch.tensor([[5]], dtype=torch.int32, device=device)
    request_ids = torch.zeros(1, dtype=torch.int32, device=device)
    topk = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32, device=device)
    lru_before = cache_handle.runtime.index_group.lru_slots.clone()

    cache_handle.runtime.begin_forward()
    indices, valid_counts = cache_handle.swap_in(
        req_id_per_token=request_ids,
        block_table=source_table,
        logical_topk_indices=topk,
        block_size=block_size,
        return_valid_counts=True,
    )
    torch.accelerator.synchronize()

    expected_indices = topk + block_size
    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(
        valid_counts, torch.tensor([topk.shape[1]], dtype=torch.int32, device=device)
    )
    torch.testing.assert_close(cache_handle.runtime.index_group.lru_slots, lru_before)
    torch.testing.assert_close(
        cache_handle.runtime.hot.attention_cache.view(-1, row_width)[
            indices.to(torch.long)
        ],
        resident_rows[topk.to(torch.long)],
    )


@requires_hisparse_ops
def test_hisparse_bf16_resident_cache_is_flat_padded():
    device = torch.device(DEVICE_TYPE)
    block_size, row_width = 4, 8
    page_bytes = block_size * row_width * torch.float32.itemsize
    cache_handle = _make_hisparse_cache_handle(
        top_k=128,
        device_buffer_size=128,
        max_num_reqs=1,
        row_width=row_width,
        block_size=block_size,
    )
    raw = torch.zeros(2 * page_bytes, dtype=torch.uint8, device=device)
    cache_handle.bind_cache(
        raw,
        byte_offset=0,
        block_stride=2 * page_bytes,
        num_blocks=1,
        block_size=block_size,
        block_table=torch.tensor([[0]], dtype=torch.int32, device=device),
        slot_mapping=torch.tensor([0], dtype=torch.int64, device=device),
    )
    assert cache_handle.view is not None
    assert cache_handle.view.attention_cache.is_contiguous()


@requires_hisparse_ops
def test_hisparse_swap_in_preserves_rows_across_eviction():
    device = torch.device(DEVICE_TYPE)
    block_size = 64
    row_width = 64
    num_blocks = 64
    top_k = 128
    buf = 256
    num_reqs = 4

    kv_pool = torch.randn(
        (num_blocks, block_size, row_width), dtype=torch.float32
    ).pin_memory()
    flat_pool = kv_pool.reshape(-1, row_width)

    runtime = _make_hisparse_runtime(
        top_k=top_k,
        device_buffer_size=buf,
        max_num_reqs=num_reqs,
        row_width=row_width,
    )
    runtime.bind_source_cache(kv_pool)
    cache = HiSparseCacheHandle(runtime)
    cache.all_context_pages_resident = False
    cache.runtime.begin_forward()

    blocks_per_req = num_blocks // num_reqs
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(
        num_reqs, blocks_per_req
    )
    req_ids = torch.arange(num_reqs, dtype=torch.int32, device=device)
    seq_len = blocks_per_req * block_size
    base = torch.arange(top_k, dtype=torch.int32, device=device)
    for step in range(3):
        cache.runtime.begin_forward()
        topk = torch.stack(
            [(base + step * top_k + row * 17) % seq_len for row in range(num_reqs)]
        )
        topk[:, -1] = -1

        hot_indices, valid_counts = cache.swap_in(
            req_id_per_token=req_ids,
            block_table=block_table,
            logical_topk_indices=topk.clone(),
            block_size=block_size,
            return_valid_counts=True,
        )
        torch.accelerator.synchronize()

        global_ref = _triton_convert_reference_impl(
            req_ids, block_table, topk, block_size, top_k
        )
        valid = hot_indices >= 0
        torch.testing.assert_close(
            valid_counts,
            (global_ref >= 0).sum(dim=1, dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        gathered = cache.runtime.hot.attention_cache.reshape(-1, row_width)[
            hot_indices[valid].to(torch.long)
        ].cpu()
        expected = flat_pool[global_ref[valid].cpu().to(torch.long)]
        torch.testing.assert_close(gathered, expected)


@requires_hisparse_ops
def test_hisparse_multi_step_swaps_match_independent():
    """The shared LRU must retain every speculative step until followers run."""
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(0)
    block_size = 64
    row_width = 64
    num_blocks = 64
    top_k = 128
    buf = 3 * top_k
    num_reqs = 4

    kv_pool = torch.randn(
        (num_blocks, block_size, row_width), dtype=torch.float32
    ).pin_memory()

    def make(
        index_group: hisparse_runtime.HiSparseIndexGroup | None = None,
    ) -> HiSparseCacheHandle:
        runtime = _make_hisparse_runtime(
            top_k=top_k,
            max_num_reqs=num_reqs,
            device_buffer_size=buf,
            row_width=row_width,
            block_size=block_size,
            max_swap_rows=2 * num_reqs,
            index_group=index_group,
        )
        runtime.bind_source_cache(kv_pool)
        cache = HiSparseCacheHandle(runtime)
        cache.all_context_pages_resident = False
        return cache

    blocks_per_req = num_blocks // num_reqs
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(
        num_reqs, blocks_per_req
    )
    req_ids = torch.arange(num_reqs, dtype=torch.int32, device=device)
    seq_len = blocks_per_req * block_size
    producer = make()
    shared = make(producer.runtime.index_group)
    indep = make()
    base = torch.arange(top_k, dtype=torch.int32, device=device)
    for iteration in range(2):
        producer.runtime.begin_forward()
        shared.runtime.begin_forward()
        indep.runtime.begin_forward()
        topks = [
            torch.stack(
                [
                    (base + (iteration * 2 + step) * top_k + row * 17) % seq_len
                    for row in range(num_reqs)
                ]
            )
            for step in range(2)
        ]
        kw = dict(
            req_id_per_token=req_ids,
            block_table=block_table,
            block_size=block_size,
        )
        producer_indices = [
            producer.swap_in(logical_topk_indices=topk.clone(), **kw) for topk in topks
        ]
        independent_indices = [
            indep.swap_in(logical_topk_indices=topk.clone(), **kw) for topk in topks
        ]
        follower_indices = [
            shared.swap_in(logical_topk_indices=topk.clone(), **kw) for topk in topks
        ]
        torch.accelerator.synchronize()

        for idx_full, idx_indep, idx_shared in zip(
            producer_indices, independent_indices, follower_indices
        ):
            torch.testing.assert_close(idx_shared, idx_full, rtol=0, atol=0)
            torch.testing.assert_close(idx_indep, idx_full, rtol=0, atol=0)
        torch.testing.assert_close(shared.runtime.hot.cache, indep.runtime.hot.cache)

        shared_rows = shared.runtime.hot.attention_cache.reshape(-1, row_width)
        host_rows = kv_pool.reshape(-1, row_width)
        for physical_rows, logical_rows in zip(follower_indices, topks):
            global_rows = _triton_convert_reference_impl(
                req_ids, block_table, logical_rows, block_size, top_k
            )
            valid = physical_rows >= 0
            torch.testing.assert_close(
                shared_rows[physical_rows[valid].to(torch.long)].cpu(),
                host_rows[global_rows[valid].cpu().to(torch.long)],
            )


@requires_hisparse_ops
def test_hisparse_multi_step_writes_request_major_output():
    device = torch.device(DEVICE_TYPE)
    block_size = 64
    row_width = 8
    num_reqs = 2
    query_len = 3
    top_k = 4
    cache = _make_hisparse_cache_handle(
        top_k=top_k,
        device_buffer_size=query_len * top_k,
        max_num_reqs=num_reqs,
        row_width=row_width,
        block_size=block_size,
        max_swap_rows=num_reqs * query_len,
    )
    host = torch.arange(num_reqs * block_size * row_width, dtype=torch.float32).view(
        num_reqs, block_size, row_width
    )
    cache.runtime.bind_source_cache(host.pin_memory())
    block_table = torch.arange(num_reqs, dtype=torch.int32, device=device).view(-1, 1)
    request_ids = torch.arange(num_reqs, dtype=torch.int32, device=device)
    logical = torch.tensor(
        [
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
            [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
        ],
        dtype=torch.int32,
        device=device,
    )
    physical = torch.full_like(logical, -1)
    valid_counts = torch.zeros((num_reqs, query_len), dtype=torch.int32, device=device)

    cache.runtime.begin_forward()
    for step in range(query_len):
        cache.swap_in(
            request_ids,
            block_table,
            logical[:, step],
            block_size=block_size,
            return_valid_counts=True,
            attention_indices_out=physical[:, step],
            valid_counts_out=valid_counts[:, step],
        )
    torch.accelerator.synchronize()

    torch.testing.assert_close(
        valid_counts,
        torch.full_like(valid_counts, top_k),
    )
    global_rows = _triton_convert_reference_impl(
        torch.arange(num_reqs, device=device, dtype=torch.int32).repeat_interleave(
            query_len
        ),
        block_table,
        logical.view(num_reqs * query_len, top_k),
        block_size,
        top_k,
    )
    hot_rows = cache.runtime.hot.attention_cache.reshape(-1, row_width)
    host_rows = host.reshape(-1, row_width)
    torch.testing.assert_close(
        hot_rows[physical.view(num_reqs * query_len, top_k).to(torch.long)].cpu(),
        host_rows[global_rows.cpu().to(torch.long)],
    )


@requires_hisparse_ops
@pytest.mark.parametrize(
    ("cudagraph_mode", "expected_layer_mirrors"),
    [
        (CUDAGraphMode.NONE, 1),
        (CUDAGraphMode.PIECEWISE, 0),
        (CUDAGraphMode.FULL, 0),
    ],
)
def test_hisparse_kv_update_uses_common_resident_write_path(
    monkeypatch, cudagraph_mode, expected_layer_mirrors
):
    device = torch.device(DEVICE_TYPE)
    block_size = 4
    row_width = 8
    cache_handle = _make_hisparse_cache_handle(
        max_num_reqs=1,
        row_width=row_width,
        block_size=block_size,
    )
    resident_slots = torch.tensor([4, 5, -1], dtype=torch.int64, device=device)
    cache_handle.bind_cache(
        cache_handle.runtime.hot.cache.view(torch.int8),
        byte_offset=0,
        block_stride=block_size * row_width * torch.float32.itemsize,
        num_blocks=cache_handle.runtime.hot.cache.shape[0],
        block_size=block_size,
        block_table=torch.tensor([[1]], dtype=torch.int32, device=device),
        slot_mapping=resident_slots,
    )
    cache_handle.mirror_staging_cache = torch.empty(
        (1, block_size, row_width), dtype=torch.float32, device=device
    )
    cache_handle.mirror_staging_slots = torch.arange(
        block_size, dtype=torch.int64, device=device
    )
    slots = torch.tensor([3, 7, -1], dtype=torch.int64, device=device)
    kv_c = torch.randn(8, row_width - 2, device=device)
    k_pe = torch.randn(8, 1, 2, device=device)
    cache_handle.num_actual_tokens = slots.numel()
    cache_handle.decode_batch = False
    cache_handle.submit_layer_mirror = MagicMock()
    monkeypatch.setattr(
        hisparse_runtime_module,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=cudagraph_mode),
    )
    source_cache = torch.zeros_like(cache_handle.view.cache)
    impl = object.__new__(FlashMLASparseImpl)
    layer = SimpleNamespace(
        hisparse_cache=cache_handle,
        use_pcp=False,
        impl=impl,
    )
    mla_attention.MLAAttention.update_kv_cache(
        layer,
        kv_c,
        k_pe,
        source_cache,
        slots,
        SimpleNamespace(num_decode_tokens=0),
        "auto",
        torch.tensor(1.0, device=device),
    )
    torch.accelerator.synchronize()

    expected = torch.cat([kv_c[:2], k_pe[:2, 0]], dim=-1).cpu()
    torch.testing.assert_close(
        cache_handle.view.cache.view(-1, row_width)[resident_slots[:2]],
        expected.to(device),
    )
    torch.testing.assert_close(source_cache, torch.zeros_like(source_cache))
    staged = cache_handle.mirror_staging_cache.view(-1, row_width)
    staged_expected = torch.cat([kv_c[:3], k_pe[:3, 0]], dim=-1)
    torch.testing.assert_close(staged[:3], staged_expected)
    assert cache_handle.submit_layer_mirror.call_count == expected_layer_mirrors


@requires_hisparse_ops
def test_hisparse_row_dma_copies_discontiguous_spans_across_layers():
    device = torch.device(DEVICE_TYPE)
    block_size = 4
    row_width = 8
    num_rows = 16
    resident_caches = tuple(
        (
            torch.arange(num_rows * row_width, dtype=torch.uint8, device=device)
            .add_(layer_index * 17)
            .view(num_rows // block_size, block_size, row_width)
        )
        for layer_index in range(2)
    )
    host_caches = tuple(
        torch.zeros((num_rows, row_width), dtype=torch.uint8, pin_memory=True)
        for _ in resident_caches
    )
    worker = object.__new__(HiSparseConnectorWorker)
    worker.is_host_writer = True
    worker.kernel_block_size = block_size
    worker.resident_caches = resident_caches
    worker.host_caches = host_caches
    worker.cache_handles = [
        SimpleNamespace(
            decode_batch=True,
            runtime=SimpleNamespace(resident_source_index=0),
        )
        for _ in resident_caches
    ]
    worker.dma_stream = torch.cuda.Stream(device=device)
    worker.host_write_event = torch.Event()
    worker._dma_free_descriptors = []
    worker._pending_dma_descriptors = deque()
    worker._pending_transfer_events = deque()
    worker._enqueued_transfer_ids = []
    worker._dma_submitted = False
    worker._set_row_mirrors(
        (
            SparseKVRowMirror((1,), 5, 2),
            SparseKVRowMirror((8,), 10, 4),
        )
    )

    worker._enqueue_row_dma(range(2))
    current_stream().wait_event(worker.host_write_event)
    torch.accelerator.synchronize()

    for resident, host in zip(resident_caches, host_caches):
        flat_resident = resident.view(num_rows, row_width).cpu()
        torch.testing.assert_close(host[5:7], flat_resident[1:3])
        torch.testing.assert_close(host[10:14], flat_resident[8:12])


@requires_hisparse_ops
def test_hisparse_remaps_strided_hma_rows_for_attention():
    device = torch.device(DEVICE_TYPE)
    block_size, row_width = 4, 8
    runtime = _make_hisparse_runtime(
        max_num_reqs=1,
        block_size=block_size,
        row_width=row_width,
    )
    page_elements = block_size * row_width
    stride_elements = 3 * page_elements
    byte_offset = page_elements * torch.float32.itemsize
    raw = torch.zeros(2 * stride_elements, dtype=torch.float32, device=device).view(
        torch.int8
    )
    runtime.bind_hot_cache(
        raw,
        byte_offset=byte_offset,
        block_stride=stride_elements * torch.float32.itemsize,
        num_blocks=2,
        block_size=block_size,
        block_table=torch.tensor([[0, 1]], dtype=torch.int32, device=device),
    )

    source = (
        torch.arange(32 * row_width, dtype=torch.float32)
        .view(8, block_size, row_width)
        .pin_memory()
    )
    runtime.bind_source_cache(source)
    cache = HiSparseCacheHandle(runtime)
    cache.all_context_pages_resident = False
    runtime.begin_forward()
    attention_indices = cache.swap_in(
        req_id_per_token=torch.tensor([0], dtype=torch.int32, device=device),
        block_table=torch.arange(8, dtype=torch.int32, device=device).view(1, 8),
        logical_topk_indices=torch.tensor(
            [[0, 1, 2, 3]], dtype=torch.int32, device=device
        ),
        block_size=block_size,
    )

    assert runtime.hot.attention_cache is not None
    assert runtime.hot.attention_cache.is_contiguous()
    physical_indices = runtime.index_group.shared_topk.device_topk_rows[0]
    expected = (
        physical_indices // block_size * (stride_elements // row_width)
        + physical_indices % block_size
    )
    torch.testing.assert_close(attention_indices[0], expected)


@requires_hisparse_ops
def test_hisparse_gather_zeroes_unaligned_destination():
    """Invalid host rows are safely zeroed even at a byte-unaligned address."""
    device = torch.device(DEVICE_TYPE)
    row_bytes = 16
    host_cache = torch.ones(1, row_bytes, dtype=torch.uint8).pin_memory()
    storage = torch.full((2 * row_bytes + 1,), 7, dtype=torch.uint8, device=device)
    hot_cache = torch.as_strided(
        storage,
        size=(2, 1, row_bytes),
        stride=(row_bytes + 1, row_bytes, 1),
    )
    global_indices = torch.tensor([[1]], dtype=torch.int32, device=device)
    hot_indices = torch.tensor([[1]], dtype=torch.int32, device=device)
    miss_mask = torch.ones_like(global_indices)

    torch.ops._C_cache_ops.hisparse_gather_plan(
        host_cache,
        hot_cache,
        global_indices,
        hot_indices,
        miss_mask,
        None,
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(hot_cache[1, 0], torch.zeros_like(hot_cache[1, 0]))
    assert storage[row_bytes].item() == 7


@requires_hisparse_ops
def test_hisparse_gather_rejects_short_attention_block_stride():
    """Reject a stride that would let the gather kernel cross block bounds."""
    device = torch.device(DEVICE_TYPE)
    host_cache = torch.ones(1, 16, dtype=torch.uint8).pin_memory()
    hot_cache = torch.empty((1, 2, 16), dtype=torch.uint8, device=device)
    global_indices = torch.zeros((1, 1), dtype=torch.int32, device=device)
    hot_indices = torch.zeros_like(global_indices)
    miss_mask = torch.ones_like(global_indices)
    attention_indices = torch.empty_like(global_indices)

    with pytest.raises(RuntimeError, match="stride must cover one hot block"):
        torch.ops._C_cache_ops.hisparse_gather_plan(
            host_cache,
            hot_cache,
            global_indices,
            hot_indices,
            miss_mask,
            None,
            attention_indices,
            1,
        )


@requires_hisparse_ops
def test_hisparse_newest_write_and_recycled_slot_invalidation():
    """Newest writes clamp padding and invalidated rows are loaded again."""
    device = torch.device(DEVICE_TYPE)
    block_size = 4
    row_width = 8
    num_blocks = 8

    kv_pool = (
        torch.arange(num_blocks * block_size * row_width, dtype=torch.float32)
        .view(num_blocks, block_size, row_width)
        .pin_memory()
    )
    flat_pool = kv_pool.reshape(-1, row_width)

    cache_handle = _make_hisparse_cache_handle(block_size=block_size)
    cache_handle.runtime.eager_host_mirror = True

    block_table = torch.tensor([[2, 0, 4]], dtype=torch.int32, device=device)
    resident_slots = torch.tensor([4, -1], dtype=torch.int64, device=device)
    cache_handle.bind_cache(
        cache_handle.runtime.hot.cache.view(torch.int8),
        byte_offset=0,
        block_stride=block_size * row_width * torch.float32.itemsize,
        num_blocks=cache_handle.runtime.hot.cache.shape[0],
        block_size=block_size,
        block_table=torch.tensor([[0, 0, 1]], dtype=torch.int32, device=device),
        slot_mapping=resident_slots,
    )
    req_ids = torch.tensor([0], dtype=torch.int32, device=device)
    newest_global = 4 * block_size
    padded_global = 1
    padded_row = flat_pool[padded_global].clone()
    slot_mapping = torch.tensor(
        [newest_global, padded_global], dtype=torch.int64, device=device
    )
    request_state_indices = torch.tensor([0, -1], dtype=torch.int32, device=device)
    cache_handle.runtime.request_state_indices = request_state_indices

    kv_c = torch.randn(3, row_width - 2, device=device)
    k_pe = torch.randn(3, 1, 2, device=device)
    cache_handle.runtime.bind_source_cache(kv_pool)
    cache_handle.num_actual_tokens = slot_mapping.numel()
    ops.concat_and_cache_mla(
        kv_c[:1],
        k_pe[:1, 0],
        cache_handle.view.cache,
        resident_slots[:1],
        kv_cache_dtype="auto",
        scale=torch.tensor(1.0, device=device),
    )
    torch.accelerator.synchronize()
    expected_row = torch.cat([kv_c[0], k_pe[0, 0]]).cpu()
    flat_pool[newest_global].copy_(expected_row)
    torch.testing.assert_close(flat_pool[newest_global], expected_row)
    torch.testing.assert_close(flat_pool[padded_global], padded_row)
    flat_hot = cache_handle.runtime.hot.cache.view(-1, row_width)
    torch.testing.assert_close(flat_hot[resident_slots[0]].cpu(), expected_row)

    topk = torch.tensor([[0, -1, -1, -1]], dtype=torch.int32, device=device)
    cache_handle.all_context_pages_resident = False
    cache_handle.runtime.begin_forward()
    hot_indices = cache_handle.swap_in(
        req_id_per_token=req_ids,
        block_table=block_table,
        logical_topk_indices=topk,
        block_size=block_size,
    )
    torch.accelerator.synchronize()
    stale_hot_slot = hot_indices[0, 0].item()
    stale_row = flat_hot[stale_hot_slot].clone()

    flat_pool[8] += 1000
    cache_handle.runtime.invalidate_written_slots(
        torch.tensor([8], dtype=torch.int64, device=device), req_ids
    )
    cache_handle.runtime.begin_forward()
    hot_indices = cache_handle.swap_in(
        req_id_per_token=req_ids,
        block_table=block_table,
        logical_topk_indices=topk,
        block_size=block_size,
    )
    torch.accelerator.synchronize()
    idx = hot_indices.cpu().tolist()[0][0]
    assert not torch.equal(flat_hot[idx], stale_row)
    torch.testing.assert_close(flat_hot[idx].cpu(), flat_pool[8])


@requires_hisparse_ops
def test_hisparse_mixed_batch_bf16_row_split(
    default_vllm_config, dist_init, workspace_init
):
    """Host-resident mixed batch on the bf16 path is row-split.

    Two long-context speculative-decode requests + one short local-prefill
    chunk: every decode step must be served from the bounded hot buffer before
    it is reused, while only the prefill rows' blocks are staged host->GPU.
    """
    ok, reason = flashmla.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)

    device = torch.device(DEVICE_TYPE)
    dtype = torch.bfloat16
    torch.manual_seed(0)

    num_heads = 64
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    head_size = kv_lora_rank + qk_rope_head_dim
    topk_tokens = 128
    block_size = 64

    # Long decode contexts + a short prefill chunk (router shortcut shape).
    batch_spec = BatchSpec(seq_lens=[2048, 2048, 192], query_lens=[2, 2, 64])
    max_seqlen = max(batch_spec.seq_lens)
    total_cache_tokens = sum(batch_spec.seq_lens)
    total_tokens = batch_spec.compute_num_tokens()

    vllm_config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-V2-Lite-Chat",
        tensor_parallel_size=1,
        max_model_len=max_seqlen,
        num_gpu_blocks=max(2048, cdiv(total_cache_tokens, block_size) + 1),
        block_size=block_size,
        hf_config_override={
            "index_topk": topk_tokens,
            "attn_module_list_cfg": [{"topk_tokens": topk_tokens}],
        },
    )
    vllm_config.attention_config.hisparse_config = HiSparseConfig(
        device_buffer_size=2 * topk_tokens,
    )
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=1
    )
    model_config = vllm_config.model_config
    model_config.hf_text_config = SimpleNamespace(
        q_lora_rank=None,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        model_type="deepseek_v2",
    )
    model_config.dtype = dtype
    model_config.get_num_attention_heads = MethodType(
        lambda self, parallel_config: num_heads, model_config
    )
    model_config.get_num_kv_heads = MethodType(
        lambda self, parallel_config: 1, model_config
    )
    model_config.get_head_size = MethodType(lambda self: head_size, model_config)
    model_config.get_sliding_window = MethodType(lambda self: None, model_config)
    # create_hisparse_cache_handle sizes hot buffers per layer.
    model_config.get_num_layers = MethodType(
        lambda self, parallel_config: 1, model_config
    )

    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        vllm_config.cache_config.block_size,
        device,
        arange_block_indices=True,
    )
    # Prepopulate every position of every sequence so the forward needs no
    # KV-cache update (the row split itself is what is under test).
    kv_c_contexts = [
        torch.rand(s_len, kv_lora_rank, dtype=dtype, device=device)
        for s_len in batch_spec.seq_lens
    ]
    k_pe_contexts = [
        torch.rand(s_len, 1, qk_rope_head_dim, dtype=dtype, device=device)
        for s_len in batch_spec.seq_lens
    ]
    kv_cache = create_and_prepopulate_kv_cache(
        kv_c_contexts=kv_c_contexts,
        k_pe_contexts=k_pe_contexts,
        block_size=vllm_config.cache_config.block_size,
        head_size=head_size,
        dtype=dtype,
        device=device,
        num_blocks=vllm_config.cache_config.num_gpu_blocks,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=False,
        kv_cache_dtype="auto",
    )

    prefill_backend = get_mla_prefill_backend(vllm_config)(
        num_heads=num_heads,
        scale=1.0 / math.sqrt(head_size),
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        vllm_config=vllm_config,
    )
    vllm_config.compilation_config.static_forward_context["placeholder"] = (
        SimpleNamespace(prefill_backend=prefill_backend)
    )

    builder_cls = FlashMLASparseBackend.get_builder_cls()
    builder = builder_cls(kv_cache_spec, ["placeholder"], vllm_config, device)
    metadata = builder.build(
        common_prefix_len=0, common_attn_metadata=common_attn_metadata
    )
    assert isinstance(metadata.prefill, SparseMLAPrefillMetadata)

    # Per-token sparse indices bounded by each token's position, with unique
    # offsets and -1 padding (same construction as the decode parity test).
    positions: list[int] = []
    for i in range(batch_spec.batch_size):
        ctx_len = batch_spec.seq_lens[i] - batch_spec.query_lens[i]
        positions.extend(ctx_len + q_idx for q_idx in range(batch_spec.query_lens[i]))
    sparse_indices = torch.empty(
        total_tokens, topk_tokens, dtype=torch.int32, device=device
    )
    for tok_idx in range(total_tokens):
        max_valid_idx = positions[tok_idx]
        offset = tok_idx * 7  # Prime number for varied offsets
        num_valid = min(topk_tokens // 2, max_valid_idx + 1)
        valid_range = torch.arange(num_valid, device=device, dtype=torch.int32)
        sparse_indices[tok_idx] = torch.cat(
            [
                (valid_range + offset) % (max_valid_idx + 1),
                torch.full(
                    (topk_tokens - num_valid,), -1, device=device, dtype=torch.int32
                ),
            ]
        )

    q = torch.rand(total_tokens, num_heads, head_size, dtype=dtype, device=device)
    mock_indexer = SimpleNamespace(
        topk_indices_buffer=sparse_indices, topk_tokens=topk_tokens
    )

    impl_cls = FlashMLASparseBackend.get_impl_cls()
    with set_current_vllm_config(vllm_config):
        impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            scale=1.0 / math.sqrt(head_size),
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            q_lora_rank=None,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_head_dim=qk_nope_head_dim + qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_b_proj=None,
            indexer=mock_indexer,
        )
    assert isinstance(impl.index_group, HiSparseMLAIndexGroup)
    cache_handle = impl.index_group.cache(impl.index_group_index)
    blocks_per_request = cdiv(cache_handle.runtime.region_stride, block_size)
    num_hot_blocks = cache_handle.runtime.max_num_reqs * blocks_per_request
    hot_cache = torch.zeros(
        num_hot_blocks * block_size * head_size,
        dtype=dtype,
        device=device,
    ).view(torch.int8)
    cache_handle.runtime.bind_hot_cache(
        hot_cache,
        byte_offset=0,
        block_stride=block_size * head_size * dtype.itemsize,
        num_blocks=num_hot_blocks,
        block_size=block_size,
        block_table=torch.arange(num_hot_blocks, dtype=torch.int32, device=device).view(
            cache_handle.runtime.max_num_reqs, blocks_per_request
        ),
    )
    cache_handle.runtime.request_state_indices = torch.arange(
        cache_handle.runtime.max_num_reqs, dtype=torch.int32, device=device
    )
    impl.prepare_for_batch(metadata)

    # Device-resident reference: the whole batch converted against the full
    # block table and run as one kernel call over the GPU cache.
    ref_topk, ref_topk_length = triton_convert_req_index_to_global_index(
        metadata.req_id_per_token,
        metadata.block_table,
        sparse_indices,
        BLOCK_SIZE=metadata.block_size,
        NUM_TOPK_TOKENS=topk_tokens,
        return_valid_counts=True,
    )
    reference, _ = impl._bf16_flash_mla_kernel(q, kv_cache, ref_topk, ref_topk_length)

    # Host-resident pool with identical contents.
    kv_pool = kv_cache.squeeze(1).cpu().pin_memory()
    cache_handle.runtime.bind_source_cache(kv_pool)

    staging_calls = []
    original_gather = cache_handle.runtime.gather_prefill_cache

    def spy_gather(self, kv, plan, resident_cache=None):
        staged = original_gather(kv, plan, resident_cache)
        staging_calls.append((plan, staged.shape))
        return staged

    cache_handle.runtime.gather_prefill_cache = MethodType(
        spy_gather, cache_handle.runtime
    )

    backend_output, _ = impl._forward_bf16_kv(
        q, kv_pool, sparse_indices, metadata, q.shape[1]
    )
    torch.accelerator.synchronize()

    # Only the prefill rows' blocks were staged: the decode rows' 2048-token
    # contexts (32 blocks each) must stay off the staging gather.
    assert len(staging_calls) == 1
    plan, staged_shape = staging_calls[0]
    assert plan is metadata.prefill.host_staging_plan
    prefill_blocks = cdiv(batch_spec.seq_lens[-1], block_size)
    assert staged_shape[0] <= prefill_blocks + 1  # +1: block-0 tail padding

    torch.testing.assert_close(backend_output, reference, rtol=0.01, atol=0.01)


def test_hisparse_prefill_staging_remap():
    """Compacted staging references the same host rows as direct indexing."""
    block_size = 4
    block_table = torch.tensor(
        [[5, 2, -1, -1], [2, 7, 3, -1], [9, -1, -1, -1]], dtype=torch.int32
    )

    new_bt, row_ids = hisparse_prefill_staging_remap(block_table, block_size)

    assert new_bt.shape == block_table.shape
    assert row_ids.dtype == torch.int32 and new_bt.dtype == torch.int32
    n_unique = len({0, 2, 3, 5, 7, 9})  # -1 clamps to block 0
    flat_rows = row_ids.flatten()
    assert (flat_rows >= 0).sum() == n_unique * block_size
    for i in range(block_table.shape[0]):
        for j in range(block_table.shape[1]):
            orig = max(int(block_table[i, j]), 0)
            staged = int(new_bt[i, j])
            for k in range(block_size):
                assert int(flat_rows[staged * block_size + k]) == orig * block_size + k


def test_hisparse_prefill_staging_plan_masks_unused_blocks():
    block_table = torch.tensor([[5, 2, 7], [9, 3, 4]], dtype=torch.int32)
    plan = build_hisparse_prefill_staging_plan(
        block_table,
        seq_lens=torch.tensor([5, 8], dtype=torch.int32),
        block_size=4,
        staging_block_capacity=4,
    )

    assert (plan.block_table[:, 2] == 0).all()
    valid_rows = plan.row_ids[plan.row_ids >= 0]
    assert set((valid_rows[::4] // 4).tolist()) == {0, 2, 3, 5, 9}
    for row, used in enumerate((2, 2)):
        for column in range(used):
            staged_block = int(plan.block_table[row, column])
            host_block = int(block_table[row, column])
            assert int(valid_rows[staged_block * 4]) == host_block * 4


def test_hisparse_prefill_staging_plan_resolves_resident_sources():
    """Per-page resident hits become device sources; misses stay host DMAs."""
    block_size = 4
    resident_block_size = 2
    block_table = torch.tensor([[5, 2, 0], [9, 3, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([5, 8], dtype=torch.int32)
    plan = build_hisparse_prefill_staging_plan(
        block_table, seq_lens, block_size, staging_block_capacity=4
    )
    # Two resident pages per host block; 0 entries are null (not resident).
    resident_table = torch.tensor(
        [[11, 12, 0, 13, 0, 0], [21, 0, 22, 23, 0, 0]], dtype=torch.int32
    )

    plan.ensure_gpu_sources(resident_table, resident_block_size)

    assert plan.gpu_row_ids is not None
    unique_hosts = (plan.row_ids[0].view(-1, block_size)[:, 0] // block_size).tolist()
    gpu_rows = plan.gpu_row_ids[0].view(-1, block_size)
    reps = {5: (0, 0), 2: (0, 1), 9: (1, 0), 3: (1, 1)}
    for u, host_id in enumerate(unique_hosts):
        for t in range(block_size):
            if host_id < 0:
                assert int(gpu_rows[u, t]) == -1
                continue
            if host_id == 0:
                assert int(gpu_rows[u, t]) == -1
                continue
            row, col = reps[host_id]
            page = t // resident_block_size
            res_block = int(resident_table[row, col * 2 + page])
            expected = (
                res_block * resident_block_size + t % resident_block_size
                if res_block > 0
                else -1
            )
            assert int(gpu_rows[u, t]) == expected
    valid_rows = plan.row_ids >= 0
    torch.testing.assert_close(
        plan.miss_mask,
        ((plan.gpu_row_ids < 0) & valid_rows).int(),
        check_dtype=False,
    )

    plan.ensure_gpu_sources(torch.zeros_like(resident_table), resident_block_size)

    assert plan.gpu_row_ids is not None
    assert (plan.gpu_row_ids == -1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hisparse_gather_prefill_cache_prefers_resident_rows():
    """Staged rows come from the resident cache when a shadow page exists."""
    if not _has_hisparse_ops():
        pytest.skip("hisparse CUDA ops unavailable")
    device = torch.device("cuda")
    block_size, resident_block_size, row_width = 4, 2, 16
    block_table = torch.tensor([[5, 2, 0], [9, 3, 0]], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([5, 8], dtype=torch.int32, device=device)
    plan = build_hisparse_prefill_staging_plan(
        block_table, seq_lens, block_size, staging_block_capacity=4
    )
    resident_table = torch.tensor(
        [[11, 12, 0, 13, 0, 0], [21, 0, 22, 23, 0, 0]],
        dtype=torch.int32,
        device=device,
    )
    plan.ensure_gpu_sources(resident_table, resident_block_size)

    num_host_blocks, num_res_blocks = 10, 24
    host_cache = (
        torch.arange(num_host_blocks * block_size * row_width, dtype=torch.float32)
        .view(num_host_blocks, block_size, row_width)
        .pin_memory()
    )
    resident_cache = (
        (
            -torch.arange(
                num_res_blocks * resident_block_size * row_width, dtype=torch.float32
            )
            - 1.0
        )
        .view(num_res_blocks, resident_block_size, row_width)
        .to(device)
    )

    staged = HiSparseRuntime.gather_prefill_cache(
        None, host_cache, plan, resident_cache=resident_cache
    )

    staged_flat = staged.view(-1, row_width).cpu()
    host_flat = host_cache.view(-1, row_width)
    resident_flat = resident_cache.reshape(-1, row_width).cpu()
    gpu_rows = plan.gpu_row_ids[0].cpu()
    host_rows = plan.row_ids[0].cpu()
    for i in range(staged_flat.shape[0]):
        if int(host_rows[i]) < 0:
            assert int(plan.miss_mask[0, i]) == 0
            continue
        expected = (
            resident_flat[int(gpu_rows[i])]
            if int(gpu_rows[i]) >= 0
            else host_flat[int(host_rows[i])]
        )
        torch.testing.assert_close(staged_flat[i], expected)


def test_hisparse_fp8_decode_resolves_steps_then_runs_batched_attention(monkeypatch):
    """FP8 decode must use active dimensions when common metadata is padded."""
    device = torch.device("cpu")
    num_decodes = 2
    query_len = 3
    num_tokens = num_decodes * query_len
    q = torch.randn(num_tokens, 2, 4, device=device)
    topk = (
        torch.arange(num_tokens, dtype=torch.int32, device=device)
        .view(-1, 1)
        .expand(-1, 4)
    )
    steps: list[int] = []
    kernel_shapes: list[torch.Size] = []

    def swap_in(req_ids, *, logical_topk_indices, **kwargs):  # noqa: ARG001
        steps.append(int(logical_topk_indices[0, 0]))
        output = kwargs["attention_indices_out"]
        if output is not None:
            output.copy_(logical_topk_indices + 10)
        return topk[:num_decodes]

    def run_kernel(self, *, q, **kwargs):  # noqa: ARG001
        kernel_shapes.append(q.shape)
        return q[..., :1], None

    runtime = SimpleNamespace(
        hot=SimpleNamespace(attention_cache=torch.empty(1, device=device))
    )
    leader_cache = SimpleNamespace(
        runtime=runtime,
        source_block_table=torch.empty(
            num_decodes, 1, dtype=torch.int32, device=device
        ),
        swap_in=MagicMock(side_effect=swap_in),
    )
    follower_cache = SimpleNamespace(
        runtime=runtime,
        source_block_table=torch.empty(
            num_decodes, 1, dtype=torch.int32, device=device
        ),
        swap_in=MagicMock(side_effect=swap_in),
    )
    index_group = object.__new__(HiSparseMLAIndexGroup)
    index_group.caches = [leader_cache, follower_cache]
    index_group.physical_topk_indices = torch.empty(
        (num_tokens + 1, 4), dtype=torch.int32, device=device
    )
    index_group.valid_topk_counts = torch.empty(
        num_tokens + 1, dtype=torch.int32, device=device
    )
    index_group.request_ids = torch.arange(
        num_decodes, dtype=torch.int32, device=device
    )
    impl = SimpleNamespace(
        kv_lora_rank=1,
        index_group=index_group,
        index_group_index=0,
    )
    impl._fp8_flash_mla_kernel = MethodType(run_kernel, impl)
    metadata = SimpleNamespace(
        num_decodes=num_tokens,
        num_decode_tokens=num_tokens,
        decode_max_query_len=1,
        query_start_loc=torch.arange(
            0,
            num_tokens + 1,
            query_len,
            dtype=torch.int32,
            device=device,
        ),
        block_table=torch.empty(num_decodes, 1, dtype=torch.int32, device=device),
        block_size=64,
    )

    output = FlashMLASparseImpl._host_backed_fp8_decode(
        impl,
        q,
        topk,
        metadata,
        SimpleNamespace(),
        num_decodes,
        query_len,
    )

    assert steps == [0, 1, 2]
    assert kernel_shapes == [(num_decodes, query_len, 2, 4)]
    assert output.shape == (num_tokens, 2, 1)
    torch.testing.assert_close(
        index_group.physical_topk_indices[:num_tokens], topk + 10
    )

    follower_indices = index_group.convert_decode_logical_to_physical_topk(
        1,
        topk,
        metadata,
        return_valid_counts=False,
        num_decodes=num_decodes,
        decode_query_len=query_len,
    )
    assert steps == [0, 1, 2, 0, 1, 2]
    torch.testing.assert_close(follower_indices, topk + 10)


def test_hisparse_single_token_decode_uses_canonical_request_rows():
    """Zero-filled graph padding must not share request 0's residency state."""
    num_decodes = 4
    topk = torch.zeros((num_decodes, 2), dtype=torch.int32)
    cache = SimpleNamespace(
        source_block_table=torch.zeros((num_decodes, 1), dtype=torch.int32),
        swap_in=MagicMock(return_value=topk),
    )
    index_group = object.__new__(HiSparseMLAIndexGroup)
    index_group.caches = [cache]
    index_group.physical_topk_indices = torch.empty_like(topk)
    index_group.request_ids = torch.arange(num_decodes, dtype=torch.int32)
    metadata = SimpleNamespace(
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        decode_max_query_len=1,
        req_id_per_token=torch.zeros(num_decodes, dtype=torch.int32),
        block_size=64,
    )

    index_group.convert_decode_logical_to_physical_topk(
        0,
        topk,
        metadata,
        return_valid_counts=False,
    )

    request_rows = cache.swap_in.call_args.args[0]
    torch.testing.assert_close(request_rows, index_group.request_ids)


def test_flashinfer_hisparse_decode_runs_batched_attention():
    device = torch.device("cpu")
    num_tokens = 6
    q = torch.randn(num_tokens, 2, 4, device=device)
    topk = torch.zeros(num_tokens, 4, dtype=torch.int32, device=device)
    physical_topk = topk + 10
    valid_counts = torch.full((num_tokens,), 4, dtype=torch.int32, device=device)
    kernel_shapes: list[torch.Size] = []

    def convert_decode(self, *args, **kwargs):  # noqa: ARG001
        return physical_topk, valid_counts

    def prepare_kernel(self, *args, **kwargs):  # noqa: ARG001
        pass

    def run_kernel(self, q, cache, indices, counts):  # noqa: ARG001
        kernel_shapes.append(q.shape)
        return q[..., :1], None

    cache_handle = SimpleNamespace(
        runtime=SimpleNamespace(
            hot=SimpleNamespace(attention_cache=torch.empty(1, device=device))
        ),
    )
    index_group = object.__new__(HiSparseMLAIndexGroup)
    index_group.caches = [cache_handle]
    index_group.convert_decode_logical_to_physical_topk = MethodType(
        convert_decode, index_group
    )
    impl = object.__new__(FlashInferMLASparseImpl)
    impl.topk_indices_buffer = topk
    impl.index_group = index_group
    impl.index_group_index = 0
    impl._prepare_mqa_kernel = MethodType(prepare_kernel, impl)
    impl._run_mqa_kernel = MethodType(run_kernel, impl)
    metadata = SimpleNamespace(num_decode_tokens=num_tokens)

    output, lse = FlashInferMLASparseImpl.forward_mqa(
        impl,
        q,
        torch.empty(1, device=device),
        metadata,
        SimpleNamespace(),
    )

    assert kernel_shapes == [q.shape]
    assert output.shape == (num_tokens, 2, 1)
    assert lse is None


def test_flashattn_hisparse_decode_uses_index_group():
    num_tokens = 4
    q_nope = torch.empty(num_tokens, 2, 3, device=DEVICE_TYPE)
    q_rope = torch.empty(num_tokens, 2, 1, device=DEVICE_TYPE)
    topk = torch.zeros(num_tokens, 4, dtype=torch.int32, device=DEVICE_TYPE)
    physical = topk.clone()
    counts = torch.full((num_tokens,), 4, dtype=torch.int32, device=DEVICE_TYPE)
    index_group = object.__new__(HiSparseMLAIndexGroup)
    index_group.convert_decode_logical_to_physical_topk = MagicMock(
        return_value=(physical, counts)
    )
    index_group.physical_kv_cache = MagicMock(
        return_value=torch.empty(1, device=DEVICE_TYPE)
    )
    impl = object.__new__(FlashAttnMLASparseImpl)
    impl.topk_indices_buffer = topk
    impl.index_group = index_group
    impl.index_group_index = 0
    impl._run_mqa_kernel = MagicMock(return_value=q_nope[..., :1])
    metadata = SimpleNamespace(num_decode_tokens=num_tokens, block_size=64)

    output, lse = FlashAttnMLASparseImpl.forward_mqa(
        impl,
        (q_nope, q_rope),
        torch.empty(1, device=DEVICE_TYPE),
        metadata,
        SimpleNamespace(),
    )

    assert output.shape == (num_tokens, 2, 1)
    assert lse is None
    index_group.convert_decode_logical_to_physical_topk.assert_called_once()
    impl._run_mqa_kernel.assert_called_once()


def test_flashinfer_sm120_hisparse_decode_uses_index_group():
    num_tokens = 4
    q = torch.empty(num_tokens, 2, 4, device=DEVICE_TYPE)
    topk = torch.zeros(num_tokens, 4, dtype=torch.int32, device=DEVICE_TYPE)
    physical = topk.clone()
    index_group = object.__new__(HiSparseMLAIndexGroup)
    index_group.convert_decode_logical_to_physical_topk = MagicMock(
        return_value=physical
    )
    index_group.physical_kv_cache = MagicMock(
        return_value=torch.empty(1, device=DEVICE_TYPE)
    )
    impl = object.__new__(FlashInferMLASparseSM120Impl)
    impl.topk_indices_buffer = topk
    impl.index_group = index_group
    impl.index_group_index = 0
    impl._run_mqa_kernel = MagicMock(return_value=q[..., :1])
    metadata = SimpleNamespace(
        num_decode_tokens=num_tokens,
        topk_tokens=topk.shape[1],
    )

    output, lse = FlashInferMLASparseSM120Impl.forward_mqa(
        impl,
        q,
        torch.empty(1, device=DEVICE_TYPE),
        metadata,
        SimpleNamespace(),
    )

    assert output.shape == (num_tokens, 2, 1)
    assert lse is None
    index_group.convert_decode_logical_to_physical_topk.assert_called_once()
    impl._run_mqa_kernel.assert_called_once()


def test_hisparse_decode_uses_group_swap_in_when_context_is_not_resident():
    resident_cache = torch.empty((2, 2, 4), dtype=torch.float32)
    expected_indices = torch.tensor([[6, 7]], dtype=torch.int32)
    expected_counts = torch.tensor([2], dtype=torch.int32)
    cache_handle = SimpleNamespace(
        decode_batch=True,
        all_context_pages_resident=False,
        runtime=SimpleNamespace(
            hot=SimpleNamespace(attention_cache=resident_cache),
        ),
        swap_in=MagicMock(return_value=(expected_indices, expected_counts)),
    )
    index_group = object.__new__(HiSparseMLAIndexGroup)
    index_group.caches = [cache_handle]
    index_group.physical_topk_indices = torch.empty((1, 2), dtype=torch.int32)
    topk = torch.tensor([[0, 1]], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_decode_tokens=1,
        num_actual_tokens=1,
        req_id_per_token=torch.tensor([0], dtype=torch.int32),
        block_table=torch.tensor([[1, 0]], dtype=torch.int32),
        block_size=2,
    )

    indices, counts = index_group.convert_logical_to_physical_topk(
        0,
        topk,
        metadata,
        block_stride_rows=None,
        return_valid_counts=True,
    )

    assert indices is expected_indices
    assert counts is expected_counts
    cache_handle.swap_in.assert_called_once()
    args = cache_handle.swap_in.call_args
    torch.testing.assert_close(args.args[0], metadata.req_id_per_token)
    assert args.kwargs["block_table"] is metadata.block_table
    torch.testing.assert_close(args.kwargs["logical_topk_indices"], topk)
    assert args.kwargs["block_size"] == metadata.block_size
    assert args.kwargs["return_valid_counts"] is True


def test_hisparse_resident_prefill_uses_attention_block_stride():
    expected = torch.tensor([[19]], dtype=torch.int32)
    cache_handle = SimpleNamespace(
        all_context_pages_resident=True,
        view=SimpleNamespace(block_size=64, attention_block_stride=832),
        block_table=torch.tensor([[3]], dtype=torch.int32),
    )
    index_group = object.__new__(HiSparseMLAIndexGroup)
    index_group.caches = [cache_handle]
    index_group.physical_topk_indices = torch.empty((1, 1), dtype=torch.int32)
    index_group._convert_once = MagicMock(return_value=expected)
    topk = torch.zeros((2, 1), dtype=torch.int32)
    metadata = SimpleNamespace(
        req_id_per_token=torch.zeros(2, dtype=torch.int32),
    )

    result = index_group.convert_logical_to_physical_topk(
        0,
        topk,
        metadata,
        block_stride_rows=None,
        return_valid_counts=False,
    )

    assert result is expected
    assert index_group._convert_once.call_args.kwargs["block_stride_rows"] == 832


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hisparse_decode_routes_resident_mapping_through_swap_in(monkeypatch):
    device = torch.device("cuda")
    expected_indices = torch.tensor([[6, 7]], dtype=torch.int32, device=device)
    expected_counts = torch.tensor([2], dtype=torch.int32, device=device)
    convert = MagicMock()
    monkeypatch.setattr(
        index_group_module,
        "triton_convert_req_index_to_global_index",
        convert,
    )
    cache_handles = [
        SimpleNamespace(
            all_context_pages_resident=True,
            swap_in=MagicMock(return_value=(expected_indices, expected_counts)),
        )
        for _ in range(2)
    ]
    index_group = object.__new__(HiSparseMLAIndexGroup)
    index_group.caches = cache_handles
    index_group.physical_topk_indices = torch.empty(
        (1, 2), dtype=torch.int32, device=device
    )
    topk = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    metadata = SimpleNamespace(
        num_decode_tokens=1,
        req_id_per_token=torch.tensor([0], dtype=torch.int32, device=device),
        block_table=torch.tensor([[3, 0]], dtype=torch.int32, device=device),
        block_size=2,
    )

    leader_result = index_group.convert_logical_to_physical_topk(
        0,
        topk,
        metadata,
        block_stride_rows=None,
        return_valid_counts=True,
    )
    follower_result = index_group.convert_logical_to_physical_topk(
        1,
        topk,
        metadata,
        block_stride_rows=None,
        return_valid_counts=True,
    )
    torch.accelerator.synchronize()

    for indices, counts in (leader_result, follower_result):
        torch.testing.assert_close(indices, expected_indices)
        torch.testing.assert_close(counts, expected_counts)
    convert.assert_not_called()
    for cache_handle in cache_handles:
        cache_handle.swap_in.assert_called_once()
        args = cache_handle.swap_in.call_args
        torch.testing.assert_close(args.args[0], metadata.req_id_per_token)
        assert args.kwargs["block_table"] is metadata.block_table
        assert args.kwargs["logical_topk_indices"] is topk


def test_hisparse_physical_cache_uses_shared_hot_view():
    hot_attention_cache = torch.empty(8, 4, 8)
    cache_handle = SimpleNamespace(
        runtime=SimpleNamespace(
            hot=SimpleNamespace(attention_cache=hot_attention_cache)
        ),
    )
    index_group = object.__new__(HiSparseMLAIndexGroup)
    index_group.caches = [cache_handle]

    selected = index_group.physical_kv_cache(0)

    assert selected is hot_attention_cache


def test_hisparse_shared_sparse_builder_routes_multi_token_chunks_to_prefill():
    """Default sparse backends must route multi-token chunks to prefill."""
    builder = object.__new__(FlashInferMLASparseMetadataBuilder)
    builder.vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=object()),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=4,
            parallel_drafting=False,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )

    builder._init_reorder_batch_threshold(
        1024,
        supports_spec_as_decode=True,
        supports_dcp_with_varlen=True,
    )

    assert builder.reorder_batch_threshold == 1


def test_hisparse_flashmla_reorders_full_speculative_window_as_decode():
    builder = object.__new__(FlashMLASparseMetadataBuilder)
    builder.vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=object()),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=3,
            parallel_drafting=False,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )

    builder._init_reorder_batch_threshold(
        256,
        supports_spec_as_decode=True,
        supports_dcp_with_varlen=True,
    )

    assert builder.reorder_batch_threshold == 4


def test_hisparse_mixed_single_token_batch_is_not_decode_only():
    cache = SimpleNamespace(
        runtime=SimpleNamespace(
            begin_forward=lambda: None,
            eager_host_mirror=True,
        ),
    )
    metadata = SimpleNamespace(
        num_decode_tokens=1,
        num_actual_tokens=2,
        max_query_len=1,
        num_reqs=2,
        req_id_per_token=torch.arange(2),
    )

    HiSparseCacheHandle._prepare_for_batch(cache, metadata)

    assert not cache.decode_batch


def test_flashmla_cache_dtype_aliases_use_ds_layout():
    # kv-cache dtype aliases are canonicalized to fp8_ds_mla before the layer
    # stores kv_cache_dtype, so they cannot bypass the gate.
    for alias in ("fp8", "fp8_e4m3"):
        assert (
            _canonicalize_sparse_mla_kv_cache_dtype(FlashMLASparseBackend, alias)
            == "fp8_ds_mla"
        )


def test_flashmla_fp8_metadata_reuses_common_batch_split():
    builder = SimpleNamespace(
        device=torch.device(DEVICE_TYPE),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=8)),
    )
    common_metadata = SimpleNamespace(
        num_actual_tokens=1,
        seq_lens_cpu_upper_bound=torch.tensor([1]),
        seq_lens=torch.tensor([1], device=DEVICE_TYPE),
        query_start_loc_cpu=torch.tensor([0, 1]),
        block_table_tensor=torch.zeros(1, 1, dtype=torch.int32, device=DEVICE_TYPE),
    )
    metadata = FlashMLASparseMetadata(
        num_reqs=1,
        max_query_len=1,
        max_seq_len=1,
        num_actual_tokens=1,
        query_start_loc=torch.tensor([0, 1], device=DEVICE_TYPE),
        slot_mapping=torch.tensor([0], device=DEVICE_TYPE),
        block_table=torch.zeros(1, 1, dtype=torch.int32, device=DEVICE_TYPE),
        req_id_per_token=torch.zeros(1, dtype=torch.int32, device=DEVICE_TYPE),
        num_decodes=0,
        num_prefills=1,
        num_decode_tokens=0,
    )

    fp8_metadata = FlashMLASparseMetadataBuilder._build_fp8_separate_prefill_decode(
        builder, common_metadata, metadata
    )

    assert fp8_metadata.num_decodes == 0
    assert fp8_metadata.num_prefills == 1
    assert fp8_metadata.num_decode_tokens == 0
    assert fp8_metadata.num_prefill_tokens == 1


def test_flashmla_common_metadata_requires_uniform_decodes():
    common_metadata = SimpleNamespace(
        max_query_len=3,
        num_reqs=3,
        num_actual_tokens=6,
        query_start_loc_cpu=torch.tensor([0, 1, 3, 6]),
        is_prefilling=None,
    )

    split = split_decodes_and_prefills(
        common_metadata,
        decode_threshold=128,
        require_uniform=FlashMLASparseMetadataBuilder.require_uniform_decodes,
    )

    assert split == (1, 2, 1, 5)


def test_flashmla_fp8_metadata_excludes_zero_token_decode_padding(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.flashmla_sparse.get_mla_metadata",
        lambda: (object(), None),
    )
    builder = SimpleNamespace(
        device=torch.device(DEVICE_TYPE),
        dummy_block_table=torch.zeros(7, 1, device=DEVICE_TYPE),
        max_model_len_tensor=torch.zeros(7, device=DEVICE_TYPE),
    )
    query_start_loc_cpu = torch.tensor([0, 110, 220, 330, 440, 550, 660, 660])
    common_metadata = SimpleNamespace(
        num_actual_tokens=660,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=torch.arange(7, device=DEVICE_TYPE),
    )
    metadata = FlashMLASparseMetadata(
        num_reqs=7,
        max_query_len=110,
        max_seq_len=110,
        num_actual_tokens=660,
        query_start_loc=query_start_loc_cpu.to(DEVICE_TYPE),
        slot_mapping=torch.arange(660, device=DEVICE_TYPE),
        block_table=torch.zeros(7, 1, dtype=torch.int32, device=DEVICE_TYPE),
        req_id_per_token=torch.zeros(660, dtype=torch.int32, device=DEVICE_TYPE),
        num_decodes=7,
        num_prefills=0,
        num_decode_tokens=660,
    )

    fp8_metadata = FlashMLASparseMetadataBuilder._build_fp8_separate_prefill_decode(
        builder, common_metadata, metadata
    )

    assert fp8_metadata.num_decodes == 6
    assert fp8_metadata.num_decode_tokens == 660
    assert fp8_metadata.decode is not None
    assert fp8_metadata.decode.decode_query_len == 110
    torch.testing.assert_close(
        fp8_metadata.decode.seq_lens, torch.arange(6, device=DEVICE_TYPE)
    )


@pytest.mark.parametrize("use_mixed_batch", [False, True])
def test_flashmla_fp8_paths_accept_decode_subset(monkeypatch, use_mixed_batch: bool):
    num_decode_tokens = 2
    num_batch_tokens = 5
    q = torch.empty(num_decode_tokens, 2, 3, device=DEVICE_TYPE)
    topk_indices = torch.empty(num_decode_tokens, 4, device=DEVICE_TYPE)
    kernel_q_shapes = []

    def convert_indices(*args, **kwargs):  # noqa: ARG001
        assert not kwargs.get("HAS_PREFILL_WORKSPACE", False)
        if not kwargs.get("return_valid_counts", False):
            return topk_indices
        valid_counts = torch.full(
            (num_decode_tokens,), 4, dtype=torch.int32, device=DEVICE_TYPE
        )
        return topk_indices, valid_counts

    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.flashmla_sparse."
        "triton_convert_req_index_to_global_index",
        convert_indices,
    )

    def run_kernel(**kwargs):
        kernel_q_shapes.append(kwargs["q"].shape)
        return kwargs["q"][..., :1], None

    if use_mixed_batch:
        fp8_metadata = FlashMLASparseMetadata.FP8KernelMetadata(
            scheduler_metadata=object(),  # type: ignore[arg-type]
            dummy_block_table=torch.empty(1, 1, dtype=torch.int32, device=DEVICE_TYPE),
            cache_lens=torch.empty(1, dtype=torch.int32, device=DEVICE_TYPE),
        )
    else:
        FP8Meta = FlashMLASparseMetadata.FP8SeparatePrefillDecode
        fp8_metadata = FP8Meta(
            num_decodes=1,
            num_prefills=1,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_batch_tokens - num_decode_tokens,
            decode=FP8Meta.Decode(
                seq_lens=torch.empty(1, dtype=torch.int32, device=DEVICE_TYPE),
                kernel_metadata=object(),  # type: ignore[arg-type]
                decode_query_len=num_decode_tokens,
            ),
            prefill=FP8Meta.Prefill(
                request_ids=torch.empty(
                    num_batch_tokens, dtype=torch.int32, device=DEVICE_TYPE
                ),
                workspace_starts=torch.empty(1, dtype=torch.int32, device=DEVICE_TYPE),
                chunks=[],
            ),
        )
    metadata = SimpleNamespace(
        fp8_extra_metadata=fp8_metadata,
        fp8_use_mixed_batch=use_mixed_batch,
        num_actual_tokens=num_batch_tokens,
        num_decode_tokens=num_decode_tokens,
        req_id_per_token=torch.empty(
            num_batch_tokens, dtype=torch.int32, device=DEVICE_TYPE
        ),
        block_table=torch.empty(1, 1, dtype=torch.int32, device=DEVICE_TYPE),
        block_size=64,
    )
    impl = SimpleNamespace(
        kv_cache_dtype="fp8_ds_mla",
        topk_indices_buffer=topk_indices,
        num_heads=2,
        kv_lora_rank=1,
        index_group=None,
        index_group_index=0,
        dcp_world_size=1,
        need_to_return_lse_for_decode=False,
        _fp8_flash_mla_kernel=run_kernel,
        _convert_logical_to_physical_topk=(
            lambda indices, metadata, **kwargs: convert_indices(
                metadata.req_id_per_token[: indices.shape[0]],
                metadata.block_table,
                indices,
                return_valid_counts=kwargs["return_valid_counts"],
            )
        ),
    )
    impl._forward_fp8_kv_mixed_batch = MethodType(
        FlashMLASparseImpl._forward_fp8_kv_mixed_batch, impl
    )
    impl._forward_fp8_kv_separate_prefill_decode = MethodType(
        FlashMLASparseImpl._forward_fp8_kv_separate_prefill_decode, impl
    )

    output, lse = FlashMLASparseImpl.forward_mqa(
        impl,
        q,
        torch.empty(0, device=DEVICE_TYPE),
        metadata,
        None,
    )

    assert kernel_q_shapes == [(1, num_decode_tokens, 2, 3)]
    assert output.shape == (num_decode_tokens, 2, 1)
    assert lse is None


def _build_sparse_dcp_vllm_config(
    local_heads: int,
    dcp_world_size: int,
    comm_backend: str = "ag_rs",
):
    """Minimal sparse-MLA VllmConfig for the FlashMLASparse DCP head-envelope
    guard. TP is simulated by mocking ``get_num_attention_heads`` to return the
    per-rank head count, as the decode-correctness test above does.
    """
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    head_size = kv_lora_rank + qk_rope_head_dim
    topk_tokens = 128

    vllm_config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-V2-Lite-Chat",
        tensor_parallel_size=1,
        max_model_len=4096,
        block_size=64,
        hf_config_override={
            "index_topk": topk_tokens,
            "attn_module_list_cfg": [{"topk_tokens": topk_tokens}],
        },
    )
    model_config = vllm_config.model_config
    model_config.dtype = torch.bfloat16
    model_config.hf_text_config = SimpleNamespace(
        q_lora_rank=None,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        model_type="deepseek_v2",
    )
    model_config.get_num_attention_heads = MethodType(
        lambda self, parallel_config: local_heads, model_config
    )
    model_config.get_num_kv_heads = MethodType(
        lambda self, parallel_config: 1, model_config
    )
    model_config.get_head_size = MethodType(lambda self: head_size, model_config)
    model_config.get_sliding_window = MethodType(lambda self: None, model_config)

    vllm_config.cache_config.cache_dtype = "fp8_ds_mla"
    vllm_config.parallel_config.decode_context_parallel_size = dcp_world_size
    vllm_config.parallel_config.dcp_comm_backend = comm_backend
    # The base builder clones the layer's dense-MHA prefill backend from
    # static_forward_context; the guard tests never run prefill.
    vllm_config.compilation_config.static_forward_context["placeholder"] = (
        SimpleNamespace(prefill_backend=None)
    )
    return vllm_config


@pytest.mark.skipif(
    torch.cuda.get_device_capability() < (9, 0),
    reason="FlashMLASparseBackend requires CUDA 9.0 or higher",
)
@pytest.mark.parametrize(
    "local_heads,dcp_world_size,should_raise",
    [
        (16, 8, True),
        (24, 4, True),
        (16, 4, False),
        (16, 1, False),
    ],
)
def test_fp8_dcp_head_envelope_guard(local_heads, dcp_world_size, should_raise):
    """The fp8 decode envelope (head padding + tile-scheduler metadata) is
    sized from the local head count while the kernel runs on the DCP-gathered
    heads, so the builder must reject configs where the two pad differently.
    """
    device = torch.device(DEVICE_TYPE)
    vllm_config = _build_sparse_dcp_vllm_config(local_heads, dcp_world_size)
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    builder_cls = FlashMLASparseBackend.get_builder_cls()

    if should_raise:
        with pytest.raises(NotImplementedError, match="envelope"):
            builder_cls(kv_cache_spec, ["placeholder"], vllm_config, device)
    else:
        builder = builder_cls(kv_cache_spec, ["placeholder"], vllm_config, device)
        gathered_heads = local_heads * dcp_world_size
        local_pad = 64 if local_heads <= 64 else 128
        gathered_pad = 64 if gathered_heads <= 64 else 128
        assert builder.fp8_decode_padded_heads == local_pad
        assert local_pad == gathered_pad


def test_fp8_mixed_batch_dcp_neutralizes_empty_rows(monkeypatch):
    """A decode row whose top-k shard holds no local candidates (all -1) has
    undefined kernel out/lse; it must come back as (0, -inf), the identity of
    the cross-rank LSE merge, or a NaN would survive the merge even at zero
    weight (0 * NaN = NaN)."""
    num_tokens, num_heads, head_dim = 3, 2, 3
    q = torch.empty(num_tokens, num_heads, head_dim, device=DEVICE_TYPE)
    local_indices = torch.tensor(
        [[0, 1, -1, -1], [-1, -1, -1, -1], [2, -1, 3, -1]],
        dtype=torch.int32,
        device=DEVICE_TYPE,
    )

    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.flashmla_sparse."
        "triton_filter_and_convert_dcp_index",
        lambda *args, **kwargs: local_indices,
    )

    def run_kernel(**kwargs):
        out = torch.full(
            (1, num_tokens, num_heads, 1), float("nan"), device=DEVICE_TYPE
        )
        lse = torch.full((1, num_heads, num_tokens), float("nan"), device=DEVICE_TYPE)
        for token_id in (0, 2):  # rows with local candidates get real values
            out[0, token_id] = float(token_id + 1)
            lse[0, :, token_id] = float(token_id + 1)
        return out, lse

    metadata = SimpleNamespace(
        fp8_extra_metadata=FlashMLASparseMetadata.FP8KernelMetadata(
            scheduler_metadata=object(),  # type: ignore[arg-type]
            dummy_block_table=torch.empty(1, 1, dtype=torch.int32, device=DEVICE_TYPE),
            cache_lens=torch.empty(1, dtype=torch.int32, device=DEVICE_TYPE),
        ),
        req_id_per_token=torch.empty(num_tokens, dtype=torch.int32, device=DEVICE_TYPE),
        block_table=torch.empty(1, 1, dtype=torch.int32, device=DEVICE_TYPE),
        block_size=64,
        cp_kv_cache_interleave_size=1,
    )
    impl = SimpleNamespace(
        index_group=None,
        index_group_index=0,
        dcp_world_size=2,
        dcp_rank=0,
        need_to_return_lse_for_decode=True,
        _fp8_flash_mla_kernel=run_kernel,
    )

    out, lse = FlashMLASparseImpl._forward_fp8_kv_mixed_batch(
        impl, q, torch.empty(0, device=DEVICE_TYPE), local_indices, metadata
    )

    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert torch.isneginf(lse[1]).all()
    for token_id in (0, 2):
        assert torch.equal(out[token_id], torch.full_like(out[token_id], token_id + 1))
        assert torch.equal(lse[token_id], torch.full_like(lse[token_id], token_id + 1))
    assert out.is_contiguous()
    assert not out.isnan().any()
    assert not lse.isnan().any()


def test_hisparse_prefill_reuses_builder_staging_plan():
    """Every layer must reuse the batch plan instead of synchronizing to dedupe."""
    plan = SimpleNamespace(
        block_table=torch.tensor([[0]], dtype=torch.int32),
        ensure_gpu_sources=MagicMock(),
    )
    staged = torch.empty((1, 1, 8))
    calls = []

    def gather(kv_cache, staging_plan, resident_cache=None):
        calls.append((kv_cache, staging_plan, resident_cache))
        return staged

    resident_cache = torch.empty((1, 1, 8))
    resident_block_table = torch.tensor([[1]], dtype=torch.int32)
    cache = SimpleNamespace(
        runtime=SimpleNamespace(
            gather_prefill_cache=gather,
        ),
        view=SimpleNamespace(cache=resident_cache, block_size=1),
        block_table=resident_block_table,
    )
    index_group = object.__new__(HiSparseMLAIndexGroup)
    index_group.caches = [cache]
    source = torch.empty((1, 1, 8))
    metadata = SimpleNamespace(
        num_decodes=0,
        num_decode_tokens=0,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        prefill=SimpleNamespace(host_staging_plan=plan),
        req_id_per_token=torch.tensor([0], dtype=torch.int32),
    )

    result, block_table, request_ids = index_group.stage_prefill_rows(
        0, source, metadata
    )

    assert result is staged
    assert block_table is plan.block_table
    torch.testing.assert_close(request_ids, metadata.req_id_per_token)
    plan.ensure_gpu_sources.assert_called_once()
    args = plan.ensure_gpu_sources.call_args.args
    torch.testing.assert_close(args[0], resident_block_table)
    assert args[1] == 1
    assert calls == [(source, plan, resident_cache)]


def test_hisparse_fp8_prefill_gather_uses_dedicated_stream(monkeypatch):
    compute_stream = object()
    prefill_stream = MagicMock()
    gather = MagicMock()
    monkeypatch.setattr(index_group_module, "current_stream", lambda: compute_stream)
    monkeypatch.setattr(
        index_group_module.ops, "cp_gather_and_upconvert_fp8_kv_cache", gather
    )

    resident_cache = torch.empty((2, 4, 656), dtype=torch.uint8)
    ready = MagicMock()
    cache = SimpleNamespace(
        view=SimpleNamespace(cache=resident_cache, block_size=4),
        block_table=torch.tensor([[2, 3]], dtype=torch.int32),
    )
    index_group = object.__new__(HiSparseMLAIndexGroup)
    index_group.caches = [cache]
    index_group.prefill_stream = prefill_stream
    index_group.prefill_ready_events = [ready]
    source = torch.empty((3, 4, 656), dtype=torch.uint8)
    dst = torch.empty((8, 576), dtype=torch.bfloat16)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    workspace_starts = torch.tensor([0], dtype=torch.int32)
    plan = SimpleNamespace(
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        row_ids=torch.arange(8, dtype=torch.int32).view(1, -1),
        gpu_row_ids=torch.full((1, 8), -1, dtype=torch.int32),
        ensure_gpu_sources=MagicMock(),
    )
    metadata = SimpleNamespace(
        prefill=SimpleNamespace(host_staging_plan=plan),
        num_decodes=0,
    )

    result = index_group.gather_fp8_prefill(
        0,
        source,
        dst,
        block_table,
        workspace_starts,
        1,
        metadata,
        0,
    )

    prefill_stream.wait_stream.assert_called_once_with(compute_stream)
    ready.record.assert_called_once_with(prefill_stream)
    assert result is ready
    gather.assert_called_once()
    args = gather.call_args
    assert args.args[0] is resident_cache
    assert args.args[1] is dst
    torch.testing.assert_close(args.args[2], plan.block_table)
    assert args.args[3] is workspace_starts
    assert args.args[4] == 1
    assert args.kwargs["host_cache"].data_ptr() == source.data_ptr()
    assert args.kwargs["host_row_ids"] is plan.row_ids
    assert args.kwargs["device_row_ids"] is plan.gpu_row_ids
    plan.ensure_gpu_sources.assert_called_once()
    ensure_args = plan.ensure_gpu_sources.call_args.args
    torch.testing.assert_close(ensure_args[0], cache.block_table)
    assert ensure_args[1] == 4
