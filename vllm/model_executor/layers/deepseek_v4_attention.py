# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DeepseekV4 MLA Attention Layer
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DeepseekV2Config, DeepseekV3Config

from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
)
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.ops.deepseek_v4_ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
    dequantize_combined_sparse_mla_decode_kv,
    fused_indexer_q_rope_quant,
    fused_inv_rope_fp8_quant,
    fused_q_kv_rmsnorm,
)
from vllm.v1.attention.ops.deepseek_v4_ops.fp8_einsum import (
    deepseek_v4_sm12_fp8_einsum,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )

from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.deepseek_compressor import DeepseekCompressor
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.input_quant_fp8 import (
    QuantFP8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
)
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    DeepseekV4FlashMLASparseBackend,
    FlashMLASparseBackend,
    FlashMLASparseMetadata,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
    get_max_prefill_buffer_size,
)
from vllm.v1.attention.backends.mla.sparse_mla_env import (
    disable_sparse_mla_reference_cudagraphs_if_enabled,
    is_sparse_mla_attention_dump_enabled,
    is_sparse_mla_reference_attention_enabled,
    sparse_mla_attention_dump_path,
    sparse_mla_matmul_decode_enabled,
    sparse_mla_reference_query_chunk_size,
    sparse_mla_reference_topk_chunk_size,
)
from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead,
    accumulate_fp8ds_paged_sparse_mla_attention_chunk_multihead,
    accumulate_indexed_sparse_mla_attention_chunk,
    build_combined_sparse_mla_decode_valid_mask,
    finish_sparse_mla_attention_with_sink,
    finish_two_sparse_mla_attention_states_with_sink,
    fp8ds_global_paged_sparse_mla_attention_with_sink_multihead,
    fp8ds_paged_sparse_mla_attention_with_sink_multihead,
    matmul_sparse_mla_attention_with_sink,
    sparse_mla_decode_head_block_size,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)


def _tensor_summary(tensor: torch.Tensor | None) -> dict[str, object] | None:
    if tensor is None:
        return None
    return {
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": str(tensor.dtype),
        "stride": [int(stride) for stride in tensor.stride()],
        "device": str(tensor.device),
        "is_contiguous": tensor.is_contiguous(),
    }


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _sparse_mla_prefill_workspace_bounds(
    seq_lens_cpu: torch.Tensor,
    gather_lens_cpu: torch.Tensor,
    compress_ratio: int,
    swa_only: bool,
) -> tuple[int, int]:
    if seq_lens_cpu.numel() == 0:
        return 0, 0

    max_gather_len = int(gather_lens_cpu.max().item())
    if swa_only:
        return 0, max_gather_len

    compressed_region_size = int((seq_lens_cpu // compress_ratio).max().item())
    return compressed_region_size, compressed_region_size + max_gather_len


def _deepseek_v4_fp8_einsum_config(
    capability_major: int,
) -> tuple[tuple[int, int, int], bool]:
    if capability_major == 10:
        return (1, 1, 128), True
    return (1, 128, 128), False


def _use_deepseek_v4_sm12_triton_fp8_einsum(
    equation: str,
    recipe: list[int],
    b_scale: torch.Tensor,
) -> bool:
    capability = current_platform.get_device_capability()
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    return (
        capability is not None
        and capability.major == 12
        and equation == "bhr,hdr->bhd"
        and tuple(recipe) == (1, 128, 128)
        and b_scale.dtype in (torch.float32, e8m0_dtype)
    )


def _dump_sparse_mla_attention_state(
    phase: str,
    prefix: str,
    compress_ratio: int,
    q: torch.Tensor,
    output: torch.Tensor,
    swa_metadata: "DeepseekSparseSWAMetadata",
    attn_metadata: FlashMLASparseMetadata | None,
    fields: dict[str, object],
) -> None:
    dump_path = sparse_mla_attention_dump_path()
    payload = {
        "phase": phase,
        "prefix": prefix,
        "compress_ratio": compress_ratio,
        "q": _tensor_summary(q),
        "output": _tensor_summary(output),
        "attn_metadata_present": attn_metadata is not None,
        "swa_metadata": {
            "block_table": _tensor_summary(swa_metadata.block_table),
            "slot_mapping": _tensor_summary(swa_metadata.slot_mapping),
            "seq_lens": _tensor_summary(swa_metadata.seq_lens),
            "query_start_loc": _tensor_summary(swa_metadata.query_start_loc),
            "is_valid_token": _tensor_summary(swa_metadata.is_valid_token),
            "token_to_req_indices": _tensor_summary(
                swa_metadata.token_to_req_indices
            ),
            "decode_swa_indices": _tensor_summary(
                swa_metadata.decode_swa_indices
            ),
            "decode_swa_lens": _tensor_summary(swa_metadata.decode_swa_lens),
            "prefill_seq_lens": _tensor_summary(swa_metadata.prefill_seq_lens),
            "prefill_gather_lens": _tensor_summary(
                swa_metadata.prefill_gather_lens
            ),
            "block_size": int(swa_metadata.block_size),
            "num_decodes": int(swa_metadata.num_decodes),
            "num_prefills": int(swa_metadata.num_prefills),
            "num_decode_tokens": int(swa_metadata.num_decode_tokens),
            "num_prefill_tokens": int(swa_metadata.num_prefill_tokens),
        },
        "flashmla_metadata": {
            "block_table": _tensor_summary(
                attn_metadata.block_table if attn_metadata is not None else None
            ),
            "slot_mapping": _tensor_summary(
                attn_metadata.slot_mapping if attn_metadata is not None else None
            ),
            "c128a_global_decode_topk_indices": _tensor_summary(
                attn_metadata.c128a_global_decode_topk_indices
                if attn_metadata is not None
                else None
            ),
            "c128a_decode_topk_lens": _tensor_summary(
                attn_metadata.c128a_decode_topk_lens
                if attn_metadata is not None
                else None
            ),
            "c128a_prefill_topk_indices": _tensor_summary(
                attn_metadata.c128a_prefill_topk_indices
                if attn_metadata is not None
                else None
            ),
            "block_size": _optional_int(
                attn_metadata.block_size if attn_metadata is not None else None
            ),
            "topk_tokens": _optional_int(
                attn_metadata.topk_tokens if attn_metadata is not None else None
            ),
        },
        "fields": fields,
    }
    with open(dump_path, "a", encoding="utf-8") as dump_file:
        dump_file.write(json.dumps(payload, sort_keys=True) + "\n")
    raise RuntimeError(
        f"DeepseekV4 sparse MLA diagnostic dump written to {dump_path}"
    )


def _write_sparse_mla_attention_state_if_enabled(
    phase: str,
    prefix: str,
    compress_ratio: int,
    q: torch.Tensor,
    output: torch.Tensor,
    swa_metadata: "DeepseekSparseSWAMetadata",
    attn_metadata: FlashMLASparseMetadata | None,
    fields: dict[str, object],
) -> None:
    if not is_sparse_mla_attention_dump_enabled():
        return
    _dump_sparse_mla_attention_state(
        phase=phase,
        prefix=prefix,
        compress_ratio=compress_ratio,
        q=q,
        output=output,
        swa_metadata=swa_metadata,
        attn_metadata=attn_metadata,
        fields=fields,
    )

# Prefill is processed in fixed-size chunks; this bounds the bf16 kv-gather
# workspace allocated at _forward_prefill (and the matching profile-time
# reservation in attention_impl's dummy-run branch).
PREFILL_CHUNK_SIZE = 4


@dataclass
class DeepseekV4MLAModules:
    """Modules used in DeepseekV4 MLA."""

    vllm_config: VllmConfig
    fused_wqa_wkv: torch.nn.Module
    q_norm: torch.nn.Module
    wq_b: torch.nn.Module
    kv_norm: torch.nn.Module
    wo_a: torch.nn.Module
    wo_b: torch.nn.Module
    attn_sink: torch.nn.Module
    rotary_emb: torch.nn.Module
    indexer: torch.nn.Module | None
    indexer_rotary_emb: torch.nn.Module
    topk_indices_buffer: torch.Tensor | None
    aux_stream: torch.cuda.Stream | None = None


# --8<-- [start:multi_head_latent_attention]
@PluggableLayer.register("deepseek_v4_multi_head_latent_attention")
class DeepseekV4MultiHeadLatentAttentionWrapper(PluggableLayer):
    """Pluggable MLA layer which allows OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently oot platforms can still use CustomOp.register_oot to
    replace MLA layer entirely, although we use PluggableLayer to register
    this layer now.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    # --8<-- [end:multi_head_latent_attention]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        o_lora_rank: int | None,
        mla_modules: DeepseekV4MLAModules,
        window_size: int,
        compress_ratio: int | None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_local_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale

        # FlashMLA sparse kernel only supports 64 or 128 heads; pad up to the
        # next supported size. Must match DeepseekV4MLAAttention.padded_heads.
        if num_heads <= 64:
            self.padded_heads = 64
        elif num_heads <= 128:
            self.padded_heads = 128
        else:
            raise ValueError(
                f"DeepseekV4 attention does not support {num_heads} heads "
                "(must be <= 128)."
            )

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.window_size = window_size
        self.compress_ratio = compress_ratio if compress_ratio is not None else 1
        self.prefix = prefix

        disable_sparse_mla_reference_cudagraphs_if_enabled(mla_modules.vllm_config)

        # Extract config from vllm_config
        config = mla_modules.vllm_config.model_config.hf_config
        tp_size = get_tensor_model_parallel_world_size()

        # DeepseekV4-specific attributes (num_heads is already TP-adjusted)
        self.eps = config.rms_norm_eps
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = head_dim - self.rope_head_dim
        self.n_local_groups = config.o_groups // tp_size
        self.o_lora_rank = config.o_lora_rank

        # Store projection modules
        self.fused_wqa_wkv = mla_modules.fused_wqa_wkv
        self.q_norm = mla_modules.q_norm
        self.wq_b = mla_modules.wq_b

        self.kv_norm = mla_modules.kv_norm
        self.wo_a = mla_modules.wo_a

        self._wo_a_act_quant = QuantFP8(
            static=False,
            group_shape=GroupShape(1, 128),
            use_ue8m0=True,
        )
        # Bypass packed-for-deepgemm path — we need FP32 scales (not packed
        # INT32) so fp8_einsum can handle layout transform internally.
        self._wo_a_act_quant.use_deep_gemm_supported = False
        self.wo_b = mla_modules.wo_b

        # Pick fp8_einsum recipe based on GPU arch:
        # SM90/SM120: FP32 block scales stay [g, r/128, d/128].
        # SM100: INT32 packed scales become [g, r, ...].
        from vllm.platforms import current_platform

        cap = current_platform.get_device_capability()
        assert cap is not None, "DeepseekV4 attention requires a CUDA device"
        self._einsum_recipe, self._tma_aligned_scales = (
            _deepseek_v4_fp8_einsum_config(cap.major)
        )

        self.rotary_emb = mla_modules.rotary_emb
        self.indexer_rotary_emb = mla_modules.indexer_rotary_emb
        self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.indexer = mla_modules.indexer

        # Per-head RMS normalization for Q (no learnable weights)
        self.q_head_norm = RMSNorm(head_dim, eps=self.eps, has_weight=False)

        # TODO(yifan): currently hardcoded for FP8 sparse, make it more generic
        head_bytes = (
            self.nope_head_dim  # 448 fp8 NoPE
            + self.rope_head_dim * 2  # 64 bf16 RoPE
            + self.nope_head_dim // 64  # 7B scale factors
            + 1  # 1B pad
        )

        self.aux_stream = mla_modules.aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        assert cache_config is not None, "DeepseekV4 attention requires cache_config"
        self.swa_cache_layer = DeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=torch.uint8,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        self.mla_attn = DeepseekV4MLAAttention(
            num_heads=self.n_local_heads,
            head_dim=self.head_dim,
            scale=self.scale,
            qk_nope_head_dim=self.nope_head_dim,
            qk_rope_head_dim=self.rope_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            compress_ratio=self.compress_ratio,
            window_size=self.window_size,
            head_bytes=head_bytes,
            swa_cache_layer=self.swa_cache_layer,
            attn_sink=mla_modules.attn_sink,  # already padded with -inf
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            indexer=self.indexer,
            topk_indices_buffer=self.topk_indices_buffer,
        )
        # Register this layer in the compilation config's static forward context
        # This allows the custom op to retrieve the layer during execution
        compilation_config = mla_modules.vllm_config.compilation_config
        # HACK
        self.layer_name = prefix + ".deepseek_v4_multi_head_latent_attention"
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self

        # Create the compressor for layers with compress_ratio > 1; after
        # creating the DeepseekV4MLAAttention layer to get its cache.
        self.compressor = None
        if self.compress_ratio > 1:
            self.compressor = DeepseekCompressor(
                vllm_config=mla_modules.vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.mla_attn.prefix,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)

        # Pre-allocate attention output with FlashMLA-padded head count.
        # The op writes into `o_padded`; we slice to n_local_heads after.
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Attention (inside custom op for torch.compile boundary)
        torch.ops.vllm.deepseek_v4_attention(
            hidden_states,
            qr,
            kv,
            positions,
            o_padded,
            self.layer_name,
        )
        o = o_padded[:, : self.n_local_heads, :]

        # O projection: inverse RoPE + FP8 quant + einsum + wo_b
        o_fp8, o_scale = fused_inv_rope_fp8_quant(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            tma_aligned_scales=self._tma_aligned_scales,
        )

        wo_a_fp8 = self.wo_a.weight
        wo_a_scale = self.wo_a.weight_scale_inv

        (z,) = current_workspace_manager().get_simultaneous(
            ((num_tokens, self.n_local_groups, self.o_lora_rank), torch.bfloat16),
        )
        torch.ops.vllm.deepseek_v4_fp8_einsum(
            o_fp8,
            o_scale,
            wo_a_fp8,
            wo_a_scale,
            z,
            "bhr,hdr->bhd",
            list(self._einsum_recipe),
        )

        return self.wo_b(z.flatten(1))

    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,  # [num_tokens, padded_heads, head_dim], written in place
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        qr, kv = fused_q_kv_rmsnorm(
            qr,
            kv,
            self.q_norm.weight.data,
            self.kv_norm.weight.data,
            self.eps,
        )
        q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)

        # Overlap kv_insert with whichever of indexer/compressor is present.
        # Indexer implies compressor; when both exist, compressor rides on the
        # aux stream alongside kv_insert so the heavy indexer owns default.
        if self.indexer is not None:
            indexer = self.indexer
            # Local ref so the closure keeps a non-None type for mypy.
            assert self.compressor is not None
            compressor = self.compressor

            def kv_insert_and_compress() -> None:
                self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
                compressor(hidden_states, positions, self.rotary_emb)

            maybe_execute_in_parallel(
                lambda: indexer(hidden_states, qr, positions, self.indexer_rotary_emb),
                kv_insert_and_compress,
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )
        elif self.compressor is not None:
            # Compressor on default, kv_insert on aux.
            compressor = self.compressor
            maybe_execute_in_parallel(
                lambda: compressor(hidden_states, positions, self.rotary_emb),
                lambda: self._fused_qnorm_rope_kv_insert(
                    q, kv, positions, attn_metadata
                ),
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )
        else:
            # SWA-only layer: no compressor, no overlap.
            self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)

        # Handle dummy run (no metadata).
        if not isinstance(attn_metadata, dict):
            out.zero_()
            return

        # Pad q to FlashMLA-required head count (64 or 128)
        if self.n_local_heads < self.padded_heads:
            pad_size = self.padded_heads - self.n_local_heads
            q = F.pad(q, (0, 0, 0, pad_size), value=0.0)

        # MLA attention writes into the pre-allocated `out` buffer
        # ([num_tokens, padded_heads, head_dim]).
        self.mla_attn(q, kv, positions, output=out)

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None
        ),
    ) -> None:
        if not isinstance(attn_metadata, dict):
            return

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_kv_cache = self.swa_cache_layer.kv_cache
        swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)

        # Horizontally fused:
        #   Q side:  q_head_norm (per-head RMSNorm, no weight) + GPT-J RoPE
        #   KV side: GPT-J RoPE + UE8M0 FP8 quant + paged cache insert
        # kv is unchanged; mla_attn reads kv solely via swa_kv_cache.
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q,
            kv,
            swa_kv_cache_2d,
            swa_metadata.slot_mapping,
            positions.to(torch.int64),
            self.rotary_emb.cos_sin_cache,
            self.eps,
            swa_metadata.block_size,
        )


def deepseek_v4_attention(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.attention_impl(hidden_states, qr, kv, positions, out)


def deepseek_v4_attention_fake(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_attention",
    op_func=deepseek_v4_attention,
    mutates_args=["out"],
    fake_impl=deepseek_v4_attention_fake,
)


def deepseek_v4_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    if equation == "bhr,hdr->bhd" and b.dim() == 2:
        num_groups = out.shape[1]
        out_rank = out.shape[2]
        hidden_size = a.shape[2]
        b = b.view(num_groups, out_rank, hidden_size)

        if b_scale.dim() == 2:
            scale_mn = recipe[1]
            scale_k_pack = 4 if b_scale.dtype == torch.int32 else 1
            scale_k = recipe[2] * scale_k_pack
            b_scale = b_scale.view(
                num_groups,
                (out_rank + scale_mn - 1) // scale_mn,
                (hidden_size + scale_k - 1) // scale_k,
            )

        if _use_deepseek_v4_sm12_triton_fp8_einsum(equation, recipe, b_scale):
            deepseek_v4_sm12_fp8_einsum(a, a_scale, b, b_scale, out)
            return

    fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))


def deepseek_v4_fp8_einsum_fake(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_fp8_einsum",
    op_func=deepseek_v4_fp8_einsum,
    mutates_args=["out"],
    fake_impl=deepseek_v4_fp8_einsum_fake,
)


class DeepseekV4MLAAttention(nn.Module, AttentionLayerBase):
    # FlashMLA FP8 sparse only supports 64 or 128 heads
    SUPPORTED_HEAD_COUNTS = (64, 128)

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        compress_ratio: int,
        window_size: int,
        head_bytes: int,
        swa_cache_layer: DeepseekV4SWACache,
        attn_sink: torch.Tensor,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        # Sparse MLA Args
        indexer: object | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream: torch.cuda.Stream | None = None,
        **extra_impl_args,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = 1
        self.head_dim = head_dim
        self.scale = scale
        self.window_size = window_size
        self.head_bytes = head_bytes
        self.compress_ratio = compress_ratio
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.nope_head_dim = qk_nope_head_dim
        self.rope_head_dim = qk_rope_head_dim
        self.indexer = indexer
        self.topk_indices_buffer = topk_indices_buffer

        self.prefix = prefix  # Alias for compatibility with compressor

        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        # Determine padded head count for FlashMLA
        if num_heads not in self.SUPPORTED_HEAD_COUNTS:
            if num_heads < 64:
                self.padded_heads = 64
            elif num_heads < 128:
                self.padded_heads = 128
            else:
                raise ValueError(
                    f"DeepseekV4MLAAttention does not support {num_heads} heads. "
                    f"Supported: <= 128 (will be padded to 64 or 128)"
                )
        else:
            self.padded_heads = num_heads

        # Store attention sink
        assert attn_sink is not None
        self.attn_sink: torch.Tensor = attn_sink
        # Store SWA cache
        assert swa_cache_layer is not None
        self.swa_cache_layer: DeepseekV4SWACache = swa_cache_layer

        # Get vllm config for cache setup
        vllm_config = get_current_vllm_config()
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        # DeepseekV4 only supports fp8 kv-cache format for now
        kv_cache_dtype = cache_config.cache_dtype if cache_config is not None else "fp8"

        assert kv_cache_dtype.startswith("fp8"), (
            f"DeepseekV4 only supports fp8 kv-cache format for now, "
            f"got {kv_cache_dtype}"
        )
        assert issubclass(self.get_attn_backend(), FlashMLASparseBackend), (
            "Only FlashMLA Sparse Attention backend is supported for DeepseekV4 for now"
        )
        # FlashMLA Sparse Attention fp8 backend uses "fp8_ds_mla" kv-cache format
        # Automatically convert fp8 kv-cache format to "fp8_ds_mla"
        if (
            issubclass(self.get_attn_backend(), FlashMLASparseBackend)
            and kv_cache_dtype.startswith("fp8")
            and kv_cache_dtype != "fp8_ds_mla"
        ):
            assert cache_config is not None
            cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"
            logger.info_once(
                "Using DeepSeek's fp8_ds_mla KV cache format. To use standard "
                "fp8 kv-cache format, please set `--attention-backend "
                "FLASHINFER_MLA_SPARSE`"
            )

        self.kv_cache_dtype = kv_cache_dtype

        # Register with compilation context for metadata lookup
        compilation_config = vllm_config.compilation_config
        if prefix and prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        if prefix:
            compilation_config.static_forward_context[prefix] = self

        self.kv_cache = torch.tensor([])

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV4FlashMLASparseBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if (
            self.compress_ratio <= 1
        ):  # SWA part. Allocated separately as DeepseekV4SWACache.
            return None
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576,  # NOTE: FlashMLA requires 576B alignment
            model_version="deepseek_v4",
        )

    def _forward_sparse_mla_swa_decode_reference(
        self,
        q: torch.Tensor,
        swa_k_cache: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        mtp_decode = num_decode_tokens != num_decodes

        swa_lens = swa_metadata.decode_swa_lens[:num_decode_tokens]
        swa_indices = swa_metadata.decode_swa_indices[:num_decode_tokens]
        max_swa_len = swa_metadata.decode_swa_indices.shape[-1]
        head_block_size = sparse_mla_decode_head_block_size(num_decode_tokens)
        if not mtp_decode:
            fp8ds_paged_sparse_mla_attention_with_sink_multihead(
                q=q,
                k_cache=swa_k_cache,
                seq_lens=swa_metadata.seq_lens[:num_decodes],
                gather_lens=swa_lens,
                block_table=swa_metadata.block_table[:num_decodes],
                block_size=swa_metadata.block_size,
                candidate_offset=0,
                num_candidates=max_swa_len,
                scale=self.scale,
                attn_sink=self.attn_sink,
                output=output,
                head_block_size=head_block_size,
                num_heads=self.num_heads,
            )
            if output.shape[1] > self.num_heads:
                output[:, self.num_heads :].zero_()
            return

        (
            swa_max_score,
            swa_denom,
            swa_acc,
        ) = current_workspace_manager().get_simultaneous(
            ((num_decode_tokens, self.num_heads), torch.float32),
            ((num_decode_tokens, self.num_heads), torch.float32),
            ((num_decode_tokens, self.num_heads, q.shape[-1]), torch.float32),
        )
        swa_max_score.fill_(float("-inf"))
        swa_denom.zero_()
        swa_acc.zero_()
        accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
            q=q,
            k_cache=swa_k_cache,
            slot_ids=swa_indices,
            lens=swa_lens,
            block_size=swa_metadata.block_size,
            scale=self.scale,
            max_score=swa_max_score,
            denom=swa_denom,
            acc=swa_acc,
            head_block_size=head_block_size,
        )
        finish_sparse_mla_attention_with_sink(
            swa_max_score,
            swa_denom,
            swa_acc,
            self.attn_sink,
            output=output,
        )
        if output.shape[1] > self.num_heads:
            output[:, self.num_heads :].zero_()

    def _forward_sparse_mla_compressed_decode_reference(
        self,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor,
        swa_k_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_lens: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata,
        output: torch.Tensor,
    ) -> None:
        if self.compress_ratio not in (4, 128):
            raise NotImplementedError(
                "Sparse MLA reference compressed decode currently supports "
                f"compress_ratio=4 or 128, got {self.compress_ratio}"
            )

        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        mtp_decode = num_decode_tokens != num_decodes

        max_swa_len = swa_metadata.decode_swa_indices.shape[-1]
        compressed_block_size = attn_metadata.block_size // self.compress_ratio
        compressed_topk = topk_indices.shape[-1]
        topk_chunk_size = min(
            compressed_topk,
            sparse_mla_reference_topk_chunk_size(),
        )
        compressed_slot_ids = topk_indices[:, 0, :]
        swa_lens = swa_metadata.decode_swa_lens[:num_decode_tokens]
        swa_indices = swa_metadata.decode_swa_indices[:num_decode_tokens]
        head_block_size = sparse_mla_decode_head_block_size(num_decode_tokens)
        if (
            not mtp_decode
            and compressed_topk <= topk_chunk_size
            and sparse_mla_matmul_decode_enabled()
        ):
            total_candidates = compressed_topk + max_swa_len
            (
                combined_kv,
                valid_tokens,
            ) = current_workspace_manager().get_simultaneous(
                (
                    (num_decode_tokens, total_candidates, q.shape[-1]),
                    torch.bfloat16,
                ),
                ((num_decode_tokens, total_candidates), torch.bool),
            )
            dequantize_combined_sparse_mla_decode_kv(
                combined_kv,
                compressed_k_cache,
                compressed_slot_ids,
                compressed_block_size,
                swa_k_cache,
                swa_metadata.seq_lens[:num_decodes],
                swa_lens,
                swa_metadata.block_table[:num_decodes],
                swa_metadata.block_size,
            )

            build_combined_sparse_mla_decode_valid_mask(
                valid_tokens,
                compressed_slot_ids,
                topk_lens,
                swa_lens,
            )
            matmul_sparse_mla_attention_with_sink(
                q=q,
                kv=combined_kv,
                valid_tokens=valid_tokens,
                scale=self.scale,
                attn_sink=self.attn_sink,
                output=output,
                num_heads=self.num_heads,
            )
            return

        if not mtp_decode and compressed_topk <= topk_chunk_size:
            fp8ds_global_paged_sparse_mla_attention_with_sink_multihead(
                q=q,
                compressed_k_cache=compressed_k_cache,
                slot_ids=compressed_slot_ids,
                topk_lens=topk_lens,
                compressed_block_size=compressed_block_size,
                swa_k_cache=swa_k_cache,
                seq_lens=swa_metadata.seq_lens[:num_decodes],
                gather_lens=swa_lens,
                block_table=swa_metadata.block_table[:num_decodes],
                swa_block_size=swa_metadata.block_size,
                num_compressed_candidates=compressed_topk,
                num_swa_candidates=max_swa_len,
                scale=self.scale,
                attn_sink=self.attn_sink,
                output=output,
                head_block_size=head_block_size,
                num_heads=self.num_heads,
            )
            if output.shape[1] > self.num_heads:
                output[:, self.num_heads :].zero_()
            return

        (
            comp_max_score,
            comp_denom,
            comp_acc,
            swa_max_score,
            swa_denom,
            swa_acc,
        ) = current_workspace_manager().get_simultaneous(
            ((num_decode_tokens, self.num_heads), torch.float32),
            ((num_decode_tokens, self.num_heads), torch.float32),
            ((num_decode_tokens, self.num_heads, q.shape[-1]), torch.float32),
            ((num_decode_tokens, self.num_heads), torch.float32),
            ((num_decode_tokens, self.num_heads), torch.float32),
            ((num_decode_tokens, self.num_heads, q.shape[-1]), torch.float32),
        )
        comp_max_score.fill_(float("-inf"))
        comp_denom.zero_()
        comp_acc.zero_()
        swa_max_score.fill_(float("-inf"))
        swa_denom.zero_()
        swa_acc.zero_()

        for chunk_start in range(0, compressed_topk, topk_chunk_size):
            chunk_end = min(chunk_start + topk_chunk_size, compressed_topk)
            accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
                q=q,
                k_cache=compressed_k_cache,
                slot_ids=compressed_slot_ids[:, chunk_start:chunk_end],
                lens=topk_lens,
                block_size=compressed_block_size,
                candidate_offset=chunk_start,
                scale=self.scale,
                max_score=comp_max_score,
                denom=comp_denom,
                acc=comp_acc,
                head_block_size=head_block_size,
            )
        if mtp_decode:
            accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
                q=q,
                k_cache=swa_k_cache,
                slot_ids=swa_indices,
                lens=swa_lens,
                block_size=swa_metadata.block_size,
                scale=self.scale,
                max_score=swa_max_score,
                denom=swa_denom,
                acc=swa_acc,
                head_block_size=head_block_size,
            )
        else:
            accumulate_fp8ds_paged_sparse_mla_attention_chunk_multihead(
                q=q,
                k_cache=swa_k_cache,
                seq_lens=swa_metadata.seq_lens[:num_decodes],
                gather_lens=swa_lens,
                block_table=swa_metadata.block_table[:num_decodes],
                block_size=swa_metadata.block_size,
                candidate_offset=0,
                num_candidates=max_swa_len,
                scale=self.scale,
                max_score=swa_max_score,
                denom=swa_denom,
                acc=swa_acc,
                head_block_size=head_block_size,
            )
        finish_two_sparse_mla_attention_states_with_sink(
            comp_max_score,
            comp_denom,
            comp_acc,
            swa_max_score,
            swa_denom,
            swa_acc,
            self.attn_sink,
            output=output,
        )
        if output.shape[1] > self.num_heads:
            output[:, self.num_heads :].zero_()

    def _forward_sparse_mla_prefill_reference(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        combined_indices: torch.Tensor,
        combined_lens: torch.Tensor,
        output: torch.Tensor,
        state_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        kv_flat = kv.reshape(-1, q.shape[-1])
        topk_chunk_size = min(
            combined_indices.shape[-1],
            sparse_mla_reference_topk_chunk_size(),
        )
        query_chunk_size = min(
            q.shape[0],
            sparse_mla_reference_query_chunk_size(),
        )
        if state_buffers is None:
            (
                max_score_buffer,
                denom_buffer,
                output_buffer,
            ) = current_workspace_manager().get_simultaneous(
                ((query_chunk_size, self.num_heads), torch.float32),
                ((query_chunk_size, self.num_heads), torch.float32),
                ((query_chunk_size, self.num_heads, q.shape[-1]), torch.float32),
            )
        else:
            max_score_buffer, denom_buffer, output_buffer = state_buffers

        for token_start in range(0, q.shape[0], query_chunk_size):
            token_end = min(token_start + query_chunk_size, q.shape[0])
            q_chunk = q[token_start:token_end]
            indices_chunk_full = combined_indices[token_start:token_end]
            lens_chunk = combined_lens[token_start:token_end]
            num_tokens = token_end - token_start
            max_score = max_score_buffer[:num_tokens]
            denom = denom_buffer[:num_tokens]
            subset_acc = output_buffer[:num_tokens]
            max_score.fill_(float("-inf"))
            denom.zero_()
            subset_acc.zero_()

            for index_start in range(0, combined_indices.shape[-1], topk_chunk_size):
                index_end = min(
                    index_start + topk_chunk_size,
                    combined_indices.shape[-1],
                )
                accumulate_indexed_sparse_mla_attention_chunk(
                    q=q_chunk,
                    kv_flat=kv_flat,
                    indices=indices_chunk_full[:, index_start:index_end],
                    lens=lens_chunk,
                    candidate_offset=index_start,
                    scale=self.scale,
                    max_score=max_score,
                    denom=denom,
                    acc=subset_acc,
                )

            finish_sparse_mla_attention_with_sink(
                max_score,
                denom,
                subset_acc,
                self.attn_sink,
                output=output[token_start:token_end],
            )
            if output.shape[1] > self.num_heads:
                output[token_start:token_end, self.num_heads :].zero_()

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        # Get SWA and indexer metadata from forward context
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            FlashMLASparseMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        # SWA-only layers (compress_ratio <= 1) don't have their own KV cache
        # allocation, so self.kv_cache may be empty after profiling cleanup.
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        # Split prefill and decode
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                assert self.topk_indices_buffer is not None
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        # We treat queries in the same seq as different queries
        # and later we only attend by generated indices.
        # q arrives pre-padded to self.padded_heads by the outer wrapper.
        q = q.unsqueeze(1)

        # Prepare SWA cache (num_blocks, swa_block_size, 1, head_bytes)
        # Use unsqueeze to preserve strides (handles padded blocks correctly)
        swa_cache = self.swa_cache_layer.kv_cache.unsqueeze(-2)
        # Reshape KV cache to (num_blocks, block_size, 1, head_bytes)
        compressed_k_cache = kv_cache
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        decode_fields = {
            "kv_cache": _tensor_summary(kv_cache),
            "swa_cache": _tensor_summary(swa_cache),
            "topk_indices": _tensor_summary(topk_indices),
            "topk_lens": _tensor_summary(topk_lens),
            "swa_indices": _tensor_summary(swa_indices),
            "swa_lens": _tensor_summary(swa_lens),
            "attn_sink": _tensor_summary(self.attn_sink),
            "scale": float(self.scale),
            "swa_only": swa_only,
            "padded_heads": int(self.padded_heads),
            "num_decodes": int(num_decodes),
            "num_decode_tokens": int(num_decode_tokens),
        }
        _write_sparse_mla_attention_state_if_enabled(
            phase="decode",
            prefix=self.prefix,
            compress_ratio=self.compress_ratio,
            q=q,
            output=output,
            swa_metadata=swa_metadata,
            attn_metadata=attn_metadata,
            fields=decode_fields,
        )

        if is_sparse_mla_reference_attention_enabled(q.device):
            if swa_only:
                self._forward_sparse_mla_swa_decode_reference(
                    q=q,
                    swa_k_cache=self.swa_cache_layer.kv_cache,
                    swa_metadata=swa_metadata,
                    output=output,
                )
                return
            if self.compress_ratio in (4, 128):
                assert compressed_k_cache is not None
                assert attn_metadata is not None
                assert topk_indices is not None
                assert topk_lens is not None
                self._forward_sparse_mla_compressed_decode_reference(
                    q=q,
                    compressed_k_cache=compressed_k_cache,
                    swa_k_cache=self.swa_cache_layer.kv_cache,
                    topk_indices=topk_indices,
                    topk_lens=topk_lens,
                    swa_metadata=swa_metadata,
                    attn_metadata=attn_metadata,
                    output=output,
                )
                return
            _dump_sparse_mla_attention_state(
                phase="decode_unsupported_compressed",
                prefix=self.prefix,
                compress_ratio=self.compress_ratio,
                q=q,
                output=output,
                swa_metadata=swa_metadata,
                attn_metadata=attn_metadata,
                fields=decode_fields,
            )

        # One FlashMLASchedMeta per layer type, shared across all same-type
        # layers within this decode step. The first forward call per type
        # triggers the in-kernel planner (allocating tile_scheduler_metadata
        # and num_splits via PyTorch's graph-aware allocator so CUDA graph
        # capture reuses the same addresses on replay); subsequent same-type
        # layers see have_initialized=True and skip the planner.
        if self.compress_ratio <= 1:
            tile_metadata = swa_metadata.tile_sched_swaonly
        elif self.compress_ratio == 4:
            tile_metadata = swa_metadata.tile_sched_c4a
        elif self.compress_ratio == 128:
            tile_metadata = swa_metadata.tile_sched_c128a
        else:
            raise ValueError(
                f"Unsupported compress_ratio={self.compress_ratio}; "
                "expected 1, 4, or 128."
            )
        assert tile_metadata is not None, (
            "swa_metadata missing tile_sched entry for "
            f"compress_ratio={self.compress_ratio}; "
            "DeepseekSparseSWAMetadataBuilder.build_tile_scheduler did not "
            "allocate one for this layer type."
        )

        out, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=swa_cache,
            block_table=None,
            head_dim_v=512,
            tile_scheduler_metadata=tile_metadata,
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_indices,
            topk_length=swa_lens,
            softmax_scale=self.scale,
            attn_sink=self.attn_sink,
            extra_k_cache=kv_cache if not swa_only else None,
            extra_indices_in_kvcache=topk_indices,
            extra_topk_length=topk_lens,
            out=output.unsqueeze(1),
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        seq_lens_cpu = swa_metadata.prefill_seq_lens_cpu
        gather_lens_cpu = swa_metadata.prefill_gather_lens_cpu
        assert seq_lens is not None
        assert gather_lens is not None
        assert seq_lens_cpu is not None
        assert gather_lens_cpu is not None

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                # C128A: pre-computed during metadata build.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
        else:
            # NOTE(woosuk): topk_indices will not be used for SWA-only layers.
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0

        N, M = _sparse_mla_prefill_workspace_bounds(
            seq_lens_cpu=seq_lens_cpu,
            gather_lens_cpu=gather_lens_cpu,
            compress_ratio=self.compress_ratio,
            swa_only=swa_only,
        )
        num_chunks = (num_prefills + PREFILL_CHUNK_SIZE - 1) // PREFILL_CHUNK_SIZE

        workspace_manager = current_workspace_manager()
        reference_attention_enabled = is_sparse_mla_reference_attention_enabled(
            q.device
        )
        if reference_attention_enabled:
            query_chunk_size = min(
                q.shape[0], sparse_mla_reference_query_chunk_size()
            )
            (
                kv,
                max_score_buffer,
                denom_buffer,
                output_buffer,
            ) = workspace_manager.get_simultaneous(
                ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
                ((query_chunk_size, self.num_heads), torch.float32),
                ((query_chunk_size, self.num_heads), torch.float32),
                ((query_chunk_size, self.num_heads, q.shape[-1]), torch.float32),
            )
            prefill_state_buffers = (
                max_score_buffer,
                denom_buffer,
                output_buffer,
            )
        else:
            kv = workspace_manager.get_simultaneous(
                ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )[0]
            prefill_state_buffers = None
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
            )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                M,
                N,
            )

            if is_sparse_mla_attention_dump_enabled():
                _dump_sparse_mla_attention_state(
                    phase="prefill",
                    prefix=self.prefix,
                    compress_ratio=self.compress_ratio,
                    q=q[query_start:query_end],
                    output=output[query_start:query_end],
                    swa_metadata=swa_metadata,
                    attn_metadata=attn_metadata,
                    fields={
                        "compressed_k_cache": _tensor_summary(compressed_k_cache),
                        "swa_k_cache": _tensor_summary(swa_k_cache),
                        "gathered_kv": _tensor_summary(kv[:chunk_size]),
                        "topk_indices": _tensor_summary(topk_indices),
                        "combined_indices": _tensor_summary(combined_indices),
                        "combined_lens": _tensor_summary(combined_lens),
                        "attn_sink": _tensor_summary(self.attn_sink),
                        "scale": float(self.scale),
                        "swa_only": swa_only,
                        "chunk_start": int(chunk_start),
                        "chunk_end": int(chunk_end),
                        "query_start": int(query_start),
                        "query_end": int(query_end),
                    },
                )

            if reference_attention_enabled:
                self._forward_sparse_mla_prefill_reference(
                    q=q[query_start:query_end],
                    kv=kv[:chunk_size],
                    combined_indices=combined_indices,
                    combined_lens=combined_lens,
                    output=output[query_start:query_end],
                    state_buffers=prefill_state_buffers,
                )
                continue

            output_chunk, _, _ = flash_mla_sparse_fwd(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                attn_sink=self.attn_sink,
                topk_length=combined_lens,
                out=output[query_start:query_end],
            )


class DeepseekV4IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        self.compress_ratio = compress_ratio
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # head_dim already carries the fp8 scale padding
        # compress_ratio=1 for V3.2, >1 for DeepseekV4; both use the same cache layout.
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
            # DeepseekV4 aligns indexer pages to FlashMLA's 576B so they can pack with
            # the indexer's compressor state cache. V3.2 keeps the legacy layout.
            alignment=576,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV4IndexerBackend


class DeepseekV4Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        compress_ratio: int = 1,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        self.compress_ratio = compress_ratio
        self.use_fp4_kv = self.vllm_config.attention_config.use_fp4_indexer_cache
        logger.info_once(
            "Using %s indexer cache for Lighening Indexer.",
            "MXFP4" if self.use_fp4_kv else "FP8",
        )

        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.n_head,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128  # TODO: get from config
        self.topk_indices_buffer = topk_indices_buffer

        self.max_model_len = (
            vllm_config.model_config.max_model_len // self.compress_ratio
        )
        self.prefix = prefix

        self.max_total_seq_len = (
            get_max_prefill_buffer_size(vllm_config) // self.compress_ratio
        )

        assert cache_config is not None, "Deepseek V4 indexer requires cache_config"
        # NOTE(yifan): FP8 indxer cache use the same layout as V3.2:
        # head_dim bytes = 128 fp8 + 4 fp32 scale = 132.
        # For FP4 indexer cache, we still allocate the same amount of memory as FP8,
        # but only use the first half of the memory.
        k_cache_head_dim = self.head_dim + self.head_dim // self.quant_block_size * 4
        self.k_cache = DeepseekV4IndexerCache(
            head_dim=k_cache_head_dim,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
            compress_ratio=self.compress_ratio,
        )
        self.compressor = DeepseekCompressor(
            vllm_config=vllm_config,
            compress_ratio=self.compress_ratio,
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            rotate=True,
            prefix=f"{prefix}.compressor",
            k_cache_prefix=self.k_cache.prefix,
            use_fp4_cache=self.use_fp4_kv,
        )

        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            skip_k_cache_insert=True,
            use_fp4_cache=self.use_fp4_kv,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        k = self.compressor(hidden_states, positions, rotary_emb)
        weights, _ = self.weights_proj(hidden_states)
        q_quant, weights = fused_indexer_q_rope_quant(
            positions,
            q,
            rotary_emb.cos_sin_cache,
            weights,
            self.softmax_scale,
            self.n_head**-0.5,
            use_fp4=self.use_fp4_kv,
        )
        return self.indexer_op(hidden_states, q_quant, k, weights)
