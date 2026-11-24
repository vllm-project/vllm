# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-local attention backend for the Beam DFlash2 drafter on ROCm.

The drafter runs a handful of query tokens per request against a long context
under a non-causal sliding window. The generic prefix-prefill kernel walks the
visible window serially with one program per (request, head), which leaves
most of the GPU idle at decode batch sizes; this backend splits the window
across segments instead (see ``dflash2_attention_ops``).

The KV cache uses the default ``[num_blocks, num_kv_heads, block_size,
2 * head_size]`` layout so the fused QK-norm/RoPE/KV-insert path can write it.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.platforms import current_platform
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.kv_cache_interface import AttentionSpec, is_quantized_kv_cache

from .dflash2_attention_ops import dflash2_split_kv_attention, split_kv_workspace

# Workspace rows are (token, segment) pairs shared by every drafter layer;
# layers run back to back on one stream, so reuse is safe.
_MIN_WORKSPACE_ROWS = 16384
_KV_CACHE_DTYPES: list[CacheDType] = ["auto", "bfloat16", "fp8", "fp8_e4m3"]
_workspaces: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _workspace(
    rows: int, num_heads: int, head_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (device, rows, num_heads, head_size)
    if key not in _workspaces:
        _workspaces[key] = split_kv_workspace(rows, num_heads, head_size, device)
    return _workspaces[key]


@dataclass
class BeamDFlash2AttentionMetadata:
    num_actual_tokens: int
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True


class BeamDFlash2AttentionMetadataBuilder(
    AttentionMetadataBuilder[BeamDFlash2AttentionMetadata]
):
    # The grid and BLOCK_M come from the host max_query_len, so a captured
    # graph is only valid for replays with the same per-request query length.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    # Metadata only references the speculator's persistent tensors.
    supports_draft_decode_metadata_update = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> BeamDFlash2AttentionMetadata:
        causal = common_attn_metadata.causal
        if not isinstance(causal, bool):
            raise NotImplementedError(
                "Beam DFlash2 attention needs one causal mode per batch."
            )
        return BeamDFlash2AttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=causal,
        )

    def update_draft_decode_metadata(
        self, _metadata: BeamDFlash2AttentionMetadata
    ) -> None:
        pass


class BeamDFlash2AttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = _KV_CACHE_DTYPES
    forward_includes_kv_cache_update: bool = False
    supports_kv_cache_update_bypass: bool = True

    @staticmethod
    def get_name() -> str:
        return "BEAM_DFLASH2_SPLIT_KV"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 256]

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @staticmethod
    def get_impl_cls() -> type["BeamDFlash2AttentionImpl"]:
        return BeamDFlash2AttentionImpl

    @staticmethod
    def get_builder_cls() -> type[BeamDFlash2AttentionMetadataBuilder]:
        return BeamDFlash2AttentionMetadataBuilder


class BeamDFlash2AttentionImpl(AttentionImpl[BeamDFlash2AttentionMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        if alibi_slopes is not None or sinks is not None:
            raise NotImplementedError(
                "Beam DFlash2 attention supports neither ALiBi nor sinks."
            )
        if logits_soft_cap:
            raise NotImplementedError("Beam DFlash2 attention has no softcap.")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Beam DFlash2 attention is decoder-only.")
        # The model passes this backend explicitly, which skips the selector's
        # dtype validation, so check here.
        model_dtype = get_current_vllm_config().model_config.dtype
        if model_dtype != torch.bfloat16:
            raise NotImplementedError(
                f"Beam DFlash2 attention needs bfloat16, not {model_dtype}."
            )
        if kv_cache_dtype not in _KV_CACHE_DTYPES:
            raise NotImplementedError(
                f"Beam DFlash2 attention has no {kv_cache_dtype} KV cache."
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads or num_heads
        self.sliding_window = sliding_window or 0
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        device = torch.device("cuda", torch.cuda.current_device())
        self.target_programs = (
            2 * torch.cuda.get_device_properties(device).multi_processor_count
        )
        max_num_seqs = get_current_vllm_config().scheduler_config.max_num_seqs
        rows = max(_MIN_WORKSPACE_ROWS, 16 * max_num_seqs)
        self.workspace = _workspace(rows, num_heads, head_size, device)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: BeamDFlash2AttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Beam DFlash2 attention does not fuse output quantization."
            )
        if attn_metadata is None:
            return output.fill_(0)

        num_tokens = attn_metadata.num_actual_tokens
        query = query[:num_tokens].view(num_tokens, self.num_heads, self.head_size)
        out = output[:num_tokens].view(num_tokens, self.num_heads, self.head_size)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            kv_cache = kv_cache.view(current_platform.fp8_dtype())
        dflash2_split_kv_attention(
            query,
            kv_cache,
            out,
            attn_metadata.query_start_loc,
            attn_metadata.seq_lens,
            attn_metadata.block_table,
            attn_metadata.num_reqs,
            attn_metadata.max_query_len,
            self.scale,
            self.sliding_window,
            attn_metadata.causal,
            self.workspace,
            self.target_programs,
            layer._k_scale,
            layer._v_scale,
        )
        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs))
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
