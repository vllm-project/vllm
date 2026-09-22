# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch
from torch import nn

from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    DispatchSpec,
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
)
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_triton,
    compress_norm_rope_store_two_stage_triton,
)
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import MXFP4_BLOCK_SIZE
from vllm.models.deepseek_v4.common.ops.save_partial_states import (
    _SAVE_PARTIAL_STATES_KERNEL,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

_C128_RATIO = 128


def _c128_ring_capacity(num_speculative_tokens: int) -> int:
    span = _C128_RATIO + num_speculative_tokens
    return _C128_RATIO * ((span + _C128_RATIO - 1) // _C128_RATIO)


@triton.jit(
    do_not_specialize=[
        "block_table_stride",
        "num_actual_tokens",
        "num_tokens",
        "num_reqs",
    ],
    do_not_specialize_on_alignment=["block_table_ptr", "positions_ptr"],
)
def _build_c128_ring_metadata_kernel(
    block_table_ptr,
    block_table_stride,
    query_start_loc_ptr,
    token_to_req_ptr,
    positions_ptr,
    slot_mapping_ptr,
    tail_slot_mapping_ptr,
    num_actual_tokens,
    num_tokens,
    num_reqs,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    query_batch_end = tl.load(query_start_loc_ptr + num_reqs)
    valid = (offsets < num_actual_tokens) & (offsets < query_batch_end)
    req = tl.load(token_to_req_ptr + offsets, mask=valid, other=0).to(tl.int64)
    query_end = tl.load(query_start_loc_ptr + req + 1, mask=valid, other=0)
    position = tl.load(positions_ptr + offsets, mask=valid, other=0)
    block = tl.load(block_table_ptr + req * block_table_stride, mask=valid, other=-1)
    valid &= block >= 0
    slot = block.to(tl.int64) * CAPACITY + position % CAPACITY
    keep_tail = valid & (offsets + CAPACITY >= query_end)
    store = offsets < num_tokens
    tl.store(slot_mapping_ptr + offsets, tl.where(valid, slot, -1), mask=store)
    tl.store(tail_slot_mapping_ptr + offsets, tl.where(keep_tail, slot, -1), mask=store)


def _build_c128_ring_metadata_warmup_inputs(*, capacity: int) -> dict[str, Any]:
    int32 = TritonWarmupTensor(torch.int32)
    int64 = TritonWarmupTensor(torch.int64)
    return dict(
        block_table=int32,
        query_start_loc=int32,
        token_to_req=int32,
        positions=int64,
        slot_mapping=int64,
        tail_slot_mapping=int64,
        num_actual_tokens=2,
        num_tokens=2,
        num_reqs=1,
        capacity=capacity,
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_build_c128_ring_metadata_kernel,
    warmup_inputs=_build_c128_ring_metadata_warmup_inputs,
)
def _build_c128_ring_metadata(
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    token_to_req: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    num_actual_tokens: int,
    num_tokens: int,
    num_reqs: int,
    capacity: int,
) -> DispatchSpec:
    return (triton.cdiv(num_tokens, 256),), dict(
        block_table_stride=block_table.stride(0),
        CAPACITY=capacity,
        BLOCK=256,
    )


_BUILD_C128_RING_METADATA_KERNEL = _build_c128_ring_metadata


def build_c128_ring_metadata(
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    num_actual_tokens: int,
    num_reqs: int,
    capacity: int,
    token_to_req_indices: torch.Tensor | None = None,
    out_slots: torch.Tensor | None = None,
    out_tail_slots: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map C128 rows to a per-request ring and select the saved suffix."""
    if capacity < _C128_RATIO or capacity % _C128_RATIO != 0:
        raise ValueError("C128 ring capacity must be a positive multiple of 128")
    if out_slots is None:
        out_slots = torch.empty_like(slot_mapping)
    if out_tail_slots is None:
        out_tail_slots = torch.empty_like(slot_mapping)
    num_tokens = slot_mapping.numel()
    if num_actual_tokens == 0 or num_reqs == 0:
        out_slots.fill_(-1)
        out_tail_slots.fill_(-1)
        return out_slots, out_tail_slots

    if token_to_req_indices is None:
        tokens = torch.arange(num_actual_tokens, device=slot_mapping.device)
        token_to_req_indices = torch.searchsorted(
            query_start_loc[1 : num_reqs + 1], tokens, right=True
        ).to(torch.int32)

    if positions.is_cuda:
        _BUILD_C128_RING_METADATA_KERNEL(
            block_table,
            query_start_loc,
            token_to_req_indices,
            positions,
            out_slots,
            out_tail_slots,
            num_actual_tokens,
            num_tokens,
            num_reqs,
            capacity,
        )
        return out_slots, out_tail_slots

    out_slots.fill_(-1)
    out_tail_slots.fill_(-1)
    rows = torch.arange(num_actual_tokens, device=slot_mapping.device)
    query_batch_end = query_start_loc[num_reqs]
    valid_rows = rows < query_batch_end
    req = token_to_req_indices[:num_actual_tokens].long().clamp(max=num_reqs - 1)
    pos = positions[:num_actual_tokens].long()
    blocks = block_table[:num_reqs, 0].index_select(0, req).long()
    valid = valid_rows & (blocks >= 0)
    slots = blocks * capacity + pos.remainder(capacity)
    out_slots[:num_actual_tokens] = torch.where(valid, slots, -1)
    query_ends = query_start_loc[1 : num_reqs + 1].index_select(0, req)
    keep_tail = valid & (rows + capacity >= query_ends)
    out_tail_slots[:num_actual_tokens] = torch.where(keep_tail, slots, -1)
    return out_slots, out_tail_slots


def _prefer_two_stage_compressor() -> bool:
    # Platforms that favor the triton variant of two-stage compressor split.
    # Currently only tested on ROCm
    return current_platform.is_rocm()


def _get_c128_boundary(metadata: CommonAttentionMetadata) -> bool | None:
    seq_lens_cpu = metadata.seq_lens_cpu_upper_bound
    if seq_lens_cpu is None:
        return None

    query_lens = metadata.query_start_loc_cpu[1:] - metadata.query_start_loc_cpu[:-1]
    starts = seq_lens_cpu - query_lens
    starts_list = starts.tolist()
    query_start_loc = metadata.query_start_loc_cpu.tolist()
    return any(
        start % 128 + query_start_loc[i + 1] - query_start_loc[i] >= 128
        for i, start in enumerate(starts_list)
    )


class CompressorBackend(AttentionBackend):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_name() -> str:
        return "CompressorBackend"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512, 1024]

    @staticmethod
    def get_builder_cls() -> type["CompressorMetadataBuilder"]:
        return CompressorMetadataBuilder


@dataclass
class CompressorMetadata:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int

    token_to_req_indices: torch.Tensor | None = None  # [num_tokens]
    tail_slot_mapping: torch.Tensor | None = None  # [num_tokens]
    query_start_loc: torch.Tensor | None = None  # [num_reqs + 1]
    is_circular: bool = False
    num_decode_tokens: int | None = None
    c128_boundary: bool | None = None


class CompressorMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(
            self.kv_cache_spec,
            SlidingWindowMLASpec | MLAAttentionSpec | CircularBufferSpec,
        )
        mla_spec = cast(
            SlidingWindowMLASpec | MLAAttentionSpec | CircularBufferSpec,
            self.kv_cache_spec,
        )
        self.block_size = mla_spec.block_size
        self.is_circular = isinstance(mla_spec, CircularBufferSpec)

        max_num_batched_tokens = (
            self.vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.token_to_req_indices = torch.zeros(
            max_num_batched_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        self.slot_mapping_buffer = torch.empty(
            max_num_batched_tokens, dtype=torch.int64, device=self.device
        )
        self.tail_slot_mapping_buffer = torch.empty(
            max_num_batched_tokens, dtype=torch.int64, device=self.device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CompressorMetadata:
        token_to_req_indices = common_attn_metadata.token_to_req_indices(
            self.token_to_req_indices
        )
        num_tokens = common_attn_metadata.slot_mapping.numel()
        if self.is_circular:
            positions = common_attn_metadata.positions
            assert positions is not None
            slot_mapping, tail_slot_mapping = build_c128_ring_metadata(
                common_attn_metadata.slot_mapping,
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.query_start_loc,
                positions,
                common_attn_metadata.num_actual_tokens,
                common_attn_metadata.num_reqs,
                self.block_size,
                token_to_req_indices=token_to_req_indices,
                out_slots=self.slot_mapping_buffer[:num_tokens],
                out_tail_slots=self.tail_slot_mapping_buffer[:num_tokens],
            )
        else:
            slot_mapping = common_attn_metadata.slot_mapping
            tail_slot_mapping = slot_mapping
        num_decode_tokens = None
        if _prefer_two_stage_compressor():
            _, _, num_decode_tokens, _ = split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=1
            )
        async_spec_decode = (
            self.vllm_config.scheduler_config.async_scheduling
            and self.vllm_config.speculative_config is not None
        )
        block_table = common_attn_metadata.block_table_tensor
        if not self.is_circular:
            block_table = block_table.clamp_(min=0)
        return CompressorMetadata(
            block_table=block_table,
            slot_mapping=slot_mapping,
            block_size=self.block_size,
            token_to_req_indices=token_to_req_indices,
            tail_slot_mapping=tail_slot_mapping,
            query_start_loc=common_attn_metadata.query_start_loc,
            is_circular=self.is_circular,
            num_decode_tokens=num_decode_tokens,
            c128_boundary=(
                _get_c128_boundary(common_attn_metadata)
                if self.is_circular and not async_spec_decode
                else None
            ),
        )


class CompressorStateCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        assert self.dtype == torch.float32
        assert compress_ratio in [4, 128]
        self.compress_ratio = compress_ratio
        coff = 1 + (compress_ratio == 4)
        self.sliding_window = coff * compress_ratio
        # C4 still shares the legacy paged layout and therefore keeps block size
        # 4. C128 replaces this value with its per-request ring capacity in the
        # cache spec.
        if compress_ratio == 4:
            self.block_size = 4
        elif compress_ratio == 128:
            self.block_size = 8
        else:
            raise ValueError(f"Invalid compress ratio: {compress_ratio}")

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C]
        self.kv_cache = kv_cache.squeeze(1)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # fp8_ds_mla is the UE8M0 paged layout and needs 576B alignment. Plain
        # full-cache rows share state pages with contiguous KV pages, so padding
        # would break page matching.
        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        if self.compress_ratio == 128:
            return CircularBufferSpec(
                block_size=_c128_ring_capacity(vllm_config.num_speculative_tokens),
                num_kv_heads=1,
                head_size=self.state_dim,
                head_size_v=0,
                dtype=self.dtype,
            )
        return SlidingWindowMLASpec(  # only has one vector instead of K + V
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=576 if uses_fp8_ds_mla_layout else 512,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return CompressorBackend


class DeepseekCompressor(nn.Module):
    """DeepSeek V4 KV/score compressor.

    Owns the linear / norm / state-cache / ape state and the shared forward
    prologue (kv/score split, save_partial_states launch). The
    compress → norm → RoPE → store step is dispatched to a triton kernel
    (``compress_norm_rope_store_triton``) by default, except for the NVIDIA
    head_dim=512 path which uses the CuTeDSL compressor kernels for better
    performance.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        k_cache_prefix="",
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rotate = rotate
        self.prefix = prefix
        self.k_cache_prefix = k_cache_prefix
        self.use_fp4_cache = use_fp4_cache

        config = vllm_config.model_config.hf_config
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.device = current_platform.device_type
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len

        self.overlap = compress_ratio == 4
        self.coff = 1 + self.overlap

        # The head=512 cr>=128 no-overlap deep gather uses the two-stage
        # compressor, which needs an fp32 scratch [max_batched, 512] for
        # the intermediate compressed_kv.
        # Currently only tested on ROCm
        self._use_two_stage_fused_compressor = (
            _prefer_two_stage_compressor() and head_dim == 512 and not self.overlap
        )
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self._compress_scratch: torch.Tensor | None = None
        if self._use_two_stage_fused_compressor:
            self._compress_scratch = torch.empty(
                self.max_num_batched_tokens,
                self.head_dim,
                dtype=torch.float32,
                device=self.device,
            )

        state_dtype = torch.float32
        self.ape = nn.Parameter(
            torch.empty(
                (compress_ratio, self.coff * self.head_dim),
                dtype=state_dtype,
                device=self.device,
            ),
            requires_grad=False,
        )

        self.fused_wkv_wgate = MergedColumnParallelLinear(
            self.hidden_size,
            [self.coff * self.head_dim, self.coff * self.head_dim],
            bias=False,
            return_bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
        )
        self.norm = RMSNorm(self.head_dim, self.rms_norm_eps)

        self.state_cache = CompressorStateCache(
            state_dim=2 * self.coff * self.head_dim,  # kv_state + score_state
            dtype=state_dtype,
            compress_ratio=compress_ratio,
            prefix=f"{prefix}.state_cache",
        )

        # Save reference to static_forward_context for forward-time KV cache lookup.
        # get_current_vllm_config() is only available during __init__, not forward.
        self._static_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )

        if self.head_dim == 512:
            assert not use_fp4_cache, (
                "MXFP4 cache is only supported for indexer (head=128)"
            )
            self._quant_block = 64
            self._token_stride = self.nope_head_dim + self.rope_head_dim * 2
            self._scale_dim = self.nope_head_dim // 64 + 1  # 7 real + 1 pad
        elif self.head_dim == 128:
            if use_fp4_cache:
                self._quant_block = MXFP4_BLOCK_SIZE
                self._token_stride = self.head_dim // 2
                self._scale_dim = self.head_dim // MXFP4_BLOCK_SIZE
            else:
                self._quant_block = 128
                self._token_stride = self.head_dim
                self._scale_dim = 4  # single float32 scale
        else:
            raise ValueError(
                f"Unsupported head_dim for fused quant+cache: {self.head_dim}"
            )

        if vllm_config.kernel_config.enable_jit_warmup:
            if self.compress_ratio == 128 and (
                current_platform.is_cuda() or current_platform.is_rocm()
            ):
                _BUILD_C128_RING_METADATA_KERNEL.register_warmup(
                    capacity=_c128_ring_capacity(vllm_config.num_speculative_tokens)
                )
            _SAVE_PARTIAL_STATES_KERNEL.register_warmup(
                head_dim=self.head_dim,
                compress_ratio=self.compress_ratio,
            )
            if current_platform.is_cuda() and self.head_dim == 512:
                from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import (  # noqa: E501
                    _SPARSE_ATTN_COMPRESS_C128_BLOCK8_KERNEL,
                    _SPARSE_ATTN_COMPRESS_NORM_ROPE_STORE_C4_KERNEL,
                    _SPARSE_ATTN_COMPRESS_NORM_ROPE_STORE_FULL_C4_KERNEL,
                    _SPARSE_ATTN_NORM_ROPE_STORE_FULL_KERNEL,
                    _SPARSE_ATTN_NORM_ROPE_STORE_KERNEL,
                )

                store_full_kv = vllm_config.cache_config.cache_dtype != "fp8_ds_mla"
                if self.compress_ratio == 4:
                    (
                        _SPARSE_ATTN_COMPRESS_NORM_ROPE_STORE_FULL_C4_KERNEL
                        if store_full_kv
                        else _SPARSE_ATTN_COMPRESS_NORM_ROPE_STORE_C4_KERNEL
                    ).register_warmup()
                else:
                    _SPARSE_ATTN_COMPRESS_C128_BLOCK8_KERNEL.register_warmup()
                    if store_full_kv:
                        _SPARSE_ATTN_NORM_ROPE_STORE_FULL_KERNEL.register_warmup()
                    else:
                        _SPARSE_ATTN_NORM_ROPE_STORE_KERNEL.register_warmup(
                            vllm_config,
                            k_cache_prefix=self.k_cache_prefix,
                            compress_ratio=self.compress_ratio,
                        )
            else:
                from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (  # noqa: E501
                    _FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_TRITON_KERNEL,
                )

                _FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_TRITON_KERNEL.register_warmup()

    def forward(
        self,
        # [num_tokens, 2 * self.coff * self.head_dim]
        kv_score: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        # Each of shape [num_tokens, coff * self.head_dim]
        # input bf16, output are fp32
        kv, score = kv_score.split(
            [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
        )

        # Get the metadata and handle dummy profiling run.
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict):
            return

        state_metadata = cast(
            CompressorMetadata, attn_metadata[self.state_cache.prefix]
        )
        token_to_req_indices = state_metadata.token_to_req_indices
        slot_mapping = state_metadata.slot_mapping
        tail_slot_mapping = state_metadata.tail_slot_mapping
        query_start_loc = state_metadata.query_start_loc
        assert token_to_req_indices is not None
        assert tail_slot_mapping is not None
        assert query_start_loc is not None
        num_actual = slot_mapping.shape[0]
        block_table = state_metadata.block_table
        block_size = state_metadata.block_size

        # [num_blocks, block_size, kv_dim+score_dim], where kv_dim == score_dim
        state_cache = self.state_cache.kv_cache
        # kv_state stored in first half, score_state stored in second half
        state_width = state_cache.shape[-1] // 2
        pdl_kwargs = (
            {}
            if current_platform.is_rocm() or current_platform.is_xpu()
            else {"launch_pdl": False}
        )

        # C4 stores before compression. C128 reads current-chunk rows directly
        # and saves only the ring tail after compression, so a long chunk cannot
        # wrap and overwrite an earlier group before that group is consumed.
        # NOTE: PDL is disabled — both this kernel and the compress kernels
        # below depend on preceding kernel outputs (kv/score from the cublas
        # GEMM; state_cache from this kernel) but neither emits/waits on PDL
        # grid dependency primitives, so launch_pdl=True caused a
        # read-after-write race and non-deterministic output.
        if not state_metadata.is_circular:
            _SAVE_PARTIAL_STATES_KERNEL(
                kv=kv,
                score=score,
                ape=self.ape,
                positions=positions,
                state_cache=state_cache,
                slot_mapping=slot_mapping,
                block_size=block_size,
                state_width=state_width,
                compress_ratio=self.compress_ratio,
                pdl_kwargs=pdl_kwargs,
            )

        # full graph cannot branch on per-step CPU metadata after capture
        if (
            current_platform.is_cuda()
            and self.head_dim == 512
            and self.compress_ratio == 128
            and forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL
            and state_metadata.c128_boundary is False
        ):
            _SAVE_PARTIAL_STATES_KERNEL(
                kv=kv,
                score=score,
                ape=self.ape,
                positions=positions,
                state_cache=state_cache,
                slot_mapping=tail_slot_mapping,
                block_size=block_size,
                state_width=state_width,
                compress_ratio=self.compress_ratio,
                pdl_kwargs=pdl_kwargs,
            )
            return

        # Fused: compress → RMSNorm → RoPE → FP8 quant → KV cache write.
        # RoPE requirements (kernel applies forward GPT-J style rotation):
        # - is_neox_style=False (interleaved pairs, NOT split-half)
        # - cos_sin_cache layout: [max_pos, rope_head_dim] with first half cos,
        #   second half sin (per-pair, length rope_head_dim // 2 each)
        # - applied to LAST rope_head_dim elements of head_dim
        # - position used: (positions // compress_ratio) * compress_ratio
        cos_sin_cache = rotary_emb.cos_sin_cache
        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        k_cache_layer = self._static_forward_context[self.k_cache_prefix]
        kv_cache = k_cache_layer.kv_cache

        # Plain-row V4 reads a contiguous bf16 / per-tensor fp8 cache row; the
        # fp8_ds_mla path uses the UE8M0 paged uint8 layout.
        store_full_kv = self.head_dim == 512 and kv_cache.dtype != torch.uint8
        store_full_fp8 = kv_cache.dtype == torch.float8_e4m3fn
        fp8_scale = (
            getattr(k_cache_layer, "_flashinfer_fp8_kv_scale", None)
            if store_full_fp8
            else None
        )

        # cutedsl (head=512) accepts the full-cache flags; triton (indexer/AMD)
        # does not, so the two callables have different signatures.
        compress_norm_rope_store_fn: Any
        if current_platform.is_cuda() and self.head_dim == 512:
            from .nvidia.ops.sparse_attn_compress_cutedsl import (
                _SPARSE_ATTN_COMPRESSOR_CUTEDSL_KERNEL,
            )

            # head=512 on CUDA always uses cutedsl, for both the fp8_ds_mla
            # layout and the plain full-cache layout. The full-cache flags
            # are consumed only here.
            compress_norm_rope_store_fn = _SPARSE_ATTN_COMPRESSOR_CUTEDSL_KERNEL
            extra_kwargs: dict[str, Any] = dict(
                store_full_kv=store_full_kv,
                store_full_fp8=store_full_fp8,
                fp8_scale=fp8_scale,
            )
        elif self._use_two_stage_fused_compressor:
            # head=512 cr>=128 (no overlap): two-pass split compressor on the
            # prefill suffix, single-pass on the decode prefix.
            assert state_metadata.num_decode_tokens is not None
            compress_norm_rope_store_fn = compress_norm_rope_store_two_stage_triton
            extra_kwargs = {
                "num_decode_tokens": state_metadata.num_decode_tokens,
                "compress_scratch": self._compress_scratch,
            }
        else:
            # Indexer path (head_dim == 128) or non-CUDA GPUs (AMD, XPU, etc.).
            compress_norm_rope_store_fn = compress_norm_rope_store_triton
            extra_kwargs = {}

        compress_norm_rope_store_fn(
            state_cache=state_cache,
            kv=kv,
            score=score,
            ape=self.ape,
            num_actual=num_actual,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            block_size=block_size,
            query_start_loc=query_start_loc,
            is_circular=state_metadata.is_circular,
            state_width=state_width,
            cos_sin_cache=cos_sin_cache,
            kv_cache=kv_cache,
            k_cache_metadata=k_cache_metadata,
            pdl_kwargs=pdl_kwargs,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            compress_ratio=self.compress_ratio,
            overlap=self.overlap,
            use_fp4_cache=self.use_fp4_cache,
            rms_norm_weight=self.norm.weight,
            rms_norm_eps=self.rms_norm_eps,
            quant_block=self._quant_block,
            token_stride=self._token_stride,
            scale_dim=self._scale_dim,
            **extra_kwargs,
        )
        if state_metadata.is_circular:
            _SAVE_PARTIAL_STATES_KERNEL(
                kv=kv,
                score=score,
                ape=self.ape,
                positions=positions,
                state_cache=state_cache,
                slot_mapping=tail_slot_mapping,
                block_size=block_size,
                state_width=state_width,
                compress_ratio=self.compress_ratio,
                pdl_kwargs=pdl_kwargs,
            )
