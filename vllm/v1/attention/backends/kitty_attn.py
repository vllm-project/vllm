# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kitty KV-cache quantization backend (https://arxiv.org/abs/2511.18643).

K is stored in 2 bits per value with the largest-magnitude channels of each
128-token page boosted to 4 bits; V is stored in 2 bits per token. The first
128 tokens (the sink), the K tail (<= 127 tokens) and the V tail (128-255
tokens once past the second page) stay unquantized in a per-request residual
block. Each step attends first, then seals the pages it completed, then writes
the tokens that stay unquantized into the residual.
"""

from dataclasses import dataclass, replace
from typing import ClassVar, cast

import numpy as np
import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    get_flash_attn_version,
)
from vllm.v1.attention.backends.utils import split_decodes_prefills_and_extends
from vllm.v1.attention.ops.kitty.decode import (
    G_PAD,
    NUM_SPLITS,
    kitty_decode_attention,
    kitty_dequant_context,
)
from vllm.v1.attention.ops.kitty.store_seal import (
    kitty_seal,
    kitty_write_residual,
    plan_kitty_step,
)
from vllm.v1.attention.ops.kitty.tiles import (
    KITTY_PAGE_SIZE,
    KITTY_RESIDUAL_ROWS,
    KITTY_RESIDUAL_SLOTS,
    kitty_d_boost,
    kitty_state_content_bytes,
)
from vllm.v1.core.single_type_kv_cache_manager import CircularBufferManager
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayerConfig,
    KVCacheLayout,
    KVQuantMode,
)
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)


def _workspace(*shapes_and_dtypes) -> list[torch.Tensor]:
    if is_workspace_manager_initialized():
        return current_workspace_manager().get_simultaneous(*shapes_and_dtypes)
    return [torch.empty(s, dtype=d, device="cuda") for s, d in shapes_and_dtypes]


class KittyAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = True  # forward method updates the kv cache

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["kitty"]

    @staticmethod
    def get_name() -> str:
        return "KITTY"

    @staticmethod
    def get_impl_cls() -> type["KittyAttentionImpl"]:
        return KittyAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["KittyMetadataBuilder"]:
        return KittyMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [128]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or block_size == KITTY_PAGE_SIZE

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype == "kitty"

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.BLHNC,)

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return False

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        if spec.kv_quant_mode != KVQuantMode.KITTY or spec.state_content_bytes:
            return spec
        ratio = get_current_vllm_config().attention_config.kitty_boost_ratio
        d_boost = kitty_d_boost(spec.head_size, ratio)
        return replace(
            spec, state_content_bytes=kitty_state_content_bytes(spec.head_size, d_boost)
        )


@dataclass
class KittyMetadata(AttentionMetadata):
    # number of tokens to process in this step across all requests
    num_actual_tokens: int
    # The batch is ordered [decodes | continuation prefills | fresh prefills].
    num_decodes: int  # number of decode requests, i.e requests with query_len = 1
    num_continuations: int  # number of prefill continuation requests
    # number of fresh prefill requests (i.e without any existing cache)
    num_fresh_prefills: int
    num_decode_tokens: int  # must be equal to num_decodes; assert check
    num_continuation_tokens: int  # total number of tokens in continuation requests
    query_start_loc: torch.Tensor  # [num_reqs + 1] cumulative query offsets
    query_start_loc_cpu: torch.Tensor  # cpu version of the query_start_loc
    seq_lens_cpu: torch.Tensor  # cache length after this step
    old_lens: torch.Tensor  # [num_reqs] cache length before this step
    old_lens_cpu: np.ndarray  # cpu version of old_lens
    # block table for the packed pages only; residual pages are handled separately
    block_table: torch.Tensor
    k_jobs: torch.Tensor  # [Jk, 2] (request, block) K blocks sealed this step
    v_jobs: torch.Tensor  # [Jv, 2] (request, block) V blocks sealed in this step
    k_writes: torch.Tensor  # [3, n] (token, request, row) residual K writes
    v_writes: torch.Tensor  # [3, n]
    # [num_decodes] max(1, cdiv(old_len, 128)), chunks for split-KV decoding
    num_chunks: torch.Tensor
    # [num_fresh_prefills + 1] int32, cumulative tokens offsets for fresh
    # prefills, starting at 0
    fresh_cu_seqlens: torch.Tensor
    fresh_max_query_len: int  # maximum query lengths for a fresh prefill request


class KittyMetadataBuilder(AttentionMetadataBuilder[KittyMetadata]):
    # no cuda graph support in this implementation. a future PR could handle it.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # no speculative decoding for now
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        if is_workspace_manager_initialized():
            model_config = vllm_config.model_config
            num_heads = model_config.get_num_attention_heads(
                vllm_config.parallel_config
            )
            head_size = kv_cache_spec.head_size
            max_num_reqs = vllm_config.scheduler_config.max_num_seqs
            context = (
                model_config.max_model_len,
                kv_cache_spec.num_kv_heads,
                head_size,
            )
            # decode requests processed in a batch; so, get workspace memory for
            # each decode request
            current_workspace_manager().get_simultaneous(
                ((max_num_reqs, num_heads, NUM_SPLITS, head_size + 1), torch.float32),
                ((max_num_reqs, num_heads), torch.float32),
            )
            # continuation prefills processed sequentially, so enough to get
            # workspace memory for the largest possible continuation request
            current_workspace_manager().get_simultaneous(
                (context, model_config.dtype), (context, model_config.dtype)
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> KittyMetadata:
        cam = common_attn_metadata
        num_reqs = cam.num_reqs
        (
            num_decodes,
            num_continuations,
            num_fresh,
            num_decode_tokens,
            num_continuation_tokens,
            _,
        ) = split_decodes_prefills_and_extends(cam, decode_threshold=1)
        assert num_decode_tokens == num_decodes

        # cumulative query offsets, in CPU
        qsl_cpu = cam.query_start_loc_cpu[: num_reqs + 1]
        assert cam.seq_lens_cpu_upper_bound is not None
        # total cache lengths of all requests AFTER this step
        seq_lens_cpu = cam.seq_lens_cpu_upper_bound[:num_reqs]
        plan = plan_kitty_step(qsl_cpu.numpy(), seq_lens_cpu.numpy())

        # number of chunks for decode requests
        num_chunks = np.maximum(-(-plan.old_lens[:num_decodes] // KITTY_PAGE_SIZE), 1)
        fresh_start = num_reqs - num_fresh  # starting index of fresh prefill jobs
        # cumulative query offsets for fresh prefills
        fresh_qsl = qsl_cpu[fresh_start:].numpy()
        fresh_cu = fresh_qsl - fresh_qsl[0]  # rebase fresh offsets so they start at 0
        parts = [
            plan.old_lens,
            num_chunks,
            plan.k_jobs.ravel(),
            plan.v_jobs.ravel(),
            plan.k_writes.ravel(),
            plan.v_writes.ravel(),
            fresh_cu,
        ]
        # pack the attention metadata together to avoid multiple CPU -> GPU
        # transfers, then unpack as needed
        packed = torch.from_numpy(np.concatenate(parts).astype(np.int64))
        packed = packed.to(self.device, non_blocking=True)
        old_lens, num_chunks_t, k_jobs, v_jobs, k_writes, v_writes, fresh_cu_t = (
            packed.split([p.size for p in parts])
        )
        return KittyMetadata(
            num_actual_tokens=cam.num_actual_tokens,
            num_decodes=num_decodes,
            num_continuations=num_continuations,
            num_fresh_prefills=num_fresh,
            num_decode_tokens=num_decode_tokens,
            num_continuation_tokens=num_continuation_tokens,
            query_start_loc=cam.query_start_loc,
            query_start_loc_cpu=qsl_cpu,
            seq_lens_cpu=seq_lens_cpu,
            old_lens=old_lens,
            old_lens_cpu=plan.old_lens,
            block_table=cam.block_table_tensor,
            k_jobs=k_jobs.view(-1, 2),
            v_jobs=v_jobs.view(-1, 2),
            k_writes=k_writes.view(3, -1),
            v_writes=v_writes.view(3, -1),
            num_chunks=num_chunks_t,
            fresh_cu_seqlens=fresh_cu_t.int(),
            fresh_max_query_len=int(np.diff(fresh_qsl).max()) if num_fresh else 0,
        )


def kitty_continuation_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    packed_cache: torch.Tensor,
    residual: torch.Tensor,
    residual_block_ids: torch.Tensor,
    block_table: torch.Tensor,
    req: int,
    old_len: int,
    sm_scale: float,
    d_boost: int,
    fa_version: int | None,
) -> None:
    """Causal attention of request `req`'s chunk (query/output [q_len, H_Q, D],
    key/value [q_len, H_KV, D]) over its dequantized cache [0, old_len)
    followed by the chunk's own K/V. Currently, only works for a single
    continuation request."""
    q_len = query.shape[0]
    seq_len = old_len + q_len
    shape = (seq_len, *key.shape[1:])
    # either allocate from the workspace (if workspace was initialized), or
    # allocate tensors now
    k_full, v_full = _workspace((shape, key.dtype), (shape, key.dtype))
    kitty_dequant_context(
        packed_cache,
        residual,
        residual_block_ids,
        block_table,
        req,
        old_len,
        k_full,
        v_full,
        d_boost,
    )
    k_full[old_len:] = key  # these are the current input keys
    v_full[old_len:] = value  # current values
    # variable length flash attention needs sequence offsets; for a single
    # request, these are just [0, length]
    cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=query.device)
    cu_k = torch.tensor([0, seq_len], dtype=torch.int32, device=query.device)
    _flash_attn(
        query, k_full, v_full, output, cu_q, cu_k, q_len, seq_len, sm_scale, fa_version
    )


def _flash_attn(q, k, v, out, cu_q, cu_k, max_q, max_k, sm_scale, fa_version):
    kwargs = {} if fa_version is None else {"fa_version": fa_version}
    flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        softmax_scale=sm_scale,
        causal=True,
        out=out,
        **kwargs,
    )


class KittyAttentionImpl(AttentionImpl[KittyMetadata]):
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
        **kwargs,
    ) -> None:
        assert not logits_soft_cap, "Kitty does not support logits soft-cap."
        assert kv_sharing_target_layer_name is None, (
            "Kitty does not support KV sharing across layers."
        )
        assert alibi_slopes is None and sliding_window is None
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads or num_heads
        assert self.num_heads // self.num_kv_heads <= G_PAD, (
            f"Kitty supports at most {G_PAD} query heads per KV head."
        )
        self.kv_cache_dtype = kv_cache_dtype
        ratio = get_current_vllm_config().attention_config.kitty_boost_ratio
        self.d_boost = kitty_d_boost(head_size, ratio)
        self.fa_version = get_flash_attn_version(head_size=head_size)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: KittyMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None  # output buffer must already exist
        assert output_scale is None and output_block_scale is None
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        # slicing q, k, v, out here, in case those tensors are bigger than the
        # actual number of tokens
        md = attn_metadata
        n = md.num_actual_tokens
        q = query[:n]
        k = key[:n]
        v = value[:n]
        out = output[:n]
        res_layer = layer.kitty_residual  # type: ignore[attr-defined]
        residual = res_layer.kv_cache
        all_md = get_forward_context().attn_metadata
        assert isinstance(all_md, dict)
        res_md = cast(KittyResidualMetadata, all_md[res_layer.prefix])
        res_ids = res_md.residual_block_ids

        # 1. Attention on the cache as it was before this step.
        nd = md.num_decodes
        if nd:
            # batching all decode attention requests
            mid_o, lse = _workspace(
                ((nd, self.num_heads, NUM_SPLITS, self.head_size + 1), torch.float32),
                ((nd, self.num_heads), torch.float32),
            )
            kitty_decode_attention(
                q[:nd],
                k[:nd],
                v[:nd],
                out[:nd],
                kv_cache,
                residual,
                res_ids,
                md.block_table,
                md.old_lens,
                md.num_chunks,
                self.scale,
                self.d_boost,
                mid_o,
                lse,
            )
        # next, processing the continuation prefill requests sequentially
        qsl = md.query_start_loc_cpu
        for req in range(nd, nd + md.num_continuations):
            start, end = int(qsl[req]), int(qsl[req + 1])
            kitty_continuation_attention(
                q[start:end],
                k[start:end],
                v[start:end],
                out[start:end],
                kv_cache,
                residual,
                res_ids,
                md.block_table,
                req,
                int(md.old_lens_cpu[req]),
                self.scale,
                self.d_boost,
                self.fa_version,
            )
        # finally, batching the fresh prefills in one kernel
        if md.num_fresh_prefills:
            start = nd + md.num_continuations
            t0 = int(qsl[start])
            _flash_attn(
                q[t0:],
                k[t0:],
                v[t0:],
                out[t0:],
                md.fresh_cu_seqlens,
                md.fresh_cu_seqlens,
                md.fresh_max_query_len,
                md.fresh_max_query_len,
                self.scale,
                self.fa_version,
            )

        # 2. Seal completed pages, reading tails before they are overwritten.
        kitty_seal(
            k,
            v,
            kv_cache,
            residual,
            res_ids,
            md.block_table,
            md.k_jobs,
            md.v_jobs,
            md.old_lens,
            md.query_start_loc,
            self.d_boost,
        )
        # 3. Write the new tokens that stay unquantized.
        kitty_write_residual(k, v, residual, res_ids, md.k_writes, md.v_writes)
        return output


class KittyResidualBackend(AttentionBackend):
    """Owns the residual's KV-cache group; runs no attention."""

    @staticmethod
    def get_name() -> str:
        return "KITTY_RESIDUAL"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type["KittyResidualMetadataBuilder"]:
        return KittyResidualMetadataBuilder


@dataclass
class KittyResidualMetadata:
    residual_block_ids: torch.Tensor  # [num_reqs]


class KittyResidualMetadataBuilder(AttentionMetadataBuilder[KittyResidualMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> KittyResidualMetadata:
        cam = common_attn_metadata
        return KittyResidualMetadata(cam.block_table_tensor[: cam.num_reqs, 0])


class KittyResidualCache(torch.nn.Module, AttentionLayerBase):
    """Per-request fp16 sink, K tail ring and V tail ring of one Kitty layer.

    Bound as [num_blocks, H, 640, D]; the row map is in ops/kitty/tiles.py.
    """

    def __init__(
        self, num_kv_heads: int, head_size: int, dtype: torch.dtype, prefix: str
    ):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H, 320, 2D] -> [B, H, 640, D]
        self.kv_cache = kv_cache.view(
            kv_cache.shape[0], kv_cache.shape[1], KITTY_RESIDUAL_ROWS, self.head_size
        )

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        del vllm_config
        spec = AttentionSpec(
            block_size=KITTY_RESIDUAL_SLOTS,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size,
            dtype=self.dtype,
        )
        return CircularBufferManager.layer_config(spec)

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return KittyResidualBackend
