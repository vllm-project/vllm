# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer task-scheduled (``prims_ts``) paged decode.

Wraps ``flashinfer.attention.prims_ts.BatchDecodePagedTSWrapper`` for uniform
query-block decode: every request contributes exactly ``max_query_len`` query
tokens over a paged KV cache. That is the DFlash/DSpark drafter shape, and the
reason this backend exists is that ``mask_type="dense"`` expresses a
*non-causal* draft block directly -- the trtllm-gen paged decode kernel is
bottom-right causal and silently ignores a supplied mask.

Layout. No KV cache change: K and V are the two halves of vLLM's ordinary
``[pages, Hkv, page_size, 2 * D]`` page, so their token stride is ``2 * D``.
The kernel builds its TMA descriptors from the tensors' own strides
(``_validate_native_hnd_tma_strides``), so the only requirement is a
contiguous head_dim. Verified numerically equal to the same data repacked
into compact planes.

Sliding window. ``prims_ts`` rejects ``window_left >= 0`` unless the mask is
causal, so a non-causal windowed layer cannot be expressed directly. The query
block always sits at the end of the sequence, so the symmetric window reduces
to a suffix, and we hand the kernel a block table restricted to the window's
pages (see :func:`_restrict_to_window`). Because the restriction is page
aligned, each query token sees up to ``page_size - 1`` extra tokens at the old
edge, and all query tokens share one window start rather than each having its
own (``max_query_len - 1`` tokens of drift).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
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
    max_decode_query_len,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec

if TYPE_CHECKING:
    from vllm.config.cache import CacheDType
    from vllm.platforms.interface import DeviceCapability

logger = init_logger(__name__)

# flashinfer/attention/prims_ts/decode.py: _SUPPORTED_PAGE_SIZES,
# _SUPPORTED_HEAD_DIMS, _SUPPORTED_COMPUTE_CAPABILITIES. (10, 7) is SM107a /
# Rubin, which the kernel additionally gates on a CuTe DSL that can emit for
# the sm_100f family (>= 4.8, else CUTE_DSL_ARCH=sm_100f in the environment).
_TS_PAGE_SIZES = (16, 32, 64, 128)
_TS_HEAD_DIMS = (64, 128, 256)
_TS_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
_TS_MAX_HEAD_RATIO = 128


def _require_prims_ts():
    """Import the prims_ts wrapper, or explain which flashinfer is needed.

    0.7.0.post1 ships prims_ts but calls ``Task._run_task_body_impl`` with a
    ``context`` kwarg that cutlass-dsl 4.8.0 removed, so it raises TypeError
    deep inside kernel codegen.
    """
    try:
        from flashinfer.attention.prims_ts import BatchDecodePagedTSWrapper
    except ImportError as exc:
        raise RuntimeError(
            "FLASHINFER_TS needs flashinfer-python>=0.7.1 "
            "(flashinfer.attention.prims_ts is missing)."
        ) from exc
    import flashinfer

    if tuple(int(p) for p in flashinfer.__version__.split(".")[:3]) < (0, 7, 1):
        raise RuntimeError(
            f"FLASHINFER_TS needs flashinfer-python>=0.7.1, found "
            f"{flashinfer.__version__}: earlier builds of prims_ts are "
            "incompatible with cutlass-dsl>=4.8."
        )
    return BatchDecodePagedTSWrapper


def _ts_output_dtype(q_dtype: torch.dtype, model_dtype: torch.dtype) -> torch.dtype:
    """The output dtype the kernel can write for this query dtype.

    An FP8 query only yields FP16 or FP8 out, so a bf16 model stages through
    FP16 and converts; every other combination writes the model dtype directly.
    """
    if q_dtype is torch.float8_e4m3fn and model_dtype is not torch.float16:
        return torch.float16
    return model_dtype


def _kv_kernel_dtype(spec: AttentionSpec) -> torch.dtype:
    """An FP8 cache is stored as uint8 and viewed as e4m3 at use."""
    if spec.dtype is torch.uint8:
        return current_platform.fp8_dtype()
    return spec.dtype


@dataclass
class FlashInferTSMetadata(AttentionMetadata):
    # Per-request K/V lengths the kernel reads, already restricted to the
    # window when one applies.
    seq_lens: torch.Tensor
    # [B, C] int32 page ids, likewise restricted.
    block_table: torch.Tensor
    num_reqs: int
    num_actual_tokens: int
    query_len: int
    # Plan bound at build time; the same object must be replayed by a captured
    # graph, which is why plans are keyed only on (batch size, query length).
    wrapper: Any


class FlashInferTSMetadataBuilder(AttentionMetadataBuilder[FlashInferTSMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert isinstance(kv_cache_spec, AttentionSpec)
        self.page_size = kv_cache_spec.block_size
        self.num_kv_heads = kv_cache_spec.num_kv_heads
        self.head_size = kv_cache_spec.head_size
        # Q and K/V dtypes are kept matched: FP8 query against the FP8 cache,
        # or model-dtype against an unquantized one. The FP8 query is produced
        # upstream (MiniMax-M3's fused QK-norm/RoPE kernel writes it through
        # q_fp8_out), so the attention layer does not quantize it again.
        self.kv_dtype = _kv_kernel_dtype(kv_cache_spec)
        self.q_dtype = self.kv_dtype
        self.out_dtype = vllm_config.model_config.dtype
        self.num_qo_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        # FullAttentionSpec records the window when the hybrid KV cache manager
        # is disabled; SlidingWindowSpec carries it directly.
        self.sliding_window: int | None = getattr(kv_cache_spec, "sliding_window", None)
        self.query_len = max_decode_query_len(vllm_config)
        self._plans: dict[tuple[int, int, bool], Any] = {}
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        # CUDA-graph-safe metadata. This builder advertises UNIFORM_BATCH, so
        # the drafter's attention is captured and replayed; the kernel records
        # the ADDRESS of seq_lens and the page table at capture time. Building
        # fresh tensors per step would leave a replayed graph reading freed
        # memory, so every build writes into these buffers in place.
        max_reqs = max(
            vllm_config.scheduler_config.max_num_seqs,
            vllm_config.scheduler_config.max_num_batched_tokens
            // max(self.query_len, 1),
        )
        self._buf_seq_lens = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._buf_block_table = torch.zeros(
            (max_reqs, self._window_pages(self.query_len)),
            dtype=torch.int32,
            device=device,
        )

    def _window_pages(self, query_len: int) -> int:
        """Page-table width a restricted (windowed) batch needs."""
        if self.sliding_window is None:
            return cdiv(self.vllm_config.model_config.max_model_len, self.page_size)
        return cdiv(self.sliding_window + query_len - 1, self.page_size) + 1

    def _max_kv_len(self) -> int:
        """Constant K/V capacity every plan compiles for.

        Keeping it constant is what makes one plan per (batch size, query
        length) safe to replay from a CUDA graph: ``build()`` can never pick a
        differently-compiled plan for a batch the graph already captured.
        """
        if self.sliding_window is not None:
            # _restrict_to_window leaves at most one partial page in front of
            # the window plus the query block itself.
            return self.sliding_window + self.query_len + self.page_size
        return self.vllm_config.model_config.max_model_len

    def _get_plan(self, num_reqs: int, query_len: int, causal: bool) -> Any:
        key = (num_reqs, query_len, causal)
        plan = self._plans.get(key)
        if plan is not None:
            return plan

        BatchDecodePagedTSWrapper = _require_prims_ts()

        window = self.sliding_window
        wrapper = BatchDecodePagedTSWrapper(kv_layout="HND")
        wrapper.plan(
            device=self.device,
            batch_size=num_reqs,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_size,
            page_size=self.page_size,
            max_kv_len=self._max_kv_len(),
            max_seq_len_q=query_len,
            packed_query=False,
            q_data_type=self.q_dtype,
            kv_data_type=self.kv_dtype,
            o_data_type=_ts_output_dtype(self.q_dtype, self.out_dtype),
            # A dense plan is what makes a non-causal draft block expressible;
            # it cannot carry a window, which _restrict_to_window applies to
            # the page table instead.
            mask_type="causal" if causal else "dense",
            window_left=window - 1 if (causal and window is not None) else -1,
        )
        self._plans[key] = wrapper
        return wrapper

    def _restrict_to_window(
        self, seq_lens: torch.Tensor, block_table: torch.Tensor, query_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cut the page table down to the pages the window can reach.

        The query block ends the sequence, so a symmetric window around it is
        the suffix ``[L - w - query_len + 1, L)``. Round that start down to a
        page so the slice is expressible as a page table, then renumber the
        lengths relative to the new first page.
        """
        assert self.sliding_window is not None
        span = self.sliding_window + query_len - 1
        num_pages = self._buf_block_table.shape[1]
        num_reqs = seq_lens.shape[0]

        start_page = torch.clamp_min(
            (seq_lens - span).div(self.page_size, rounding_mode="floor"), 0
        )
        cols = torch.arange(num_pages, device=block_table.device, dtype=torch.int32)
        idx = start_page.unsqueeze(1).to(torch.int64) + cols.unsqueeze(0)
        idx = idx.clamp_max(block_table.shape[1] - 1)

        assert num_reqs <= self._buf_seq_lens.shape[0], (
            f"FLASHINFER_TS metadata buffer holds {self._buf_seq_lens.shape[0]} "
            f"requests, got {num_reqs}"
        )
        out_lens = self._buf_seq_lens[:num_reqs]
        out_table = self._buf_block_table[:num_reqs]
        out_lens.copy_(seq_lens - start_page.to(seq_lens.dtype) * self.page_size)
        out_table.copy_(block_table.gather(1, idx))
        return out_lens, out_table

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashInferTSMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        causal = common_attn_metadata.causal
        if isinstance(causal, torch.Tensor):
            raise NotImplementedError(
                "FLASHINFER_TS has no per-sequence causality; it plans one "
                "mask for the whole batch."
            )
        # The drafter contributes the same number of query tokens per request,
        # so the block length is the quotient; max_query_len is only an upper
        # bound. Startup profiling also builds ragged batches that no real
        # draft step produces (e.g. 5 tokens over 2 requests). Rounding down
        # covers a whole number of blocks and leaves the remainder untouched,
        # which keeps those builds in bounds instead of failing the engine.
        query_len = num_tokens // num_reqs if num_reqs else 0
        num_tokens = num_reqs * query_len
        # The model runner builds metadata for every KV cache group on every
        # batch, so this is also called with the target's prefill shapes for
        # the drafter's own group -- metadata the speculator never uses, since
        # it builds its own. A drafter block is never longer than
        # max_decode_query_len, so anything wider is not a batch this backend
        # runs; planning it would compile a prefill-shaped plan and fail
        # validation (seq_len_q > max_kv_len).
        if query_len > self.query_len:
            query_len = 0

        seq_lens = common_attn_metadata.seq_lens[:num_reqs]
        block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        if self.sliding_window is not None and not causal:
            seq_lens, block_table = self._restrict_to_window(
                seq_lens, block_table, query_len
            )
        else:
            num_pages = min(block_table.shape[1], self._buf_block_table.shape[1])
            out_lens = self._buf_seq_lens[:num_reqs]
            out_table = self._buf_block_table[:num_reqs, :num_pages]
            out_lens.copy_(seq_lens)
            out_table.copy_(block_table[:, :num_pages])
            seq_lens, block_table = out_lens, out_table

        if query_len == 0:
            return FlashInferTSMetadata(
                seq_lens=common_attn_metadata.seq_lens[:num_reqs],
                block_table=common_attn_metadata.block_table_tensor[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=0,
                query_len=0,
                wrapper=None,
            )

        return FlashInferTSMetadata(
            seq_lens=seq_lens,
            block_table=block_table,
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            query_len=query_len,
            wrapper=self._get_plan(num_reqs, query_len, causal),
        )


class FlashInferTSImpl(AttentionImpl[FlashInferTSMetadata]):
    forward_includes_kv_cache_update: bool = False

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
        if sinks is not None:
            raise NotImplementedError("FLASHINFER_TS does not support attention sinks.")
        if alibi_slopes is not None:
            raise NotImplementedError("FLASHINFER_TS does not support alibi.")
        if logits_soft_cap:
            raise NotImplementedError("FLASHINFER_TS does not support softcap.")
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("FLASHINFER_TS does not support KV sharing.")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                f"FLASHINFER_TS does not support attention type {attn_type}."
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads or num_heads
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        # The kernel wants the query in the cache's dtype. For FP8 the caller
        # supplies it already quantized (query_pre_quantized on the layer), so
        # advertise support rather than making the layer quantize a second time.
        self.supports_quant_query_input = kv_cache_dtype.startswith("fp8")

    def _kv_views(self, kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """HND K/V views of vLLM's ``[pages, Hkv, page, 2 * D]`` page.

        Their token stride is ``2 * D`` rather than ``D``; the kernel builds its
        TMA descriptors from the tensor's own strides, so no repacking and no
        layout change are needed.
        """
        if self.kv_cache_dtype.startswith("fp8"):
            kv_cache = kv_cache.view(current_platform.fp8_dtype())
        return kv_cache[..., : self.head_size], kv_cache[..., self.head_size :]

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        from vllm._custom_ops import reshape_and_cache_flash

        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferTSMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            # Profiling run: the KV cache is not bound yet.
            return output
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "FLASHINFER_TS does not support fused output quantization."
            )

        n = attn_metadata.num_actual_tokens
        if n == 0:
            return output
        b, sq = attn_metadata.num_reqs, attn_metadata.query_len
        k_cache, v_cache = self._kv_views(kv_cache)
        q = query[:n].view(b, sq, self.num_heads, self.head_size)

        out = output[:n].view(b, sq, self.num_heads, self.head_size)
        kernel_dtype = _ts_output_dtype(q.dtype, out.dtype)
        needs_cast = kernel_dtype is not out.dtype
        kernel_out = torch.empty_like(out, dtype=kernel_dtype) if needs_cast else out

        attn_metadata.wrapper.run(
            q=q,
            paged_kv_cache=(k_cache, v_cache),
            seq_lens=attn_metadata.seq_lens,
            block_tables=attn_metadata.block_table,
            bmm1_scale=self.scale * layer._q_scale_float * layer._k_scale_float,
            bmm2_scale=layer._v_scale_float,
            out=kernel_out,
            validate=False,
        )
        if needs_cast:
            out.copy_(kernel_out)
        return output


class FlashInferTSBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list["CacheDType"]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_TS"

    @staticmethod
    def get_impl_cls() -> type[FlashInferTSImpl]:
        return FlashInferTSImpl

    @staticmethod
    def get_builder_cls() -> type[FlashInferTSMetadataBuilder]:
        return FlashInferTSMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return list(_TS_HEAD_DIMS)

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec=None,
    ) -> list[int | MultipleOf]:
        return list(_TS_PAGE_SIZES)

    @classmethod
    def supports_sliding_window(cls) -> bool:
        # Causal layers use the kernel's own window_left; non-causal ones get a
        # page-restricted block table (see the module docstring).
        return True

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_compute_capability(cls, capability: "DeviceCapability") -> bool:
        return (capability.major, capability.minor) in _TS_CAPABILITIES

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: "CacheDType | None",
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: "DeviceCapability",
    ) -> str | None:
        if use_mla or use_sparse or use_mm_prefix:
            return "FLASHINFER_TS supports dense non-MLA attention only"
        if has_sink:
            return "FLASHINFER_TS does not support attention sinks"
        return None
