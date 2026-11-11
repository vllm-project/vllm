# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention for gfx1151 (opt-in).

Decodes of up to MAX_M query tokens per sequence run
csrc/rocm/rdna_decode_attn.cu, other batches
csrc/rocm/rdna_prefill_attn.cu, on TRITON_ATTN's KV cache. Anything
the kernels do not support runs TRITON_ATTN.
"""

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.rocm_rdna_hip import (
    MAX_M,
    PREFILL_MIN_M,
    SUPPORTED_HEAD_SIZES,
    KernelVariant,
    expected_kv_cache_strides,
    load,
    load_prefill,
    make_prefill_scratch,
    make_scratch,
    plan_prefill,
    prefill_batch_variant_for,
    prefill_target_workgroups,
    prefill_variant_for,
    prefill_variants,
    scratch_bytes,
    variant_for,
)
from vllm.v1.kv_cache_interface import KVCacheLayout, KVQuantMode

if TYPE_CHECKING:
    from vllm.platforms.interface import DeviceCapability

logger = init_logger(__name__)


def _on_gfx1151() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1151

    return on_gfx1151()


# A kernel and its scratch buffers.
_Built = tuple[Any, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]

# Split-KV scratch shared by all layers; each launch leaves the counters zeroed.
# A larger batch than the budget holds runs on Triton.
_SCRATCH: dict[tuple[KernelVariant, torch.device], tuple[int, Any]] = {}
_SCRATCH_BUDGET = 64 * 1024**2
_PREFILL_SCRATCH: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
# The decode kernel's multimodal ranges, a row [count, start0, end0, ...] per
# request, per device. The metadata builder rewrites them every step, so CUDA
# graphs replay the current ones.
MM_RANGES_CAPACITY = 32
_MM_RANGES: dict[tuple[torch.device, int], torch.Tensor] = {}


def mm_ranges_buffer(device: torch.device, rows: int) -> torch.Tensor:
    if (device, rows) not in _MM_RANGES:
        _MM_RANGES[device, rows] = torch.zeros(
            rows, 1 + 2 * MM_RANGES_CAPACITY, dtype=torch.int32, device=device
        )
    return _MM_RANGES[device, rows]


# Side stream for a mixed batch's decodes, per device.
_SIDE_STREAMS: dict[torch.device, torch.cuda.Stream] = {}


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    if device not in _SIDE_STREAMS:
        _SIDE_STREAMS[device] = torch.cuda.Stream(device)
    return _SIDE_STREAMS[device]


def mm_query_overlap(md: CommonAttentionMetadata) -> tuple[bool, int]:
    """How the step's query tokens sit in its multimodal ranges.

    Returns:
        (whether a query token's range has later keys, which only the
        bidirectional mask shows; how far back the farthest in-range query
        token reaches into its range, -1 if none is in one). Ranges are
        inclusive; start >= end is empty.

    """
    ranges = md.mm_req_doc_ranges
    if not ranges:
        return False, -1
    seq_lens = md.seq_lens_cpu_upper_bound
    if seq_lens is None:
        return True, 1 << 30
    starts = md.query_start_loc_cpu
    future_keys, reach = False, -1
    for req_idx, spans in ranges.items():
        end = int(seq_lens[req_idx])
        begin = end - int(starts[req_idx + 1] - starts[req_idx])
        for start, stop in spans:
            if start >= stop:
                continue
            # The in-range query tokens: max(start, begin) .. min(stop, end - 1).
            first, last = max(start, begin), min(stop, end - 1)
            if first <= last:
                future_keys |= first < last
                reach = max(reach, last - start)
    return future_keys, reach


def mm_decode_ranges(
    md: CommonAttentionMetadata,
) -> list[list[tuple[int, int]]] | None:
    """Per request the decode kernel may serve (at most MAX_M query tokens),
    its non-empty ranges that may hold one of its query tokens; none for the
    others, which run on the prefill kernel.

    The kernel only bounds the mask of query tokens, so the others change
    nothing. On async spec-decode rows seq_lens_cpu_upper_bound may overshoot
    by the rejected drafts, M - 1 while the draft count holds, so the span is
    widened by M - 1; at most 2M - 1 disjoint ranges then remain. None if the
    model has no ranges.
    """
    if md.mm_req_doc_ranges is None:
        return None
    starts = md.query_start_loc_cpu
    per_request: list[list[tuple[int, int]]] = [[] for _ in range(len(starts) - 1)]
    for i, ranges in md.mm_req_doc_ranges.items():
        m = int(starts[i + 1] - starts[i])
        if m > MAX_M:
            continue
        spans = [(s, e) for s, e in ranges if s < e]
        if md.seq_lens_cpu_upper_bound is not None:
            end = int(md.seq_lens_cpu_upper_bound[i])
            spans = [(s, e) for s, e in spans if s <= end - 1 and e >= end - 2 * m + 1]
        per_request[i] = spans
    return per_request


@dataclass
class RocmRdnaMetadata(TritonAttentionMetadata):
    """Triton's metadata plus the host's copy of the batch for the prefill
    plan."""

    query_start_loc_cpu: torch.Tensor | None = None
    # Upper bounds; they only balance the plan.
    seq_lens_cpu: torch.Tensor | None = None
    # Leading decodes of one query length.
    num_decodes: int = 0
    num_decode_tokens: int = 0
    # (window, first sequence) -> (build, host plan, device plan, head group,
    # query starts); the host plan outlives its non-blocking upload.
    plans: dict = field(default_factory=dict)
    # mm_query_overlap of the step.
    mm_future_keys: bool = False
    mm_reach: int = -1
    # The step's ranges for the decode kernel (mm_ranges_buffer); None if the
    # model has none or they do not fit.
    mm_ranges_decode: torch.Tensor | None = None


class RocmRdnaMetadataBuilder(TritonAttentionMetadataBuilder):
    """Triton's builder, decodes first and counted. CUDA graphs only for
    uniform decode batches: a prefill plan follows the host."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        if self.reorder_batch_threshold is not None:
            self.reorder_batch_threshold = min(self.reorder_batch_threshold, MAX_M)
        self._mm_rows = self.vllm_config.scheduler_config.max_num_seqs

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> RocmRdnaMetadata:
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        num_decodes = num_decode_tokens = 0
        if self.reorder_batch_threshold is not None:
            num_decodes, _, num_decode_tokens, _ = split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        future_keys, reach = mm_query_overlap(common_attn_metadata)
        return RocmRdnaMetadata(
            **{f.name: getattr(md, f.name) for f in fields(TritonAttentionMetadata)},
            query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu,
            seq_lens_cpu=common_attn_metadata.seq_lens_cpu_upper_bound,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            mm_future_keys=future_keys,
            mm_reach=reach,
            mm_ranges_decode=self._mm_ranges_decode(common_attn_metadata),
        )

    def _mm_ranges_decode(self, md: CommonAttentionMetadata) -> torch.Tensor | None:
        """Write mm_decode_ranges to mm_ranges_buffer, a row per request."""
        per_request = mm_decode_ranges(md)
        if per_request is None:
            return None
        buf = mm_ranges_buffer(self.device, self._mm_rows)
        if len(per_request) > buf.shape[0] or any(
            len(spans) > MM_RANGES_CAPACITY for spans in per_request
        ):
            # A replayed CUDA graph must not read an earlier step's ranges.
            buf[:, 0].fill_(0)
            return None
        rows = torch.zeros(len(per_request), buf.shape[1], dtype=torch.int32)
        for i, spans in enumerate(per_request):
            if spans:
                rows[i, 0] = len(spans)
                rows[i, 1 : 1 + 2 * len(spans)] = torch.tensor(spans).flatten()
        buf[: len(per_request)].copy_(rows, non_blocking=True)
        return buf


class RocmRdnaAttentionBackend(TritonAttentionBackend):
    """TRITON_ATTN with the RDNA3.5 kernels."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "ROCM_RDNA_ATTN"

    @staticmethod
    def get_impl_cls() -> type["RocmRdnaAttentionImpl"]:
        return RocmRdnaAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[RocmRdnaMetadataBuilder]:
        return RocmRdnaMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # Sizes without a built kernel run on Triton.
        return [32, 64, 80, 96, 128, 160, 192, 224, 256, 512]

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in cls.get_supported_head_sizes()

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_rswa(cls) -> bool:
        return False

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_compute_capability(cls, capability: "DeviceCapability") -> bool:
        return _on_gfx1151()

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # The kernels address a page as [H, N, K|V].
        return (KVCacheLayout.LBHNC,)


class RocmRdnaAttentionImpl(TritonAttentionImpl):
    """Runs the RDNA3.5 kernels when they support the call, else Triton."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._variant: KernelVariant | None = None
        self._built: _Built | None = None
        self._rejected: str | None = None
        # Call counters for tests and tools.
        self.kernel_calls = 0
        self.fallback_calls = 0
        self.prefill_calls = 0
        vllm_config = get_current_vllm_config_or_none()
        self._max_seqs = (
            vllm_config.scheduler_config.max_num_seqs if vllm_config is not None else 1
        )
        # Sequences the batched scratch holds.
        self._cap = 0

    def _reject(self, reason: str) -> None:
        """Log why this call falls back to Triton."""
        if self._rejected != reason:
            self._rejected = reason
            logger.info_once(
                "ROCM_RDNA_ATTN falling back to Triton: %s",
                reason,
                scope="local",
            )

    def _served_window(self, kwargs: dict) -> int | None:
        """The sliding window in keys (0: none), or None if unsupported."""
        if kwargs["alibi_slopes"] is not None or kwargs["sinks"] is not None:
            self._reject("alibi/sinks unsupported")
            return None
        if kwargs["softcap"] or not kwargs["causal"]:
            self._reject("softcap or non-causal unsupported")
            return None
        # vLLM passes a causal window of w keys as (w - 1, 0).
        window = kwargs["window_size"]
        win = 0
        if window is not None and window[0] >= 0:
            if window[1] != 0:
                self._reject(f"only causal sliding windows, got {window}")
                return None
            win = window[0] + 1
        if kwargs.get("rswa_prefix_lens") is not None:
            self._reject("rswa unsupported")
            return None
        if kwargs.get("chunk_lookback", -1) >= 0:
            self._reject("chunk lookback unsupported")
            return None
        # Not the descales: unquantized, they are a 1.0 broadcast.
        if kwargs["kv_quant_mode"] != KVQuantMode.NONE:
            self._reject(f"KV quant mode {kwargs['kv_quant_mode']!r} unsupported")
            return None
        # The kernels bound a multimodal range's earlier keys by the window, as
        # the clamp does. Unclamped, a query token farther than the window into
        # its range also sees the range's first keys.
        if (
            win
            and not kwargs.get("mm_prefix_clamp")
            and kwargs.get("mm_reach", -1) >= win
        ):
            self._reject("unclamped multimodal range reaches past the window")
            return None
        return win

    def _served_tensors(
        self, kv_cache: torch.Tensor, kwargs: dict, head_sizes: tuple[int, ...] | None
    ) -> bool:
        """Whether the kernels support these tensors."""
        dtype = kwargs["q"].dtype
        if dtype not in (torch.float16, torch.bfloat16):
            self._reject(f"kernel is fp16 or bf16 only, got {dtype}")
            return False
        # Q is strided when RoPE runs in place on the QKV projection.
        if not (kwargs["q"].is_contiguous() and kwargs["out"].is_contiguous()):
            self._reject("query or output is not contiguous")
            return False
        if kv_cache.dtype != dtype:
            self._reject(f"KV cache is {kv_cache.dtype}, query is {dtype}")
            return False
        if head_sizes is not None and self.head_size not in head_sizes:
            self._reject(f"head_size {self.head_size} not built")
            return False
        if self.num_heads % self.num_kv_heads:
            self._reject("q heads must divide evenly over kv heads")
            return False
        if kv_cache.shape[2] % 16:
            self._reject(f"block size {kv_cache.shape[2]} is not a multiple of 16")
            return False
        return True

    def _prepare(self, kv_cache: torch.Tensor, **kwargs) -> _Built | None:
        """The decode kernel and its scratch, or None to fall back."""
        win = self._served_window(kwargs)
        if win is None:
            return None
        # The decode kernel reads the ranges from mm_ranges_decode; without it,
        # causal is right unless a query token's range has later keys.
        if kwargs.get("mm_future_keys") and kwargs.get("mm_ranges_decode") is None:
            self._reject("decode: multimodal ranges with later keys do not fit")
            return None
        # One build per M.
        nseq = kwargs["seqused_k"].shape[0]
        max_query_len = kwargs["max_seqlen_q"]
        if kwargs["q"].shape[0] != nseq * max_query_len:
            self._reject("sequences of unequal query length")
            return None
        if max_query_len > MAX_M:
            self._reject(f"{max_query_len} query tokens, more than {MAX_M}")
            return None
        if not self._served_tensors(kv_cache, kwargs, SUPPORTED_HEAD_SIZES):
            return None

        # Logical KV cache order is (num_blocks, num_kv_heads, block_size, 2*hs).
        q = kwargs["q"]
        variant = variant_for(
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            max_query_len,
            kv_cache.shape[2],
            0 if kv_cache.stride(1) < kv_cache.stride(2) else 1,
            win,
            q.dtype,
            nseq > 1,
        )
        if variant is None:
            self._reject("no build in _rocm_C for this shape and layout")
            return None
        expected = expected_kv_cache_strides(variant)
        actual = (kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2))
        if actual != expected:
            self._reject(f"KV strides {actual} != {expected} for this layout")
            return None

        if self._variant != variant:
            module = load(variant)
            if module is None:
                self._reject(f"{variant.name} is not in this _rocm_C")
                return None
            key = (variant, q.device)
            if key not in _SCRATCH:
                cap = 1
                if variant.batched:
                    per_seq = scratch_bytes(variant)
                    cap = min(self._max_seqs, max(1, _SCRATCH_BUDGET // per_seq))
                _SCRATCH[key] = (
                    cap,
                    make_scratch(variant, q.device, cap if variant.batched else 0),
                )
            self._cap, scratch = _SCRATCH[key]
            self._built = (module, scratch)
            self._variant = variant
        if nseq > self._cap:
            self._reject(f"{nseq} sequences, the decode scratch holds {self._cap}")
            return None
        return self._built

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The kernels when they can serve the call, else Triton's forward."""
        if (
            attn_metadata is None
            or output_scale is not None
            or output_block_scale is not None
        ):
            # Profiling and quantized output: Triton's.
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        assert isinstance(attn_metadata, RocmRdnaMetadata)
        n = attn_metadata.num_actual_tokens
        args = {
            "q": query[:n],
            "out": output[:n],
            "seqused_k": attn_metadata.seq_lens,
            "block_table": attn_metadata.block_table,
            "max_seqlen_q": attn_metadata.max_query_len,
            "softmax_scale": self.scale,
            "causal": attn_metadata.causal,
            "window_size": self.sliding_window,
            "alibi_slopes": self.alibi_slopes,
            "sinks": self.sinks,
            "softcap": self.logits_soft_cap,
            "mm_prefix_range": attn_metadata.mm_prefix_range_tensor,
            "mm_future_keys": attn_metadata.mm_future_keys,
            "mm_reach": attn_metadata.mm_reach,
            "mm_ranges_decode": attn_metadata.mm_ranges_decode,
            "mm_prefix_clamp": getattr(layer, "mm_prefix_clamp_sliding_window", False),
            "rswa_prefix_lens": attn_metadata.rswa_prefix_lens,
            "chunk_lookback": self.chunk_lookback,
            "kv_quant_mode": self._kv_quant_mode,
        }
        nseq = attn_metadata.seq_lens.shape[0]
        max_query_len = attn_metadata.max_query_len
        if max_query_len <= MAX_M and n == nseq * max_query_len:
            # Decode: every sequence the same M.
            built = self._prepare(kv_cache, **args)
            if built is not None:
                self.kernel_calls += 1
                self._launch(built, kv_cache, args)
                return output
        elif self._serve_mixed(kv_cache, attn_metadata, args) or self._serve_prefill(
            kv_cache, attn_metadata, args
        ):
            self.kernel_calls += 1
            self.prefill_calls += 1
            return output
        self.fallback_calls += 1
        return super().forward(
            layer, query, key, value, kv_cache, attn_metadata, output
        )

    def _serve_mixed(
        self, kv_cache: torch.Tensor, md: TritonAttentionMetadata, kwargs: dict
    ) -> bool:
        """Leading decodes on the decode kernel, the rest on the prefill
        kernel; False if a kernel cannot take its part."""
        nd = getattr(md, "num_decodes", 0)
        ndt = getattr(md, "num_decode_tokens", 0)
        nseq = kwargs["seqused_k"].shape[0]
        if not 0 < nd < nseq or torch.cuda.is_current_stream_capturing():
            return False
        decodes, prefills = dict(kwargs), dict(kwargs)
        for k in ("q", "out"):
            decodes[k], prefills[k] = kwargs[k][:ndt], kwargs[k][ndt:]
        for k in ("seqused_k", "block_table"):
            decodes[k], prefills[k] = kwargs[k][:nd], kwargs[k][nd:]
        decodes["max_seqlen_q"] = ndt // nd
        built = self._prepare(kv_cache, **decodes)
        if built is None:
            return False
        launch_prefills = self._prepare_prefill(kv_cache, md, prefills, first=nd)
        if launch_prefills is None:
            return False
        # Decodes on a side stream fill the SIMDs a narrow prefill leaves.
        main = torch.cuda.current_stream()
        side = _side_stream(main.device)
        side.wait_stream(main)
        with torch.cuda.stream(side):
            self._launch(built, kv_cache, decodes)
        launch_prefills()
        main.wait_stream(side)
        return True

    def _serve_prefill(
        self, kv_cache: torch.Tensor, md: RocmRdnaMetadata, kwargs: dict
    ) -> bool:
        """Run the prefill kernel; False if it cannot serve the call."""
        launch = self._prepare_prefill(kv_cache, md, kwargs)
        if launch is None:
            return False
        launch()
        return True

    def _prepare_prefill(
        self,
        kv_cache: torch.Tensor,
        md: TritonAttentionMetadata,
        kwargs: dict,
        first: int = 0,
    ) -> Callable[[], None] | None:
        """The prefill launch for the sequences from `first` on, or None.

        Not under CUDA-graph capture: the launch depends on the host's M and S.
        """
        if torch.cuda.is_current_stream_capturing():
            return None
        win = self._served_window(kwargs)
        if win is None or not self._served_tensors(kv_cache, kwargs, None):
            return None
        block_size = kv_cache.shape[2]
        kv_row = 2 * self.head_size
        hnd = (block_size * self.num_kv_heads * kv_row, block_size * kv_row, kv_row)
        if (kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2)) != hnd:
            self._reject("prefill: KV cache is not the packed HND layout")
            return None
        q, out = kwargs["q"], kwargs["out"]
        m = q.shape[0]
        nseq = kwargs["seqused_k"].shape[0]
        # A batch, or fewer tokens than the first build serves: a plan.
        planned = nseq > 1 or m < PREFILL_MIN_M
        # Ranges whose query tokens see no later keys leave the mask causal.
        mm_ranges = kwargs["mm_prefix_range"] if kwargs.get("mm_future_keys") else None
        if mm_ranges is not None:
            mm_ranges = mm_ranges[first : first + nseq] if planned else mm_ranges[first]
        if planned:
            # The host's copy must be this step's.
            starts = getattr(md, "query_start_loc_cpu", None)
            if (
                starts is None
                or starts.shape[0] != first + nseq + 1
                or int(starts[-1]) - int(starts[first]) != m
            ):
                self._reject("prefill batch without the host's query starts")
                return None
        hq, hkv, d = self.num_heads, self.num_kv_heads, self.head_size
        if planned:
            variant, plan, head_group, starts = self._plan(
                cast(RocmRdnaMetadata, md), block_size, win, first, q
            )
        else:
            variant = prefill_variant_for(hq, hkv, d, block_size, 1, win, q.dtype, m)
        module = load_prefill(variant) if variant is not None else None
        if variant is None or module is None:
            self._reject(
                f"prefill of {m} query tokens: no build in _rocm_C for "
                f"{hq}/{hkv}/{d} window {win} page {block_size} {q.dtype}"
            )
            return None
        # Sized for every build of the shape.
        shape = (hq, hkv, d, block_size, win, q.dtype)
        scratch_key = (shape, q.device)
        scratch = _PREFILL_SCRATCH.get(scratch_key)
        if scratch is None or scratch[1].numel() < 2 * variant.scratch_rows(q.device):
            scratch = make_prefill_scratch(
                [*prefill_variants(*shape), variant], q.device
            )
            _PREFILL_SCRATCH[scratch_key] = scratch
        args = (
            q,
            kv_cache,
            kwargs["block_table"],
            out,
            *scratch,
            kwargs["seqused_k"],
        )
        if not planned:
            # Only sizes the KV split.
            return lambda: module.prefill_attn(
                *args, max(md.max_seq_len, m), kwargs["softmax_scale"], mm_ranges
            )
        if not plan.shape[0]:
            return lambda: None
        return lambda: module.prefill_attn(
            *args,
            md.max_seq_len,
            kwargs["softmax_scale"],
            mm_ranges,
            plan=plan,
            query_start_loc=starts,
            head_group=head_group,
        )

    def _plan(
        self,
        md: RocmRdnaMetadata,
        block_size: int,
        win: int,
        first: int,
        q: torch.Tensor,
    ) -> tuple[Any, torch.Tensor, int, torch.Tensor]:
        """This step's build, plan, head group and device query starts for the
        sequences from `first` on; made by the first layer, shared by the
        rest."""
        key = (win, first)
        cached = md.plans.get(key)
        if cached is None:
            assert md.query_start_loc_cpu is not None
            base = int(md.query_start_loc_cpu[first])
            starts = (md.query_start_loc_cpu[first:] - base).numpy()
            nseq = starts.shape[0] - 1
            seq_lens = (
                md.seq_lens_cpu[first : first + nseq].numpy()
                if md.seq_lens_cpu is not None
                else np.full(nseq, md.max_seq_len)
            )
            variant, head_group = prefill_batch_variant_for(
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                block_size,
                1,
                win,
                q.dtype,
                starts,
                seq_lens,
            )
            plan = device_plan = None
            if variant is not None:
                plan, head_group = plan_prefill(
                    starts,
                    seq_lens,
                    variant,
                    win,
                    prefill_target_workgroups(q.device),
                    head_group,
                )
                plan = torch.from_numpy(plan).pin_memory()
                device_plan = plan.to(q.device, non_blocking=True)
            cached = md.plans[key] = (
                variant,
                plan,
                device_plan,
                head_group,
                md.query_start_loc[first:] - base if first else md.query_start_loc,
            )
        variant, _, device_plan, head_group, device_starts = cached
        return variant, device_plan, head_group, device_starts

    def _launch(self, built: _Built, kv_cache: torch.Tensor, kwargs: dict) -> None:
        module, (acc, softmax_max, softmax_sum, arrivals) = built
        # Called directly: this runs under CUDA-graph capture.
        module.decode_attn(
            kwargs["q"],
            kv_cache,
            kwargs["block_table"],
            kwargs["out"],
            acc,
            softmax_max,
            softmax_sum,
            arrivals,
            # A device tensor, so CUDA graphs follow the sequence length.
            kwargs["seqused_k"],
            kwargs["softmax_scale"],
            kwargs.get("mm_ranges_decode"),
        )
