# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-sequence attention for gfx1151 (max_num_seqs=1, opt-in).

Calls of up to MAX_M query tokens run csrc/rocm/rdna_singleseq_decode_attn.cu, longer
ones csrc/rocm/rdna_singleseq_prefill_attn.cu, on TRITON_ATTN's KV cache. Anything the
kernels do not support runs TRITON_ATTN.
"""

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, ClassVar

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
from vllm.v1.attention.ops.rocm_rdna_singleseq_hip import (
    MAX_M,
    PREFILL_MIN_M,
    SUPPORTED_HEAD_SIZES,
    KernelVariant,
    expected_kv_cache_strides,
    load,
    load_prefill,
    make_prefill_scratch,
    make_scratch,
    prefill_variant_for,
    prefill_variants,
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
_SCRATCH: dict[tuple[KernelVariant, torch.device], Any] = {}
_PREFILL_SCRATCH: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
# The decode kernel's multimodal ranges, [count, start0, end0, ...] per
# device. The metadata builder rewrites them every step, so CUDA graphs replay
# the current ones.
MM_RANGES_CAPACITY = 32
_MM_RANGES: dict[torch.device, torch.Tensor] = {}


def mm_ranges_buffer(device: torch.device) -> torch.Tensor:
    if device not in _MM_RANGES:
        _MM_RANGES[device] = torch.zeros(
            1 + 2 * MM_RANGES_CAPACITY, dtype=torch.int32, device=device
        )
    return _MM_RANGES[device]


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


def mm_decode_ranges(md: CommonAttentionMetadata) -> list[tuple[int, int]] | None:
    """Request 0's non-empty ranges that may hold one of the step's query tokens.

    The kernel only bounds the mask of query tokens, so the others change
    nothing. On async spec-decode rows seq_lens_cpu_upper_bound may overshoot
    by the rejected drafts, M - 1 while the draft count holds, so the span is
    widened by M - 1; at most 2M - 1 disjoint ranges then remain. None if the
    model has no ranges.
    """
    if md.mm_req_doc_ranges is None:
        return None
    spans = [(s, e) for s, e in md.mm_req_doc_ranges.get(0, []) if s < e]
    if md.seq_lens_cpu_upper_bound is not None:
        end = int(md.seq_lens_cpu_upper_bound[0])
        m = int(md.query_start_loc_cpu[1] - md.query_start_loc_cpu[0])
        spans = [(s, e) for s, e in spans if s <= end - 1 and e >= end - 2 * m + 1]
    return spans


@dataclass
class RocmRdnaSingleSeqMetadata(TritonAttentionMetadata):
    # mm_query_overlap of the step.
    mm_future_keys: bool = False
    mm_reach: int = -1
    # The step's ranges for the decode kernel (mm_ranges_buffer); None if the
    # model has none or they do not fit.
    mm_ranges_decode: torch.Tensor | None = None


class RocmRdnaSingleSeqMetadataBuilder(TritonAttentionMetadataBuilder):
    # The decode kernel is built for a fixed M and ignores query_start_loc.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> RocmRdnaSingleSeqMetadata:
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        future_keys, reach = mm_query_overlap(common_attn_metadata)
        return RocmRdnaSingleSeqMetadata(
            **{f.name: getattr(md, f.name) for f in fields(md)},
            mm_future_keys=future_keys,
            mm_reach=reach,
            mm_ranges_decode=self._mm_ranges_decode(common_attn_metadata),
        )

    def _mm_ranges_decode(self, md: CommonAttentionMetadata) -> torch.Tensor | None:
        """Write mm_decode_ranges to mm_ranges_buffer."""
        spans = mm_decode_ranges(md)
        if spans is None:
            return None
        buf = mm_ranges_buffer(self.device)
        if len(spans) > MM_RANGES_CAPACITY:
            # A replayed CUDA graph must not read an earlier step's ranges.
            buf[:1].fill_(0)
            return None
        flat = [len(spans), *(x for span in spans for x in span)]
        buf[: len(flat)].copy_(torch.tensor(flat, dtype=torch.int32), non_blocking=True)
        return buf


class RocmRdnaSingleSeqAttentionBackend(TritonAttentionBackend):
    """TRITON_ATTN with the RDNA3.5 kernels; requires max_num_seqs=1."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "ROCM_RDNA_SINGLESEQ_ATTN"

    @staticmethod
    def get_impl_cls() -> type["RocmRdnaSingleSeqAttentionImpl"]:
        return RocmRdnaSingleSeqAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["RocmRdnaSingleSeqMetadataBuilder"]:
        return RocmRdnaSingleSeqMetadataBuilder

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
    def supports_combination(cls, *args, **kwargs) -> str | None:
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None or vllm_config.scheduler_config.max_num_seqs != 1:
            return "max_num_seqs must be 1"
        return None

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBHNC,)


class RocmRdnaSingleSeqAttentionImpl(TritonAttentionImpl):
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

    def _reject(self, reason: str) -> None:
        """Log why this call falls back to Triton."""
        if self._rejected != reason:
            self._rejected = reason
            logger.info_once(
                "ROCM_RDNA_SINGLESEQ_ATTN falling back to Triton: %s",
                reason,
                scope="local",
            )

    def _served_window(self, kwargs: dict) -> int | None:
        """The sliding window in keys (0: none), or None if unsupported."""
        nseq = kwargs["seqused_k"].shape[0]
        if nseq != 1:
            self._reject(f"{nseq} sequences in one batch, the kernels take one")
            return None
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
        max_query_len = kwargs["max_seqlen_q"]
        if kwargs["q"].shape[0] != max_query_len:
            self._reject("query tokens do not match max_query_len")
            return None
        if max_query_len > MAX_M:
            self._reject(f"{max_query_len} query tokens, more than {MAX_M}")
            return None
        if not self._served_tensors(kv_cache, kwargs, SUPPORTED_HEAD_SIZES):
            return None

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
                _SCRATCH[key] = make_scratch(variant, q.device)
            self._built = (module, _SCRATCH[key])
            self._variant = variant
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
        if (
            attn_metadata is None
            or output_scale is not None
            or output_block_scale is not None
        ):
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
        assert isinstance(attn_metadata, RocmRdnaSingleSeqMetadata)
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
        if n >= PREFILL_MIN_M:
            if self._serve_prefill(kv_cache, attn_metadata, args):
                self.kernel_calls += 1
                self.prefill_calls += 1
                return output
        else:
            built = self._prepare(kv_cache, **args)
            if built is not None:
                self.kernel_calls += 1
                self._launch(built, kv_cache, args)
                return output
        self.fallback_calls += 1
        return super().forward(
            layer, query, key, value, kv_cache, attn_metadata, output
        )

    def _serve_prefill(
        self, kv_cache: torch.Tensor, md: RocmRdnaSingleSeqMetadata, kwargs: dict
    ) -> bool:
        """Run the prefill kernel; False if it cannot serve the call.

        Not under CUDA-graph capture: the launch depends on the host's M and S.
        """
        if torch.cuda.is_current_stream_capturing():
            return False
        win = self._served_window(kwargs)
        if win is None or not self._served_tensors(kv_cache, kwargs, None):
            return False
        # Ranges whose query tokens see no later keys leave the mask causal.
        mm_ranges = kwargs["mm_prefix_range"] if kwargs["mm_future_keys"] else None
        block_size = kv_cache.shape[2]
        kv_row = 2 * self.head_size
        hnd = (block_size * self.num_kv_heads * kv_row, block_size * kv_row, kv_row)
        if (kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2)) != hnd:
            self._reject("prefill: KV cache is not the packed HND layout")
            return False
        q, out = kwargs["q"], kwargs["out"]
        m = q.shape[0]
        hq, hkv, d = self.num_heads, self.num_kv_heads, self.head_size
        variant = prefill_variant_for(hq, hkv, d, block_size, 1, win, q.dtype, m)
        module = load_prefill(variant) if variant is not None else None
        if variant is None or module is None:
            self._reject(
                f"prefill of {m} query tokens: no build in _rocm_C for "
                f"{hq}/{hkv}/{d} window {win} page {block_size} {q.dtype}"
            )
            return False
        shape = (hq, hkv, d, block_size, win, q.dtype)
        scratch_key = (shape, q.device)
        scratch = _PREFILL_SCRATCH.get(scratch_key)
        if scratch is None or scratch[1].numel() < 2 * variant.scratch_rows(q.device):
            scratch = make_prefill_scratch(
                [*prefill_variants(*shape), variant], q.device
            )
            _PREFILL_SCRATCH[scratch_key] = scratch
        module.prefill_attn(
            q,
            kv_cache,
            kwargs["block_table"],
            out,
            *scratch,
            kwargs["seqused_k"],
            # Only sizes the KV split.
            max(md.max_seq_len, m),
            kwargs["softmax_scale"],
            mm_ranges[0] if mm_ranges is not None else None,
        )
        return True

    def _launch(self, built: _Built, kv_cache: torch.Tensor, kwargs: dict) -> None:
        module, (acc, softmax_max, softmax_sum, arrivals) = built
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
