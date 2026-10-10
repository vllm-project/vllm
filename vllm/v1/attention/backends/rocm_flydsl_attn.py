# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL paged attention backend for gfx950 (CDNA4).

This backend is a hybrid, and selecting it does not mean every attention call
runs a FlyDSL kernel. Causal prefill-shaped batches go to the FlyDSL
``DUALWAVE_SWP`` gfx950 kernel through ``kernels.attention.
flash_attn_interface.flydsl_flash_attn_func``; everything else, decode
included, is served by the AITER kernel inherited from
:class:`RocmAiterUnifiedAttentionBackend`. Inheriting also means the KV-cache
layout, metadata builder and KV-cache update path are shared, so the two
backends are interchangeable at the selector level.

Two things about FlyDSL make this backend different from every other entry in
the registry, and both are load-bearing:

* **The kernels are not packaged.** The published ``flydsl`` wheel contains the
  DSL compiler and runtime only; the attention kernels live in the FlyDSL
  source checkout under a top-level ``kernels`` package that no wheel installs.
  ``_resolve_flydsl_flash_attn`` therefore searches for it, and
  ``VLLM_FLYDSL_KERNELS_PATH`` points at a checkout root when it is not already
  importable.
* **AITER's ``flydsl_flash_attn_func`` is not the same kernel.**
  ``aiter.ops.flydsl.fmha_kernels`` wraps ``flash_attn_func_gfx1201``, an RDNA4
  kernel built on ``llvm.amdgcn.wmma.*`` intrinsics that do not exist on CDNA.
  Its ``gfx1201`` check is a real capability gate, not an allowlist, so this
  backend deliberately does not go through it.

Decode is routed away because the dualwave kernel tiles ``BLOCK_M=256`` over the
query dimension, so a single-token query would waste 255/256 of every tile. The
predicate is ``attn_metadata.max_query_len``, a batch-level scalar, chosen
because it is constant for any one CUDA-graph capture key (full graphs are
captured for uniform single-token decode only), so the branch a graph records is
the branch its replays need.

Being batch-level, it sends whole mixed chunked-prefill batches to FlyDSL,
single-token rows included. Those rows are correct -- per-request lengths come
from ``cu_seqlens_q`` -- but each spends a mostly empty tile. Splitting them out
would need the ``reorder_batch_threshold`` plus ``split_decodes_and_prefills``
machinery, which reorders batches for the inherited AITER path as well.
"""

import math
import os
import sys
from typing import ClassVar

import torch

import vllm.envs as envs
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionType, MultipleOf
from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
    RocmAiterUnifiedAttentionBackend,
    RocmAiterUnifiedAttentionImpl,
)
from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadata

logger = init_logger(__name__)

# Page sizes the gfx950 dualwave paged path accepts. It walks KV in 64-token
# tiles, and a page must be a whole number of them; 128 is what vLLM serves
# MiniMax-M3 with.
FLYDSL_SUPPORTED_PAGE_SIZES: tuple[int, ...] = (64, 128)

# Head dims the paged path compiles for.
FLYDSL_SUPPORTED_HEAD_SIZES: tuple[int, ...] = (64, 128)

# The kernel caches the block table in LDS, 2048 tile entries per split, and
# runs with num_kv_splits=1 here.
FLYDSL_MAX_SEQ_LEN: int = 2048 * 64

# Shortest context worth launching the dualwave kernel for.
#
# Its runtime is flat over the short end (0.124ms at 1024, 0.127ms at 2048 for
# 8x1024) because fixed setup dominates before the KV walk gets long enough to
# matter, while the split-KV AITER kernel tracks context down (0.092 -> 0.179).
# Measured 0.74x at 1024, 0.98x at 1408, crossing to 1.09x at 1536.
FLYDSL_MIN_SEQ_LEN: int = 1536

# Decode rows tolerated in an otherwise prefill batch, per unit of context.
#
# A single-token row occupies a whole q-block workgroup and still streams its
# row's entire KV, so FlyDSL's cost grows linearly in the count while AITER's
# stays flat. AITER pays for that KV too, and the deeper the walk the less its
# splitting buys, so the tolerable count rises with context: at ~8k, 8 rows
# measure 1.11x and 16 measure 0.82x; at ~60k, 15 rows still measure 1.29x.
# A flat cap tuned at 8k therefore gives up the long-context case.
FLYDSL_MAX_DECODE_ROWS_BASE: int = 8
FLYDSL_DECODE_CTX_UNIT: int = 8192

# Query length at or below which the parent AITER kernel is used instead. The
# dualwave kernel's BLOCK_M is 256, so decode-shaped batches would run it at
# 1/256 tile occupancy.
FLYDSL_MIN_QUERY_LEN: int = 2

# M-dimension tile of the dualwave kernel.
FLYDSL_BLOCK_M: int = 256

# Fewest q-block workgroups worth launching the dualwave kernel for.
#
# The grid is (num_heads_q, ceil(max_query_len / BLOCK_M), num_reqs) and every
# workgroup streams its row's whole KV, so a batch with few q-blocks leaves most
# of the 256 CUs idle while each of the survivors walks the full context. The
# kernel has no split-K on this path to widen the grid (paged varlen rejects
# num_kv_splits > 1), so below this many q-block rows it is latency-bound and the
# AITER kernel, which does split KV, is faster: measured 0.25x-0.96x against it
# from 16 to 32 rows, crossing over to 1.31x at 48.
#
# The test is on num_heads_q * num_actual_tokens rather than on the grid product
# because a row shorter than BLOCK_M only partly fills its tile -- 4 rows of 128
# tokens make the same 64-workgroup grid as 1 row of 1024 but a quarter of the
# work, and measured 0.89x where the 1024 measures 1.71x. Counting tokens
# separates every shape swept; counting grid rows does not.
FLYDSL_MIN_QBLOCK_ROWS: int = 48


def _max_decode_rows(max_seq_len: int) -> int:
    """Decode rows the kernel stays ahead of at this context depth."""
    return FLYDSL_MAX_DECODE_ROWS_BASE * max(1, max_seq_len // FLYDSL_DECODE_CTX_UNIT)


def _resolve_flydsl_flash_attn():
    """Import ``flydsl_flash_attn_func`` from a FlyDSL source checkout.

    Search order, first hit wins:

    1. ``VLLM_FLYDSL_KERNELS_PATH`` — a FlyDSL checkout root (the directory
       holding ``kernels/attention/``), loaded straight off that path rather
       than through ``sys.path``. Ordering cannot express what this variable
       means: appending loses to an installed ``kernels`` package, which is
       how a stale install silently answers for a checkout under test, while
       prepending would let a checkout root's generic top-level names
       (``lib``, ``tools``, ``tests``) shadow unrelated imports. Loading the
       file by location wins for this one module and shadows nothing.
    2. ``kernels.attention.flash_attn_interface`` already importable.

    Raises ImportError with both attempts named, so a misconfigured path is
    distinguishable from an absent one.
    """
    tried: list[str] = []

    path = envs.VLLM_FLYDSL_KERNELS_PATH
    if path:
        root = os.path.abspath(os.path.expanduser(path))
        pkg_dir = os.path.join(root, "kernels", "attention")
        if not os.path.isdir(pkg_dir):
            tried.append(
                f"VLLM_FLYDSL_KERNELS_PATH={path!r} has no kernels/attention/ "
                "subdirectory"
            )
        else:
            # The interface module imports its siblings as `kernels.attention.*`,
            # so the checkout has to be importable under that name for the load
            # to resolve; put it first only for the duration of the import.
            sys.path.insert(0, root)
            try:
                stale = sys.modules.get("kernels")
                stale_file = str(getattr(stale, "__file__", "") or "")
                if stale is not None and not stale_file.startswith(root):
                    # An installed `kernels` got imported first; drop it so the
                    # explicitly requested checkout is what actually loads.
                    for mod in [m for m in sys.modules if m.split(".")[0] == "kernels"]:
                        del sys.modules[mod]
                from kernels.attention.flash_attn_interface import (  # noqa: PLC0415
                    flydsl_flash_attn_func,
                )

                return flydsl_flash_attn_func
            except ImportError as exc:
                tried.append(f"VLLM_FLYDSL_KERNELS_PATH={path!r} ({exc})")
            finally:
                if sys.path and sys.path[0] == root:
                    del sys.path[0]

    try:
        from kernels.attention.flash_attn_interface import (  # noqa: PLC0415
            flydsl_flash_attn_func,
        )
    except ImportError as exc:
        tried.append("import kernels.attention.flash_attn_interface")
        raise ImportError(
            "ROCM_FLYDSL_DUALWAVE_ATTN needs the FlyDSL attention kernels, which the "
            "flydsl wheel does not ship (it carries the DSL compiler and "
            "runtime only). Point VLLM_FLYDSL_KERNELS_PATH at a FlyDSL "
            f"checkout root. Attempts: {'; '.join(tried)}. Last error: {exc}"
        ) from exc

    return flydsl_flash_attn_func


def flydsl_attention_available() -> bool:
    """Whether this build can run the FlyDSL gfx950 attention kernels."""
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        return False
    try:
        _resolve_flydsl_flash_attn()
    except ImportError as exc:
        logger.debug_once("FlyDSL attention unavailable: %s", exc)
        return False
    return True


class RocmFlyDSLAttentionBackend(RocmAiterUnifiedAttentionBackend):
    """Same KV-cache ABI as ROCM_AITER_UNIFIED_ATTN, different kernel.

    Inheriting is what makes this a swap rather than a rewrite: the KV cache
    shape ``(num_blocks, num_kv_heads, block_size, 2 * head_size)``, the
    ``RocmAttentionMetadataBuilder``, and the ``reshape_and_cache_flash`` /
    fused RoPE update path are all shared, so switching between the two
    backends does not re-lay-out the cache.
    """

    # No fp8: the FlyDSL paged path compiles for bf16/f16 KV only. Its dense
    # path takes fp8 descales, the paged one does not.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "ROCM_FLYDSL_DUALWAVE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["RocmFlyDSLAttentionImpl"]:
        return RocmFlyDSLAttentionImpl

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return list(FLYDSL_SUPPORTED_PAGE_SIZES)

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        # Not MultipleOf semantics: the page size is a compile-time constant of
        # the kernel and only these two are built, so 256 is *not* acceptable
        # even though it is a multiple of 64.
        return block_size is None or block_size in FLYDSL_SUPPORTED_PAGE_SIZES

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        if cls.supports_block_size(default_block_size):
            return default_block_size
        return max(FLYDSL_SUPPORTED_PAGE_SIZES)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return list(FLYDSL_SUPPORTED_HEAD_SIZES)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in FLYDSL_SUPPORTED_HEAD_SIZES

    @classmethod
    def supports_sink(cls) -> bool:
        # The dualwave kernel has no sink slot; the parent claims support
        # because the AITER Triton kernel does.
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        # Partial-multimodal full attention needs the non-causal path, which
        # here is the AITER fallback rather than FlyDSL. Decline it so the
        # selector never picks this backend for a model that needs it.
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        # Decoder only: the encoder paths in the parent bypass the paged cache
        # entirely, so routing them here would gain nothing.
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_combination(cls, *args, **kwargs) -> str | None:
        parent = super().supports_combination(*args, **kwargs)
        if parent is not None:
            return parent
        from vllm.platforms.rocm import on_gfx950

        if not on_gfx950():
            return "ROCM_FLYDSL_DUALWAVE_ATTN requires gfx950 (CDNA4)"
        try:
            _resolve_flydsl_flash_attn()
        except ImportError as exc:
            return f"FlyDSL attention kernels not importable: {exc}"
        return None


class RocmFlyDSLAttentionImpl(RocmAiterUnifiedAttentionImpl):
    """Causal decoder attention on the FlyDSL gfx950 dualwave kernel."""

    def __init__(self, *args, **kwargs) -> None:
        # The parent binds the AITER Triton kernel, which stays as the
        # fallback for decode-shaped and non-causal batches.
        super().__init__(*args, **kwargs)
        self.flydsl_flash_attn_func = _resolve_flydsl_flash_attn()
        # cu_seqlens_kv is the one argument the FlyDSL varlen paged ABI needs
        # that RocmAttentionMetadata does not carry. Built into a persistent
        # buffer so replay reads the values the current step wrote, and so the
        # hot path allocates nothing.
        self._cu_seqlens_kv: torch.Tensor | None = None
        # flydsl_flash_attn_func takes no softmax_scale: the kernel computes
        # rsqrt(head_dim) itself. Any layer with a different scale (logit
        # softcapping variants, MiniMax-M2 style scaled attention) must not go
        # to FlyDSL until the entry point grows the argument.
        self._scale_is_default = math.isclose(
            float(self.scale), self.head_size**-0.5, rel_tol=1e-6
        )
        # Whether this layer's KV cache views satisfy the paged ABI. Decided on
        # the first forward, when a real cache tensor is in hand; the layout is
        # a property of the backend and the config, so it never changes after.
        self._layout_ok: bool | None = None
        # causal_lpt is a build flag, so each value is a separately JIT-compiled
        # module. Compiling one inside a graph capture would capture the compile,
        # so both are forced eagerly on the first forward that is not capturing.
        self._lpt_variants_built: set[bool] = set()
        logger.info_once(
            "Using FlyDSL gfx950 dualwave attention for "
            "RocmFlyDSLAttentionImpl (page sizes %s, query_len >= %d; "
            "shorter queries and non-causal batches use the AITER kernel)",
            FLYDSL_SUPPORTED_PAGE_SIZES,
            FLYDSL_MIN_QUERY_LEN,
        )

    def _cu_seqlens_kv_for(self, seq_lens: torch.Tensor) -> torch.Tensor:
        """Exclusive prefix sum of ``seq_lens``, in a persistent buffer.

        ``torch.cumsum`` with ``out=`` is a plain device kernel reading the
        persistent ``attn_metadata.seq_lens``, so capturing it records real GPU
        work that re-runs against fresh sequence lengths on every replay.

        Element 0 is written once, at allocation. Re-zeroing it per call reads
        as harmless but is a host-to-device scalar copy, which CUDA-graph
        capture rejects outright ("Cannot copy between CPU and CUDA tensors
        during CUDA graph capture") — the buffer is grown, never re-zeroed.
        """
        n = seq_lens.shape[0]
        buf = self._cu_seqlens_kv
        if buf is None or buf.shape[0] < n + 1 or buf.device != seq_lens.device:
            buf = torch.zeros(n + 1, dtype=torch.int32, device=seq_lens.device)
            self._cu_seqlens_kv = buf
        out = buf[: n + 1]
        torch.cumsum(seq_lens, 0, dtype=torch.int32, out=out[1:])
        return out

    def _kv_layout_supported(self, kv_cache: torch.Tensor) -> bool:
        """Whether the FlyDSL paged ABI can address this KV cache in place.

        The kernel computes ``page_base + token * stride_kv_n + head * D``:
        the token stride is a runtime argument but the head stride is baked to
        ``D``, and the page stride is assumed to be ``page_size * stride_kv_n``.
        vLLM's cache is logically ``(num_blocks, num_kv_heads, block_size,
        2 * head_size)``, so a K view has head stride ``block_size * 2 *
        head_size`` and page stride ``num_kv_heads * block_size * 2 *
        head_size``. Both disagree with the kernel's assumptions unless
        ``num_kv_heads == 1``, which is the case that never reads them — and is
        what MiniMax-M3 serves at TP=4 (4 KV heads over 4 ranks).

        Rather than assert ``num_kv_heads == 1``, check the strides the kernel
        actually constrains, so this starts accepting multi-head caches the day
        ``flydsl_flash_attn_func`` grows runtime head and page strides.
        """
        if self._layout_ok is not None:
            return self._layout_ok
        k, v = self._split_kv_cache(kv_cache)
        page_size = k.shape[1]
        sk, sv = k.stride(), v.stride()
        ok = (
            sk[3] == 1
            and sv[3] == 1
            and sk[1] == sv[1]
            and sk[0] == page_size * sk[1]
            and sv[0] == page_size * sv[1]
            and (self.num_kv_heads == 1 or sk[2] == self.head_size)
        )
        if not ok:
            logger.warning_once(
                "ROCM_FLYDSL_DUALWAVE_ATTN: KV cache strides K=%s V=%s are outside the "
                "FlyDSL paged ABI (head stride must be head_size and page "
                "stride page_size*token_stride), which holds only for "
                "num_kv_heads==1 per rank; got num_kv_heads=%d. Falling back "
                "to the AITER unified attention kernel for this layer.",
                tuple(sk),
                tuple(sv),
                self.num_kv_heads,
            )
        self._layout_ok = ok
        return ok

    @staticmethod
    def _use_causal_lpt(num_short_query_rows: int, max_query_len: int) -> bool:
        """Pick between the causal_lpt-ordered kernel and the plain one.

        causal_lpt reverses the q-block grid axis so causal work issues
        heaviest-first, which packs the makespan of a prefill chunk and is worth
        up to 1.23x on a batch that is all prefill. It is a build flag, so the two
        orderings are separate compiled modules and choosing between them is a
        host-side decision that keeps one launch sequence per shape.

        It is not always a win. The grid is
        (num_heads_q, ceil(max_query_len / BLOCK_M), num_reqs) with the request
        axis slowest, so a short row's workgroups are always dispatched last.
        Without the reversal the long row's heaviest q-blocks are also last, and
        the short row's memory latency hides inside them; with it, the long row
        ends on its cheapest blocks and the short row's latency is fully exposed.
        Measured, that costs up to 1.29x -- larger the deeper the short row's own
        context, since that is what it has to walk while nothing overlaps it.

        Thresholds are read off a measured chunk-length x short-row-count map:

          no short rows          keep it; 1.05x-1.23x, and this is the common
                                 chunked-prefill and ragged two-sequence case.
          >= 96 q-blocks         keep it. Not because it provably wins, but a
                                 >=24k chunk runs for milliseconds and the
                                 arm-to-arm measurement floor there is ~10%, so
                                 the 1-4% the map hints at is unresolvable.
          1 short row, >= 48     the makespan gain still pays, 1.01x-1.06x.
          >= 8 short rows, <= 32 they fill the launch tail and overlap each
                                 other, so nothing is left exposed; 1.02x-1.04x.
          otherwise              it is a 3-28% loss. Drop it.
        """
        num_q_blocks = math.ceil(max_query_len / FLYDSL_BLOCK_M)
        if num_short_query_rows == 0:
            return True
        if num_q_blocks >= 96:
            return True
        if num_short_query_rows == 1 and num_q_blocks >= 48:
            return True
        return num_short_query_rows >= 8 and num_q_blocks <= 32

    def _flydsl_eligible(
        self,
        query: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
    ) -> bool:
        """Whether this batch goes to FlyDSL.

        Every term is a shape or dtype fact, never a value read back from the
        device, so the decision is fixed for a CUDA-graph capture key.
        """
        return (
            bool(attn_metadata.causal)
            and attn_metadata.max_query_len >= FLYDSL_MIN_QUERY_LEN
            and self.num_heads * attn_metadata.num_actual_tokens
            >= FLYDSL_MIN_QBLOCK_ROWS * FLYDSL_BLOCK_M
            and attn_metadata.max_seq_len <= FLYDSL_MAX_SEQ_LEN
            and attn_metadata.max_seq_len >= FLYDSL_MIN_SEQ_LEN
            and attn_metadata.num_decode_rows
            <= _max_decode_rows(attn_metadata.max_seq_len)
            and not is_quantized_kv_cache(self.kv_cache_dtype)
            and query.dtype in (torch.bfloat16, torch.float16)
            and self.head_size in FLYDSL_SUPPORTED_HEAD_SIZES
            and self._scale_is_default
            and self.alibi_slopes is None
            and self.sinks is None
            and not self.logits_soft_cap
            and self.sliding_window == (-1, -1)
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            attn_metadata is None
            or output_scale is not None
            or output_block_scale is not None
            or self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER)
            or not self._flydsl_eligible(query, attn_metadata)
            or not self._kv_layout_supported(kv_cache)
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

        assert attn_metadata.use_cascade is False

        num_actual_tokens = attn_metadata.num_actual_tokens
        # (B, H, N, 2*hs) -> two (B, N, H, hs) strided views. The FlyDSL paged
        # path reads the token stride off the tensor, so these are passed
        # through as-is; calling .contiguous() would repack the whole cache.
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        seq_lens = attn_metadata.seq_lens

        causal_lpt = self._use_causal_lpt(
            attn_metadata.num_short_query_rows,
            attn_metadata.max_query_len,
        )

        def _run(lpt: bool, out: torch.Tensor) -> None:
            self.flydsl_flash_attn_func(
                query[:num_actual_tokens],
                key_cache,
                value_cache,
                causal=True,
                num_kv_heads=self.num_kv_heads,
                block_table=attn_metadata.block_table,
                seqlen_k=seq_lens,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_kv=self._cu_seqlens_kv_for(seq_lens),
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_kv=attn_metadata.max_seq_len,
                cross_seqlen=True,
                kv_cache_layout="linear",
                num_kv_splits=1,
                causal_lpt=lpt,
                out=out,
            )

        # Force the ordering this batch does not want to compile too, so a later
        # capture of a batch that does want it never compiles mid-capture. Only
        # outside a capture, and only until both exist.
        if (
            len(self._lpt_variants_built) < 2
            and not torch.cuda.is_current_stream_capturing()
        ):
            for lpt in (causal_lpt, not causal_lpt):
                if lpt not in self._lpt_variants_built:
                    _run(lpt, output[:num_actual_tokens])
                    self._lpt_variants_built.add(lpt)

        _run(causal_lpt, output[:num_actual_tokens])
        return output
