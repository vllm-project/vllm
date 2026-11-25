# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Overlap of the sparse-attention indexer with independent same-layer work.

Enabled by `VLLM_INDEXER_OVERLAP`, and only while a FULL CUDA graph of a decode
batch is captured: an eager segment of a breakable (piecewise) graph cannot
wait on an event recorded inside a captured segment. The moved kernels and
their inputs are unchanged, so the outputs are bit-identical.

Deferred top-k (Kimi K3.1): the indexer op runs its scan and top-k on another
stream and returns without joining; the index group's conversion, the first
reader of the kept tokens, waits for them.

Deferred candidates (DeepSeek-V4.1): the candidate source layer selects its
candidate blocks, and expands them into its first consumer's sparse block ids,
on a side stream that the consumers join.
"""

import threading
from collections.abc import Callable
from typing import Any, TypeVar

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

T = TypeVar("T")


def full_cuda_graph_capture() -> bool:
    """Whether a plain (FULL) CUDA graph is being captured."""
    if not torch.cuda.is_current_stream_capturing():
        return False
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.forward_context import in_piecewise_cudagraph

    return not BreakableCUDAGraphCapture.is_active() and not in_piecewise_cudagraph()


def decode_only(metadata: Any) -> bool:
    return (
        getattr(metadata, "num_prefills", 1) == 0
        and getattr(metadata, "num_decodes", 0) > 0
    )


def captured_decode_metadata(layer_name: str) -> Any:
    """`layer_name`'s metadata while a FULL CUDA graph of a decode-only batch is
    captured, else None."""
    if not full_cuda_graph_capture():
        return None
    from vllm.forward_context import get_forward_context

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return None
    metadata = attn_metadata.get(layer_name)
    return metadata if decode_only(metadata) else None


def index_group_joins_topk(impl: Any, metadata: Any, num_tokens: int) -> bool:
    """Whether sparse attention reads the top-k only through its index group's
    conversion (`_convert_once`), the one reader that waits for a deferred top-k."""
    from vllm.model_executor.layers.attention.sparse_mla_attention import (
        SparseMLACommonImpl,
    )
    from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup

    group = getattr(impl, "index_group", None)
    return (
        isinstance(impl, SparseMLACommonImpl)
        and type(impl).forward_mqa is SparseMLACommonImpl.forward_mqa
        and group is not None
        and not isinstance(group, HiSparseMLAIndexGroup)
        and impl.dcp_world_size == 1
        and not getattr(metadata, "topk_start", 0)
        and not (impl.group_conversion_decode_only and group.num_layers <= 1)
        and num_tokens <= group.physical_topk_indices.shape[0]
    )


# Deferred top-k.
_deferred = threading.local()
_stock_stream: torch.cuda.Stream | None = None


def request_deferred_topk() -> None:
    """Called right before an indexer call whose top-k may land late."""
    _deferred.requested = True
    _deferred.stream = _deferred.keep = None


def consume_deferred_topk_request() -> bool:
    """Called by the indexer op: whether it may return before its top-k is
    written (it then says where with `set_deferred_topk`)."""
    requested = getattr(_deferred, "requested", False)
    _deferred.requested = False
    return requested and full_cuda_graph_capture()


def set_deferred_topk(stream: torch.cuda.Stream, keep: tuple) -> None:
    """The top-k is written on `stream`, which reads `keep`."""
    _deferred.stream, _deferred.keep = stream, keep
    logger.info_once("Sparse indexer overlap: deferred top-k active.")


def defer_stock_topk(work: Callable[[], None], keep: tuple) -> None:
    """Run the stock scan and top-k `work` after the current stream's work, on a
    high-priority stream, without joining: the top-k is on the critical path and
    the work moved beside it competes for the SMs the scan leaves free."""
    global _stock_stream
    if _stock_stream is None:
        _stock_stream = torch.cuda.Stream(priority=-1)
    _stock_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(_stock_stream):
        work()
    set_deferred_topk(_stock_stream, keep)


def take_deferred_topk() -> tuple[torch.cuda.Stream | None, tuple | None]:
    """After the indexer call: the stream that writes its top-k (None if the
    call joined) and the tensors that stream reads, which the caller keeps
    referenced until its stream has waited, so their memory is not reused."""
    taken = getattr(_deferred, "stream", None), getattr(_deferred, "keep", None)
    _deferred.requested = False
    _deferred.stream = _deferred.keep = None
    return taken


# Deferred candidates.
class CandidateStream:
    """A side stream for work that only the candidate consumers read. `defer`
    runs it after the current stream's work so far, so it cannot overtake earlier
    reads of the buffers it writes; each consumer calls `join` before reading.
    The side stream issues no collectives."""

    def __init__(self) -> None:
        self.stream = torch.cuda.Stream()
        self.done = torch.cuda.Event()
        self.pending = False
        # Tensors the work reads, kept alive until the join: no early memory reuse.
        self.keep: tuple = ()

    def defer(self, work: Callable[[], T], keep: tuple = ()) -> T:
        """Run `work()` on the side stream, without joining."""
        main = torch.cuda.current_stream()
        self.stream.wait_stream(main)
        try:
            with torch.cuda.stream(self.stream):
                result = work()
                self.done.record()
        except BaseException:
            # Rejoin so that a capture reports this error, not unjoined work.
            main.wait_stream(self.stream)
            raise
        self.pending, self.keep = True, keep
        return result

    def join(self) -> None:
        """Make the current stream wait for the deferred work, if any."""
        if self.pending:
            torch.cuda.current_stream().wait_event(self.done)
            self.pending, self.keep = False, ()


_candidates: CandidateStream | None = None
# The sparse decode rows whose sparse block ids the deferred work expanded.
_expanded: Any = None


def init_candidate_stream(allowed: bool) -> None:
    """Create the candidate stream at model construction if `allowed` (not with
    micro-batching, whose threads would share it, nor speculative decoding)."""
    global _candidates
    if (
        envs.VLLM_INDEXER_OVERLAP
        and allowed
        and _candidates is None
        and torch.cuda.is_available()
    ):
        _candidates = CandidateStream()


def candidate_defer_allowed(metadata: Any) -> bool:
    """Whether the candidate source may defer: a FULL CUDA graph capture of a
    decode-only batch whose consumers read the sparse decode rows."""
    return (
        _candidates is not None
        and decode_only(metadata)
        and getattr(metadata, "sparse_decode", None) is not None
        and full_cuda_graph_capture()
    )


def defer_candidates(work: Callable[[], None], keep: tuple) -> None:
    """Run `work()` on the candidate stream without joining; `keep` holds the
    caller-allocated tensors it reads until `join_candidates`."""
    assert _candidates is not None
    assert not _candidates.pending, "deferred candidates were never joined"
    _candidates.defer(work, keep)
    logger.info_once("Sparse indexer overlap: deferred candidate selection active.")


def join_candidates() -> None:
    """Called by each reader of the candidates: wait for deferred work, if any."""
    if _candidates is not None:
        _candidates.join()


def take_expanded(rows: Any) -> bool:
    """True once for the rows the deferred work expanded, while unscheduled."""
    global _expanded
    if rows is not _expanded or rows.kernel_metadata is not None:
        return False
    _expanded = None
    return True


def expand_sparse_decode_rows(
    metadata: Any, candidate_blocks: torch.Tensor, candidate_block_size: int
) -> None:
    """Expand the candidates into the sparse decode rows' block ids, as their
    first consumer would; that consumer then builds only the schedule."""
    global _expanded
    from vllm.model_executor.layers.sparse_mqa_indexer import expand_decode_candidates

    rows = metadata.sparse_decode
    if rows.row_ke.shape[0] == 0 or rows.kernel_metadata is not None:
        return
    expand_decode_candidates(
        rows, candidate_blocks, candidate_block_size, metadata.sparse_block_kv
    )
    _expanded = rows
