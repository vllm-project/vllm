# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The logits processor of the specialized decode: the stock processor, with
the target's logits gathered by `LogitsAllGather` and the MTP draft passes'
greedy token chosen by `TokenChoice`, once those are bound.

The target of a `latency_specialized` deployment builds its
`LogitsProcessor` as this class, and so does the MTP draft model when the
speculative config sets `use_local_argmax_reduction` (without it, the draft
never asks for its tokens and keeps the stock processor); until a kernel is
bound (and for every call it does not take) the processor is the stock one.
The kernels are collective over the TP group, so they are built at points
every rank reaches together:

- the target's `LogitsAllGather` with the executor (`bind_logits_gather`,
  from `Qwen4ExpForCausalLM.build_specialized_decode`), at TP > 1, whatever
  the speculative config;
- the draft's `TokenChoice` at the end of the draft model's weight loading
  (`bind_token_choice`, from `Qwen4ExpMTP.load_weights`), which Model
  Runner V2 runs before memory profiling and every CUDA graph capture.

Which calls they take is decided from host state every rank shares (row
counts and shapes), so every rank takes the same path in every call. The
token choice serves the speculator's `get_top_tokens` calls, which Model
Runner V2 makes when the speculative config sets
`use_local_argmax_reduction`; it computes the head's logits with the stock
`_apply_head` (the same GEMM, so the same logits) and returns the token
`greedy_argmax` returns on the gathered logits.

With `draft_head_fp8_prefilter` (TP1), the model state also builds the
exact FP8-prefiltered `DraftHead` for the target's LM head, which the MTP
drafter shares (`bind_draft_head`, before memory profiling, which counts
its FP8 copy). The draft passes' `get_top_tokens` calls of up to 8 rows on
that head take it before the token choice; it returns the same tokens without
the BF16 logits (`draft_head.py`).
"""

import weakref
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.utils import (
    cuda_unquantized_gemm,
    dispatch_unquantized_gemm,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform

from .. import MAX_DECODE_REQUESTS

if TYPE_CHECKING:
    from .choice import TokenChoice
    from .draft_head import DraftHead
    from .gather import LogitsAllGather

logger = init_logger(__name__)

# The FP8-prefiltered draft heads, by the LM head module they were built for
# (the target's, which the drafter shares); weak, so a dropped model frees its
# FP8 copy.
_DRAFT_HEADS: "weakref.WeakKeyDictionary[VocabParallelEmbedding, DraftHead]" = (
    weakref.WeakKeyDictionary()
)


def _shard(processor: LogitsProcessor, lm_head: VocabParallelEmbedding):
    """The head's shard of the vocabulary as the token choice takes it:
    `(logits a row holds, how many of them are entries, the first entry's
    index)`, or why the choice cannot serve it. Whatever the head's weights
    (the drafter may run the target's head or a quantized copy of it), the
    choice needs only its BF16 logits, which `get_top_tokens` checks."""
    if processor.scale != 1.0 or processor.soft_cap is not None:
        return "logits without scaling or soft cap"
    if processor.head_dtype not in (None, torch.bfloat16):
        return f"BF16 logits (head_dtype {processor.head_dtype})"
    indices = getattr(lm_head, "shard_indices", None)
    if indices is None:
        return f"a vocab-parallel LM head (not {type(lm_head).__name__})"
    if indices.num_added_elements:
        return "a head without added vocabulary"
    start = indices.org_vocab_start_index
    # Entries past the processor's vocabulary or in the padding never win.
    valid = max(0, min(indices.num_org_elements, processor.org_vocab_size - start))
    width = lm_head.num_embeddings_per_partition
    if valid <= 0 or width % 8:
        return "a shard of whole 16-byte rows"
    return width, valid, start


class SpecializedDecodeLogitsProcessor(LogitsProcessor):
    """`LogitsProcessor` with the specialized decode's gather and token
    choice once bound (module docstring)."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__(vocab_size)
        # The target's logits gather (TP > 1), bound with the executor.
        self.all_gather: LogitsAllGather | None = None
        # The draft passes' token choice and the head shard it was built
        # for, bound at the end of the draft model's weight loading.
        self.token_choice: TokenChoice | None = None
        self.choice_shard: tuple[int, int, int] | None = None

    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """The stock gather, or the bound `LogitsAllGather` for logits it
        accepts (a decision from the shape alone)."""
        gather = self.all_gather
        if self.use_all_gather and gather is not None and gather.accepts(logits):
            return gather(logits)
        return super()._gather_logits(logits)

    def get_top_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The greedy token of every row: the `DraftHead` built for
        `lm_head` when it takes the call (up to 8 rows), else the bound
        `TokenChoice` on this rank's logits when it takes the call (its head
        shard, row count, BF16 logits), the stock vocab-parallel argmax
        otherwise. Every check reads shapes and dtypes, the same on every
        rank."""
        head = _DRAFT_HEADS.get(lm_head)
        if (
            head is not None
            and embedding_bias is None
            and head.takes(hidden_states)
            and _shard(self, lm_head) == (head.vocab, head.valid, 0)
        ):
            return head(hidden_states)
        choice = self.token_choice
        rows = hidden_states.shape[0]
        if (
            choice is not None
            and embedding_bias is None
            and hidden_states.dim() == 2
            and choice.takes(rows)
            and _shard(self, lm_head) == self.choice_shard
        ):
            logits = self._apply_head(lm_head, hidden_states, None)
            if (
                logits.dtype == torch.bfloat16
                and logits.shape == (rows, choice.shard)
                and logits.stride(1) == 1
            ):
                return choice(logits)
            logger.warning_once(
                "Qwen4Exp specialized decode: the MTP draft head returned %s "
                "logits of shape %s; the draft passes keep the stock token "
                "choice, which recomputes them.",
                logits.dtype,
                tuple(logits.shape),
            )
        return super().get_top_tokens(lm_head, hidden_states, embedding_bias)


def bind_token_choice(
    processor: LogitsProcessor,
    lm_head: VocabParallelEmbedding,
    max_num_seqs: int,
) -> None:
    """Build the draft model's `TokenChoice` for `lm_head`'s shard and
    bind it to `processor`; log once why not when the head or processor is
    not one it serves. It takes up to `min(max_num_seqs,
    MAX_DECODE_REQUESTS)` rows, the most the executor's decode steps run
    (and below the SM count, which its liveness needs: `choice.py`); larger
    draft passes keep the stock choice. Collective over the TP group (every
    rank's processor is built alike, so every rank reaches the same
    branch)."""
    assert isinstance(processor, SpecializedDecodeLogitsProcessor)
    shard = _shard(processor, lm_head)
    if isinstance(shard, str):
        logger.info_once(
            "Qwen4Exp specialized decode: the MTP draft passes keep the stock "
            "token choice, which needs %s.",
            shard,
        )
        return
    # Imported here: CuTe DSL kernels (CUDA builds only).
    from .choice import TokenChoice

    width, valid, start = shard
    group = get_tp_group()
    capacity = min(max_num_seqs, MAX_DECODE_REQUESTS[group.world_size])
    processor.token_choice = TokenChoice(
        group.device_group if group.world_size > 1 else None,
        width,
        valid,
        start,
        capacity,
        torch.device("cuda", torch.accelerator.current_device_index()),
    )
    processor.choice_shard = shard
    processor.token_choice.warm_up()
    logger.info_once(
        "Qwen4Exp specialized decode: the MTP draft passes choose their tokens "
        "on the vocabulary shard (%d logits a row, up to %d rows); it serves "
        "speculative configs with use_local_argmax_reduction.",
        width,
        capacity,
    )


def _draft_head_unsupported(
    processor: LogitsProcessor, lm_head: VocabParallelEmbedding
) -> str | None:
    """Why `DraftHead` cannot serve `lm_head`'s draft passes, or None: its
    tokens are exactly the stock ones only where the stock head is the whole
    vocabulary through cuBLAS's `F.linear` on GB300 (`draft_head.py`)."""
    if get_tp_group().world_size != 1:
        return "tensor parallel size 1"
    if not current_platform.is_device_capability(103):
        return "a GB300 (SM103) GPU"
    # A Qwen4Exp skinny-GEMM plan replaces the quant method; quantized heads
    # have their own.
    if type(lm_head.quant_method) is not UnquantizedEmbeddingMethod:
        return (
            "a BF16 head computed by F.linear (not "
            f"{type(lm_head.quant_method).__name__})"
        )
    if (
        getattr(lm_head, "use_skinny_gemm", False)
        or dispatch_unquantized_gemm() is not cuda_unquantized_gemm
        or envs.VLLM_BATCH_INVARIANT
    ):
        return (
            "a BF16 head computed by F.linear (the default CUDA GEMM; not "
            "VLLM_BATCH_INVARIANT's)"
        )
    shard = _shard(processor, lm_head)
    if isinstance(shard, str):
        return shard
    weight = lm_head.weight
    if (
        shard[2] != 0
        or weight.dtype != torch.bfloat16
        or weight.dim() != 2
        or tuple(weight.shape) != (shard[0], 2560)
        or not weight.is_contiguous()
    ):
        return "a contiguous BF16 head of hidden size 2560 over the whole vocabulary"
    return None


def bind_draft_head(
    processor: LogitsProcessor, lm_head: VocabParallelEmbedding
) -> None:
    """Build the FP8-prefiltered `DraftHead` for the target's `lm_head`
    (which the MTP drafter shares) and register it for the draft passes'
    `get_top_tokens`; raise when it cannot serve the head. `processor` is
    the target's: the draft passes' processor checks its own shard against
    the head's on every call. Called by the model state before memory
    profiling, at TP1 only (`draft_head_fp8_prefilter`)."""
    reason = _draft_head_unsupported(processor, lm_head)
    if reason is not None:
        raise ValueError(f"draft_head_fp8_prefilter needs {reason}.")
    # Imported here: CuTe DSL kernels (CUDA builds only).
    from .draft_head import DraftHead

    shard = _shard(processor, lm_head)
    assert not isinstance(shard, str)
    head = DraftHead(lm_head.weight, shard[1])
    head.warm_up()
    _DRAFT_HEADS[lm_head] = head
    logger.info_once(
        "Qwen4Exp specialized decode: MTP draft passes of up to 8 rows choose "
        "their tokens with the FP8-prefiltered head (%.2f GiB FP8 copy).",
        head.nbytes / 2**30,
    )


def bind_logits_gather(
    processor: LogitsProcessor, lm_head: VocabParallelEmbedding, max_rows: int
) -> None:
    """Build the target's `LogitsAllGather` for up to `max_rows` rows of
    `lm_head`'s shard and bind it to `processor` (TP > 1 only).
    Collective over the TP group."""
    assert isinstance(processor, SpecializedDecodeLogitsProcessor)
    group = get_tp_group()
    width = lm_head.num_embeddings_per_partition
    if group.world_size == 1 or width % 8:
        return
    # Imported here: CuTe DSL kernels (CUDA builds only).
    from .gather import LogitsAllGather

    gather = LogitsAllGather(
        group.device_group,
        width,
        max_rows,
        torch.device("cuda", torch.accelerator.current_device_index()),
    )
    gather.warm_up()
    processor.all_gather = gather
    logger.info_once(
        "Qwen4Exp specialized decode: the target's logits of up to %d rows are "
        "gathered over NVLink (%.1f MiB mailbox a rank).",
        max_rows,
        gather.mailbox_bytes / 2**20,
    )
