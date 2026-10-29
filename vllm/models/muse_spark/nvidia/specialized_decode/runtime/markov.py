# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Muse Spark DSpark draft's fused Markov sampling loop.

``DSparkSpeculator`` samples the draft tokens left to right, adding each
step's Markov bias to the base draft logits. ``MarkovSampler`` runs that loop
with each step's tail as one CuTe DSL kernel (``cutedsl/draft/markov.py``),
bitwise equal to the speculator's eager loop on rows without NaN.

Each row samples the lowest vocab id whose noised logit is NaN, else the
lowest id of the maximum (the ``torch.argmax`` order), at every block size and
vocab shard. On a row with a NaN the eager loop's pick depends on its block
layout, so the two can differ there.

With a vocab-parallel Markov w2 each rank samples its own vocab range
(``VocabParallelMarkovHead``) and the ranks exchange one packet per request
and step. The draft model builds it on the first call of its
``sample_sequential`` hook, from the speculator's sampling state.
"""

import dataclasses
import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
from torch import nn

import vllm.envs as envs
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.common.ops.sequence_parallel import sp_all_gather
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv, round_up

from ..cutedsl.common.policy import batch_table

logger = init_logger(__name__)

# gumbel_sample's BLOCK_SIZE.
MARKOV_BLOCK = 1024


@dataclass(frozen=True)
class FusedMarkovParams:
    """Static inputs of the fused loop, fixed once the draft model loads."""

    w1: torch.Tensor
    """Markov w1 table ``[V, R]`` (bf16, R a power of two)."""
    vocab_size: int
    output_multiplier: float
    """Rounded to fp32, as torch multiplies an fp32 tensor by it."""
    soft_cap: float | None
    """Rounded to fp32; ``None`` without a soft cap."""
    inv_soft_cap: float
    """``fp32(1) / fp32(soft_cap)``: torch divides by a scalar this way."""
    vocab_shard: "MarkovVocabShard | None" = None
    """Set when ``markov_w2`` is vocab-parallel."""


@dataclass(frozen=True)
class MarkovVocabShard:
    """This rank's contiguous vocab range of the draft lm_head and Markov w2."""

    tp_size: int
    size: int
    offset: int


# Rows up to which markov_step launches with programmatic dependent launch,
# so that its blocks start while the w2 GEMV runs. Measured on GB300, the
# overlap pays at one and two rows and costs from three rows on.
STEP_PDL_MAX_ROWS = 2

# markov_step's vocab entries per CTA, by row count; other row counts take
# MARKOV_BLOCK. Smaller blocks spread a step over more SMs; on GB300 at the
# TP8 vocab shard 512 beat MARKOV_BLOCK at most row counts up to 16. Every row
# samples the same id at any block size: each argmax takes the first NaN, else
# the first maximum.
STEP_BLOCKS = batch_table("0-16:512")
MIN_STEP_BLOCK = min(MARKOV_BLOCK, *STEP_BLOCKS.values())


def step_block(num_rows: int) -> int:
    """markov_step's block for a call over ``num_rows`` rows."""
    return STEP_BLOCKS.get(num_rows, MARKOV_BLOCK)


class VocabParallelMarkovHead(DSparkMarkovHead):
    """``DSparkMarkovHead`` with ``markov_w2`` sharded like the draft lm_head:
    each rank computes its vocab range of the transition bias
    (``local_bias``)."""

    def __init__(
        self,
        vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        # DSparkMarkovHead.__init__ would first allocate the replicated w2:
        # the same modules are built here, with the w2 sharded from the start.
        nn.Module.__init__(self)
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        # The same parameter name and loader as the replicated w2, sharded.
        self.markov_w2 = ParallelLMHead(
            draft_vocab_size,
            markov_rank,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "markov_w2"),
        )
        self.markov_w2._retain_weight_for_gather = False
        self.markov_w2.is_w4a16_nvfp4 = False
        # Rank-local w2 rows padded to a 16-byte row count, set by
        # align_vocab_shard() once the weights are loaded.
        self._local_w2: torch.Tensor | None = None

    def align_vocab_shard(self) -> None:
        """Back the rank's unquantized ``markov_w2`` rows with storage padded
        by zero rows to a 16-byte multiple, read by :meth:`local_bias`.

        cuBLAS picks its GEMV kernel by shape; the padded shard keeps the
        replicated head's sums, an unaligned one does not.
        """
        if not isinstance(self.markov_w2.quant_method, UnquantizedEmbeddingMethod):
            return
        weight = self.markov_w2.weight
        if (
            self._local_w2 is not None
            and weight.data_ptr() == self._local_w2.data_ptr()
        ):
            # A reload wrote into the padded storage, which captured CUDA
            # graphs keep reading.
            return
        rows = weight.shape[0]
        per_16_bytes = 16 // weight.element_size()
        padded = weight.new_zeros((round_up(rows, per_16_bytes), weight.shape[1]))
        padded[:rows].copy_(weight)
        weight.data = padded[:rows]
        self._local_w2 = padded

    def local_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        """This rank's vocab shard of the transition bias ([B, r] -> [B, V/tp]).

        The result is a view with the row stride of the padded storage.
        """
        assert self._local_w2 is not None
        rows = self.markov_w2.weight.shape[0]
        return dispatch_unquantized_gemm()(
            self.markov_w2, markov_embed, self._local_w2
        )[:, :rows]


def markov_vocab_shard(model: Any) -> tuple[int, int] | None:
    """This rank's ``[start, end)`` draft-vocab range of a vocab-parallel
    Markov w2, or None when the w2 is replicated.

    Called once the weights are loaded; prepares the padded w2 storage
    ``VocabParallelMarkovHead.local_bias`` reads. Fails where the local
    computations would not return this range's columns of
    ``model.compute_draft_logits`` and ``model.markov_bias`` on the default
    GEMM (under ``--linear-backend`` they match to rounding only).
    """
    head = model.model.markov_head
    w2 = head.markov_w2
    if w2.tp_size == 1:
        return None
    head.align_vocab_shard()
    if head._local_w2 is None or w2.weight.dtype != torch.bfloat16:
        raise ValueError(
            "The vocab-parallel Markov head needs an unquantized bf16 Markov w2"
        )
    lm_head = model.lm_head
    w2_range = w2.shard_indices
    lm_range = lm_head.shard_indices
    vocab_size = model.config.draft_vocab_size
    if (
        # The local bias runs the default GEMM that markov_bias runs through
        # the logits processor and UnquantizedEmbeddingMethod.apply unless
        # --linear-backend selects another one.
        model.logits_processor.scale != 1.0
        or model.logits_processor.soft_cap is not None
        or envs.VLLM_BATCH_INVARIANT
        or lm_head.tp_size != w2.tp_size
        or w2.num_embeddings_padded != vocab_size
        or lm_head.num_embeddings_padded != vocab_size
        or w2_range.added_vocab_end_index != w2_range.added_vocab_start_index
        or lm_range.added_vocab_end_index != lm_range.added_vocab_start_index
        or (w2_range.org_vocab_start_index, w2_range.org_vocab_end_index)
        != (lm_range.org_vocab_start_index, lm_range.org_vocab_end_index)
    ):
        raise ValueError(
            "The vocab-parallel Markov head needs a logits processor without "
            "scale or soft cap, no VLLM_BATCH_INVARIANT, and a draft lm_head "
            "and Markov w2 split into the same unpadded vocab ranges"
        )
    return w2_range.org_vocab_start_index, w2_range.org_vocab_end_index


def compute_local_draft_logits(model: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    """This rank's vocab shard of ``model.compute_draft_logits``, ungathered."""
    return model.logits_processor(
        model.lm_head, hidden_states.to(model.lm_head.weight.dtype), skip_gather=True
    )


def vocab_shard(params: FusedMarkovParams, start: int, end: int) -> MarkovVocabShard:
    """The sharded loop's view of this rank's draft-vocab range.

    The packets and cache shards are gathered in TP rank order, so rank k
    must hold the k-th equal slice of the draft vocabulary. The w2 is
    never replicated in this mode, so a draft the loop does not take fails
    at its first draft step instead of running the eager loop on gathered
    shards.
    """
    tp_size = get_tensor_model_parallel_world_size()
    size = end - start
    if tp_size & (tp_size - 1):
        reason = f"a power-of-two TP size, not {tp_size}"
    elif size * tp_size != params.vocab_size:
        reason = (
            f"a draft vocabulary ({params.vocab_size}) that TP {tp_size} "
            "divides without padding"
        )
    elif start != size * get_tensor_model_parallel_rank():
        reason = "TP rank k to hold the k-th slice of the draft vocabulary"
    else:
        return MarkovVocabShard(tp_size=tp_size, size=size, offset=start)
    raise ValueError(f"The vocab-parallel Markov head needs {reason}")


def unsupported(
    model: Any,
    *,
    draft_logits: torch.Tensor | None,
    draft_topk: int | None,
    watermarked: bool,
    use_fp64_gumbel: bool,
    sample_from_anchor: bool,
) -> str | None:
    """Why only the speculator's eager Markov loop computes this draft's
    tokens and draft-logits cache, or None when the fused loop does."""
    d2t = getattr(model, "draft_id_to_target_id", None)
    if not current_platform.is_cuda():
        return "the platform is not CUDA"
    if getattr(model, "output_logit_transform_params", None) is None:
        return "the draft has no output-logit transform"
    if draft_topk is not None:
        return "dspark_draft_topk is set"
    if watermarked:
        return "the draft is watermarked"
    if use_fp64_gumbel:
        return "the draft samples with fp64 Gumbel noise"
    if not sample_from_anchor:
        return "the draft does not sample from the anchor"
    if (
        draft_logits is None
        or draft_logits.dtype != torch.bfloat16
        or not draft_logits.is_contiguous()
    ):
        return "drafting does not keep a contiguous bf16 draft-logits cache"
    # No map (None) is the identity map, as map_draft_to_target reads it.
    if d2t is not None and bool((d2t != 0).any()):
        return "the draft's draft-to-target id map is not the identity"
    if model.logits_processor.head_dtype not in (None, torch.bfloat16):
        return "the draft lm_head does not run in bf16"
    vocab_size = draft_logits.shape[-1]
    w1 = model.model.markov_head.markov_w1.weight
    rank = w1.shape[1]
    if (
        w1.dtype != torch.bfloat16
        or not w1.is_contiguous()
        or rank & (rank - 1)
        or w1.shape[0] != vocab_size
    ):
        return "the Markov w1 is not a contiguous bf16 power-of-two-rank matrix"
    if model.config.draft_vocab_size != vocab_size:
        return "the draft and target vocabularies differ in size"
    return None


def fused_markov_params(model: Any, draft_logits: torch.Tensor) -> FusedMarkovParams:
    """Inputs of the fused Markov loop for a draft ``unsupported`` accepts.

    The fused loop rounds each step's logits to bf16 before sampling, as the
    eager loop does through the bf16 d2t scatter buffer, and it applies the
    draft's output-logit transform in its epilogue. A draft without a d2t
    map runs the same loop as one with the all-zero (identity) map.
    """
    multiplier, soft_cap = model.output_logit_transform_params()

    def fp32(x: float) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32)

    cap = None if soft_cap is None else fp32(soft_cap)
    return FusedMarkovParams(
        w1=model.model.markov_head.markov_w1.weight,
        vocab_size=draft_logits.shape[-1],
        output_multiplier=float(fp32(multiplier)),
        soft_cap=None if cap is None else float(cap),
        inv_soft_cap=1.0 if cap is None else float(fp32(1.0) / cap),
    )


def build_markov_sampler(
    model: Any,
    *,
    draft_logits: torch.Tensor | None,
    draft_topk: int | None,
    watermarked: bool,
    use_fp64_gumbel: bool,
    sample_from_anchor: bool,
    max_num_reqs: int,
    num_steps: int,
) -> "MarkovSampler":
    """The fused loop over ``model``, which the specialized decode's draft
    runs: a draft it does not reproduce fails at its first draft step, as the
    vocab-parallel Markov w2 has no eager loop."""
    reason = unsupported(
        model,
        draft_logits=draft_logits,
        draft_topk=draft_topk,
        watermarked=watermarked,
        use_fp64_gumbel=use_fp64_gumbel,
        sample_from_anchor=sample_from_anchor,
    )
    if reason is not None:
        raise ValueError(
            "The Muse Spark specialized decode runs the fused DSpark Markov "
            f"loop, which does not run here: {reason}"
        )
    assert draft_logits is not None
    params = fused_markov_params(model, draft_logits)
    local_range = markov_vocab_shard(model)
    if local_range is not None:
        shard = vocab_shard(params, *local_range)
        params = dataclasses.replace(params, vocab_shard=shard)
        logger.info_once(
            "Using the vocab-parallel DSpark Markov sampling loop.",
            scope="process",
        )
    logger.info_once("Using the fused DSpark Markov sampling loop.", scope="process")
    return MarkovSampler(model, params, max_num_reqs, num_steps, draft_logits)


class MarkovCall(NamedTuple):
    """The views one call of the fused loop runs on (``MarkovSampler.call_views``)."""

    width: int
    """Vocab entries per rank: the whole vocab, or this rank's shard."""
    offset: int
    """The first vocab id of this rank's shard (0 without a shard)."""
    base_logits: torch.Tensor
    """``[rows, n_spec, width]`` BF16."""
    markov_bias: Callable[[torch.Tensor], torch.Tensor]
    cache: torch.Tensor
    """The draft-logits cache, or with a shard ``[rows, n_spec, width]``."""
    flat_cache: torch.Tensor | None
    """With a shard, the flat cache the all-gather moves (else None)."""
    packets: torch.Tensor
    """``[rows, 4]`` FP32: a row's packet (with a shard)."""
    idx_map: torch.Tensor
    pos: torch.Tensor
    markov_embeds: torch.Tensor
    """``[n_spec + 1, rows, rank]``."""
    arrivals: torch.Tensor
    block: int
    num_blocks: int
    block_argmax: torch.Tensor
    block_max: torch.Tensor
    step_pdl: bool


class MarkovSampler:
    """The fused Markov loop of one draft model, with its per-call buffers."""

    def __init__(
        self,
        model: Any,
        params: FusedMarkovParams,
        max_num_reqs: int,
        num_steps: int,
        draft_logits: torch.Tensor,
    ) -> None:
        self.model = model
        self.params = params
        self.num_steps = num_steps
        # Launch the per-step kernels with programmatic dependent launch
        # (the step kernel up to STEP_PDL_MAX_ROWS rows).
        self._pdl = current_platform.is_arch_support_pdl()
        # The per-call buffers, flat and sized for max_num_reqs rows; a call
        # takes views of their leading elements.
        rows, n_spec, w1 = max_num_reqs, num_steps, params.w1
        shard = params.vocab_shard
        width = params.vocab_size if shard is None else shard.size
        num_blocks = cdiv(width, MIN_STEP_BLOCK)

        def empty(numel: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.empty(numel, dtype=dtype, device=w1.device)

        self._markov_embeds = empty((n_spec + 1) * rows * w1.shape[1], w1.dtype)
        self._markov_block_argmax = empty(rows * num_blocks, torch.int64)
        self._markov_block_max = empty(rows * num_blocks, torch.float32)
        # Blocks of each row that finished their step, zeroed per call.
        self._markov_arrivals = empty(rows, torch.int32)
        # The rows' packets, exchanged with a vocab shard; with it also the
        # draft-logits shard the custom all-gather moves in 16-byte vectors.
        sharded = shard is not None
        self._markov_packets = empty(rows * 4, torch.float32)
        self._markov_cache = empty(
            round_up(rows * n_spec * width, 8) if sharded else 0, torch.bfloat16
        )
        from ..cutedsl.draft import markov as kernels

        self._kernels = kernels
        self._loop: Any = None
        if not w1.is_cuda:
            return
        # The CuTe kernels compile here, before any CUDA graph capture: the
        # step at every (block, PDL, vector) a call over 1 to max_num_reqs
        # rows takes.
        self._loop = kernels.MarkovKernels(
            w1,
            self._markov_embeds,
            width=width,
            has_cap=params.soft_cap is not None,
            tp=None if shard is None else shard.tp_size,
            steps=self._step_variants(rows, width, draft_logits),
            pick_pdl=self._pdl,
            fill_vector=kernels.fill_vectorized(
                # The gathered shards: a fresh all-gather output, its rank
                # stride the padded flat shard (a multiple of 8 elements).
                gathered=0,
                src_s0=round_up(n_spec * width, 8),
                cache=kernels.Operand.of(draft_logits),
                shard=width,
                vocab=params.vocab_size,
            ),
        )

    def _heads(self) -> tuple[Callable, Callable]:
        """``(base logits, Markov bias)`` of a call: the draft's full-vocab
        heads, or with a vocab shard this rank's (neither all-gathers)."""
        if self.params.vocab_shard is None:
            return self.model.compute_draft_logits, self.model.markov_bias
        return (
            functools.partial(compute_local_draft_logits, self.model),
            self.model.model.markov_head.local_bias,
        )

    def _step_variants(
        self, max_rows: int, width: int, draft_logits: torch.Tensor
    ) -> set[tuple[int, bool, bool]]:
        """The ``(block, PDL, vector)`` of every step a call over 1 to
        ``max_rows`` rows launches (``MarkovKernels.step`` picks ``vector``
        from its operands).

        The base logits and the bias of a call are fresh outputs of the
        heads (``_heads``): their row strides are the heads' own (a padded
        lm_head or Markov w2 leaves its padded rows' stride under the
        ``[..., :vocab]`` view the logits processor returns), read here from
        one call of each on zeros, and they start as aligned as these
        probes do (allocator blocks)."""
        kernels, n_spec, w1 = self._kernels, self.num_steps, self.params.w1
        base_head, bias_head = self._heads()
        with torch.no_grad():
            base_probe = base_head(w1.new_zeros(2, self.model.config.hidden_size))
            bias_probe = bias_head(w1.new_zeros(2, w1.shape[1]))
        base_row = base_probe.stride(0)
        bias = kernels.Operand(bias_probe.data_ptr(), (bias_probe.stride(0), 1))
        if self.params.vocab_shard is not None:
            cache = kernels.Operand(
                self._markov_cache.data_ptr(), (n_spec * width, width, 1)
            )
        else:
            cache = kernels.Operand.of(draft_logits)
        variants = set()
        for rows in range(1, max_rows + 1):
            block = step_block(rows)
            pdl = self._pdl and rows <= STEP_PDL_MAX_ROWS
            for col in range(n_spec):
                vector = kernels.vectorized(
                    block,
                    width=width,
                    col=col,
                    # Column col of the [rows, n_spec, width] view.
                    base=kernels.Operand(
                        base_probe.data_ptr() + col * base_row * 2,
                        (n_spec * base_row, 1),
                    ),
                    bias=bias,
                    cache=cache,
                )
                variants.add((block, pdl, vector))
        return variants

    def call_views(
        self,
        num_reqs: int,
        head_hidden: torch.Tensor,
        *,
        sample_indices: torch.Tensor,
        sample_idx_mapping: torch.Tensor,
        sample_pos: torch.Tensor,
        draft_logits: torch.Tensor,
    ) -> MarkovCall:
        """The base logits of a call over ``num_reqs`` rows and the views of
        its buffers that the loop's kernels run on (``sample``)."""
        n_spec = self.num_steps
        num_sample = num_reqs * n_spec
        shard = self.params.vocab_shard
        rank = self.params.w1.shape[1]
        base_head, markov_bias = self._heads()
        base_logits = base_head(head_hidden[sample_indices[:num_sample]])
        flat_cache = None
        if shard is None:
            width, offset = self.params.vocab_size, 0
            cache = draft_logits
        else:
            width, offset = shard.size, shard.offset
            # The custom all-gather moves 16-byte vectors: pad the flat shard.
            numel = num_sample * width
            flat_cache = self._markov_cache[: round_up(numel, 8)]
            cache = flat_cache[:numel].view(num_reqs, n_spec, width)
        block = step_block(num_reqs)
        num_blocks = cdiv(width, block)
        call = MarkovCall(
            width=width,
            offset=offset,
            base_logits=base_logits.view(num_reqs, n_spec, width),
            markov_bias=markov_bias,
            cache=cache,
            flat_cache=flat_cache,
            packets=self._markov_packets[: num_reqs * 4].view(num_reqs, 4),
            idx_map=sample_idx_mapping[:num_sample].view(num_reqs, n_spec),
            pos=sample_pos[:num_sample].view(num_reqs, n_spec),
            # Row i holds the Markov embedding that step i reads.
            markov_embeds=self._markov_embeds[: (n_spec + 1) * num_reqs * rank].view(
                n_spec + 1, num_reqs, rank
            ),
            arrivals=self._markov_arrivals[:num_reqs],
            block=block,
            num_blocks=num_blocks,
            block_argmax=self._markov_block_argmax[: num_reqs * num_blocks],
            block_max=self._markov_block_max[: num_reqs * num_blocks],
            step_pdl=self._pdl and num_reqs <= STEP_PDL_MAX_ROWS,
        )
        for tensor, dtype in (
            (call.base_logits, torch.bfloat16),
            (call.cache, torch.bfloat16),
            (call.idx_map, torch.int64),
            (call.pos, torch.int64),
        ):
            assert tensor.dtype == dtype, (tensor.dtype, dtype)
        return call

    def sample(
        self,
        num_reqs: int,
        head_hidden: torch.Tensor,
        *,
        sample_indices: torch.Tensor,
        sample_idx_mapping: torch.Tensor,
        sample_pos: torch.Tensor,
        input_ids: torch.Tensor,
        anchor_idx: torch.Tensor,
        seeds: torch.Tensor,
        temperature: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor,
    ) -> None:
        """The eager Markov loop with each step's tail as one kernel.

        Writes ``draft_tokens`` and the ``draft_logits`` cache for the first
        ``num_reqs`` requests, as ``DSparkSpeculator``'s eager loop does over
        the same buffers. (The specialized decode refuses adaptive
        verification, so there is no confidence head to run.)

        With a vocab shard each rank samples its own vocab range, the ranks
        exchange one packet per request and step, and one all-gather after the
        loop fills the full-vocab draft-logits cache.
        """
        params = self.params
        call = self.call_views(
            num_reqs,
            head_hidden,
            sample_indices=sample_indices,
            sample_idx_mapping=sample_idx_mapping,
            sample_pos=sample_pos,
            draft_logits=draft_logits,
        )
        for tensor, dtype in (
            (seeds, torch.int64),
            (temperature, torch.float32),
            (draft_tokens, torch.int64),
        ):
            assert tensor.dtype == dtype, (tensor.dtype, dtype)
        loop = self._loop
        markov_embeds, packets = call.markov_embeds, call.packets
        self._kernels.anchor_embed(
            input_ids, anchor_idx, params.w1, markov_embeds[0], call.arrivals
        )
        for i in range(self.num_steps):
            bias = call.markov_bias(markov_embeds[i])
            assert bias.dtype == torch.bfloat16, bias.dtype
            loop.step(
                call.base_logits[:, i],
                bias,
                call.cache,
                i,
                call.idx_map[:, i],
                call.pos[:, i],
                seeds,
                temperature,
                call.block_argmax,
                call.block_max,
                call.arrivals,
                draft_tokens,
                markov_embeds[i + 1],
                packets.view(torch.int32),
                call.offset,
                params.output_multiplier,
                params.inv_soft_cap,
                1.0 if params.soft_cap is None else params.soft_cap,
                block=call.block,
                pdl=call.step_pdl,
            )
            if call.flat_cache is None:
                continue
            loop.pick(
                sp_all_gather(packets).view(torch.int32),
                num_reqs,
                draft_tokens,
                i,
                markov_embeds[i + 1],
            )

        if call.flat_cache is not None:
            self._kernels.cache_fill(
                sp_all_gather(call.flat_cache),
                call.flat_cache.numel(),
                draft_logits,
                sample_idx_mapping,
                num_reqs,
                self.num_steps,
                call.width,
                params.vocab_size,
            )
