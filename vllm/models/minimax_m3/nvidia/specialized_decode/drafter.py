# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark drafter's sequential Markov sampling in K + 1 launches.

The stock loop (`DSparkSpeculator._sample_markov_chain`) runs, for each of
the K draft positions, about ten kernels: the Markov embedding lookup, the
`[rows, r] x [r, V]` bias GEMM, the bias add, two index copies, the Gumbel
block argmax (which also stores the draft logits cache column), the argmax
over the blocks, and two token copies, most of the time in small kernels
around one stream of W2 (V x r, BF16).

Here one kernel per position does all of it, for every request at once:

- its prologue reads the previous position's token from a packed argmax key
  per request (position 0 reads the anchor token instead); program 0 also
  stores that token into `draft_tokens` and re-arms the key buffer the next
  position reduces into;
- it gathers the Markov embedding `W1[prev]` and, tile by tile over its
  vocabulary slice, computes the bias `W1[prev] @ W2^T` (fp32 accumulate,
  rounded to the dtype the stock GEMM returns), adds the base logits (rounded
  to the dtype the stock add returns), stores the draft logits cache column,
  and runs the Gumbel-max sample (the stock helpers, so the noise is
  bit-identical) or the plain argmax at temperature 0;
- it folds its (max, leftmost index) per request into the key with one
  atomic max: the key orders (value, -index), so the reduction returns the
  leftmost global maximum, which is what the stock block argmax plus
  `torch.argmax` return.

A last one-program launch stores position K-1's token.

Only the bias GEMM's accumulation order differs from the stock path (it is
rounded to the same dtype), so the drafts are those of the stock sampler up to
that rounding. The base draft logits (draft LM head) stay stock, gathered
through peer memory (`peer_logits.py`) instead of NCCL when supported.

The sampler engages only on configurations it reproduces exactly (otherwise
the stock loop runs): probabilistic drafting with a draft logits cache, no
reduced draft vocabulary, watermarker, acceptance estimator or confidence
head, FP32 Gumbel sampling, an unquantized Markov head, at most
`MARKOV_MAX_REQUESTS` requests.
"""

import weakref
from typing import Any

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import (
    _DRAFT_NOISE_SALT,
    _log1p_neg_stable,
    murmur3_uniform32,
    spec_step_seed,
)
from vllm.v1.worker.gpu.spec_decode.draft_logits import draft_logits_row

from .peer_logits import PeerPushedLogits, peer_logits, peer_logits_unsupported

logger = init_logger(__name__)

# Requests per launch: the rows of the bias MMA (padded to at least 16).
MARKOV_MAX_REQUESTS = 16
# Launch geometry of the step kernel: vocabulary entries per MMA tile, programs per SM
# (each loops over its tiles), warps, pipeline stages, and whether the MMA runs
# W2-tile-major (M = BLOCK_V) instead of rows-major. Chosen by a sweep on GB300 (V =
# 200064, r = 256, CUDA-graph replays, with the active-row sampling), at 1 to 8
# requests, greedy and at temperature 1.
BLOCK_V = 64
PROGRAMS_PER_SM = 3
NUM_WARPS = 4
NUM_STAGES = 3
SWAP_MMA = False


# The empty key (int64 min): below every _argmax_key.
_KEY_EMPTY = tl.constexpr(-(2**63))


@triton.jit
def _argmax_key(value, index):
    """(value, -index) packed into an int64 ordered like the pair: an atomic
    max over keys picks the leftmost maximum."""
    bits = (value + 0.0).to(tl.int32, bitcast=True)  # + 0.0: -0.0 -> +0.0
    # The float order as a signed int32 order: flip the magnitude of negatives.
    ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.int64)
    return (ordered << 32) | (0xFFFFFFFF - index)


@triton.jit
def _key_index(key):
    return 0xFFFFFFFF - (key & 0xFFFFFFFF)


@triton.jit(do_not_specialize=["num_reqs", "step"])
def _markov_step_kernel(
    base_ptr,
    base_stride_req,
    base_stride_step,
    anchor_ptr,
    anchor_stride,
    w1_ptr,
    w1_stride,
    w2_ptr,
    w2_stride,
    cache_ptr,
    cache_stride_0,
    cache_stride_1,
    row_mapping_ptr,
    idx_mapping_ptr,
    pos_ptr,
    temp_ptr,
    seeds_ptr,
    keys_ptr,
    out_ptr,
    out_stride,
    num_reqs,
    step,
    vocab_size,
    N_SPEC: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_V: tl.constexpr,
    TILES: tl.constexpr,
    BIAS_DTYPE: tl.constexpr,
    LOGITS_DTYPE: tl.constexpr,
    FIRST: tl.constexpr,
    SWAP: tl.constexpr,
    R_ACT: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    row_mask = rows < num_reqs
    # keys[3][BLOCK_R]: position i reduces into keys[i % 3]; it reads keys[(i - 1) % 3]
    # and re-arms keys[(i + 1) % 3] (last read at i - 1).
    if FIRST:
        prev = tl.load(anchor_ptr + rows * anchor_stride, mask=row_mask, other=0)
        prev = prev.to(tl.int64)
    else:
        prev_key = tl.load(keys_ptr + ((step - 1) % 3) * BLOCK_R + rows)
        prev = tl.where(row_mask, _key_index(prev_key), 0)
        if pid == 0:
            tl.store(out_ptr + rows * out_stride + (step - 1), prev, mask=row_mask)
    if pid == 0:
        tl.store(
            keys_ptr + ((step + 1) % 3) * BLOCK_R + rows,
            tl.full([BLOCK_R], _KEY_EMPTY, tl.int64),
        )

    ks = tl.arange(0, RANK)
    embed = tl.load(
        w1_ptr + prev[:, None] * w1_stride + ks[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )

    # Everything after the MMA runs on its first R_ACT rows (num_reqs <= R_ACT <=
    # BLOCK_R): the padded rows of the MMA are never sampled.
    rows_a = tl.arange(0, R_ACT)
    row_mask_a = rows_a < num_reqs
    # Sampling parameters, as gumbel_block_argmax reads them.
    flat = rows_a * N_SPEC + step
    req_state_idx = tl.load(idx_mapping_ptr + flat, mask=row_mask_a, other=-1).to(
        tl.int64
    )
    is_valid_req = req_state_idx >= 0
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(
        tl.float32
    )
    seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
    seed = spec_step_seed(seed, step)
    # The draw is keyed by the position before the sampled token.
    pos = tl.load(pos_ptr + flat, mask=row_mask_a, other=1) - 1
    noisy = temp != 0.0
    pos = tl.where(noisy, pos + _DRAFT_NOISE_SALT, pos)
    any_noisy = tl.max(noisy.to(tl.int32), axis=0) > 0
    cache_row = draft_logits_row(req_state_idx, row_mapping_ptr)

    start = pid * TILES * BLOCK_V
    best_val = tl.full([R_ACT], float("-inf"), tl.float32)
    best_idx = tl.full([R_ACT], 0, tl.int64) + start
    # Which R_ACT-row block of the MMA rows a row lies in (block 0 is kept).
    blk = (tl.arange(0, BLOCK_R // R_ACT) == 0)[:, None, None]
    for t in range(TILES):
        vocab = start + t * BLOCK_V + tl.arange(0, BLOCK_V)
        vocab_mask = vocab < vocab_size
        w2 = tl.load(
            w2_ptr + vocab[:, None].to(tl.int64) * w2_stride + ks[None, :],
            mask=vocab_mask[:, None],
            other=0.0,
        )
        if SWAP:
            acc = tl.trans(tl.dot(w2, tl.trans(embed)))
        else:
            acc = tl.dot(embed, tl.trans(w2))
        if R_ACT < BLOCK_R:
            # The first R_ACT rows (a max with -inf is exact, -0.0 included).
            acc = tl.max(
                tl.where(
                    blk,
                    tl.reshape(acc, [BLOCK_R // R_ACT, R_ACT, BLOCK_V]),
                    float("-inf"),
                ),
                axis=0,
            )
        bias = acc.to(BIAS_DTYPE).to(tl.float32)
        mask = row_mask_a[:, None] & vocab_mask[None, :]
        base = tl.load(
            base_ptr
            + rows_a[:, None] * base_stride_req
            + step * base_stride_step
            + vocab[None, :],
            mask=mask,
            other=float("-inf"),
        )
        logits = (base.to(tl.float32) + bias).to(LOGITS_DTYPE)
        # The cache keeps the logits before temperature, in their own dtype.
        tl.store(
            cache_ptr
            + cache_row[:, None] * cache_stride_0
            + step * cache_stride_1
            + vocab[None, :],
            logits,
            mask=mask & (cache_row[:, None] >= 0),
        )
        x = logits.to(tl.float32)
        x = tl.where(noisy[:, None], x / temp[:, None], x)
        if any_noisy:
            u = murmur3_uniform32(seed[:, None], pos[:, None], vocab[None, :])
            noise = -tl.log(-_log1p_neg_stable(u))
            x = tl.where(
                noisy[:, None],
                tl.where(vocab_mask[None, :], x + noise, float("-inf")),
                x,
            )
        tile_max, tile_arg = tl.max(x, axis=1, return_indices=True)
        better = tile_max > best_val
        best_idx = tl.where(
            better, (start + t * BLOCK_V + tile_arg).to(tl.int64), best_idx
        )
        best_val = tl.where(better, tile_max, best_val)

    tl.atomic_max(
        keys_ptr + (step % 3) * BLOCK_R + rows_a,
        _argmax_key(best_val, best_idx),
        mask=row_mask_a,
        sem="relaxed",
    )


@triton.jit(do_not_specialize=["num_reqs", "step"])
def _markov_finish_kernel(
    keys_ptr, out_ptr, out_stride, num_reqs, step, BLOCK_R: tl.constexpr
):
    rows = tl.arange(0, BLOCK_R)
    row_mask = rows < num_reqs
    key = tl.load(keys_ptr + (step % 3) * BLOCK_R + rows)
    tl.store(out_ptr + rows * out_stride + step, _key_index(key), mask=row_mask)
    # Re-arm the buffer the next call's position 0 reduces into.
    tl.store(keys_ptr + rows, tl.full([BLOCK_R], _KEY_EMPTY, tl.int64))


def _tl_dtype(dtype: torch.dtype):
    return {
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
        torch.float32: tl.float32,
    }[dtype]


def markov_keys(device: torch.device) -> torch.Tensor:
    """The step kernels' reduction buffer ([3, MARKOV_MAX_REQUESTS] int64 keys)."""
    return torch.full(
        (3 * MARKOV_MAX_REQUESTS,), -(2**63), dtype=torch.int64, device=device
    )


def markov_sample(
    base_logits: torch.Tensor,  # [num_reqs, n_spec, V], last dim contiguous
    anchor_ids: torch.Tensor,  # strided [num_reqs] view of the anchor tokens
    w1: torch.Tensor,  # [>= target vocab, r]
    w2: torch.Tensor,  # [>= V, r]
    bias_dtype: torch.dtype,
    logits_cache: torch.Tensor,  # [max_reqs, n_spec, >= V]
    row_mapping: torch.Tensor | None,
    idx_mapping: torch.Tensor,  # [num_reqs * n_spec]
    sample_pos: torch.Tensor,  # [num_reqs * n_spec]
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    out_tokens: torch.Tensor,  # [>= num_reqs, n_spec] int64
    keys: torch.Tensor,  # markov_keys()
    num_programs: int,
    block_v: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    swap: bool | None = None,
) -> None:
    """Sample every draft position; see the module docstring."""
    num_reqs, n_spec, vocab_size = base_logits.shape
    assert num_reqs <= MARKOV_MAX_REQUESTS
    rank = w1.shape[1]
    logits_dtype = torch.promote_types(base_logits.dtype, bias_dtype)
    block_v = block_v or BLOCK_V
    tiles = triton.cdiv(triton.cdiv(vocab_size, num_programs), block_v)
    nblk = triton.cdiv(vocab_size, tiles * block_v)
    for step in range(n_spec):
        _markov_step_kernel[(nblk,)](
            base_logits,
            base_logits.stride(0),
            base_logits.stride(1),
            anchor_ids,
            anchor_ids.stride(0),
            w1,
            w1.stride(0),
            w2,
            w2.stride(0),
            logits_cache,
            logits_cache.stride(0),
            logits_cache.stride(1),
            row_mapping,
            idx_mapping,
            sample_pos,
            temperature,
            seeds,
            keys,
            out_tokens,
            out_tokens.stride(0),
            num_reqs,
            step,
            vocab_size,
            N_SPEC=n_spec,
            RANK=rank,
            BLOCK_R=MARKOV_MAX_REQUESTS,
            BLOCK_V=block_v,
            TILES=tiles,
            BIAS_DTYPE=_tl_dtype(bias_dtype),
            LOGITS_DTYPE=_tl_dtype(logits_dtype),
            FIRST=step == 0,
            SWAP=SWAP_MMA if swap is None else swap,
            R_ACT=min(MARKOV_MAX_REQUESTS, triton.next_power_of_2(num_reqs)),
            num_warps=num_warps or NUM_WARPS,
            num_stages=num_stages or NUM_STAGES,
        )
    _markov_finish_kernel[(1,)](
        keys,
        out_tokens,
        out_tokens.stride(0),
        num_reqs,
        n_spec - 1,
        BLOCK_R=MARKOV_MAX_REQUESTS,
        num_warps=1,
    )


class FusedMarkovSampler:
    """`sample_sequential` of a MiniMax-M3 DSpark draft model under the
    specialized decode (see `DSparkSpeculator._sample_sequential`)."""

    def __init__(self, model: nn.Module) -> None:
        # A weak handle would do as well; the draft model owns this object.
        self._model = model
        self._keys: torch.Tensor | None = None
        self._num_programs = 0
        self._checked: bool | None = None
        self._peer: PeerPushedLogits | None = None
        self._peer_checked = False

    def _supported(self, speculator: Any) -> bool:
        """Static checks, the same on every call (capture and replay)."""
        if self._checked is not None:
            return self._checked
        model = self._model
        head = model.model.markov_head
        w1 = head.markov_w1.weight
        w2 = getattr(head.markov_w2, "weight", None)
        lp = model.logits_processor
        reasons = []
        if speculator.draft_logits is None:
            reasons.append("greedy drafting (no draft logits cache)")
        if speculator.use_fp64_gumbel:
            reasons.append("FP64 Gumbel sampling")
        if getattr(speculator, "_d2t_scatter_index", None) is not None or (
            getattr(model, "draft_id_to_target_id", None) is not None
        ):
            reasons.append("a reduced draft vocabulary")
        if getattr(speculator, "draft_watermarker", None) is not None:
            reasons.append("a draft watermarker")
        if getattr(speculator, "acceptance_estimator", None) is not None:
            reasons.append("the online acceptance estimator")
        if getattr(speculator, "use_confidence_head", False):
            reasons.append("the confidence head")
        if hasattr(model, "apply_output_logit_transform"):
            reasons.append("an output logit transform")
        if getattr(model, "has_recurrent_markov_head", False):
            reasons.append("a recurrent Markov head")
        if w2 is None or w2.dim() != 2:
            reasons.append("a quantized Markov head")
        elif w2.dtype not in (torch.bfloat16, torch.float16):
            reasons.append(f"a {w2.dtype} Markov head")
        elif w1.dtype != w2.dtype or w1.shape[1] != w2.shape[1]:
            reasons.append("mismatched Markov weights")
        elif w1.shape[1] not in (64, 128, 256, 512):
            reasons.append(f"Markov rank {w1.shape[1]}")
        if lp.scale != 1.0 or lp.soft_cap is not None:
            reasons.append("scaled or soft-capped logits")
        if reasons:
            logger.info(
                "DSpark Markov sampling runs stock under the specialized decode: %s.",
                "; ".join(reasons),
            )
        else:
            logger.info(
                "DSpark Markov sampling: fused (%d launches per draft step).",
                speculator.num_speculative_steps + 1,
            )
        self._checked = not reasons
        return self._checked

    def _bias_dtype(self, embed_dtype: torch.dtype) -> torch.dtype:
        # LogitsProcessor._apply_head: an fp32 head accumulates into fp32.
        head_dtype = self._model.logits_processor.head_dtype
        if head_dtype is None or head_dtype == embed_dtype:
            return embed_dtype
        return head_dtype

    def __call__(self, speculator: Any, num_reqs: int, head_hidden: torch.Tensor):
        if num_reqs > MARKOV_MAX_REQUESTS or not self._supported(speculator):
            return False
        model = self._model
        n_spec = speculator.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[speculator.sample_indices[:num_sample]]
        peer = self._peer_logits(n_spec)
        base_logits = None if peer is None else peer(sample_hidden)
        if base_logits is None:
            base_logits = model.compute_draft_logits(sample_hidden)
            if base_logits.stride(-1) != 1:
                base_logits = base_logits.contiguous()
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)
        head = model.model.markov_head
        w1 = head.markov_w1.weight
        w2 = head.markov_w2.weight
        bias_dtype = self._bias_dtype(w1.dtype)
        cache = speculator.draft_logits
        logits_dtype = torch.promote_types(base_logits.dtype, bias_dtype)
        assert cache.values.dtype == logits_dtype, (
            f"draft logits cache is {cache.values.dtype}, logits {logits_dtype}"
        )
        assert cache.values.shape[-1] >= vocab_size and w2.shape[0] >= vocab_size
        device = base_logits.device
        if self._keys is None:
            sms = torch.cuda.get_device_properties(device).multi_processor_count
            self._num_programs = PROGRAMS_PER_SM * sms
            self._keys = markov_keys(device)
        markov_sample(
            base_logits,
            speculator.input_buffers.input_ids[
                : num_reqs * speculator.num_query_per_req : speculator.num_query_per_req
            ],
            w1,
            w2,
            bias_dtype,
            cache.values,
            cache.row_mapping,
            speculator.sample_idx_mapping[:num_sample],
            speculator.sample_pos[:num_sample],
            speculator.temperature,
            speculator.seeds,
            speculator.draft_tokens,
            self._keys,
            self._num_programs,
        )
        return True

    def warm_up(self, n_spec: int) -> None:
        """Collective, at start-up (`warm_up_drafters`): the draft LM head's
        peer logits, allocated and compiled before memory profiling."""
        peer = self._peer_logits(n_spec)
        if peer is not None:
            peer.warm_up()

    def _peer_logits(self, n_spec: int) -> "PeerPushedLogits | None":
        """The LM head through peer memory, or None for the NCCL gather."""
        if self._peer is not None or self._peer_checked:
            return self._peer
        self._peer_checked = True
        model = self._model
        lm_head = getattr(model, "lm_head", None)
        reason = peer_logits_unsupported(lm_head, model.logits_processor)
        if reason:
            logger.info("DSpark base logits stay gathered: %s.", reason)
            return None
        self._peer = peer_logits(
            lm_head,
            model.logits_processor,
            MARKOV_MAX_REQUESTS * n_spec,
            "draft_logits",
        )
        logger.info("DSpark base logits: through peer memory.")
        return self._peer


class ShardedContextFc:
    """`combine_hidden_states` with the context projection `fc` split over
    the TP ranks.

    `fc` ([hidden, 5 x hidden], BF16, unquantized) is replicated: every rank
    streams all of it (377 MB) for the 8 rows of a verify step. Here
    each rank multiplies its 1/TP slice of the input features (strided views,
    no copies) into FP32, the FP32 partials are all-reduced and rounded to the
    model dtype once, as a split-K GEMM would. Larger (prefill) batches keep
    the replicated GEMM.
    """

    # Rows up to which the split runs: the decode batches.
    MAX_ROWS = 64

    def __init__(self, model: nn.Module, stock_combine) -> None:
        self._model = model
        self._stock = stock_combine
        self._logged = False

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
            tensor_model_parallel_all_reduce,
        )

        inner = self._model.model
        fc = getattr(inner, "fc", None)
        tp = get_tensor_model_parallel_world_size()
        weight = getattr(fc, "weight", None)
        if (
            not getattr(inner, "use_aux_hidden_state", False)
            or tp == 1
            or hidden_states.dim() != 2
            or hidden_states.shape[0] > self.MAX_ROWS
            or weight is None
            or weight.dtype not in (torch.bfloat16, torch.float16)
            or getattr(fc, "bias", None) is not None
            or weight.shape[1] % tp
            or hidden_states.shape[-1] != weight.shape[1]
        ):
            return self._stock(hidden_states)
        if not self._logged:
            logger.info("DSpark context fc: split over %d TP ranks.", tp)
            self._logged = True
        width = weight.shape[1] // tp
        cols = slice(get_tensor_model_parallel_rank() * width, None)
        cols = slice(cols.start, cols.start + width)
        x = hidden_states.to(weight.dtype)[:, cols]
        partial = torch.mm(x, weight[:, cols].t(), out_dtype=torch.float32)
        return tensor_model_parallel_all_reduce(partial).to(weight.dtype)


@triton.jit
def _gemma_rmsnorm_kernel(
    x_ptr,
    x_stride,
    res_ptr,
    res_stride,
    w_ptr,
    out_ptr,
    out_stride,
    n,
    eps,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n
    x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0)
    if HAS_RESIDUAL:
        r = tl.load(res_ptr + row * res_stride + cols, mask=mask, other=0.0)
        # The sum is rounded to the residual's dtype, stored, and normalized.
        x = (x.to(tl.float32) + r.to(tl.float32)).to(r.dtype)
        tl.store(res_ptr + row * res_stride + cols, x, mask=mask)
    xf = x.to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(xf * xf, axis=0) / n + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = xf * rstd * (1.0 + w)
    tl.store(
        out_ptr + row * out_stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask
    )


class GemmaRMSNormRows:
    """`forward` of the draft model's hidden-size Gemma RMSNorms
    (`x * rsqrt(mean(x^2) + eps) * (1 + w)`, optionally fused with the
    residual add, in place like the FlashInfer path): one program per row.

    It replaces the FlashInfer CuTe-DSL fused add + norm for the 28-56 rows of
    a draft block (10 calls per draft step), which is slower at these rows.
    """

    def __init__(self, norm: nn.Module) -> None:
        self._norm = norm

    def __call__(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        norm = self._norm
        n = x.shape[-1]
        if (
            x.dim() != 2
            or x.stride(-1) != 1
            or x.shape[0] == 0
            or (residual is not None and residual.stride(-1) != 1)
        ):
            return type(norm).forward(norm, x, residual)
        block = triton.next_power_of_2(n)
        if residual is None:
            out = torch.empty_like(x)
            _gemma_rmsnorm_kernel[(x.shape[0],)](
                x,
                x.stride(0),
                x,
                0,
                norm.weight,
                out,
                out.stride(0),
                n,
                norm.variance_epsilon,
                HAS_RESIDUAL=False,
                BLOCK=block,
                num_warps=8,
            )
            return out
        _gemma_rmsnorm_kernel[(x.shape[0],)](
            x,
            x.stride(0),
            residual,
            residual.stride(0),
            norm.weight,
            x,
            x.stride(0),
            n,
            norm.variance_epsilon,
            HAS_RESIDUAL=True,
            BLOCK=block,
            num_warps=8,
        )
        return x, residual


def _patch_norms(model: nn.Module) -> int:
    hidden = model.config.hidden_size
    count = 0
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if (
            type(module).__name__ == "MiniMAXGemmaRMSNorm"
            and weight is not None
            and weight.numel() == hidden
        ):
            module.forward = GemmaRMSNormRows(module)
            count += 1
    return count


# The samplers attach_drafter installed in this process; the model state warms them up
# at start-up (warm_up_drafters).
_ATTACHED: list[weakref.ref] = []


def attach_drafter(model: nn.Module, vllm_config: Any) -> None:
    """Give a MiniMax-M3 DSpark draft model the fused Markov sampling when the
    specialized decode is on (the caller checks the switch; the stock draft
    model is left as it is otherwise)."""
    # Batch-sharded sampling shards the Markov loop across ranks instead; a model-level
    # sample_sequential would turn that off.
    if vllm_config.parallel_config.enable_batch_sharded_sampling:
        return
    sampler = FusedMarkovSampler(model)
    model.sample_sequential = sampler
    model._specialized_decode_drafter = True
    _ATTACHED.append(weakref.ref(sampler))
    model.combine_hidden_states = ShardedContextFc(model, model.combine_hidden_states)
    logger.info(
        "DSpark drafter: %d hidden-size norms on the row kernel.", _patch_norms(model)
    )
    from .draft_norm import attach_fused_norms

    # The norms' forward (the row kernel above) on an already reduced input.
    attach_fused_norms(model, lambda norm: norm.forward)


def warm_up_drafters(n_spec: int) -> None:
    """Collective (every rank, at start-up): warm up every attached sampler."""
    for ref in list(_ATTACHED):
        sampler = ref()
        if sampler is not None:
            sampler.warm_up(n_spec)
