# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Muse Spark DSpark Markov chain sharded by vocabulary over TP (ROCm).

`VLLM_MUSE_DSPARK_MARKOV_VOCAB_SHARD=1`. vLLM's loop all-gathers the draft
logits, then every rank reads the whole replicated `markov_w2` ([V, r]) for
the bias of each draft position. Here rank `k` keeps the draft lm_head's
vocabulary shard `[k V / tp, (k + 1) V / tp)`:

    head          local logits [R, n, V / tp] (no logits all-gather)
    position i    bias = embed @ markov_w2[shard].T            (1/tp of the bytes)
                  per 1024-block (max, index) of the shard     (`_shard_sample_kernel`)
                  all-gather of the (index, value) keys        (R x 2 x blocks fp64)
                  first NaN block, else first maximum          (`_shard_combine_kernel`)
    probabilistic one all-gather of the cached shards, then a copy into the
                  draft logits cache rows                      (`_cache_scatter_kernel`)

The draws equal vLLM's loop: the Gumbel noise is keyed by the global token id,
the logits are the same elements, and the winner is the lowest-index maximum
in either block partition. Only a NaN logit can choose differently, because the
block boundaries move.

Greedy drafts at TP 4 replace the keys all-gather and the combine launch with
one FlyDSL launch a position (`mono/kernels/markov_combine.py`): each rank's
shard argmax goes to every peer's mailbox and every rank picks the first
maximum. It takes the argmax of `base + bias` before Muse's output transform,
which is increasing, so the pick is the same unless the transform rounds two
different logits to one fp32 value.

The model installs `sample_sequential` from `attach`. Every check that sends a
step back is the same on all ranks, so the ranks always enter the all-gathers
together.
"""

from collections.abc import Callable

import torch
from torch._inductor.runtime.triton_helpers import libdevice

from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.models.muse_spark.amd.mono.decoder import (
    HEALTH_POLL_INTERVAL,
    register_health_check,
)
from vllm.models.muse_spark.amd.mono.peer import PeerBuffer
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax, spec_step_seed
from vllm.v1.worker.gpu.spec_decode.draft_logits import draft_logits_row

logger = init_logger(__name__)

BLOCK_SIZE = 1024
SCATTER_BLOCK = 4096


@triton.jit(do_not_specialize=["cache_col", "vocab_start"])
def _shard_sample_kernel(
    keys_ptr,
    keys_stride,
    num_blocks,
    base_ptr,
    base_stride,
    bias_ptr,
    bias_stride,
    stage_ptr,
    stage_stride_0,
    stage_stride_1,
    cache_col,
    idx_map_ptr,
    idx_map_stride,
    pos_ptr,
    pos_stride,
    seeds_ptr,
    temp_ptr,
    multiplier,
    inv_soft_cap,
    soft_cap,
    shard_size,
    vocab_start,
    BLOCK_SIZE: tl.constexpr,
    HAS_SOFT_CAP: tl.constexpr,
    USE_FP64: tl.constexpr,
    GREEDY: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < shard_size

    base = tl.load(
        base_ptr + token_idx * base_stride + block, mask=mask, other=float("-inf")
    )
    bias = tl.load(bias_ptr + token_idx * bias_stride + block, mask=mask, other=0.0)
    # `base_logits[:, i] + bias` rounds the sum to the logits dtype.
    logits = (base.to(tl.float32) + bias.to(tl.float32)).to(base.dtype)
    logits = logits.to(tl.float32) * multiplier
    if HAS_SOFT_CAP:
        # torch divides by a scalar as a multiply by its fp32 reciprocal.
        logits = soft_cap * libdevice.tanh(logits * inv_soft_cap)

    if GREEDY:
        # greedy_argmax: NaN is -inf, ties resolve to the first index.
        values = tl.where(mask, logits, float("-inf"))
        values = tl.where(values != values, float("-inf"), values)
        idx = tl.argmax(values, axis=0)
        value = tl.max(values, axis=0)
    else:
        # Cached and sampled in the cache dtype, as vLLM's loop casts first.
        logits = logits.to(stage_ptr.dtype.element_ty)
        tl.store(
            stage_ptr + token_idx * stage_stride_0 + cache_col * stage_stride_1 + block,
            logits,
            mask=mask,
        )
        keys = vocab_start + block
        req_state_idx = tl.load(idx_map_ptr + token_idx * idx_map_stride).to(tl.int64)
        is_valid_req = req_state_idx >= 0
        temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(
            tl.float32
        )
        seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
        seed = spec_step_seed(seed, cache_col)
        # Draws are keyed by the position before the sampled token.
        pos = tl.load(pos_ptr + token_idx * pos_stride) - 1
        value, idx = gumbel_noised_argmax(
            logits.to(tl.float32),
            keys,
            mask,
            seed,
            pos,
            temp,
            IS_DRAFTING=True,
            USE_FP64=USE_FP64,
            APPLY_TEMPERATURE=True,
        )
    # fp64 holds both exactly: the global token id and an fp32/fp64 score.
    gidx = vocab_start + block_idx * BLOCK_SIZE + idx
    tl.store(keys_ptr + token_idx * keys_stride + block_idx, gidx.to(tl.float64))
    tl.store(
        keys_ptr + token_idx * keys_stride + num_blocks + block_idx,
        value.to(tl.float64),
    )


@triton.jit
def _shard_combine_kernel(
    gathered_ptr,
    gathered_stride,
    num_blocks,
    total_blocks,
    out_ptr,
    out_stride,
    w1_ptr,
    w1_stride,
    embed_ptr,
    embed_stride,
    markov_rank,
    PADDED_TOTAL: tl.constexpr,
    RANK_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    # Global block g = rank * num_blocks + b, ascending in vocabulary order.
    g = tl.arange(0, PADDED_TOTAL)
    gmask = g < total_blocks
    rank = g // num_blocks
    b = g - rank * num_blocks
    base = gathered_ptr + row * gathered_stride + rank * (2 * num_blocks)
    values = tl.load(base + num_blocks + b, mask=gmask, other=float("-inf"))
    # torch.argmax: the first NaN wins, otherwise the first maximum.
    nan_block = tl.min(tl.where(values != values, g, PADDED_TOTAL), axis=0)
    maximum = tl.max(values, axis=0)
    first_max = tl.min(tl.where(gmask & (values == maximum), g, PADDED_TOTAL), axis=0)
    winner = tl.where(nan_block < PADDED_TOTAL, nan_block, first_max)
    w_rank = winner // num_blocks
    w_b = winner - w_rank * num_blocks
    token = tl.load(
        gathered_ptr + row * gathered_stride + w_rank * (2 * num_blocks) + w_b
    ).to(tl.int64)
    tl.store(out_ptr + row * out_stride, token)

    r = tl.arange(0, RANK_BLOCK)
    rmask = r < markov_rank
    embed = tl.load(w1_ptr + token * w1_stride + r, mask=rmask)
    tl.store(embed_ptr + row * embed_stride + r, embed, mask=rmask)


@triton.jit
def _cache_scatter_kernel(
    full_ptr,
    full_stride,
    cache_ptr,
    cache_stride_0,
    cache_stride_1,
    cache_row_mapping_ptr,
    idx_map_ptr,
    idx_map_stride_0,
    idx_map_stride_1,
    num_steps,
    vocab_size,
    BLOCK: tl.constexpr,
):
    flat = tl.program_id(0).to(tl.int64)
    req = flat // num_steps
    step = flat - req * num_steps
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < vocab_size
    req_state_idx = tl.load(
        idx_map_ptr + req * idx_map_stride_0 + step * idx_map_stride_1
    ).to(tl.int64)
    row = draft_logits_row(req_state_idx, cache_row_mapping_ptr)
    vals = tl.load(full_ptr + flat * full_stride + cols, mask=mask)
    tl.store(
        cache_ptr + row * cache_stride_0 + step * cache_stride_1 + cols,
        vals,
        mask=mask & (row >= 0),
    )


@triton.jit
def _bias_round_kernel(
    base_ptr,
    base_stride,
    bias_ptr,
    bias_stride,
    out_ptr,
    shard_size,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < shard_size
    b = tl.load(base_ptr + row * base_stride + cols, mask=mask, other=0.0)
    s = tl.load(bias_ptr + row * bias_stride + cols, mask=mask, other=0.0)
    # `_shard_sample_kernel`'s logits before the output transform, widened.
    v = (b.to(tl.float32) + s.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + row * shard_size + cols, v, mask=mask)


# One mailbox region a draft position: 16 B a (rank, row) record.
_REGION_BYTES = 4096
_COMBINE_WORLD = 4


class _GreedyCombine:
    """The fused combine's buffers and launches, built at load on every rank."""

    def __init__(
        self, kernel, group, rows: int, n_spec: int, shard: int, device
    ) -> None:
        self.rank = group.rank_in_group
        self.peers = PeerBuffer(_REGION_BYTES * n_spec, group.cpu_group)
        self.epoch = torch.zeros(1, dtype=torch.int32, device=device)
        self.health = torch.zeros(8 + 8 * rows, dtype=torch.int32, device=device)
        self.out = torch.zeros(n_spec, rows, dtype=torch.int32, device=device)
        self.logits = torch.empty(rows, shard, dtype=torch.float32, device=device)
        self.shard = shard
        self.launch = {
            r: kernel.compile_combine(group.world_size, r, shard)
            for r in range(1, rows + 1)
        }
        self._health_host = torch.zeros_like(self.health, device="cpu").pin_memory()
        self._health_event: torch.cuda.Event | None = None
        self._calls = 0

    def run(
        self, position: int, base: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """The draft tokens of `position` (int32 `[rows]`)."""
        rows = base.shape[0]
        _bias_round_kernel[(rows, triton.cdiv(self.shard, 1024))](
            base,
            base.stride(0),
            bias,
            bias.stride(0),
            self.logits,
            self.shard,
            BLOCK=1024,
        )
        pointers = [p + position * _REGION_BYTES for p in self.peers.pointers]
        self.launch[rows](
            self.logits.data_ptr(),
            self.out[position].data_ptr(),
            self.epoch.data_ptr(),
            self.health.data_ptr(),
            *pointers,
            *([0] * (8 - len(pointers))),
            self.shard,
            self.rank,
            torch.cuda.current_stream(),
        )
        return self.out[position, :rows]

    def poll_health(self) -> None:
        """Raise if a launch timed out waiting for a peer rank (an asynchronous
        copy every HEALTH_POLL_INTERVAL calls, checked one interval later)."""
        if torch.cuda.is_current_stream_capturing():
            return
        self._calls += 1
        if self._health_event is not None and self._health_event.query():
            self._health_event = None
            if int(self._health_host[0]) != 0:
                raise RuntimeError(
                    f"DSpark fused greedy combine: rank {self.rank}: a launch "
                    "timed out waiting for a peer rank"
                )
        if self._health_event is None and self._calls % HEALTH_POLL_INTERVAL == 0:
            self._health_host.copy_(self.health, non_blocking=True)
            self._health_event = torch.cuda.Event()
            self._health_event.record()


def _greedy_combine(model, shard: int) -> _GreedyCombine | None:
    """The fused combine for greedy drafts at TP 4, or None. Collective over
    the TP group: every rank calls it and all take the same answer."""
    import torch.distributed as dist

    group = get_tp_group()
    config = model.vllm_config
    rows = config.scheduler_config.max_num_seqs
    n_spec = config.speculative_config.num_speculative_tokens
    reasons = []
    if group.world_size != _COMBINE_WORLD:
        reasons.append(f"tp {group.world_size} (kernel: {_COMBINE_WORLD})")
    if not 1 <= rows <= 56 or 16 * group.world_size * rows > _REGION_BYTES:
        reasons.append(f"max_num_seqs {rows}")
    if shard % 4 != 0:
        reasons.append(f"vocab shard {shard}")
    if not model.output_multiplier > 0:
        reasons.append("an output transform that is not increasing")
    kernel = None
    if not reasons:
        from vllm.models.muse_spark.amd.mono import _ensure_writable_flydsl_cache

        _ensure_writable_flydsl_cache()
        try:
            from vllm.models.muse_spark.amd.mono.kernels import markov_combine

            kernel = markov_combine
        except ImportError as e:
            reasons.append(f"needs FlyDSL and AITER's FlyDSL helpers ({e})")
    verdict = torch.tensor([0 if reasons else 1], dtype=torch.int32)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN, group=group.cpu_group)
    if int(verdict.item()) == 0:
        logger.info(
            "DSpark fused greedy combine off: %s",
            "; ".join(reasons) or "a peer rank refused it",
        )
        return None
    device = torch.device("cuda", torch.accelerator.current_device_index())
    combine = _GreedyCombine(kernel, group, rows, n_spec, shard, device)
    register_health_check(id(combine), combine.poll_health)
    logger.info(
        "DSpark fused greedy combine engaged: %d positions, shard %d.",
        n_spec,
        shard,
    )
    return combine


def _all_gather_last(local: torch.Tensor) -> torch.Tensor:
    """`[M, N]` a rank -> `[M, tp * N]` in rank order."""
    local = local.contiguous()
    if local.dtype == torch.float64:
        # The ROCm TP all-gather takes AITER's path for 16/32-bit dtypes only;
        # fp64 would fall back to RCCL inside the draft graph. Same bytes.
        return _all_gather_last(local.view(torch.float32)).view(torch.float64)
    return tensor_model_parallel_all_gather(local, -1)


class _HeadShard:
    """A row slice of a replicated head, for `LogitsProcessor._apply_head`."""

    def __init__(self, head: torch.nn.Module, start: int, size: int) -> None:
        self.weight = head.weight[start : start + size]
        self.quant_method = head.quant_method


def static_refusal(model: torch.nn.Module) -> str | None:
    """Why this deployment cannot shard the chain; None when it can."""
    tp = get_tp_group()
    if tp.world_size == 1:
        return "tp == 1"
    if model.vllm_config.parallel_config.enable_batch_sharded_sampling:
        return "batch-sharded sampling is on"
    if model.draft_id_to_target_id.any().item():
        return "the draft->target map is not the identity"
    lm_head = model.lm_head
    lp = model.logits_processor
    if not isinstance(lm_head, ParallelLMHead):
        return "no vocab-parallel draft lm_head"
    if lp.logits_as_input or lp.soft_cap is not None or lp.scale != 1.0:
        return "the draft logits processor transforms its logits"
    idx = lm_head.shard_indices
    vocab = model.config.draft_vocab_size
    shard = vocab // tp.world_size
    if (
        lm_head.tp_size != tp.world_size
        or idx.num_org_vocab_padding != 0
        or vocab % tp.world_size != 0
        or idx.org_vocab_start_index != tp.rank_in_group * shard
        or idx.org_vocab_end_index - idx.org_vocab_start_index != shard
    ):
        return "the draft lm_head shards are not equal contiguous vocab slices"
    head = model.model.markov_head
    w1, w2 = head.markov_w1.weight, head.markov_w2
    if tuple(w2.weight.shape) != (vocab, w1.shape[1]) or w2.tp_size != 1:
        return "markov_w2 is not a replicated [V, r] weight"
    if not w2.weight.is_contiguous() or not isinstance(
        w2.quant_method, UnquantizedEmbeddingMethod
    ):
        return "markov_w2 is not a plain contiguous weight"
    return None


class MarkovVocabShard:
    """The draft model's `sample_sequential` with the chain on a vocab shard."""

    def __init__(
        self,
        model: torch.nn.Module,
        fallback: Callable[..., bool] | None,
    ) -> None:
        idx = model.lm_head.shard_indices
        self.vocab_size = model.config.draft_vocab_size
        self.vocab_start = idx.org_vocab_start_index
        self.shard_size = idx.org_vocab_end_index - idx.org_vocab_start_index
        head = model.model.markov_head
        self.w1 = head.markov_w1.weight
        self.w2_shard = _HeadShard(head.markov_w2, self.vocab_start, self.shard_size)
        self.num_blocks = triton.cdiv(self.shard_size, BLOCK_SIZE)
        self.greedy = (
            model.vllm_config.speculative_config.draft_sample_method != "probabilistic"
        )
        # The step to hand back to: the --monokernel Markov rounds or vLLM's loop.
        self.fallback = fallback
        self.combine = _greedy_combine(model, self.shard_size) if self.greedy else None
        self.keys: torch.Tensor | None = None
        self.stage: torch.Tensor | None = None
        self.embeds: torch.Tensor | None = None

    def _refusal(self, spec, num_reqs: int) -> str | None:
        if spec.draft_watermarker is not None:
            return "a draft watermarker"
        if spec.acceptance_estimator is not None:
            return "the acceptance estimator"
        if (spec.draft_logits is None) != self.greedy:
            return "the draft logits cache does not match the draft sample method"
        cache = spec.draft_logits
        if cache is not None and (
            cache.row_mapping is not None or cache.values.shape[-1] != self.vocab_size
        ):
            return "the draft logits cache layout"
        if num_reqs > spec.max_num_reqs:
            return f"{num_reqs} requests"
        return None

    def _buffers(self, spec) -> None:
        """Allocated at the first (warm-up) step and kept, so CUDA graphs keep
        their addresses."""
        if self.keys is not None:
            return
        rows, n_spec = spec.max_num_reqs, spec.num_speculative_steps
        self.keys = torch.empty(
            rows, 2 * self.num_blocks, dtype=torch.float64, device=spec.device
        )
        if not self.greedy:
            self.stage = torch.empty(
                rows,
                n_spec,
                self.shard_size,
                dtype=spec.draft_logits.values.dtype,
                device=spec.device,
            )
        # Step i writes the embedding that step i + 1's bias reads; slot 0 keeps
        # the anchor's for the confidence head.
        self.embeds = torch.empty(
            n_spec + 1, rows, self.w1.shape[1], dtype=self.w1.dtype, device=spec.device
        )

    def sample_sequential(self, spec, num_reqs: int, head_hidden: torch.Tensor) -> bool:
        why = self._refusal(spec, num_reqs)
        if why is not None:
            logger.debug_once(
                "DSpark vocab-sharded Markov chain off for a step: %s", why
            )
            return self.fallback is not None and self.fallback(
                spec, num_reqs, head_hidden
            )
        self._buffers(spec)
        assert self.keys is not None and self.embeds is not None
        model = spec.model
        n_spec = spec.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[spec.sample_indices[:num_sample]]
        # compute_draft_logits without the all-gather.
        base = model.logits_processor(
            model.lm_head,
            sample_hidden.to(model.lm_head.weight.dtype),
            skip_gather=True,
        ).view(num_reqs, n_spec, self.shard_size)
        prev = spec.input_buffers.input_ids[spec._anchor_idx[:num_reqs]]
        idx_map = spec.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = spec.sample_pos[:num_sample].view(num_reqs, n_spec)
        out_tokens = spec.draft_tokens[:num_reqs]
        embeds = self.embeds[:, :num_reqs]
        keys = self.keys[:num_reqs]
        stage = None if self.stage is None else self.stage[:num_reqs]
        soft_cap = model.output_soft_cap_temp
        markov_embed = model.markov_embed(prev)
        if spec.use_confidence_head:
            embeds[0].copy_(markov_embed)
        combine = self.combine
        if combine is not None:
            combine.poll_health()
            combine.epoch.add_(1)
        for i in range(n_spec):
            # `markov_bias` on this rank's rows of markov_w2.
            bias = model.logits_processor._apply_head(self.w2_shard, markov_embed, None)
            if combine is not None:
                tokens = combine.run(i, base[:, i], bias)
                embeds[i + 1] = torch.index_select(self.w1, 0, tokens.long())
                markov_embed = embeds[i + 1]
                continue
            _shard_sample_kernel[(num_reqs, self.num_blocks)](
                keys,
                keys.stride(0),
                self.num_blocks,
                base[:, i],
                base.stride(0),
                bias,
                bias.stride(0),
                keys if stage is None else stage,
                0 if stage is None else stage.stride(0),
                0 if stage is None else stage.stride(1),
                i,
                idx_map[:, i],
                idx_map.stride(0),
                sample_pos[:, i],
                sample_pos.stride(0),
                spec.seeds,
                spec.temperature,
                float(model.output_multiplier),
                0.0 if soft_cap is None else 1.0 / float(soft_cap),
                0.0 if soft_cap is None else float(soft_cap),
                self.shard_size,
                self.vocab_start,
                BLOCK_SIZE=BLOCK_SIZE,
                HAS_SOFT_CAP=soft_cap is not None,
                USE_FP64=bool(spec.use_fp64_gumbel),
                GREEDY=stage is None,
            )
            gathered = _all_gather_last(keys)
            total_blocks = gathered.shape[1] // 2
            _shard_combine_kernel[(num_reqs,)](
                gathered,
                gathered.stride(0),
                self.num_blocks,
                total_blocks,
                out_tokens[:, i],
                out_tokens.stride(0),
                self.w1,
                self.w1.stride(0),
                embeds[i + 1],
                embeds.stride(1),
                self.w1.shape[1],
                PADDED_TOTAL=triton.next_power_of_2(total_blocks),
                RANK_BLOCK=triton.next_power_of_2(self.w1.shape[1]),
            )
            markov_embed = embeds[i + 1]
        if combine is not None:
            out_tokens.copy_(combine.out[:n_spec, :num_reqs].t())
        if stage is not None:
            full = _all_gather_last(stage.reshape(num_sample, self.shard_size))
            values = spec.draft_logits.values
            _cache_scatter_kernel[
                (num_sample, triton.cdiv(self.vocab_size, SCATTER_BLOCK))
            ](
                full,
                full.stride(0),
                values,
                values.stride(0),
                values.stride(1),
                spec.draft_logits.row_mapping,
                idx_map,
                idx_map.stride(0),
                idx_map.stride(1),
                n_spec,
                self.vocab_size,
                BLOCK=SCATTER_BLOCK,
            )
        if spec.use_confidence_head:
            spec.draft_token_confidence_probs[:num_reqs] = spec._confidence(
                sample_hidden, embeds[:n_spec].transpose(0, 1), out_tokens.shape
            )
        return True


def attach(model: torch.nn.Module) -> None:
    """Install the sharded chain as the draft model's `sample_sequential`,
    ahead of a `sample_sequential` already there (the --monokernel Markov
    rounds), which then runs only the steps the sharded chain sends back."""
    fallback = getattr(model, "sample_sequential", None)
    if isinstance(getattr(fallback, "__self__", None), MarkovVocabShard):
        return  # a weight reload
    reason = static_refusal(model)
    if reason is not None:
        logger.warning(
            "VLLM_MUSE_DSPARK_MARKOV_VOCAB_SHARD=1 ignored: %s.",
            reason,
        )
        return
    shard = MarkovVocabShard(model, fallback)
    model.sample_sequential = shard.sample_sequential
    logger.info(
        "DSpark Markov chain vocab-sharded: shard [%d, %d) of %d, %d blocks, %s%s.",
        shard.vocab_start,
        shard.vocab_start + shard.shard_size,
        shard.vocab_size,
        shard.num_blocks,
        "greedy" if shard.greedy else "probabilistic",
        ", ahead of the --monokernel Markov rounds" if fallback is not None else "",
    )
