# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculative verification of Muse Spark's drafts against vocab-sharded logits.

With every request of a verify batch greedy, or sampling with a top-k of at
most 64 and no other logits processing, the masked target logits of each row
are among its 64 largest logits. Instead of all-gathering `[rows, vocab]`
logits, every TP rank soft-caps its head shard and multicasts that shard's 64
largest (`ShardTopK`, 512 bytes per row) into every rank's
`CandidateMailbox`, and every rank masks the ranks' candidates by the
request's temperature, min-p, top-k and top-p and verifies the drafts from
them (`ShardedVerify`), as the stock sampler and `rejection_sample`, which run
for every other batch, do.
Greedy is identical to stock; sampling follows the same distribution but not
the same random draws as the stock sampler, so a seeded request can differ
between the two paths.
"""

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState, use_penalty
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

from ..cutedsl.common.symmetric import NoMulticastError
from ..cutedsl.sample.shard_top_k import CandidateMailbox, K, ShardTopK
from ..cutedsl.sample.verify import BLOCK, STANDARD, SYNTHETIC, ShardedVerify
from ..runtime.nvfp4_lm_head import processor_is_identity

logger = init_logger(__name__)

# Logits rows a batch may verify here: up to eight requests of seven drafts.
MAX_ROWS = 64
# The logits processors every `Sampler` builds, which `plain_top_k` requests
# skip.
BUILT_IN_PROCESSORS = (LogitBiasState, PenaltiesState, BadWordsState)


def has_custom_logits_processors(sampler: Sampler) -> bool:
    return not all(type(p) in BUILT_IN_PROCESSORS for p in sampler.logits_processors)


def _unsupported_reason(vllm_config: VllmConfig, model, sampler: Sampler) -> str | None:
    """Why the model's verification cannot be vocab-sharded, or None. The
    specialized decode's own requirements (`runtime.unsupported`) hold."""
    tp = get_tp_group()
    spec = vllm_config.speculative_config
    vocab = vllm_config.model_config.get_vocab_size()
    indices = model.lm_head.shard_indices
    processor = model.logits_processor
    # The specialized decode's own requirements guarantee a DSpark draft and
    # a TP size above one.
    assert spec is not None
    checks = (
        (
            "no ShardedVerify kernels for the TP size, vocab or draft count",
            not ShardedVerify.supports(
                tp.world_size, vocab, spec.num_speculative_tokens
            ),
        ),
        (
            "an uneven or padded vocab split",
            vocab % tp.world_size != 0
            or indices.num_org_elements != vocab // tp.world_size
            or indices.num_org_elements_padded != indices.num_org_elements
            or indices.num_added_elements != 0
            or indices.org_vocab_start_index
            != tp.rank_in_group * indices.num_org_elements,
        ),
        (
            "no ShardTopK kernel for the shard",
            not ShardTopK.supports(vocab // tp.world_size),
        ),
        ("no output soft cap", model.mul_tanh_soft_cap is None),
        ("a logits processor transform", not processor_is_identity(processor)),
        (
            "FLy verification",
            spec.rejection_sample_method == "fly"
            or (
                spec.rejection_sample_method == "synthetic"
                and spec.fly_window_size is not None
            ),
        ),
        (
            (
                "sampler options (NaN counts, sampling masks, trace replay or "
                "custom logits processors)"
            ),
            sampler.compute_nans
            or sampler.return_sampling_mask
            or sampler.trace_replay_state is not None
            or has_custom_logits_processors(sampler),
        ),
    )
    return next((reason for reason, failed in checks if failed), None)


def plain_top_k(params: SamplingParams) -> bool:
    """Whether a request has no logits processing and wants no logprobs.
    Its temperature and top-k are checked on the sampler's state
    (`VocabShardedVerifier.apply_staged_writes`)."""
    return (
        params.logprobs is None
        and not params.logprob_token_ids
        and not use_penalty(params)
        and not params.logit_bias
        and not params.allowed_token_ids
        and not params.bad_words_token_ids
        and not (params.min_tokens and params.all_stop_token_ids)
        and params.thinking_token_budget is None
        and params.structured_outputs is None
    )


class VocabShardedVerifier:
    """Per-rank kernels and buffers of the vocab-sharded verification.

    Every rank must run the same candidate exchanges (`CandidateMailbox`) in
    the same order: a `ShardTopK` launch, then one select over its rows. A
    step's exchange runs on every rank or none: whether a batch verifies here
    (`plan`) depends only on the batch and its requests' parameters, which
    every rank of the group shares, and `logits` posts only for a planned
    batch that `verify` then selects. `warm_up` runs the same exchanges on
    every rank (the dummy run's rows are the same on each).
    """

    def __init__(
        self,
        model,
        sampler: Sampler,
        stock: RejectionSampler,
        mailbox: CandidateMailbox,
        num_speculative_steps: int,
        max_concurrent_batches: int,
        device: torch.device,
    ) -> None:
        tp = get_tp_group()
        self.model = model
        self.sampler = sampler
        self.stock = stock
        self.shard = model.lm_head.shard_indices.num_org_elements
        self.vocab_offset = tp.rank_in_group * self.shard
        self.vocab = tp.world_size * self.shard
        mode = STANDARD
        if stock.use_block_verification:
            mode = BLOCK
        elif stock.synthetic_conditional_rates is not None:
            mode = SYNTHETIC
        self.top_k = ShardTopK(self.shard, MAX_ROWS, device)
        self.kernels = ShardedVerify(
            tp.world_size, self.vocab, MAX_ROWS, num_speculative_steps, mode, device
        )
        self.rank = tp.rank_in_group
        self.mailbox = mailbox
        # What `logits` returns for a batch's packets, by row count: views of
        # a token of its own that `owns` knows by identity (`verify` polls the
        # mailbox), so nothing reaches the mailbox through it.
        token = torch.zeros(MAX_ROWS, 1, dtype=torch.int32, device=device)
        self._handles = [token[:rows] for rows in range(MAX_ROWS + 1)]
        self.rows_per_request = num_speculative_steps + 1
        self.device_index = self.top_k.device_index
        self.side_stream = torch.cuda.Stream(device=device)
        self.forked, self.joined = torch.cuda.Event(), torch.cuda.Event()
        # A set of verify outputs per batch in flight, taken in turn: a step's
        # async output copy ends before the engine (`max_concurrent_batches`
        # in its batch queue) submits the step that reuses its set.
        self.outputs = [
            (
                torch.empty(
                    MAX_ROWS, self.rows_per_request, dtype=torch.int64, device=device
                ),
                torch.empty(MAX_ROWS, dtype=torch.int32, device=device),
                torch.empty(MAX_ROWS, dtype=torch.int32, device=device),
            )
            for _ in range(max_concurrent_batches)
        ]
        # The first `num_reqs` rows of each output set, by (set, num_reqs).
        self._output_views: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}
        self.step = 0
        # Per request slot: verifies here, and is greedy. Set from the
        # request's parameters and, once the sampler holds them, its sampling
        # state, which no later update changes.
        max_num_reqs = sampler.sampling_states.max_num_reqs
        self.eligible = np.zeros(max_num_reqs, dtype=bool)
        self.greedy = np.zeros(max_num_reqs, dtype=bool)
        self.added: list[int] = []
        # The planned batch, and whether it verifies here (`plan`).
        self._batch: InputBatch | None = None
        # Packets posted by `logits` that `verify` has not selected yet.
        self._posted = False
        self._takes = False
        self._all_greedy = False

    @staticmethod
    def build(
        vllm_config: VllmConfig, model, sampler: Sampler, device: torch.device
    ) -> "VocabShardedVerifier | None":
        reason = _unsupported_reason(vllm_config, model, sampler)
        if reason is not None:
            logger.info_once("Vocab-sharded verification disabled: %s.", reason)
            return None
        # Collective; every rank gets a multicast mapping or none.
        tp = get_tp_group()
        try:
            mailbox = CandidateMailbox(tp.world_size, MAX_ROWS, device, tp.device_group)
        except NoMulticastError as error:
            logger.warning("Vocab-sharded verification disabled: %s.", error)
            return None
        spec = vllm_config.speculative_config
        assert spec is not None
        stock = RejectionSampler(sampler, spec, device)
        verifier = VocabShardedVerifier(
            model,
            sampler,
            stock,
            mailbox,
            spec.num_speculative_tokens,
            max(2, vllm_config.max_concurrent_batches),
            device,
        )
        logger.info_once("Vocab-sharded verification enabled.")
        return verifier

    def add_request(self, req_index: int, params: SamplingParams | None) -> None:
        self.eligible[req_index] = False
        self.greedy[req_index] = False
        if params is not None and plain_top_k(params):
            self.added.append(req_index)

    def apply_staged_writes(self) -> None:
        """Complete the added requests' eligibility from the sampler's state:
        greedy, or a top-k of at most K (any temperature, top-p and min-p)."""
        if not self.added:
            return
        states = self.sampler.sampling_states
        added = np.array(self.added)
        greedy = states.temperature.np[added] == 0.0
        self.greedy[added] = greedy
        self.eligible[added] = greedy | (
            (states.top_k.np[added] <= K) & (states.min_p.np[added] < 1.0)
        )
        self.added.clear()

    def plan(self, input_batch: InputBatch) -> None:
        """Note this step's batch and whether it verifies here."""
        slots = input_batch.idx_mapping_np
        self._batch = input_batch
        self._takes = (
            input_batch.num_draft_tokens > 0
            and not input_batch.has_structured_output_reqs
            and input_batch.num_admitted_draft_tokens is None
            and input_batch.pcp is None
            and 0 < input_batch.logits_indices.shape[0] <= MAX_ROWS
            and bool(self.eligible[slots].all())
        )
        self._all_greedy = self._takes and bool(self.greedy[slots].all())

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Post this rank's candidate packets of the planned batch and return
        their handle (what `compute_sample_logits` returns for it), or None
        to compute the logits."""
        if not self._takes:
            self._batch = None
            return None
        # A post without its select would pair every later post with the
        # wrong select on this rank (`CandidateMailbox`).
        if self._posted:
            raise RuntimeError("candidate packets posted but never verified")
        self._posted = True
        return self._post(self.model.compute_head_logits_local(hidden_states))

    def warm_up(self, hidden_states: torch.Tensor) -> None:
        """Run the head, `ShardTopK` and `select` (the exchange, on every rank)
        on one row and on every full verify batch's rows (whole requests of
        every draft, up to `MAX_ROWS`), and `draft_stats` and `verify` on one
        dummy request, so that their kernels load and the profiled peak
        includes them before serving; check the head's output layout. Every
        rank runs the same exchanges: the dummy run's rows are the same on
        each."""
        if hidden_states.shape[0] == 0:
            return
        device = hidden_states.device

        def full(value, dtype, n=1):
            return torch.full((n,), value, dtype=dtype, device=device)

        int32, int64, fp32 = torch.int32, torch.int64, torch.float32
        # One sampling request state (top-k 1, temperature 1).
        temperature = full(1.0, fp32)
        x = hidden_states.new_zeros(MAX_ROWS, hidden_states.shape[1])
        x[: min(hidden_states.shape[0], MAX_ROWS)] = hidden_states[:MAX_ROWS]
        rows = self.rows_per_request
        for count in (1, *range(rows, MAX_ROWS + 1, rows)):
            raw = self.model.compute_head_logits_local(x[:count])
            assert self.top_k.takes(raw), "the head's local logits layout changed"
            self._post(raw)
            self.kernels.select(
                self.mailbox,
                count,
                full(0, int64, count),
                temperature,
                full(1, int32),
                full(1.0, fp32),
                full(0.0, fp32),
            )
        draft = torch.zeros(1, rows, self.vocab, dtype=torch.bfloat16, device=device)
        idx_mapping = full(0, int64, rows)
        local_pos = torch.arange(rows, dtype=int32, device=device)
        self.kernels.draft_stats(draft, idx_mapping, local_pos, temperature)
        self.kernels.verify(
            draft,
            torch.tensor([0, rows], dtype=int32, device=device),
            full(0, int64),
            torch.arange(rows, dtype=int64, device=device),
            full(0, int64, rows),
            full(0, int32, rows),
            local_pos,
            full(0, int32),
            full(0, int32),
            temperature,
            full(0, int64),
            self.stock.synthetic_conditional_rates,
            out=self._outputs(0, 1),
        )

    def _outputs(self, index: int, num_reqs: int) -> tuple[torch.Tensor, ...]:
        """The first `num_reqs` rows of output set `index`."""
        key = (index, num_reqs)
        views = self._output_views.get(key)
        if views is None:
            views = tuple(t[:num_reqs] for t in self.outputs[index])
            self._output_views[key] = views
        return views

    def _post(self, raw: torch.Tensor) -> torch.Tensor:
        """Post this rank's candidate packets of its head logits shard `raw`
        to every rank's mailbox; returns their handle."""
        soft_cap = self.model.mul_tanh_soft_cap
        self.top_k.launch(
            raw,
            soft_cap.multiplier,
            soft_cap.soft_cap_temp,
            self.vocab_offset,
            self.mailbox,
            self.rank,
            torch.cuda.current_stream(self.device_index).cuda_stream,
        )
        return self._handles[raw.shape[0]]

    def owns(self, logits: torch.Tensor) -> bool:
        rows = logits.shape[0]
        return rows <= MAX_ROWS and logits is self._handles[rows]

    def verify(
        self,
        packets: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: DraftLogitsCache | None,
    ) -> SamplerOutput:
        if input_batch is not self._batch:
            raise RuntimeError("candidate packets of another batch")
        self._batch = None
        self._posted = False
        assert draft_logits is not None and draft_logits.row_mapping is None
        draft = draft_logits.values
        assert draft.dtype == torch.bfloat16 and draft.shape[-1] == self.vocab
        rows = packets.shape[0]
        main = torch.cuda.current_stream()
        states = self.sampler.sampling_states
        # Greedy rows never read the draft stats.
        if not self._all_greedy:
            self.forked.record(main)
            self.side_stream.wait_event(self.forked)
            self.kernels.draft_stats(
                draft,
                input_batch.expanded_idx_mapping,
                input_batch.expanded_local_pos,
                states.temperature.gpu,
                self.side_stream.cuda_stream,
            )
            self.joined.record(self.side_stream)
        stream = main.cuda_stream
        self.kernels.select(
            self.mailbox,
            rows,
            input_batch.expanded_idx_mapping,
            states.temperature.gpu,
            states.top_k.gpu,
            states.top_p.gpu,
            states.min_p.gpu,
            stream,
        )
        if not self._all_greedy:
            main.wait_event(self.joined)
        num_reqs = input_batch.idx_mapping.shape[0]
        out = self._outputs(self.step % len(self.outputs), num_reqs)
        self.step += 1
        sampled, num_sampled, num_rejected = self.kernels.verify(
            draft,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            input_batch.logits_indices,
            input_batch.positions,
            input_batch.input_ids,
            input_batch.expanded_local_pos,
            self.sampler.req_states.prefill_len.gpu,
            input_batch.seq_lens,
            states.temperature.gpu,
            states.seeds.gpu,
            self.stock.synthetic_conditional_rates,
            stream,
            out,
        )
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            sampling_mask_tensors=None,
        )


class VocabShardedRejectionSampler:
    """The stock rejection sampler, but for the batches `verifier` planned,
    whose logits are its candidate packets."""

    # The specialized decode does not support adaptive verification.
    enable_adaptive_verification = False

    def __init__(self, verifier: VocabShardedVerifier):
        self.verifier = verifier

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: DraftLogitsCache | None = None,
    ) -> SamplerOutput:
        if self.verifier.owns(logits):
            return self.verifier.verify(logits, input_batch, draft_logits)
        return self.verifier.stock(logits, input_batch, draft_logits)
