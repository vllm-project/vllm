# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Server adapter for target-conditioned own-KV speculators.

Protocol commands become draft steps the way a co-located model runner
would schedule them:

- A prompt chunk that is not the last runs at once. Its final position is
  completed by the next prompt token, carried as NEXT_PREFILL_TOKENS.
- The last prompt chunk waits for the sequence's first round, because its
  final position needs the token the target samples from it. That round
  carries no target tokens for the sequence and supplies the sampled token
  as its recovery token.
- A later round covers the target tokens verified in the previous step:
  the accepted drafts and the recovery token were sampled, the remaining
  drafts were rejected.

The device work (block tables, attention metadata, the draft forward) is
delegated to a DraftStepExecutor.
"""

import itertools
from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence

import torch

from vllm.v1.spec_decode.remote.capabilities import (
    SpeculatorPlacementCapabilities,
    TargetFeatureKind,
)
from vllm.v1.spec_decode.remote.protocol import (
    SequenceKey,
    SpeculatorStatusCode,
    TargetFeatureSchema,
)
from vllm.v1.spec_decode.remote.runtime.step import (
    DraftStepPlan,
    DraftStepPlanner,
    DraftStepRow,
)
from vllm.v1.spec_decode.remote.server import (
    AdapterError,
    PrefillInputs,
    StandaloneDraftAdapter,
    StandaloneProposalBatch,
    StandaloneProposalResult,
)
from vllm.v1.spec_decode.remote.session import frame_to_tensor, tensor_to_frame
from vllm.v1.spec_decode.remote.transport import DataFrame, frame_to_ints

Features = list[torch.Tensor | None]


class DraftStepExecutor(ABC):
    """Runs planned draft steps on the drafter's device."""

    def warmup(self) -> None:  # noqa: B027
        """Complete device initialization before the server accepts requests."""

    @property
    def max_num_tokens(self) -> int | None:
        """Maximum tokens per execution, or None for an unbounded executor."""
        return None

    @abstractmethod
    def execute(
        self,
        plan: DraftStepPlan,
        features: Features,
        *,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        """Run one draft step and return ``[plan.num_reqs, K]`` draft tokens.

        ``features`` follow the negotiated schema: token-level tensors hold
        ``plan.num_tokens`` rows in plan order, per-sequence tensors hold
        ``plan.num_reqs`` rows, and omitted optional slots are None. The
        executor applies ``plan.block_table_updates`` to its block tables
        before running the step.

        ``num_speculative_tokens`` limits draft generation without skipping
        context updates; None uses the configured maximum. Return width stays
        at that maximum even when the effective budget is zero.
        """


class TargetConditionedDraftAdapter(StandaloneDraftAdapter):
    """Feeds protocol rows through a DraftStepPlanner into an executor."""

    def __init__(
        self,
        executor: DraftStepExecutor,
        planner: DraftStepPlanner,
        *,
        capabilities: SpeculatorPlacementCapabilities,
        feature_schema: TargetFeatureSchema,
        methods: Collection[str],
        draft_checkpoint_fingerprint: str = "",
    ) -> None:
        self._lookahead_slot = _lookahead_slot(
            feature_schema, planner.num_prefill_lookahead
        )
        if capabilities.num_prefill_lookahead != planner.num_prefill_lookahead:
            raise ValueError(
                f"capabilities declare {capabilities.num_prefill_lookahead} "
                f"prefill lookahead tokens, the planner uses "
                f"{planner.num_prefill_lookahead}"
            )
        self.executor = executor
        self.planner = planner
        self._capabilities = capabilities
        self._schema = feature_schema
        self._methods = frozenset(methods)
        self._fingerprint = draft_checkpoint_fingerprint
        self._last_chunks: dict[SequenceKey, tuple[int, Features]] = {}
        """Final prompt chunks waiting for the token sampled from them."""

    def capabilities(self) -> SpeculatorPlacementCapabilities:
        return self._capabilities

    def warmup(self) -> None:
        self.executor.warmup()

    def feature_schema(self) -> TargetFeatureSchema:
        return self._schema

    def supported_methods(self) -> Collection[str]:
        return self._methods

    def draft_checkpoint_fingerprint(self) -> str:
        return self._fingerprint

    def open_sequence(self, key: SequenceKey) -> None:
        if not self.planner.open(key):
            raise AdapterError(
                SpeculatorStatusCode.QUEUE_FULL,
                f"all {self.planner.max_num_reqs} draft request slots are in use",
            )

    def prefill(self, key: SequenceKey, inputs: PrefillInputs) -> None:
        if inputs.num_tokens <= 0:
            raise AdapterError(
                SpeculatorStatusCode.PROTOCOL_ERROR,
                "a prompt chunk must contain at least one token",
            )
        features = _tensors(inputs.features)
        if inputs.is_final:
            self._last_chunks[key] = (inputs.num_tokens, features)
            return
        lookahead = features[self._lookahead_slot]
        if lookahead is None:
            raise AdapterError(
                SpeculatorStatusCode.PROTOCOL_ERROR,
                "a prompt chunk that is not the last must carry next_prefill_tokens",
            )
        row = DraftStepRow(
            key,
            num_tokens=inputs.num_tokens,
            num_sampled=0,
            next_prefill_tokens=tuple(int(t) for t in lookahead[0].tolist()),
        )
        plan, failures = self.planner.plan([row], num_speculative_tokens=0)
        if failures:
            raise AdapterError(*failures[0])
        # A prompt chunk only fills draft KV; its drafts are never used.
        self.executor.execute(plan, features, num_speculative_tokens=0)
        self.planner.commit(plan)

    def advance_and_propose(
        self, batch: StandaloneProposalBatch
    ) -> StandaloneProposalResult:
        num_rows = len(batch.keys)
        effective_k = batch.effective_num_speculative_tokens
        if effective_k is None:
            effective_k = batch.num_speculative_tokens
        if not 0 <= effective_k <= batch.num_speculative_tokens:
            raise ValueError("round draft budget exceeds the negotiated maximum")
        accepted = frame_to_ints(batch.accepted_counts)
        recovery = frame_to_ints(batch.recovery_tokens)
        features = _tensors(batch.features)
        starts = list(itertools.accumulate(batch.query_lens, initial=0))
        statuses = [SpeculatorStatusCode.OK] * num_rows
        rows: list[DraftStepRow] = []
        batch_rows: list[int] = []
        sources: list[Features] = []
        for i in batch.active_rows:
            key = batch.keys[i]
            query_len = batch.query_lens[i]
            last_chunk = self._last_chunks.get(key)
            if last_chunk is not None:
                # The round only delivers the token sampled from the prompt.
                if query_len or accepted[i]:
                    statuses[i] = SpeculatorStatusCode.SEQUENCE_DESYNC
                    continue
                num_tokens, source = last_chunk
                row = DraftStepRow(
                    key, num_tokens, num_sampled=1, last_sampled=recovery[i]
                )
            else:
                if not 0 <= accepted[i] < query_len:
                    statuses[i] = SpeculatorStatusCode.SEQUENCE_DESYNC
                    continue
                source = [
                    None
                    if tensor is None or spec.per_sequence
                    else tensor[starts[i] : starts[i] + query_len]
                    for spec, tensor in zip(self._schema.slots, features)
                ]
                row = DraftStepRow(
                    key,
                    query_len,
                    num_sampled=accepted[i] + 1,
                    num_rejected=query_len - 1 - accepted[i],
                    last_sampled=recovery[i],
                )
            rows.append(row)
            batch_rows.append(i)
            sources.append(source)

        max_num_tokens = self.executor.max_num_tokens
        groups: list[list[int]] = [[]]
        num_tokens = 0
        for i, row in enumerate(rows):
            if max_num_tokens is not None:
                if row.num_tokens > max_num_tokens:
                    statuses[batch_rows[i]] = SpeculatorStatusCode.OUT_OF_MEMORY
                    continue
                if num_tokens + row.num_tokens > max_num_tokens:
                    groups.append([])
                    num_tokens = 0
            groups[-1].append(i)
            num_tokens += row.num_tokens
        token_ids = torch.zeros(
            num_rows, batch.num_speculative_tokens, dtype=torch.int64
        )
        for group in groups:
            plan, failures = self.planner.plan(
                [rows[i] for i in group], num_speculative_tokens=effective_k
            )
            for index, (status, _) in failures.items():
                statuses[batch_rows[group[index]]] = status
            planned = [
                (batch_rows[i], sources[i])
                for index, i in enumerate(group)
                if index not in failures
            ]
            if not planned:
                continue
            drafts = self.executor.execute(
                plan,
                self._step_features(features, planned),
                num_speculative_tokens=effective_k,
            )
            self.planner.commit(plan)
            row_index = torch.tensor([i for i, _ in planned], dtype=torch.long)
            token_ids[row_index, :effective_k] = drafts[:, :effective_k].to(
                "cpu", torch.int64
            )
            for row in plan.rows:
                self._last_chunks.pop(row.key, None)
        valid_lengths = torch.tensor(
            [
                effective_k if status is SpeculatorStatusCode.OK else 0
                for status in statuses
            ],
            dtype=torch.int32,
        )
        return StandaloneProposalResult(
            token_ids=tensor_to_frame(0, token_ids),
            valid_lengths=tensor_to_frame(0, valid_lengths),
            row_statuses=tuple(statuses),
        )

    def close_sequences(self, keys: tuple[SequenceKey, ...]) -> None:
        for key in keys:
            self._last_chunks.pop(key, None)
            self.planner.close(key)

    def _step_features(
        self, round_features: Features, planned: Sequence[tuple[int, Features]]
    ) -> Features:
        """Assemble the step's features in plan order.

        Token-level rows come from each row's source (a buffered prompt
        chunk or its segment of the round); per-sequence rows always come
        from the round. An optional token-level slot is passed on only when
        every row has it.
        """
        step: Features = []
        for j, spec in enumerate(self._schema.slots):
            if spec.per_sequence:
                tensor = round_features[j]
                step.append(
                    None
                    if tensor is None
                    else tensor[torch.tensor([i for i, _ in planned])]
                )
                continue
            segments = [source[j] for _, source in planned]
            if any(segment is None for segment in segments):
                step.append(None)
            else:
                step.append(torch.cat(segments))  # type: ignore[arg-type]
        return step


def _tensors(frames: Sequence[DataFrame | None]) -> Features:
    return [None if frame is None else frame_to_tensor(frame) for frame in frames]


def _lookahead_slot(schema: TargetFeatureSchema, num_lookahead: int) -> int:
    for i, spec in enumerate(schema.slots):
        if spec.kind == TargetFeatureKind.NEXT_PREFILL_TOKENS.value:
            if (
                not spec.per_sequence
                or not spec.optional
                or tuple(spec.trailing_shape) != (num_lookahead,)
            ):
                raise ValueError(
                    "next_prefill_tokens must be an optional per-sequence slot "
                    f"of {num_lookahead} tokens; got {spec}"
                )
            return i
    raise ValueError("the feature schema needs a next_prefill_tokens slot")
