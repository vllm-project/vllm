# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

import numpy as np
import torch

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.distributed.aux_output_connector.connector import AuxRequestOutput
    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorWorkerMetadata
    from vllm.distributed.ec_transfer.ec_connector.metrics import ECConnectorStats
    from vllm.distributed.kv_events import KVConnectorKVEvents
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorWorkerMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
else:
    AuxRequestOutput = object
    KVConnectorStats = object
    KVConnectorWorkerMetadata = object
    KVConnectorKVEvents = object
    ECConnectorWorkerMetadata = object


class LogprobsLists(NamedTuple):
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprob_token_ids: np.ndarray
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprobs: np.ndarray
    # [num_reqs x num_generated_tokens]
    sampled_token_ranks: np.ndarray
    # [num_reqs]
    # Used for slicing the logprobs in cases like speculative
    # decoding where the number of generated tokens may be
    # different for each request.
    cu_num_generated_tokens: list[int] | None = None

    def slice_request(self, req_idx: int, num_positions: int):
        if self.cu_num_generated_tokens is not None:
            req_idx = self.cu_num_generated_tokens[req_idx]
        end_idx = req_idx + num_positions
        return LogprobsLists(
            self.logprob_token_ids[req_idx:end_idx],
            self.logprobs[req_idx:end_idx],
            self.sampled_token_ranks[req_idx:end_idx],
            None,
        )


class SamplingMaskLists(NamedTuple):
    """CSR sampling masks; a request slice may contain multiple positions."""

    # [num_kept_tokens]
    token_ids: np.ndarray
    # [num_positions + 1], or None for a single position
    offsets: np.ndarray | None = None
    # [num_requests + 1] for multi-position request batches.
    cu_num_generated_tokens: list[int] | None = None

    def slice_request(self, req_idx: int, num_positions: int) -> "SamplingMaskLists":
        assert self.offsets is not None
        cu = self.cu_num_generated_tokens
        start = req_idx if cu is None else cu[req_idx]
        end = start + num_positions
        lo, hi = self.offsets[start], self.offsets[end]
        if num_positions == 1:
            return SamplingMaskLists(self.token_ids[lo:hi])
        return SamplingMaskLists(
            self.token_ids[lo:hi], self.offsets[start : end + 1] - lo
        )

    def to_nested_list(self) -> list[list[int]]:
        token_ids = self.token_ids.tolist()
        if self.offsets is None:
            return [token_ids]
        offsets = self.offsets.tolist()
        return [token_ids[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]


class LogprobsTensors(NamedTuple):
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprob_token_ids: torch.Tensor
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprobs: torch.Tensor
    # [num_reqs x num_generated_tokens]
    selected_token_ranks: torch.Tensor
    # [num_reqs + 1]
    cu_num_generated_tokens: list[int] | None = None
    # [num_reqs + 1]. Set instead of cu_num_generated_tokens when the
    # boundaries only exist on device (adaptive verification); rides along on
    # the async D2H copy.
    cu_num_generated_tokens_tensor: torch.Tensor | None = None

    def tolists(self, cu_num_generated_tokens: list[int] | None = None):
        if cu_num_generated_tokens is None:
            if self.cu_num_generated_tokens_tensor is not None:
                cu_num_generated_tokens = self.cu_num_generated_tokens_tensor.tolist()
            else:
                cu_num_generated_tokens = self.cu_num_generated_tokens
        return LogprobsLists(
            self.logprob_token_ids.cpu().numpy(),
            self.logprobs.cpu().numpy(),
            self.selected_token_ranks.cpu().numpy(),
            cu_num_generated_tokens,
        )

    def to_cpu_nonblocking(self) -> "LogprobsTensors":
        if self.logprob_token_ids.device.type == "cpu":
            return self
        cu_tensor = self.cu_num_generated_tokens_tensor
        if cu_tensor is not None:
            cu_tensor = cu_tensor.to("cpu", non_blocking=True)
        return LogprobsTensors(
            self.logprob_token_ids.to("cpu", non_blocking=True),
            self.logprobs.to("cpu", non_blocking=True),
            self.selected_token_ranks.to("cpu", non_blocking=True),
            self.cu_num_generated_tokens,
            cu_tensor,
        )

    def filter(self, mask: torch.Tensor) -> "LogprobsTensors":
        """Filter the logprobs tensors with the given bool mask."""
        assert self.cu_num_generated_tokens is None, (
            "filter can't be used with cu_num_generated_tokens"
        )
        assert self.cu_num_generated_tokens_tensor is None, (
            "filter can't be used with cu_num_generated_tokens_tensor"
        )
        return LogprobsTensors(
            self.logprob_token_ids[mask],
            self.logprobs[mask],
            self.selected_token_ranks[mask],
        )

    @staticmethod
    def cat(
        tensors: Sequence["LogprobsTensors"],
        cu_num_generated_tokens: list[int] | None = None,
    ) -> "LogprobsTensors":
        """Concatenate flattened logprob tensors."""
        assert tensors
        assert cu_num_generated_tokens is not None or all(
            tensor.cu_num_generated_tokens is None for tensor in tensors
        )
        if len(tensors) == 1:
            tensor = tensors[0]
            if cu_num_generated_tokens is None:
                return tensor
            return tensor._replace(cu_num_generated_tokens=cu_num_generated_tokens)
        # The multi-chunk path rebuilds boundaries from the CPU layout, which
        # the device-only boundaries of adaptive verification never use.
        assert all(tensor.cu_num_generated_tokens_tensor is None for tensor in tensors)
        return LogprobsTensors(
            logprob_token_ids=torch.cat(
                [tensor.logprob_token_ids for tensor in tensors]
            ),
            logprobs=torch.cat([tensor.logprobs for tensor in tensors]),
            selected_token_ranks=torch.cat(
                [tensor.selected_token_ranks for tensor in tensors]
            ),
            cu_num_generated_tokens=cu_num_generated_tokens,
        )

    @staticmethod
    def empty_cpu(
        num_positions: int, num_tokens_per_position: int
    ) -> "LogprobsTensors":
        """Create empty LogprobsTensors on CPU."""
        logprob_token_ids = torch.empty(
            (num_positions, num_tokens_per_position),
            dtype=torch.int32,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )
        logprobs = logprob_token_ids.new_empty(
            (num_positions, num_tokens_per_position),
            dtype=torch.float32,
            pin_memory=PIN_MEMORY,
        )
        selected_token_ranks = logprob_token_ids.new_empty(
            num_positions, pin_memory=PIN_MEMORY
        )
        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids,
            logprobs=logprobs,
            selected_token_ranks=selected_token_ranks,
        )


# [num_reqs, <dynamic>]
# The shape of each element depends on the pooler used
PoolerOutput: TypeAlias = torch.Tensor | list[torch.Tensor] | list[torch.Tensor | None]


@dataclass
class SamplerOutput:
    # [num_reqs, max_num_generated_tokens]
    # Different requests can have different number of generated tokens.
    # All requests are padded to max_num_generated_tokens.
    # PLACEHOLDER_TOKEN_ID (-1 by default) is used for padding.
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None


@dataclass(slots=True)
class KVLoadFailureReport:
    """One failed external-KV region, reported by the connector's worker side.

    Timing contract: identical to get_block_ids_with_load_errors()
    (sync-loading failures are reported in the forward pass in which they
    are detected). A report implies the failure is COMPLETE (no in-flight
    reads for the region).
    """

    request_id: str
    # The flat-set members (single KV cache group in v1).
    failed_block_ids: list[int]
    # [corrupt_start, corrupt_end) in token positions.
    token_span: tuple[int, int]
    # B_loaded: the repair's floor; <= token_span[0].
    rollback_boundary: int
    # The per-layer failure verdicts' union (physical layer indices).
    observed_failed_layers: list[int]
    # The above-span external ranges landed byte-exact.
    healthy_above_span: bool
    # The pre-failure external computed count C: the restoration target
    # at the repair commit.
    restored_extent: int
    # Diagnostic only (AUTHORITATIVELY set by the TP output
    # aggregator; a connector-supplied value is overwritten): the index
    # of the reporting worker in the gathered per-rank outputs. Two
    # reports of one fault from different ranks are identical
    # duplicates that differ only in this tag; the scheduler's evidence
    # comparison excludes it.
    source_rank: int | None = None
    # The identity of the transfer attempt whose failure this report
    # describes (see KVLoadAttestation.transfer_generation). Additive:
    # None marks evidence from a connector that has not implemented
    # the transfer-identity contract, which the repair policy's
    # evidence gates reject.
    transfer_generation: int | None = None
    # Hybrid attention failures use group-qualified physical IDs. Every named
    # recurrent group was fully delivered at restored_extent; a missing state
    # must never be reported through this attention-only recovery contract.
    failed_blocks_by_group: dict[int, list[int]] | None = None
    healthy_recurrent_group_ids: tuple[int, ...] = ()


@dataclass(slots=True)
class KVLoadAttestation:
    """One rank's positive, terminal statement about its portion of one
    external-KV load attempt ('kv_load_failure_policy="repair"').

    Reports are negative evidence: a rank that loaded everything
    successfully reports NOTHING, so a silent rank -- one whose GET
    'succeeded' against corrupt or stale content, or one that failed
    before reporting -- is indistinguishable from a healthy one. This
    attestation closes that gap: every rank that participated in a
    transfer attempt positively attests its completion, its expected
    object set, and the integrity certification it actually applied.
    The scheduler refuses to authorize a recovery unless every rank of
    the TP group supplied a terminal attestation: the missing /
    incomplete / unverifiable state never authorizes.

    Timing contract: identical to get_load_failure_reports(). Sync
    loads attest in the forward pass of the transfer; async loads
    attest no later than the pass in which the request is returned by
    get_transfer_results().

    Object counts are in the connector's own store-object units (the
    per-rank, per-layer, per-block cells it GETs); the contract fixes
    only the completeness arithmetic below, not the unit.

    Publication policy: under the repair policy, reusing an
    external claim -- publishing its blocks to the prefix cache, or
    restoring its above-span extent after a repair -- requires that
    every TP rank's terminal attestation also PROVE CONTENT IDENTITY
    for the objects it loaded (certifies_content()). Transfer
    completion alone authorizes nothing: a successful GET proves only
    that bytes arrived. The "size" and "none" levels are the connector
    explicitly reporting that it could NOT verify content; they are
    supported values with defined semantics (fail closed to the local
    recompute fallback), not a weaker publication mode -- there is no
    configuration under which unverified content becomes reusable.
    """

    request_id: str
    # The rank's expected object set size for this attempt: the
    # complete set the transfer promised this rank, not only the
    # subset a GET touched.
    expected_objects: int
    # Objects whose GET acknowledged bytes and did not fail; the
    # failed objects are counted separately (loaded + failed must
    # cover the complete expected set).
    loaded_objects: int
    # Objects whose GET failed; each must be covered by a failure
    # report for the same request and generation.
    failed_objects: int
    # The integrity certification the connector actually applied to
    # the loaded objects:
    #  - "sha256": per-object digest certification against a
    #    producer-recorded manifest (the strong reference mode).
    #  - "content": content-addressed immutable keys -- GET success
    #    implies correct content iff the store's key-to-data binding is
    #    trustworthy (the store-trust assumption).
    #  - "size": only length checks (no content verification).
    #  - "none": no verification beyond GET success.
    certification: str
    # Objects that passed certification (0 when certification is
    # "size"/"none": absence of proof, not a failure).
    certified_objects: int
    # Objects that FAILED certification: loaded bytes whose content
    # provably diverges from the producer's manifest.
    certification_failures: int
    # True when the attempt is complete: no in-flight reads remain and
    # the counts above are final for this generation.
    terminal: bool
    # The identity of the load attempt this attestation completes
    # (echoed from the scheduler through the connector metadata). None
    # marks a connector that has not implemented the transfer-identity
    # contract.
    transfer_generation: int | None = None
    # Diagnostic only (AUTHORITATIVELY set by the TP output
    # aggregator; a connector-supplied value is overwritten): the
    # index of the attesting worker in the gathered per-rank outputs.
    source_rank: int | None = None

    def well_formed(self) -> bool:
        """The structural invariants of a single attestation: a
        terminal attestation must account for its complete expected
        object set (loaded + failed == expected), its certified count
        cannot exceed what it loaded, no count may be negative, the
        certification must be one of the enumerated levels, and a
        certification failure can only appear alongside a certification
        capable of detecting it."""
        if (
            self.expected_objects < 0
            or self.loaded_objects < 0
            or self.failed_objects < 0
            or self.certified_objects < 0
            or self.certification_failures < 0
        ):
            return False
        if self.certification not in ("sha256", "content", "size", "none"):
            return False
        if self.terminal and (
            self.loaded_objects + self.failed_objects != self.expected_objects
        ):
            return False
        if self.certified_objects + self.certification_failures > self.loaded_objects:
            return False
        if self.certification_failures > 0:
            return self.certification in ("sha256", "content")
        return True

    def certifies_content(self) -> bool:
        """Whether this attestation independently proves content
        identity for every object it LOADED: a real certification
        level ("sha256" or "content"), zero certification failures, and
        every loaded object certified. The failed objects are excluded
        by construction -- they are the failure report's regions (the
        scheduler reconciles the counts), and those regions are what
        the repair recomputes; everything ABOVE the repair span is
        covered by the loaded-and-certified set. A vacuous attestation
        (no expected objects) proves nothing: a rank with a real
        external claim holds at least one object."""
        if self.certification not in ("sha256", "content"):
            return False
        if self.expected_objects < 1:
            return False
        return (
            self.certification_failures == 0
            and self.certified_objects == self.loaded_objects
        )


@dataclass(slots=True)
class RepairResultReport:
    """Worker-built verdict of one repair-only execution, consumed by the
    scheduler at update_from_output of the repair step (commit vs the
    recompute fallback).

    The production commit gate is the summary `structural_check`;
    `commit_check` is its per-element detail (diagnostics).
    """

    request_id: str
    # The plan identity echoed from the executed RepairWork -- the stale /
    # duplicate rejection key (with `epoch`).
    plan_id: int
    # The execution epoch (the scheduler step at which the plan was
    # admitted). A result echoing a different epoch is stale.
    epoch: int
    # The production commit-check verdict (the summary).
    structural_check: bool
    # How many layers the repair stream actually executed (== layer_end on
    # a clean run; < layer_end only on an aborted run).
    executed_layers: int
    # The physical repair span, for the scheduler's cross-check.
    span: tuple[int, int]
    # True when the repair was aborted by a forced/operational failure
    # after partial writes (as opposed to a commit-check failure).
    aborted: bool = False
    # The per-element commit-check detail (diagnostics).
    commit_check: dict[str, bool] | None = None
    # Diagnostic only: the type and message of the first rank-local
    # exception the worker's failure envelope contained (None on every
    # clean run). The full traceback lives in the worker log; this field
    # survives to the scheduler side so an abort's root cause is visible
    # even when worker logs are lost.
    error_detail: str | None = None


@dataclass
class KVConnectorOutput:
    # [req_ids]
    finished_sending: set[str] | None = None
    finished_recving: set[str] | None = None
    kv_connector_stats: KVConnectorStats | None = None
    kv_cache_events: KVConnectorKVEvents | None = None
    kv_connector_worker_meta: KVConnectorWorkerMetadata | None = None
    # IDs of externally computed KV blocks that failed to load.
    # Requests referencing these blocks should be rescheduled to recompute them
    invalid_block_ids: set[int] = field(default_factory=set)
    # Receive failures keyed by request identity. This remains unambiguous for
    # hybrid/multi-pool cache layouts where numeric block IDs overlap.
    failed_recving: set[str] = field(default_factory=set)
    # Structured load-failure regions reported by connectors that support
    # the hole-repair policy (`kv_load_failure_policy="repair"`). Absent
    # or empty for every other connector: behavior is byte-identical.
    load_failure_reports: list[KVLoadFailureReport] | None = None
    # Positive per-rank transfer attestations (the hole-repair policy's
    # completeness evidence): one terminal KVLoadAttestation per rank
    # per completed load attempt, produced even when the rank loaded
    # everything successfully. Absent or empty for connectors that do
    # not implement the attestation contract.
    load_attestations: list[KVLoadAttestation] | None = None
    # The repair-only step's commit verdicts (one per repair work item).
    # None on every ordinary step.
    repair_results: list[RepairResultReport] | None = None
    # Configuration describing how many finished sending/receiving
    # notifications should be expected for each request. This allows
    # handshake-based connectors like Nixl to update the KVOutputAggregator.
    # It captures a static setup info and should almost always remain constant
    # for a given connector after discovery. Default value entails no change.
    expected_finished_count: int = 0

    def is_empty(self):
        return (
            not self.finished_sending
            and not self.finished_recving
            and not self.kv_connector_stats
            and not self.kv_cache_events
            and not self.invalid_block_ids
            and not self.failed_recving
            and not self.load_failure_reports
            and not self.load_attestations
            and not self.repair_results
            and not self.kv_connector_worker_meta
        )


@dataclass
class ECConnectorOutput:
    # [mm_hash]
    finished_sending: set[str] | None = None
    finished_recving: set[str] | None = None
    ec_connector_stats: "ECConnectorStats | None" = None
    ec_connector_worker_meta: ECConnectorWorkerMetadata | None = None

    def is_empty(self):
        return (
            not self.finished_sending
            and not self.finished_recving
            and not self.ec_connector_stats
            and not self.ec_connector_worker_meta
        )


# ModelRunnerOutput is serialized and sent to the scheduler process.
# This is expensive for torch.Tensor so prefer to use list instead.
@dataclass
class ModelRunnerOutput:
    # [num_reqs]
    req_ids: list[str]

    # num_reqs x num_generated_tokens
    # num_generated_tokens is the number of tokens
    # generated in the current step. It can be different for
    # each request due to speculative/jump decoding.
    sampled_token_ids: list[list[int]] = field(default_factory=list)

    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs]
    logprobs: LogprobsLists | None = None

    # req_id -> (token_ids, logprobs, ranks)
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len]
    prompt_logprobs_dict: dict[str, LogprobsTensors | None] = field(
        default_factory=dict
    )

    # req_id -> [num_scored_rows, num_token_ids] prompt_logprob_token_ids scores.
    prompt_token_id_logprobs_dict: dict[str, torch.Tensor] = field(default_factory=dict)

    # [num_reqs, hidden_size]
    pooler_output: list[torch.Tensor | None] | None = None

    kv_connector_output: KVConnectorOutput | None = None

    ec_connector_output: ECConnectorOutput | None = None

    # req_id -> num_nans_in_logits
    num_nans_in_logits: dict[str, int] | None = None

    # information related to cudagraph execution
    cudagraph_stats: CUDAGraphStat | None = None

    aux_output_connector_output: dict[str, AuxRequestOutput] | None = None

    # ``None`` when ``return_sampling_mask`` is off.
    sampling_masks: SamplingMaskLists | None = None

    @staticmethod
    def with_kv_conn_output_only(
        kv_connector_output: KVConnectorOutput | None,
    ) -> "ModelRunnerOutput":
        """Return ModelRunnerOutput containing the provided KVConnectorOutput,
        otherwise empty. Returns None if kv_connector_output is passed as None.
        """
        if kv_connector_output is None or kv_connector_output.is_empty():
            return EMPTY_MODEL_RUNNER_OUTPUT
        output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output

    @staticmethod
    def with_ec_conn_output_only(
        ec_connector_output: ECConnectorOutput | None,
    ) -> "ModelRunnerOutput":
        """Return an otherwise-empty output carrying `ec_connector_output`."""
        return ModelRunnerOutput.with_ec_conn_output(
            EMPTY_MODEL_RUNNER_OUTPUT, ec_connector_output
        )

    @staticmethod
    def with_ec_conn_output(
        output: "ModelRunnerOutput",
        ec_connector_output: ECConnectorOutput | None,
    ) -> "ModelRunnerOutput":
        """Return `output` carrying `ec_connector_output`.

        The shared empty output is copied rather than written to, so callers
        must use the return value.
        """
        if ec_connector_output is None or ec_connector_output.is_empty():
            return output
        if output is EMPTY_MODEL_RUNNER_OUTPUT:
            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.ec_connector_output = ec_connector_output
        return output


# ModelRunnerOutput wrapper for async scheduling.
class AsyncModelRunnerOutput(ABC):
    @abstractmethod
    def get_output(self) -> ModelRunnerOutput:
        """Get the ModelRunnerOutput for this async output.

        This is a blocking call that waits until the results are ready, which
        might involve copying device tensors to the host.
        This method should only be called once per AsyncModelRunnerOutput.
        """
        pass

    @abstractmethod
    def get_kv_connector_output(self) -> KVConnectorOutput | None:
        """Get the underlying kv_connector_output.

        When get_output() fails, this is used to salvage the step's KV
        output instead of losing it with the failure.
        """
        pass


@dataclass
class DraftTokenIds:
    # [num_reqs]
    req_ids: list[str]
    # num_reqs x num_draft_tokens
    draft_token_ids: list[list[int]]


def make_empty_encoder_model_runner_output(
    scheduler_output: "SchedulerOutput",
) -> ModelRunnerOutput:
    """Create a ModelRunnerOutput stub that contains the correct
    per-request bookkeeping but no generated data yet.
    """
    if not scheduler_output.num_scheduled_tokens:
        return EMPTY_MODEL_RUNNER_OUTPUT

    # Convert to list so we get a deterministic, indexable sequence
    req_ids: list[str] = list(scheduler_output.num_scheduled_tokens.keys())

    # An encoder instance never samples, so it emits no tokens at all. The
    # scheduler finishes these requests once their prompt is fully encoded
    # (see `Scheduler.update_from_output`).
    sampled_token_ids: list[list[int]] = [[] for _ in req_ids]

    # Pooler outputs are not available yet ⇒ use None placeholders
    pooler_output: list[torch.Tensor | None] = [None for _ in req_ids]

    return ModelRunnerOutput(
        req_ids=req_ids,
        sampled_token_ids=sampled_token_ids,
        pooler_output=pooler_output,
    )


EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(req_ids=[])
