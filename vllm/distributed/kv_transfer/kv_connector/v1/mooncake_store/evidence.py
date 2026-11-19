# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-attempt transfer evidence for the MooncakeStore worker.

The production-side evidence producer of the hole-repair policy
(`kv_load_failure_policy="repair"`). One `AttemptEvidence` instance is
created by a receive thread at the START of every load attempt, carries
the scheduler-assigned `transfer_generation` captured once from the
load spec, and accumulates the ACTUAL outcomes of every store GET the
attempt issued.

Accounting unit
---------------
A Mooncake store object is one (group, shard, logical-block) chunk whose
payload packs EVERY KV-cache layer of its group for that block. The
scheduler evidence contract reconciles reports and attestations in
`(block, layer)` cells: `failed_objects == len(failed_block_ids) *
len(observed_failed_layers)`. This module therefore counts in the same
cell unit:

* `expected_cells` -- blocks this rank expected to load, times the
  layer count. A shard of a block is NOT a separate cell: any failed
  shard makes the whole block unusable, because the repair writes whole
  blocks and a partially-headed block is corrupt for attention.
* `failed_cells` -- failed blocks times the layer count.
* `certified_cells` -- blocks fully loaded under the claimed
  certification mode, times the layer count.

Truthfulness rules
-------------------
* The generation is captured once at attempt start and echoed unchanged
  in both the report and the attestation. It is never re-read at
  emission time, so a delayed old attempt can never inherit a newer
  attempt's identity.
* Certification is only ever one of the modes actually applied:
  `"content"` -- the store keys embed the block's chained token hash
  (an identity key, NOT a digest of the KV payload bytes), so GET
  success means the bytes stored under that identity key were returned:
  correct content iff the store's key-to-data binding is trustworthy
  (the documented store-trust assumption); `"none"` -- no verification
  beyond GET success. Digest verification (`"sha256"`) is never
  claimed: the Mooncake GET path performs no independent digest check.
  Deployments that do not trust the store key-to-data binding should
  configure `evidence_certification="none"`; the scheduler then fails
  publication closed, exactly as designed.
* An attempt that did not complete (interrupted, exception before
  terminality) never reports success: it stays non-terminal and its
  attestation is not emitted; the scheduler completeness gate then fails
  the delivery closed to recomputation.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field

from vllm.v1.outputs import KVLoadAttestation, KVLoadFailureReport


def validate_evidence_certification(mode: str) -> str:
    """Validate the configured evidence certification mode.

    Only `content` (content-addressed store keys under the documented
    store-trust assumption) and `none` (no verification beyond GET
    success) are producible by this worker: the GET path performs no
    independent digest check, so `sha256` is rejected -- never claim a
    verification that did not happen.

    Args:
        mode: The configured mode string.

    Returns:
        The validated mode.

    Raises:
        ValueError: For any mode other than `"content"`/`"none"`.

    """
    if mode not in ("content", "none"):
        raise ValueError(
            "MooncakeStore evidence_certification must be "
            '"content" or "none" (got '
            f"{mode!r}); the sha256 mode "
            "requires manifest support that the store GET path does "
            "not provide"
        )
    return mode


def evidence_enabled(
    kv_load_failure_policy: str | None,
    num_prefix_cacheable_groups: int,
    is_hma_required: bool,
) -> bool:
    """Whether production evidence emission may activate.

    Activation gates:

    * the repair policy must be explicitly configured -- under any other
      policy (including the default `fail`) the connector's evidence
      output would raise in the scheduler's evidence handler and the
      flat-channel suppression would destroy the only working failure
      channel, so the evidence channels stay fully off;
    * exactly ONE prefix-cacheable group AND a single config group
      (non-HMA): store block ids are group-local, so evidence from a
      multi-group scheduler cannot be represented without conflation,
      and the scheduler's report consumer is single-group-shaped.

    Args:
        kv_load_failure_policy: The configured load-failure policy.
        num_prefix_cacheable_groups: Number of prefix-cacheable groups.
        is_hma_required: Whether the scheduler runs a hybrid (multi-group)
            KV cache layout.

    Returns:
        True when the evidence channels may activate.

    """
    return (
        kv_load_failure_policy == "repair"
        and num_prefix_cacheable_groups == 1
        and not is_hma_required
    )


@dataclass
class AttemptEvidence:
    """Mutable per-attempt bookkeeping owned by one receive thread."""

    req_id: str
    transfer_generation: int | None
    num_layers: int
    block_size: int
    restored_extent: int
    certification_mode: str
    vllm_cached_tokens: int = 0
    # Logical (group-local) block indices this rank expected to load.
    expected_blocks: set[int] = field(default_factory=set)
    # Blocks with at least one failed store GET (or an exception mid-batch).
    failed_blocks: set[int] = field(default_factory=set)
    # Blocks fully loaded (every shard-object GET returned success).
    loaded_blocks: set[int] = field(default_factory=set)
    # Logical index -> physical block id (the scheduler's block table for
    # this request), captured during key enumeration.
    block_ids: dict[int, int] = field(default_factory=dict)
    terminal: bool = False

    def note_expected_block(self, logical_block: int) -> None:
        self.expected_blocks.add(logical_block)

    def note_block_outcome(self, logical_block: int, ok: bool) -> None:
        """Record the outcome of one store object covering `logical_block`.

        A block is loaded only when EVERY shard-object of that block
        succeeded (a partially-headed block is corrupt); it is failed as
        soon as any object failed.
        """
        if ok:
            self.loaded_blocks.add(logical_block)
        else:
            self.failed_blocks.add(logical_block)
            self.loaded_blocks.discard(logical_block)

    def note_blocks_failed(self, logical_blocks: Iterable[int]) -> None:
        for block in logical_blocks:
            self.note_block_outcome(block, False)

    def finalize(self) -> None:
        """Mark the attempt terminal. Called exactly once, when the receive
        handler for the request completes on any path (success, partial GET
        failure, or contained exception)."""
        self.terminal = True

    # ---- derived accounting (cell unit: (block, layer)) ----

    @property
    def expected_cells(self) -> int:
        return len(self.expected_blocks) * self.num_layers

    @property
    def loaded_cells(self) -> int:
        return len(self.loaded_blocks - self.failed_blocks) * self.num_layers

    @property
    def failed_cells(self) -> int:
        return len(self.failed_blocks) * self.num_layers

    @property
    def certified_cells(self) -> int:
        # The certification covers exactly the LOADED set (the failed
        # objects are the report's region -- excluded by construction:
        # certifies_content() on the attestation checks certified ==
        # loaded). Only "content" is producible here: "sha256" requires
        # per-object digest verification the store GET path does not
        # perform (validate_evidence_certification rejects it at
        # configuration time), so the accounting refuses to ever count it
        # as certified -- even a directly-constructed AttemptEvidence
        # cannot claim a verification that did not happen.
        if self.certification_mode == "content":
            return self.loaded_cells
        return 0

    @property
    def certifies_content(self) -> bool:
        """Whether the loaded set is fully certified under the mode
        (mirrors KVLoadAttestation.certifies_content: a real mode, every
        loaded object certified; the failed objects are the report's
        region and are excluded by construction). Only `content` can
        certify here -- see certified_cells."""
        if self.certification_mode != "content":
            return False
        return self.certified_cells > 0 and self.certified_cells == self.loaded_cells

    def to_attestation(self) -> KVLoadAttestation:
        return KVLoadAttestation(
            request_id=self.req_id,
            expected_objects=self.expected_cells,
            loaded_objects=self.loaded_cells,
            failed_objects=self.failed_cells,
            certification=self.certification_mode,
            certified_objects=self.certified_cells,
            certification_failures=0,
            terminal=self.terminal,
            transfer_generation=self.transfer_generation,
        )

    def to_failure_report(self) -> KVLoadFailureReport | None:
        """Build the v1 single-region failure report for this attempt.

        Returns `None` when the failure geometry cannot be represented by
        the single-region contract (more than one failed block, or a failed
        block without a captured physical id): the caller then emits no
        report and the scheduler's reconciliation fails the delivery closed
        to recomputation -- never an invented repairable region.

        The rollback boundary is the largest block-aligned prefix of
        CONTIGUOUSLY loaded blocks before the failed one: blocks the
        attempt never loaded are not healthy, so any hole before the
        failure forces the full-recompute floor (rollback 0). The locally
        cached prefix is already valid and does not require a store GET.
        """
        if len(self.failed_blocks) != 1:
            return None
        logical = next(iter(self.failed_blocks))
        block_id = self.block_ids.get(logical)
        if block_id is None:
            return None
        span_start = logical * self.block_size
        span_end = span_start + self.block_size
        rollback = span_start
        for idx in range(self.vllm_cached_tokens // self.block_size, logical):
            if idx not in self.loaded_blocks or idx in self.failed_blocks:
                rollback = 0
                break
        return KVLoadFailureReport(
            request_id=self.req_id,
            failed_block_ids=[block_id],
            token_span=(span_start, span_end),
            rollback_boundary=rollback,
            observed_failed_layers=list(range(self.num_layers)),
            healthy_above_span=self.certifies_content,
            restored_extent=self.restored_extent,
            transfer_generation=self.transfer_generation,
        )


class EvidenceBuffer:
    """Thread-safe hand-off of completed attempt evidence from the receive
    threads to the worker `get_load_*` drain (called once per step in
    `post_forward`)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._attempts: dict[str, AttemptEvidence] = {}
        # Terminal evidence displaced by a newer in-flight attempt (kept
        # for the next drain instead of being destroyed).
        self._drained_ready: list[AttemptEvidence] = []

    def start_attempt(self, evidence: AttemptEvidence) -> None:
        """Register a new attempt, replacing any stale entry for the same
        request, EXCEPT when the registered entry must win:

        * it carries a strictly NEWER generation (the multi-recv-thread
          race: an old attempt's handler can register after a new
          attempt's; the stale one must never clobber the current
          attempt's evidence); or
        * it is IDENTIFIED while the new one is identity-less (an
          identity-less attempt can never produce admissible evidence
          -- the scheduler's partition rejects it -- so replacing an
          identified attempt with it would only destroy the request's
          potentially valid evidence; the identity-less attempt still
          fails the delivery closed when nothing replaces it).
        """
        with self._lock:
            existing = self._attempts.get(evidence.req_id)
            if existing is not None:
                if (
                    existing.transfer_generation is not None
                    and evidence.transfer_generation is not None
                    and evidence.transfer_generation < existing.transfer_generation
                ):
                    return
                if (
                    existing.transfer_generation is not None
                    and evidence.transfer_generation is None
                ):
                    return
                if existing.terminal and not evidence.terminal:
                    # A TERMINAL-but-undrained entry's evidence is
                    # complete and about to be consumed by this step's
                    # drain; a newer NON-terminal registration (an
                    # in-flight attempt that just started) must not
                    # destroy it -- the delivery would silently demote to
                    # a full recompute (availability loss). Move the
                    # terminal evidence to the drain-ready list and
                    # register the new attempt: BOTH survive, each in its
                    # own channel.
                    self._drained_ready.append(existing)
                    self._attempts[evidence.req_id] = evidence
                    return
            self._attempts[evidence.req_id] = evidence

    def get(self, req_id: str) -> AttemptEvidence | None:
        with self._lock:
            return self._attempts.get(req_id)

    def drain_terminal(self) -> list[AttemptEvidence]:
        """Remove and return every terminal attempt. Non-terminal entries
        stay for a later pass (their requests are still in flight)."""
        with self._lock:
            terminal = [ev for ev in self._attempts.values() if ev.terminal]
            for ev in terminal:
                del self._attempts[ev.req_id]
            if self._drained_ready:
                terminal.extend(self._drained_ready)
                self._drained_ready = []
            return terminal

    def drop(self, req_id: str) -> None:
        """Forget a request pending evidence (finished/preempted without
        terminality, or connector shutdown)."""
        with self._lock:
            self._attempts.pop(req_id, None)

    def get_attempt(self, req_id: str) -> AttemptEvidence | None:
        """Return the request's currently registered attempt (read-only
        peek for identity-aware drop decisions; never mutates state)."""
        with self._lock:
            return self._attempts.get(req_id)
