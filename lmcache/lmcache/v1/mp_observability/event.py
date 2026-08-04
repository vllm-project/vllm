# SPDX-License-Identifier: Apache-2.0

"""Unified event model for the MP observability system."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(Enum):
    """All observable event types in the MP system.

    Naming convention: ``<COMPONENT>_<OPERATION>`` or
    ``<COMPONENT>_<OPERATION>_<PHASE>`` for start/end pairs.
    """

    # L1 Manager events
    L1_READ_RESERVED = "l1.read.reserved"
    L1_READ_FINISHED = "l1.read.finished"
    L1_WRITE_RESERVED = "l1.write.reserved"
    L1_WRITE_FINISHED = "l1.write.finished"
    L1_WRITE_FINISHED_AND_READ_RESERVED = "l1.write_finished_and_read_reserved"
    L1_KEYS_EVICTED = "l1.keys.evicted"
    L1_KEYS_ACCESSED = "l1.keys.accessed"
    L1_EVICTION_LOOP_TICK = "l1.eviction.loop_tick"

    # L1 failure events (LM-291 health monitoring)
    L1_ALLOCATION_FAILED = "l1.allocation.failed"
    L1_READ_FAILED = "l1.read.failed"

    # StorageManager events
    SM_READ_PREFETCHED = "sm.read.prefetched"
    SM_READ_PREFETCHED_FINISHED = "sm.read.prefetched_finished"
    SM_WRITE_RESERVED = "sm.write.reserved"
    SM_WRITE_FINISHED = "sm.write.finished"

    # L2 Store Controller events
    L2_STORE_SUBMITTED = "l2.store.submitted"
    L2_STORE_COMPLETED = "l2.store.completed"

    # L2 Prefetch Controller events
    L2_PREFETCH_LOOKUP_SUBMITTED = "l2.prefetch.lookup.submitted"
    L2_PREFETCH_LOOKUP_COMPLETED = "l2.prefetch.lookup.completed"
    L2_PREFETCH_LOAD_SUBMITTED = "l2.prefetch.load.submitted"
    L2_PREFETCH_LOAD_COMPLETED = "l2.prefetch.load.completed"
    # Per-adapter load task events, for throughput correlation.  Fire once
    # per (request_id, adapter_index) pair, unlike the request-level
    # L2_PREFETCH_LOAD_* events above which aggregate across adapters.
    L2_LOAD_TASK_SUBMITTED = "l2.load_task.submitted"
    L2_LOAD_TASK_COMPLETED = "l2.load_task.completed"

    # L2 Eviction Controller events
    L2_KEYS_EVICTED = "l2.keys.evicted"

    # L2 adapter key-level events.
    L2_KEYS_STORED = "l2.keys.stored"
    L2_KEYS_ACCESSED = "l2.keys.accessed"
    L2_KEYS_DELETED = "l2.keys.deleted"

    # L2 failure events (LM-291 health monitoring)
    L2_PREFETCH_FAILED = "l2.prefetch.failed"

    # MP Server request-level events (start/end pairs)
    MP_STORE_START = "mp.store.start"
    MP_STORE_END = "mp.store.end"
    MP_RETRIEVE_START = "mp.retrieve.start"
    MP_RETRIEVE_END = "mp.retrieve.end"
    MP_LOOKUP_PREFETCH_START = "mp.lookup_prefetch.start"
    MP_LOOKUP_PREFETCH_END = "mp.lookup_prefetch.end"

    # Chunk hash logging events
    MP_LOOKUP = "mp.lookup"

    # MP Server lifecycle sentinels (CPU-synchronous)
    MP_REQUEST_START = "mp.request.start"
    MP_RETRIEVE_SUBMITTED = "mp.retrieve.submitted"
    MP_STORE_SUBMITTED = "mp.store.submitted"
    MP_REQUEST_END = "mp.request.end"

    # vLLM block allocation events
    MP_VLLM_BLOCK_ALLOCATION = "mp.vllm.block_allocation"

    # vLLM end session events
    MP_VLLM_END_SESSION = "mp.vllm.end_session"

    # Published by LMCacheTimeoutError on construction (see errors.py).
    # Metadata: message (str), exception_type (str), stacktrace (str, mapped
    # to the OTel exception.stacktrace span attribute).
    TIMEOUT_RAISED = "timeout.raised"

    # Trace recording — unified function-call entry event used by the
    # ``@enable_tracing`` decorator.  Metadata layout:
    #   ``qualname`` (str):   fully-qualified function name
    #   ``args``     (dict):  name -> raw Python value (codec-encoded at
    #                         record time by the recorder)
    #   ``t_mono``   (float): ``time.monotonic()`` captured at publish
    #                         time, so it is comparable to
    #                         ``Event.timestamp`` (wall-clock) even
    #                         though the drain thread processes the
    #                         event later
    TRACE_CALL = "trace.call"

    # Cache Blending (CB) events — GPU operation start/end pairs
    CB_LOOKUP_START = "cb.lookup.start"
    CB_LOOKUP_END = "cb.lookup.end"
    CB_STORE_PRE_COMPUTED_START = "cb.store_pre_computed.start"
    CB_STORE_PRE_COMPUTED_END = "cb.store_pre_computed.end"
    CB_RETRIEVE_START = "cb.retrieve.start"
    CB_RETRIEVE_END = "cb.retrieve.end"
    CB_STORE_FINAL_START = "cb.store_final.start"
    CB_STORE_FINAL_END = "cb.store_final.end"
    CB_FINGERPRINTS_REGISTERED = "cb.fingerprints.registered"
    CB_CHUNKS_EVICTED = "cb.chunks.evicted"

    # CB V3 lookup sub-spans (CPU) — nest under cb.lookup. Submitted-once but
    # END may fire on a later poll (the non-blocking lookup re-issues), so the
    # span captures submit→resident incl. poll-wait.
    CB_FINGERPRINT_MATCH_START = "cb.fingerprint_match.start"
    CB_FINGERPRINT_MATCH_END = "cb.fingerprint_match.end"
    # Prefix leg: blend_v3 owns the submit/poll (direct storage_manager), so the
    # prefix lookup is a CB-namespace span under cb.lookup. Its hit tokens ride
    # the CB hit-rate metric via CB_LOOKUP_END (CB requests no longer feed the
    # MP mp.lookup_prefetch span / hit-rate aggregate).
    CB_PREFIX_LOOKUP_START = "cb.prefix_lookup.start"
    CB_PREFIX_LOOKUP_END = "cb.prefix_lookup.end"
    # Coordinator (fleet directory) match leg — cross-server analogue of
    # cb.fingerprint_match. Async: START at submit, END on the resolving poll.
    CB_COORDINATOR_MATCH_START = "cb.coordinator_match.start"
    CB_COORDINATOR_MATCH_END = "cb.coordinator_match.end"
    CB_SPARSE_PREFETCH_START = "cb.sparse_prefetch.start"
    CB_SPARSE_PREFETCH_END = "cb.sparse_prefetch.end"

    # CB V3 retrieve sub-span (GPU) — nest under cb.retrieve. Emitted via
    # publish_on_stream for GPU-accurate timing of the L1->paged scatter.
    CB_SCATTER_START = "cb.scatter.start"
    CB_SCATTER_END = "cb.scatter.end"

    # Cache Blending (CB) events — lifecycle sentinels (CPU-synchronous)
    CB_REQUEST_START = "cb.request.start"
    CB_STORE_PRE_COMPUTED_SUBMITTED = "cb.store_pre_computed.submitted"
    CB_RETRIEVE_SUBMITTED = "cb.retrieve.submitted"
    CB_STORE_FINAL_SUBMITTED = "cb.store_final.submitted"
    CB_REQUEST_END = "cb.request.end"


@dataclass
class Event:
    """A single observable event in the MP system.

    Attributes:
        event_type: The type of event.
        timestamp: Wall-clock time (``time.time()``) stamped by
            ``EventBus.publish()`` at the moment it is called — not when the
            drain thread processes the event.  For CUDA host-callback events
            this captures GPU-accurate timing.
        metadata: Flat key-value payload.  Contents depend on ``event_type``;
            see the metadata contracts in
            ``docs/design/v1/mp_observability/event-bus.md`` Section 2.7.
        session_id: Caller-provided ID for correlating start/end pairs.
    """

    event_type: EventType
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
