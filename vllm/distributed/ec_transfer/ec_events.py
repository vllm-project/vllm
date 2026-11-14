# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder cache (EC) events.

Mirror the KV cache events in `vllm.distributed.kv_events` so routers and
cache-aware orchestrators can track which instances hold encoder outputs for
which multimodal inputs. Events are keyed by `mm_hash` instead of block hash.
"""

import msgspec

from vllm.distributed.kv_events import EventBatch


class ECCacheEvent(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag=True,
):
    """Base class for all encoder cache-related events."""


class ECStored(ECCacheEvent):
    mm_hashes: list[str]
    medium: str | None


class ECRemoved(ECCacheEvent):
    mm_hashes: list[str]
    medium: str | None


class AllECCleared(ECCacheEvent):
    pass


class ECEventBatch(EventBatch):
    events: list[ECStored | ECRemoved | AllECCleared]
