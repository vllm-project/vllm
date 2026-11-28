# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-affinity routing for the internal data-parallel load balancer.

Requests that share a prompt prefix are hashed to the same DP engine so that
prefix-cache blocks accumulate there. Affinity is honored only while the
chosen engine's load score stays within a bounded slack of the mean score
(consistent hashing with bounded loads); otherwise the caller falls back to
its load-based choice. The hash is a routing key only and is not required to
match the engine-side block hashes.
"""

import zlib
from collections import Counter, OrderedDict, deque
from collections.abc import Sequence

import numpy as np

from vllm.v1.engine import EngineCoreRequest

# Adaptive hash length: escalate the key over at most this many doublings of
# the base block count, and only once this many requests have been observed.
MAX_LEVELS = 8
_MIN_WINDOW = 32
# Routing keys whose owning engine the front-end remembers (from coordinator
# prefix summaries); oldest entries are dropped first.
_OWNER_CAPACITY = 65536


def routing_keys(
    tokens: Sequence[int],
    cache_salt: str | None,
    lora_name: str | None,
    block_size: int,
    num_blocks: int,
    levels: int,
) -> list[int]:
    """Routing keys over the first num_blocks * 2**i blocks, i < levels,
    stopping when the prompt runs out of whole blocks. Shared by the
    front-end router and the engine-side prefix summary so both agree."""
    salt = 0
    if cache_salt:
        salt = zlib.crc32(cache_salt.encode(), salt)
    if lora_name is not None:
        salt = zlib.crc32(lora_name.encode(), salt)
    keys: list[int] = []
    nb = num_blocks
    for _ in range(levels):
        n = min(nb * block_size, len(tokens))
        n -= n % block_size
        keys.append(zlib.crc32(np.asarray(tokens[:n], dtype=np.int64).tobytes(), salt))
        if nb * block_size >= len(tokens):
            break
        nb *= 2
    return keys


class PrefixAffinityRouter:
    def __init__(
        self,
        num_blocks: int = 1,
        load_slack: float = 0.25,
        kv_cap: float = 0.9,
        adaptive: bool = True,
        window: int = 1024,
    ):
        self.num_blocks = num_blocks
        self.load_slack = load_slack
        self.kv_cap = kv_cap
        self.adaptive = adaptive
        self.window = window
        # Sliding window of per-request key tuples (one key per level) and
        # their counts, used to detect hot prefixes that need a longer key.
        self._recent: deque[tuple[int, ...]] = deque()
        self._counts: Counter[tuple[int, int]] = Counter()
        # Routing key -> engine that most recently admitted it, as reported
        # by the coordinator. Observed ownership beats the hash guess.
        self.prefix_owner: OrderedDict[int, int] = OrderedDict()

    def observe_owners(self, engine_index: int, keys: list[int]) -> None:
        owner = self.prefix_owner
        for key in keys:
            owner[key] = engine_index
            owner.move_to_end(key)
        while len(owner) > _OWNER_CAPACITY:
            owner.popitem(last=False)

    def _keys(
        self, request: EngineCoreRequest, block_size: int, levels: int
    ) -> list[int]:
        tokens = request.prompt_token_ids
        assert tokens is not None
        lora_name = (
            request.lora_request.lora_name if request.lora_request is not None else None
        )
        return routing_keys(
            tokens, request.cache_salt, lora_name, block_size, self.num_blocks, levels
        )

    def _select_level(self, keys: list[int], num_engines: int) -> int:
        """Smallest level whose key takes no more than its fair share of the
        recent window; a hotter key is lengthened so a shared prefix (e.g. a
        common system prompt) stops collapsing every request onto one engine."""
        if not self.adaptive or len(self._recent) < _MIN_WINDOW:
            return 0
        hot = len(self._recent) / num_engines
        for level, key in enumerate(keys):
            if self._counts[(level, key)] <= hot:
                return level
        return len(keys) - 1

    def _observe(self, keys: list[int]) -> None:
        entry = tuple(keys)
        self._recent.append(entry)
        for level, key in enumerate(entry):
            self._counts[(level, key)] += 1
        if len(self._recent) > self.window:
            old = self._recent.popleft()
            for level, key in enumerate(old):
                self._counts[(level, key)] -= 1
                if self._counts[(level, key)] <= 0:
                    del self._counts[(level, key)]

    def candidates(
        self, request: EngineCoreRequest, block_size: int, num_engines: int
    ) -> list[int]:
        """Preferred engines for this request: the hash of the first
        `num_blocks` prompt blocks, then of the first `2 * num_blocks`
        blocks so an overloaded primary can shed to a finer-grained group.
        With `adaptive` the base level is raised for hot prefixes."""
        tokens = request.prompt_token_ids
        if not tokens or len(tokens) < block_size:
            # A sub-block prompt cannot hit the prefix cache anyway.
            return []
        keys = self._keys(request, block_size, MAX_LEVELS if self.adaptive else 2)
        level = self._select_level(keys, num_engines)
        if self.adaptive:
            self._observe(keys)
        out: list[int] = []
        for key in keys[level : level + 2]:
            idx = self.prefix_owner.get(key, key % num_engines)
            if idx < num_engines and idx not in out:
                out.append(idx)
        return out

    def pick(
        self,
        scores: list[float],
        kv_usages: list[float],
        candidates: list[int],
        client_count: int,
    ) -> int | None:
        """First candidate whose load score is within the bounded-load cap,
        or None when every candidate is too loaded.

        A candidate whose live KV usage is at or above `kv_cap` is skipped
        only while some other engine still has KV headroom: it is about to
        preempt, and shedding to an engine with room avoids that. When every
        engine is saturated, spilling gains nothing and only loses locality,
        so affinity is kept."""
        if not candidates:
            return None
        cap = (1.0 + self.load_slack) * (sum(scores) / len(scores)) + client_count
        kv_headroom = min(kv_usages) < self.kv_cap
        for idx in candidates:
            if scores[idx] <= cap and (kv_usages[idx] < self.kv_cap or not kv_headroom):
                return idx
        return None
