# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observe explicit request associations without treating them as GPU ownership."""

import hashlib
import struct
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import msgspec

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.kv_hints.control import ControlError
from vllm.v1.kv_hints.native import (
    AssociatePayload,
    ContentSpan,
    HashTarget,
    RequestTarget,
)
from vllm.v1.kv_hints.state import (
    ContinuationKey,
    ContinuationState,
    SessionPrefixIndex,
    TokenPrefixCheckpoint,
)
from vllm.v1.request import Request


def _token_bytes(request: Request, start: int, end: int) -> bytes:
    return struct.pack(f"<{end - start}Q", *request.all_token_ids[start:end])


@dataclass
class _Binding:
    request: Request
    target: RequestTarget
    incarnation: int
    count: int
    hash_count: int
    computed_count: int
    token_hasher: Any


class RequestContinuationObserver:
    """A bounded request-to-continuation subscription, backed by the shared index."""

    def __init__(
        self,
        index: SessionPrefixIndex,
        manager: KVCacheManager,
        lookup: Callable[[str], Request | None],
    ):
        self.index = index
        self.manager = manager
        self.lookup = lookup
        self._attached: dict[str, Request] = {}
        self._bindings: dict[ContinuationKey, _Binding] = {}

    @contextmanager
    def attach(self, request: Request) -> Iterator[None]:
        self._attached[request.request_id] = request
        try:
            yield
        finally:
            del self._attached[request.request_id]

    def request(self, target: RequestTarget, request_id: str | None) -> Request:
        identity = request_id if target.request_id == "$request" else target.request_id
        request = (
            None
            if identity is None
            else self._attached.get(identity) or self.lookup(identity)
        )
        if request is None or request.is_finished():
            raise ControlError("missing", "Request is not active on this engine")
        if (
            request.prompt_embeds is not None
            or request.mm_features
            or request.lora_request is not None
            or request.resumable
        ):
            raise ControlError(
                "unsupported",
                "Request targeting requires a finite text-token request without LoRA",
            )
        return request

    def resolve(self, target: RequestTarget, request_id: str | None) -> HashTarget:
        request = self.request(target, request_id)
        return self._resolve(request, target)

    def resolve_finished(self, request: Request, target: RequestTarget) -> HashTarget:
        """Resolve completed token content, including detached cached checkpoints."""
        if not request.is_finished():
            raise ControlError(
                "invalid_payload", "Completion scope requires a finished request"
            )
        return self._resolve(request, target)

    def session_for(self, target: RequestTarget, request_id: str | None) -> str | None:
        return self.request(target, request_id).session_id

    def _resolve(self, request: Request, target: RequestTarget) -> HashTarget:
        end = request.num_tokens if target.end_token is None else target.end_token
        if end > request.num_tokens or target.start_token >= end:
            raise ControlError("missing", "Request range is not yet observed")
        unit = self.manager.block_pool.hash_block_size
        cached = self.manager.block_pool.cached_block_hash_to_block
        blocks = []
        for group_id, group_manager in enumerate(
            self.manager.coordinator.single_type_managers
        ):
            spec = self.manager.kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
            if not spec.prefix_cacheable:
                continue
            size = group_manager.block_size
            for hash_index, block_hash in enumerate(
                request.block_hashes[: end // unit]
            ):
                position = (hash_index + 1) * unit
                start = (position - 1) // size * size
                if position <= target.start_token:
                    continue
                if position % size and not cached.get_blocks(
                    make_block_hash_with_group_id(block_hash, group_id)
                ):
                    continue
                blocks.append(ContentSpan(block_hash.hex(), group_id, start, position))
                if len(blocks) > self.index.max_associations:
                    raise ControlError(
                        "capacity", "Request target exceeds logical index capacity"
                    )
        return HashTarget(self.index.namespace, tuple(blocks), target.start_token, end)

    def associate(
        self,
        payload: AssociatePayload,
        request_id: str | None,
        blocks: tuple[ContentSpan, ...],
    ) -> None:
        key = payload.session_id, payload.continuation_id
        try:
            previous = self.index.get(*key)
        except ControlError:
            previous = None
        revision = 0 if previous is None else previous.revision
        if payload.expected_revision != revision:
            raise ControlError(
                "stale_revision", f"Expected current revision {revision}"
            )
        binding = self._bindings.get(key)
        if not isinstance(payload.target, RequestTarget):
            if binding is not None:
                raise ControlError(
                    "busy", "Continuation is observing an active request"
                )
            self.index.associate(
                *key,
                blocks,
                expected_revision=payload.expected_revision,
                task_parent_id=payload.task_parent_id,
                parent_prefix=payload.parent_prefix,
            )
            return
        request = self.request(payload.target, request_id)
        if binding is not None and binding.request is not request:
            raise ControlError(
                "busy", "Use a new continuation for a concurrent request"
            )
        count = (
            request.num_tokens
            if payload.target.end_token is None
            else payload.target.end_token
        )
        context = hashlib.sha256(msgspec.json.encode((request.cache_salt,))).hexdigest()
        if previous is None and payload.parent_prefix is not None:
            parent = self.index.get(
                payload.parent_prefix.session_id, payload.parent_prefix.continuation_id
            )
            inherited = tuple(
                b
                for b in parent.blocks
                if b.end_token <= payload.parent_prefix.end_token
            )
            self._check_content(request, inherited)
            if (
                not any(
                    b.end_token == payload.parent_prefix.end_token for b in inherited
                )
                and payload.parent_prefix.end_token
            ):
                raise ControlError(
                    "unsupported",
                    "Parent boundary has no verifiable content checkpoint",
                )
        if previous is not None:
            old = previous.checkpoint
            if old is not None:
                if (
                    request.num_tokens < old.token_count
                    or context != old.context_digest
                    or hashlib.sha256(
                        _token_bytes(request, 0, old.token_count)
                    ).hexdigest()
                    != old.token_digest
                ):
                    raise ControlError(
                        "conflict",
                        "Request rewrites the token prefix; use a new continuation ID",
                    )
                count = max(count, old.token_count)
            self._check_content(request, previous.blocks)
            blocks = tuple(set(previous.blocks) | set(blocks))
        hasher = hashlib.sha256(_token_bytes(request, 0, count))
        checkpoint = TokenPrefixCheckpoint(count, hasher.hexdigest(), context)
        state = self.index.associate(
            *key,
            blocks,
            expected_revision=payload.expected_revision,
            task_parent_id=payload.task_parent_id,
            parent_prefix=payload.parent_prefix,
            checkpoint=checkpoint,
        )
        if payload.observe_growth:
            hash_count, computed_count = self._hash_counts(request, payload.target)
            self._bindings[key] = _Binding(
                request,
                payload.target,
                state.incarnation,
                count,
                hash_count,
                computed_count,
                hasher,
            )
        else:
            self._bindings.pop(key, None)

    def _hash_counts(self, request: Request, target: RequestTarget) -> tuple[int, int]:
        end = (
            request.num_tokens
            if target.end_token is None
            else min(request.num_tokens, target.end_token)
        )
        unit = self.manager.block_pool.hash_block_size
        known = min(len(request.block_hashes), end // unit)
        completed = (
            max(0, request.num_computed_tokens - request.num_in_flight_tokens) // unit
        )
        return known, min(known, completed)

    def _check_content(self, request: Request, blocks: tuple[ContentSpan, ...]) -> None:
        unit = self.manager.block_pool.hash_block_size
        for block in blocks:
            index = block.end_token // unit - 1
            if (
                block.end_token % unit
                or not 0 <= index < len(request.block_hashes)
                or request.block_hashes[index].hex() != block.block_hash
            ):
                raise ControlError(
                    "conflict", "Request changes previously associated content"
                )

    def observe(self) -> None:
        for key, binding in tuple(self._bindings.items()):
            try:
                state = self.index.get(*key)
            except ControlError:
                del self._bindings[key]
                continue
            if state.incarnation != binding.incarnation:
                del self._bindings[key]
                continue
            request = binding.request
            count = request.num_tokens
            if binding.target.end_token is not None:
                count = min(count, binding.target.end_token)
            try:
                if request.num_tokens < binding.count:
                    raise ControlError(
                        "conflict", "Observed request truncated its token prefix"
                    )
                assert state.checkpoint is not None
                count = max(count, state.checkpoint.token_count)
                hash_count, computed_count = self._hash_counts(request, binding.target)
                refresh = (hash_count, computed_count) != (
                    binding.hash_count,
                    binding.computed_count,
                )
                if count != binding.count or refresh or request.is_finished():
                    binding.token_hasher.update(
                        _token_bytes(request, binding.count, count)
                    )
                    blocks = state.blocks
                    if refresh or request.is_finished():
                        blocks = self._growth_blocks(binding, state)
                    self.index.associate(
                        *key,
                        blocks,
                        expected_revision=state.revision,
                        parent_prefix=state.parent_prefix,
                        checkpoint=TokenPrefixCheckpoint(
                            count,
                            binding.token_hasher.hexdigest(),
                            state.checkpoint.context_digest,
                        ),
                    )
                    binding.count, binding.hash_count, binding.computed_count = (
                        count,
                        hash_count,
                        computed_count,
                    )
            except ControlError as error:
                self.index.report_observation_failure(
                    *key, reason=f"{request.request_id}: {error.code}: {error}"
                )
                del self._bindings[key]
                continue
            if request.is_finished():
                del self._bindings[key]

    def _growth_blocks(
        self, binding: _Binding, state: ContinuationState
    ) -> tuple[ContentSpan, ...]:
        request = binding.request
        self._check_content(request, state.blocks)
        blocks = set(state.blocks)
        blocks.update(self._resolve(request, binding.target).blocks)
        if blocks == set(state.blocks):
            return state.blocks
        return tuple(blocks)
