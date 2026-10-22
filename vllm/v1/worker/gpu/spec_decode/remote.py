# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 adapter for a target-conditioned drafter on a dedicated device."""

import itertools
import time
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.utils import random_uuid
from vllm.v1.spec_decode.remote.capabilities import TargetFeatureKind
from vllm.v1.spec_decode.remote.errors import (
    cuda_context_fatal,
    host_allocation_failure,
)
from vllm.v1.spec_decode.remote.protocol import SequenceKey, SpeculatorStatusCode
from vllm.v1.spec_decode.remote.runtime.schema import create_runtime_contract
from vllm.v1.spec_decode.remote.session import RemoteDraftSession, VerifierIdentity
from vllm.v1.spec_decode.remote.state import (
    FeatureBatch,
    RemoteProposalBatch,
    SequenceState,
)
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

logger = init_logger(__name__)


@dataclass
class _Request:
    key: SequenceKey
    num_computed: int = 0


class RemoteSpeculator(BaseSpeculator):
    """Keep all draft weights and KV on the server; TP rank zero owns its session."""

    supports_mm_inputs = False
    draft_logits = None

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self._validate_config(vllm_config)
        spec = vllm_config.speculative_config
        assert spec is not None and spec.remote_draft is not None
        assert spec.method is not None
        self.device = device
        self.num_speculative_steps = spec.num_speculative_tokens
        self._contract = create_runtime_contract(vllm_config)
        self._tp = get_tp_group()
        self._session: RemoteDraftSession | None = None
        self._requests: dict[str, _Request] = {}
        self._terminal_requests: set[str] = set()
        self._sequence_ids = itertools.count()
        self._batch_ids = itertools.count()
        self._stats: dict[str, int | float] = {
            "proposal_calls": 0,
            "feature_bytes": 0,
            "proposal_seconds": 0.0,
            "session_seconds": 0.0,
        }
        error = None
        if self._tp.rank_in_group == 0:
            identity = VerifierIdentity(
                verifier_instance_id=random_uuid(),
                target_fingerprint=self._contract.target_fingerprint,
                tokenizer_fingerprint=self._contract.tokenizer_fingerprint,
                draft_checkpoint_fingerprint=self._contract.draft_fingerprint,
                method=spec.method,
                num_speculative_tokens=self.num_speculative_steps,
                parallel_drafting=False,
                provided_features=tuple(s.kind for s in self._contract.schema.slots),
                max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
                max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
                max_model_len=vllm_config.model_config.max_model_len,
                vocab_size=vllm_config.model_config.get_vocab_size(),
            )
            self._session = RemoteDraftSession.from_config(spec.remote_draft, identity)
            try:
                self._session.connect()
                if self._session.feature_schema != self._contract.schema:
                    raise ValueError(
                        "remote feature schema differs from checkpoint contract"
                    )
            except Exception as exc:
                self.close()
                error = f"{type(exc).__name__}: {exc}"
        error = self._tp.broadcast_object(error, src=0)
        if error is not None:
            raise RuntimeError(f"remote drafter startup failed: {error}")

    @staticmethod
    def _validate_config(config: VllmConfig) -> None:
        assert config.speculative_config is not None
        parallel = config.parallel_config
        unsupported = []
        if parallel.tensor_parallel_size not in (1, 2):
            unsupported.append("target tensor_parallel_size must be 1 or 2")
        if any(
            getattr(parallel, name) != 1
            for name in (
                "pipeline_parallel_size",
                "data_parallel_size",
                "decode_context_parallel_size",
                "prefill_context_parallel_size",
            )
        ):
            unsupported.append("PP, DP and context parallelism must be disabled")
        if config.cache_config.enable_prefix_caching:
            unsupported.append("enable_prefix_caching must be False")
        if config.scheduler_config.async_scheduling:
            unsupported.append("async_scheduling must be False")
        if config.lora_config is not None:
            unsupported.append("LoRA is unsupported")
        if config.model_config.is_multimodal_model:
            unsupported.append("multimodal models are unsupported")
        if getattr(config.model_config, "enable_prompt_embeds", False):
            unsupported.append("prompt embeddings are unsupported")
        if getattr(config, "watermark_config", None) is not None:
            unsupported.append("watermarking is unsupported")
        if config.speculative_config.enable_adaptive_verification:
            unsupported.append("adaptive verification is unsupported")
        if unsupported:
            raise ValueError("remote drafter: " + "; ".join(unsupported))

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        pass

    def capture(self) -> None:
        pass

    def close_requests(self, req_ids: Collection[str]) -> None:
        if not req_ids:
            return
        self._terminal_requests.difference_update(req_ids)
        error = None
        if self._session is not None:
            try:
                keys = tuple(
                    self._requests.pop(req_id).key
                    for req_id in req_ids
                    if req_id in self._requests
                )
                if keys:
                    self._session.close_sequences(keys)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        error = self._tp.broadcast_object(error, src=0)
        if error is not None:
            raise RuntimeError(f"remote drafter request cleanup failed: {error}")

    def close(self) -> None:
        if self._session is not None:
            session = self._session
            self._session = None
            try:
                session.close()
            except Exception:
                logger.exception("Failed to close remote drafter session")
            finally:
                self._requests.clear()
                self._terminal_requests.clear()
                logger.info("Remote drafter verifier statistics: %s", self._stats)

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        if is_profile and self._tp.rank_in_group == 0:
            self._profile_features(
                input_batch,
                last_hidden_states,
                aux_hidden_states,
                next_prefill_tokens,
            )
        if dummy_run or is_profile:
            return torch.full(
                (input_batch.num_reqs, self.num_speculative_steps),
                -1,
                dtype=torch.int64,
                device=self.device,
            )
        error = None
        tokens = None
        if self._tp.rank_in_group == 0:
            started = time.perf_counter()
            try:
                assert self._session is not None
                preparation_deadline = self._session.preparation_deadline()
                if num_speculative_tokens is not None and not (
                    0 <= num_speculative_tokens <= self.num_speculative_steps
                ):
                    raise ValueError(
                        "round draft budget exceeds the configured maximum"
                    )
                if torch.any(temperature[input_batch.idx_mapping] != 0).item():
                    raise ValueError(
                        "remote drafter currently requires greedy target sampling"
                    )
                try:
                    tokens = self._propose(
                        input_batch,
                        last_hidden_states,
                        aux_hidden_states,
                        num_sampled,
                        num_rejected,
                        last_sampled,
                        next_prefill_tokens,
                        num_speculative_tokens,
                        preparation_deadline=preparation_deadline,
                    ).tolist()
                except Exception as exc:
                    if cuda_context_fatal(exc) or not (
                        isinstance(exc, torch.OutOfMemoryError)
                        or host_allocation_failure(exc)
                    ):
                        raise
                    # Feature preparation can advance local request positions
                    # before all chunks reach the drafter. Never resume these
                    # own-KV sequences after dropping the failed round.
                    assert self._session is not None
                    self._terminal_requests.update(input_batch.req_ids)
                    self._session.fail_sequences(
                        tuple(
                            self._requests[req_id].key
                            for req_id in input_batch.req_ids
                            if req_id in self._requests
                        ),
                        status=SpeculatorStatusCode.OUT_OF_MEMORY,
                        detail=f"remote feature preparation failed: {exc}",
                        deadline=preparation_deadline,
                    )
                    tokens = [
                        [-1] * self.num_speculative_steps for _ in input_batch.req_ids
                    ]
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                self._stats["proposal_calls"] += 1
                self._stats["proposal_seconds"] += time.perf_counter() - started
        # This small CPU message shares failures and tokens together. A device
        # broadcast can initialize NCCL after KV profiling, exhausting the
        # target GPU memory that profiling has already assigned to its cache.
        error, tokens = self._tp.broadcast_object((error, tokens), src=0)
        if error is not None:
            raise RuntimeError(f"remote drafter proposal failed: {error}")
        assert tokens is not None
        return torch.tensor(tokens, dtype=torch.int64, device=self.device).reshape(
            input_batch.num_reqs, self.num_speculative_steps
        )

    def _profile_features(
        self,
        batch: InputBatch,
        hidden: torch.Tensor,
        aux: list[torch.Tensor] | None,
        next_prefill_tokens: torch.Tensor,
    ) -> None:
        # Full-size compaction bounds both all-active and mixed fallback batches.
        tokens = list(range(batch.num_tokens))
        rows = list(range(batch.num_reqs))
        features = self._features(
            batch,
            hidden,
            aux,
            next_prefill_tokens,
            token_rows=tokens,
            sequence_rows=rows,
        )
        feature_bytes = self._stats["feature_bytes"]
        self._select(features, tokens, rows)
        self._stats["feature_bytes"] = feature_bytes

    def _propose(
        self,
        batch: InputBatch,
        hidden: torch.Tensor,
        aux: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        num_speculative_tokens: int | None = None,
        *,
        preparation_deadline: float | None = None,
    ) -> torch.Tensor:
        session = self._session
        assert session is not None
        session_started = time.perf_counter()
        starts = batch.query_start_loc_np
        new_requests = []
        for row, req_id in enumerate(batch.req_ids):
            if req_id in self._terminal_requests:
                continue
            if req_id not in self._requests:
                if (
                    not batch.is_prefilling_np[row]
                    or int(batch.positions[int(starts[row])]) != 0
                ):
                    raise ValueError(
                        "new remote sequence requires prefill from position zero"
                    )
                new_requests.append(req_id)
        if new_requests:
            opened_keys: tuple[SequenceKey, ...] = ()
            try:
                opened_keys = session.open_sequences(
                    tuple(next(self._sequence_ids) for _ in new_requests),
                    deadline=preparation_deadline,
                )
                for req_id, key in zip(new_requests, opened_keys):
                    self._requests[req_id] = _Request(key)
            except MemoryError:
                # Remote opens may already exist without a local request owner.
                # Fence the stream before dropping their local records; do not
                # spend a second close-RPC budget after the failed preparation.
                session.close()
                session.close_sequences(opened_keys)
                raise
        active_rows = [
            row
            for row, req_id in enumerate(batch.req_ids)
            if req_id not in self._terminal_requests
            and session.sequence_state(self._requests[req_id].key)
            in (SequenceState.PREFILLING, SequenceState.READY)
        ]
        output = torch.full(
            (batch.num_reqs, self.num_speculative_steps), -1, dtype=torch.int64
        )
        if not active_rows:
            self._stats["session_seconds"] += time.perf_counter() - session_started
            return output
        session_seconds = time.perf_counter() - session_started
        active_tokens = (
            None
            if len(active_rows) == batch.num_reqs
            else [
                token
                for row in active_rows
                for token in range(int(starts[row]), int(starts[row + 1]))
            ]
        )
        features = self._features(
            batch,
            hidden,
            aux,
            next_prefill_tokens,
            token_rows=active_tokens,
            sequence_rows=(None if active_tokens is None else active_rows),
        )
        positions = features[TargetFeatureKind.POSITIONS.value]
        sampled = num_sampled.cpu().tolist()
        rejected = num_rejected.cpu().tolist()
        recovery = last_sampled[batch.idx_mapping].reshape(-1).cpu()
        session_started = time.perf_counter()
        chunks = []
        rows, keys, query_lens, accepted, sequence_rows = [], [], [], [], []
        token_indices: list[int] = []
        end = 0
        for feature_row, row in enumerate(active_rows):
            req_id = batch.req_ids[row]
            request = self._requests[req_id]
            query_len = int(starts[row + 1]) - int(starts[row])
            start, end = end, end + query_len
            if int(positions[start]) != request.num_computed:
                raise ValueError(f"remote context discontinuity for request {req_id}")
            if batch.is_prefilling_np[row]:
                remaining = int(batch.prefill_len_np[row]) - request.num_computed
                if remaining <= 0:
                    raise ValueError("remote prefill has no remaining prompt tokens")
                prompt_len = min(query_len, remaining)
                is_final = prompt_len == remaining
                # The scheduler may append speculative placeholders to a final
                # prompt chunk. Keep the physical end for the following row,
                # but never send those rejected positions as prompt context.
                padding = query_len - prompt_len
                if (
                    padding > self.num_speculative_steps
                    or rejected[row] != padding
                    or sampled[row] != int(is_final)
                ):
                    raise ValueError("unexpected sampling counts during remote prefill")
                chunks.append(
                    (
                        request.key,
                        self._select(
                            features, range(start, start + prompt_len), [feature_row]
                        ),
                        is_final,
                    )
                )
                request.num_computed += prompt_len
                if not is_final:
                    continue
                query_len = 0
            else:
                if sampled[row] < 1 or rejected[row] != query_len - sampled[row]:
                    raise ValueError(
                        "remote verification counts do not cover the query"
                    )
                token_indices.extend(range(start, end))
                request.num_computed += sampled[row]
            rows.append(row)
            keys.append(request.key)
            query_lens.append(query_len)
            accepted.append(sampled[row] - 1)
            sequence_rows.append(feature_row)
        session.prefill_many(chunks, deadline=preparation_deadline)
        if rows:
            proposal = RemoteProposalBatch(
                batch_id=next(self._batch_ids),
                keys=tuple(keys),
                accepted_counts=torch.tensor(accepted, dtype=torch.int32),
                recovery_tokens=recovery[rows],
                features=self._select(features, token_indices, sequence_rows),
                query_lens=torch.tensor(query_lens, dtype=torch.int32),
                num_speculative_tokens=num_speculative_tokens,
            )
            result = session.collect(
                session.dispatch(proposal, deadline=preparation_deadline)
            ).output
            valid = (
                torch.arange(self.num_speculative_steps)[None, :]
                < result.valid_lengths[:, None]
            )
            output[rows] = torch.where(valid, result.token_ids, -1)
        self._stats["session_seconds"] += (
            session_seconds + time.perf_counter() - session_started
        )
        return output

    def _features(
        self,
        batch: InputBatch,
        hidden: torch.Tensor,
        aux: list[torch.Tensor] | None,
        next_prefill_tokens: torch.Tensor,
        *,
        token_rows: Collection[int] | None = None,
        sequence_rows: Collection[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        token_rows = None if token_rows is None else list(token_rows)
        sequence_rows = None if sequence_rows is None else list(sequence_rows)
        source = {
            TargetFeatureKind.TOKEN_IDS.value: batch.input_ids[: batch.num_tokens],
            TargetFeatureKind.POSITIONS.value: batch.positions[: batch.num_tokens],
            TargetFeatureKind.LAST_HIDDEN_STATES.value: hidden[: batch.num_tokens],
        }
        result = {}
        for slot in self._contract.schema.slots:
            if slot.kind == TargetFeatureKind.AUX_HIDDEN_STATES.value:
                assert aux is not None
                tensor = torch.cat(
                    [
                        state[: batch.num_tokens]
                        if token_rows is None
                        else state[token_rows]
                        for state in aux
                    ],
                    dim=-1,
                )
            elif slot.kind == TargetFeatureKind.NEXT_PREFILL_TOKENS.value:
                mapping = batch.idx_mapping
                if sequence_rows is not None:
                    mapping = mapping[sequence_rows]
                tensor = next_prefill_tokens[:1, mapping].T
            else:
                tensor = source[slot.kind]
                if token_rows is not None:
                    tensor = tensor[token_rows]
            if tuple(tensor.shape[1:]) != slot.trailing_shape:
                raise ValueError(
                    f"unexpected remote feature shape for {slot.kind}: {tensor.shape}"
                )
            result[slot.kind] = tensor.to(
                device="cpu", dtype=getattr(torch, slot.dtype)
            )
        return result

    def _select(
        self,
        features: dict[str, torch.Tensor],
        token_rows: Collection[int],
        sequence_rows: Collection[int],
    ) -> FeatureBatch:
        slots = tuple(
            features[slot.kind][
                list(sequence_rows if slot.per_sequence else token_rows)
            ]
            for slot in self._contract.schema.slots
        )
        self._stats["feature_bytes"] += sum(t.numel() * t.element_size() for t in slots)
        return FeatureBatch(self._contract.schema.schema_id, slots)
