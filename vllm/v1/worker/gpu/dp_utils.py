# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch
import torch.distributed as dist

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_dp_group
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)


@dataclass(frozen=True)
class DPSyncState:
    """What a `dispatch_cg_and_sync_dp` call agreed across DP ranks.

    Every field is identical on every rank, so callers can branch on them
    without disagreeing about whether to run a collective. Never add a per-rank
    value here.
    """

    # Per-rank token counts for the forward context. An execution contract
    # rewrites every entry to its rank-identical execution capacity; legacy
    # eager dispatch retains live per-rank counts.
    num_tokens_across_dp: torch.Tensor
    # Agreed uniform decode length. None means the ranks disagreed, which is an
    # answer, not "unknown".
    uniform_token_count: int | None
    # Whether the ranks agreed to run eager. A dispatch reusing this must run
    # eager too; no shape was agreed, so picking a graph per rank would diverge.
    eager: bool
    # Sequence number within this coordinator's collective lane.
    generation: int | None = None
    # Target generation that owns this speculative child agreement.
    parent_generation: int | None = None
    # Rank-local work before execution-contract padding. These are immutable
    # snapshots because ``num_tokens_across_dp`` may be rewritten in place.
    live_num_tokens_across_dp: tuple[int, ...] = ()
    live_num_reqs_across_dp: tuple[int, ...] = ()
    # Request capacity selected by a target execution contract. Unlike
    # ``batch_desc.num_reqs``, this is populated for PIECEWISE execution too.
    execution_num_reqs: int | None = None
    # Cached contracts intentionally omit current cross-rank occupancy. The
    # execution descriptor remains exact, but the live vectors are unavailable.
    live_facts_exact: bool = True
    contract_epoch: int | None = None


@dataclass(frozen=True)
class _CachedDPExecutionContract:
    batch_desc: BatchExecutionDescriptor
    execution_num_tokens: int
    execution_num_reqs: int
    uniform_token_count: int
    num_tokens_across_dp: torch.Tensor
    epoch: int

    @property
    def signature(self) -> tuple[object, ...]:
        desc = self.batch_desc
        return (
            desc.cg_mode,
            desc.num_tokens,
            desc.num_reqs,
            desc.uniform_token_count,
            desc.max_query_len,
            desc.num_active_loras,
            self.execution_num_tokens,
            self.execution_num_reqs,
            self.uniform_token_count,
        )


_SYNC_NUM_TOKENS = 0
_SYNC_CG_MODE = 1
_SYNC_UNIFORM_TOKEN_COUNT = 2
_SYNC_MAX_QUERY_LEN = 3
_SYNC_NUM_REQS = 4
_SYNC_GENERATION = 5
_SYNC_PARENT_GENERATION = 6
_SYNC_NEED_EAGER = 7
_SYNC_NUM_ACTIVE_LORAS = 8
_SYNC_NUM_FIELDS = 9

_DP_SYNC_GROUPS: dict[tuple[int, str], dist.ProcessGroup] = {}


def _get_dp_sync_group(
    lane: Literal["target", "speculator"],
) -> dist.ProcessGroup:
    """Return the isolated CPU process group for a collective lane."""
    base_group = get_dp_group().cpu_group
    if lane == "target":
        return base_group

    key = (id(base_group), lane)
    group = _DP_SYNC_GROUPS.get(key)
    if group is None:
        # Only members of this DP group construct the clone. Every member
        # creates the speculator lane on its first speculative agreement.
        group = dist.new_group(
            ranks=dist.get_process_group_ranks(base_group),
            backend="gloo",
            use_local_synchronization=True,
        )
        _DP_SYNC_GROUPS[key] = group
    return group


class DPSyncFuture:
    """An in-flight DP shape agreement backed by a reusable CPU buffer."""

    def __init__(
        self,
        coordinator: DPSyncCoordinator,
        local_batch_desc: BatchExecutionDescriptor,
        num_tokens: int,
        num_reqs: int,
        num_active_loras: int,
        generation: int,
        parent_generation: int | None,
        work: dist.Work | None,
        contract_epoch: int | None = None,
        contract_capacity_num_reqs: int | None = None,
        has_prefill: bool = False,
        cached_result: tuple[BatchExecutionDescriptor, DPSyncState] | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._local_batch_desc = local_batch_desc
        self._num_tokens = num_tokens
        self._num_reqs = num_reqs
        self._num_active_loras = num_active_loras
        self.generation = generation
        self.parent_generation = parent_generation
        self.contract_epoch = contract_epoch
        self.contract_capacity_num_reqs = contract_capacity_num_reqs
        self.has_prefill = has_prefill
        self._work = work
        self._waited = work is None
        self._result: tuple[BatchExecutionDescriptor, DPSyncState | None] | None = (
            cached_result
        )
        self._released = False

    def result(
        self, cudagraph_manager: CudaGraphManager | None
    ) -> tuple[BatchExecutionDescriptor, DPSyncState | None]:
        """Wait for and resolve the agreement without releasing its buffer.

        The returned ``DPSyncState`` may contain a view into the coordinator's
        reusable buffer and is valid only until :meth:`release` is called.
        """
        if self._released:
            raise RuntimeError("Cannot resolve a released DP sync future")
        if self._result is not None:
            return self._result

        if self._work is None:
            self._result = self._local_batch_desc, None
        else:
            self._work.wait()
            self._waited = True
            tensor = self._coordinator._tensor
            assert tensor is not None
            _validate_sync_generation(tensor, self.generation, self.parent_generation)
            if self._coordinator.execution_contract:
                self._result = _finish_dp_execution_contract(
                    cudagraph_manager,
                    tensor,
                    self.generation,
                    self.parent_generation,
                )
                batch_desc, sync = self._result
                if sync is not None:
                    sync = replace(sync, contract_epoch=self.contract_epoch)
                    self._result = batch_desc, sync
                self._coordinator._observe_execution_contract(
                    self,
                    self._result,
                    cudagraph_manager,
                )
            else:
                live_num_tokens = tuple(
                    int(value) for value in tensor[_SYNC_NUM_TOKENS].tolist()
                )
                live_num_reqs = tuple(
                    int(value) for value in tensor[_SYNC_NUM_REQS].tolist()
                )
                batch_desc, sync = _finish_cudagraph_and_dp_padding(
                    cudagraph_manager,
                    self._local_batch_desc,
                    tensor,
                    self._num_tokens,
                    self._num_reqs,
                    self._num_active_loras,
                )
                if sync is not None:
                    sync = replace(
                        sync,
                        generation=self.generation,
                        parent_generation=self.parent_generation,
                        live_num_tokens_across_dp=live_num_tokens,
                        live_num_reqs_across_dp=live_num_reqs,
                    )
                self._result = batch_desc, sync
        return self._result

    def release(self) -> None:
        """Wait for unfinished work and return the backing buffer."""
        if self._released:
            return
        if self._work is not None and not self._waited:
            self._work.wait()
            self._waited = True
        self._coordinator._release(self)
        self._released = True


class DPSyncCoordinator:
    """Own one persistent CPU buffer for asynchronous DP shape agreement."""

    def __init__(
        self,
        dp_size: int,
        dp_rank: int,
        *,
        group: dist.ProcessGroup | None = None,
        lane: Literal["target", "speculator"] = "target",
        execution_contract: bool = False,
        cache_execution_contract: bool = False,
        cache_stability_steps: int = 2,
    ) -> None:
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.group = group
        self.lane = lane
        self.execution_contract = execution_contract
        if cache_execution_contract and (lane != "target" or not execution_contract):
            raise ValueError(
                "Only the target execution-contract coordinator may cache results"
            )
        if cache_stability_steps <= 0:
            raise ValueError("cache_stability_steps must be positive")
        self.cache_execution_contract = cache_execution_contract
        self.cache_stability_steps = cache_stability_steps
        self._cached_contract: _CachedDPExecutionContract | None = None
        self._cache_candidate: (
            tuple[tuple[object, ...], _CachedDPExecutionContract, int] | None
        ) = None
        self._tensor = (
            torch.zeros(_SYNC_NUM_FIELDS, dp_size, dtype=torch.int64, device="cpu")
            if dp_size > 1
            else None
        )
        self._active_future: DPSyncFuture | None = None
        self._next_generation = 0

    def start(
        self,
        cudagraph_manager: CudaGraphManager | None,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        max_query_len: int | None = None,
        need_eager: bool = False,
        num_active_loras: int = 0,
        parent_generation: int | None = None,
        force_refresh: bool = False,
        contract_epoch: int | None = None,
        contract_capacity_num_reqs: int | None = None,
        has_prefill: bool = False,
    ) -> DPSyncFuture:
        """Dispatch locally and issue the DP collective without waiting."""
        if self._active_future is not None:
            raise RuntimeError("A DP sync is already in flight")

        batch_desc = _dispatch_local_batch(
            cudagraph_manager,
            num_reqs,
            num_tokens,
            uniform_token_count,
            max_query_len,
            need_eager,
            num_active_loras,
        )

        generation = self._next_generation
        self._next_generation += 1
        cached_result = None
        cached = self._cached_contract
        if (
            self.cache_execution_contract
            and contract_epoch is not None
            and cached is not None
            and not force_refresh
        ):
            self._validate_cached_contract(
                cached,
                batch_desc,
                num_reqs=num_reqs,
                num_tokens=num_tokens,
                uniform_token_count=uniform_token_count,
                max_query_len=max_query_len,
                need_eager=need_eager,
                num_active_loras=num_active_loras,
                has_prefill=has_prefill,
                contract_epoch=contract_epoch,
            )
            cached_result = self._make_cached_result(
                cached,
                generation,
                contract_epoch,
            )

        work = None
        if self._tensor is not None and cached_result is None:
            tensor = self._tensor
            tensor.zero_()
            tensor[_SYNC_NUM_TOKENS][self.dp_rank] = num_tokens
            tensor[_SYNC_CG_MODE][self.dp_rank] = batch_desc.cg_mode.value
            tensor[_SYNC_UNIFORM_TOKEN_COUNT][self.dp_rank] = uniform_token_count or 0
            tensor[_SYNC_MAX_QUERY_LEN][self.dp_rank] = max_query_len or -1
            tensor[_SYNC_NUM_REQS][self.dp_rank] = num_reqs
            tensor[_SYNC_GENERATION][self.dp_rank] = generation
            tensor[_SYNC_PARENT_GENERATION][self.dp_rank] = (
                parent_generation if parent_generation is not None else -1
            )
            tensor[_SYNC_NEED_EAGER][self.dp_rank] = int(need_eager)
            tensor[_SYNC_NUM_ACTIVE_LORAS][self.dp_rank] = num_active_loras
            group = self.group
            if group is None:
                group = _get_dp_sync_group(self.lane)
            work = dist.all_reduce(tensor, group=group, async_op=True)

        future = DPSyncFuture(
            self,
            batch_desc,
            num_tokens,
            num_reqs,
            num_active_loras,
            generation,
            parent_generation,
            work,
            contract_epoch=contract_epoch,
            contract_capacity_num_reqs=contract_capacity_num_reqs,
            has_prefill=has_prefill,
            cached_result=cached_result,
        )
        self._active_future = future
        return future

    def _make_cached_result(
        self,
        cached: _CachedDPExecutionContract,
        generation: int,
        contract_epoch: int,
    ) -> tuple[BatchExecutionDescriptor, DPSyncState]:
        return cached.batch_desc, DPSyncState(
            num_tokens_across_dp=cached.num_tokens_across_dp,
            uniform_token_count=cached.uniform_token_count,
            eager=False,
            generation=generation,
            live_num_tokens_across_dp=(),
            live_num_reqs_across_dp=(),
            execution_num_reqs=cached.execution_num_reqs,
            live_facts_exact=False,
            contract_epoch=contract_epoch,
        )

    def _validate_cached_contract(
        self,
        cached: _CachedDPExecutionContract,
        local_desc: BatchExecutionDescriptor,
        *,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        max_query_len: int | None,
        need_eager: bool,
        num_active_loras: int,
        has_prefill: bool,
        contract_epoch: int,
    ) -> None:
        violations = []
        if contract_epoch != cached.epoch:
            violations.append(f"epoch={contract_epoch}, cached={cached.epoch}")
        if need_eager:
            violations.append("eager execution")
        if has_prefill:
            violations.append("prefill work")
        if num_active_loras:
            violations.append(f"num_active_loras={num_active_loras}")

        if num_tokens > 0:
            if num_reqs <= 0 or num_reqs > num_tokens:
                violations.append(f"requests/tokens={num_reqs}/{num_tokens}")
            if num_tokens > cached.execution_num_tokens:
                violations.append(
                    f"num_tokens={num_tokens}, capacity={cached.execution_num_tokens}"
                )
            if num_reqs > cached.execution_num_reqs:
                violations.append(
                    f"num_reqs={num_reqs}, capacity={cached.execution_num_reqs}"
                )
            if local_desc.cg_mode != CUDAGraphMode.FULL:
                violations.append(f"local_cg_mode={local_desc.cg_mode.name}")
            if local_desc.num_tokens > cached.execution_num_tokens:
                violations.append(
                    "local graph token capacity="
                    f"{local_desc.num_tokens}, cached={cached.execution_num_tokens}"
                )
            if (
                local_desc.num_reqs is not None
                and local_desc.num_reqs > cached.execution_num_reqs
            ):
                violations.append(
                    "local graph request capacity="
                    f"{local_desc.num_reqs}, cached={cached.execution_num_reqs}"
                )
            if uniform_token_count != cached.uniform_token_count:
                violations.append(
                    "uniform_token_count="
                    f"{uniform_token_count}, cached={cached.uniform_token_count}"
                )
            if max_query_len != cached.uniform_token_count:
                violations.append(
                    f"max_query_len={max_query_len}, "
                    f"cached={cached.uniform_token_count}"
                )
            if num_tokens != num_reqs * cached.uniform_token_count:
                violations.append(
                    f"num_tokens={num_tokens}, expected="
                    f"{num_reqs * cached.uniform_token_count}"
                )

        if violations:
            raise RuntimeError(
                "Cached target DP execution contract violation; a local fallback "
                "would change collective ordering: " + "; ".join(violations)
            )

    def _observe_execution_contract(
        self,
        future: DPSyncFuture,
        result: tuple[BatchExecutionDescriptor, DPSyncState | None],
        cudagraph_manager: CudaGraphManager | None,
    ) -> None:
        if not self.cache_execution_contract or future.contract_epoch is None:
            return

        batch_desc, sync = result
        capacity_num_reqs = future.contract_capacity_num_reqs
        candidate = None
        if (
            sync is not None
            and not sync.eager
            and not future.has_prefill
            and sync.uniform_token_count is not None
            and batch_desc.cg_mode == CUDAGraphMode.FULL
            and capacity_num_reqs is not None
            and capacity_num_reqs > 0
        ):
            uniform_token_count = sync.uniform_token_count
            capacity_num_tokens = capacity_num_reqs * uniform_token_count
            assert cudagraph_manager is not None
            capacity_desc = cudagraph_manager.dispatch(
                capacity_num_reqs,
                capacity_num_tokens,
                uniform_token_count,
                num_active_loras=0,
                max_query_len=uniform_token_count,
            )
            if (
                capacity_desc.cg_mode == CUDAGraphMode.FULL
                and capacity_desc.num_tokens >= capacity_num_tokens
                and (capacity_desc.num_reqs or 0) >= capacity_num_reqs
            ):
                execution_num_reqs = capacity_desc.num_reqs or capacity_num_reqs
                candidate = _CachedDPExecutionContract(
                    batch_desc=capacity_desc,
                    execution_num_tokens=capacity_desc.num_tokens,
                    execution_num_reqs=execution_num_reqs,
                    uniform_token_count=uniform_token_count,
                    num_tokens_across_dp=torch.full(
                        (self.dp_size,),
                        capacity_desc.num_tokens,
                        dtype=torch.int64,
                        device="cpu",
                    ),
                    epoch=future.contract_epoch,
                )
        self._observe_cache_candidate(candidate)

    def _observe_cache_candidate(
        self, candidate: _CachedDPExecutionContract | None
    ) -> None:
        if candidate is None:
            self._cached_contract = None
            self._cache_candidate = None
            return

        active = self._cached_contract
        if active is not None and active.signature == candidate.signature:
            self._cached_contract = candidate
            self._cache_candidate = None
            return

        previous = self._cache_candidate
        count = (
            previous[2] + 1
            if previous is not None and previous[0] == candidate.signature
            else 1
        )
        if count >= self.cache_stability_steps:
            self._cached_contract = candidate
            self._cache_candidate = None
        else:
            self._cached_contract = None
            self._cache_candidate = (candidate.signature, candidate, count)

    def _release(self, future: DPSyncFuture) -> None:
        if self._active_future is not future:
            raise RuntimeError("DP sync future is not owned by this coordinator")
        self._active_future = None


def _validate_sync_generation(
    tensor: torch.Tensor,
    generation: int,
    parent_generation: int | None,
) -> None:
    generations = tensor[_SYNC_GENERATION]
    if not bool(torch.all(generations == generation).item()):
        raise RuntimeError(
            "DP sync generation mismatch: "
            f"local={generation}, observed={generations.tolist()}"
        )

    expected_parent = parent_generation if parent_generation is not None else -1
    parents = tensor[_SYNC_PARENT_GENERATION]
    if not bool(torch.all(parents == expected_parent).item()):
        raise RuntimeError(
            "DP sync parent generation mismatch: "
            f"local={expected_parent}, observed={parents.tolist()}"
        )


def _finish_dp_execution_contract(
    cudagraph_manager: CudaGraphManager | None,
    tensor: torch.Tensor,
    generation: int,
    parent_generation: int | None,
) -> tuple[BatchExecutionDescriptor, DPSyncState | None]:
    """Resolve rank-local facts into one target execution descriptor."""
    live_num_tokens = tensor[_SYNC_NUM_TOKENS]
    live_num_reqs = tensor[_SYNC_NUM_REQS]
    live_tokens_tuple = tuple(int(value) for value in live_num_tokens.tolist())
    live_reqs_tuple = tuple(int(value) for value in live_num_reqs.tolist())

    active = live_num_tokens > 0
    if not bool(torch.any(active).item()):
        return (
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=0,
                num_reqs=0,
            ),
            None,
        )

    active_num_tokens = live_num_tokens[active]
    active_num_reqs = live_num_reqs[active]
    if bool(torch.any(active_num_reqs > active_num_tokens).item()):
        raise RuntimeError(
            "DP execution contract observed requests exceeding tokens: "
            f"requests={active_num_reqs.tolist()}, "
            f"tokens={active_num_tokens.tolist()}"
        )

    exec_num_tokens = int(active_num_tokens.max().item())
    exec_num_reqs = int(active_num_reqs.max().item())
    if exec_num_reqs <= 0:
        raise RuntimeError("DP execution contract has active tokens but no requests")

    uniform_counts = tensor[_SYNC_UNIFORM_TOKEN_COUNT][active]
    uniform_token_count: int | None = int(uniform_counts[0].item())
    if uniform_token_count == 0 or not bool(
        torch.all(uniform_counts == uniform_token_count).item()
    ):
        uniform_token_count = None

    max_query_lens = tensor[_SYNC_MAX_QUERY_LEN][active]
    max_query_len: int | None = None
    if bool(torch.all(max_query_lens != -1).item()):
        max_query_len = int(max_query_lens.max().item())

    num_active_loras = int(tensor[_SYNC_NUM_ACTIVE_LORAS][active].max().item())
    need_eager = bool(torch.any(tensor[_SYNC_NEED_EAGER][active] != 0).item())
    if need_eager:
        batch_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=exec_num_tokens,
            num_reqs=exec_num_reqs,
            uniform_token_count=uniform_token_count,
            max_query_len=max_query_len,
            num_active_loras=num_active_loras,
        )
    else:
        assert cudagraph_manager is not None
        batch_desc = cudagraph_manager.dispatch(
            exec_num_reqs,
            exec_num_tokens,
            uniform_token_count,
            num_active_loras=num_active_loras,
            max_query_len=max_query_len,
        )

    exec_num_reqs = batch_desc.num_reqs or exec_num_reqs
    if batch_desc.num_tokens <= 0 or exec_num_reqs <= 0:
        raise RuntimeError(
            "DP execution contract resolved invalid geometry: "
            f"tokens={batch_desc.num_tokens}, requests={exec_num_reqs}"
        )

    # Forward-context DP metadata describes execution capacity. Preserve live
    # occupancy above before rewriting this reusable row in place.
    live_num_tokens.fill_(batch_desc.num_tokens)
    return batch_desc, DPSyncState(
        num_tokens_across_dp=live_num_tokens,
        uniform_token_count=uniform_token_count,
        eager=batch_desc.cg_mode == CUDAGraphMode.NONE,
        generation=generation,
        parent_generation=parent_generation,
        live_num_tokens_across_dp=live_tokens_tuple,
        live_num_reqs_across_dp=live_reqs_tuple,
        execution_num_reqs=exec_num_reqs,
    )


def _dispatch_local_batch(
    cudagraph_manager: CudaGraphManager | None,
    num_reqs: int,
    num_tokens: int,
    uniform_token_count: int | None,
    max_query_len: int | None,
    need_eager: bool,
    num_active_loras: int,
) -> BatchExecutionDescriptor:
    if need_eager:
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=num_active_loras,
        )

    assert cudagraph_manager is not None, (
        "cudagraph_manager should only be None during profile run, "
        "where need_eager must be True"
    )
    return cudagraph_manager.dispatch(
        num_reqs,
        num_tokens,
        uniform_token_count,
        num_active_loras=num_active_loras,
        max_query_len=max_query_len,
    )


def sync_cudagraph_and_dp_padding(
    cudagraph_manager: CudaGraphManager | None,
    desired_batch_desc: BatchExecutionDescriptor,
    num_tokens: int,
    num_reqs: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    max_query_len: int | None = None,
    num_active_loras: int = 0,
) -> tuple[BatchExecutionDescriptor, DPSyncState | None]:
    """
    Coordinates the batch descriptor and DP padding across all ranks.

    Returns (synced_batch_desc, sync). `sync` is None when no rank has work.
    """
    assert dp_size > 1, "DP size must be greater than 1"
    group = get_dp_group().cpu_group
    tensor = torch.zeros(4, dp_size, dtype=torch.int32, device="cpu")
    tensor[0][dp_rank] = num_tokens
    tensor[1][dp_rank] = desired_batch_desc.cg_mode.value
    tensor[2][dp_rank] = uniform_token_count or 0  # (0 means None)
    tensor[3][dp_rank] = max_query_len or -1  # (-1 means None)
    dist.all_reduce(tensor, group=group)

    return _finish_cudagraph_and_dp_padding(
        cudagraph_manager,
        desired_batch_desc,
        tensor,
        num_tokens,
        num_reqs,
        num_active_loras,
    )


def _finish_cudagraph_and_dp_padding(
    cudagraph_manager: CudaGraphManager | None,
    desired_batch_desc: BatchExecutionDescriptor,
    tensor: torch.Tensor,
    num_tokens: int,
    num_reqs: int,
    num_active_loras: int,
) -> tuple[BatchExecutionDescriptor, DPSyncState | None]:
    num_tokens_across_dp = tensor[0]
    cg_mode_across_dp = tensor[1]
    uniform_token_counts_across_dp = tensor[2]
    max_query_lens_across_dp = tensor[3]

    # If ranks disagree on the uniform token count, or its 0 (means None) set to None
    synced_uniform_token_count: int | None = int(uniform_token_counts_across_dp[0])
    if synced_uniform_token_count == 0 or not torch.all(
        uniform_token_counts_across_dp == synced_uniform_token_count
    ):
        synced_uniform_token_count = None

    if torch.all(num_tokens_across_dp == 0).item():
        synced_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE, num_tokens=0, num_reqs=0
        )
        return synced_desc, None

    synced_cg_mode = CUDAGraphMode(int(cg_mode_across_dp.min().item()))

    # If any rank wants to run eager, all ranks run eager
    if synced_cg_mode == CUDAGraphMode.NONE:
        return (
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                num_active_loras=desired_batch_desc.num_active_loras,
            ),
            DPSyncState(
                num_tokens_across_dp=num_tokens_across_dp,
                uniform_token_count=synced_uniform_token_count,
                eager=True,
            ),
        )

    assert cudagraph_manager is not None, (
        "cudagraph_manager should only be None during profile run, "
        "where synced_cg_mode must be NONE across all DP ranks"
    )
    synced_num_tokens = int(num_tokens_across_dp.max().item())

    # Varlen decode graphs are selected by the query-length bound, so ranks must agree
    # on it or they pad to different token counts below.
    synced_max_query_len: int | None = None
    if bool(torch.all(max_query_lens_across_dp != -1).item()):
        synced_max_query_len = int(max_query_lens_across_dp.max().item())

    # Dispatch for the final synced values, use num_reqs instead of synced_num_reqs
    # so we don't perform request padding for PIECEWISE graphs.
    # num_active_loras is per-rank and doesn't need cross-rank agreement.
    synced_desc = cudagraph_manager.dispatch(
        num_reqs,
        synced_num_tokens,
        synced_uniform_token_count,
        num_active_loras=num_active_loras,
        max_query_len=synced_max_query_len,
    )

    # Update num_tokens_across_dp to reflect padded size.
    num_tokens_across_dp[:] = synced_desc.num_tokens

    return synced_desc, DPSyncState(
        num_tokens_across_dp=num_tokens_across_dp,
        uniform_token_count=synced_uniform_token_count,
        eager=False,
    )


def dispatch_cg_and_sync_dp(
    cudagraph_manager: CudaGraphManager | None,
    num_reqs: int,
    num_tokens: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    max_query_len: int | None = None,
    need_eager: bool = False,
    num_active_loras: int = 0,
    dp_sync: DPSyncState | None = None,
) -> tuple[BatchExecutionDescriptor, DPSyncState | None]:
    """Pick a cudagraph descriptor for this batch, agreeing it across DP ranks.

    Runs a collective when dp_size > 1 so every rank dispatches to the same
    shape. Pass `dp_sync` from a dispatch already made over this same batch (a
    drafter's prefill runs the target's batch shape) to reuse that agreement
    instead, with no collective.

    Args:
        cudagraph_manager: Manager to dispatch against. May be None only when
            `need_eager` is True (profile run).
        num_reqs: Requests in this rank's batch.
        num_tokens: Tokens in this rank's batch, already padded by the caller.
        uniform_token_count: Per-request token count if this rank's batch is a
            uniform decode, else None. `dp_sync.uniform_token_count` takes its
            place when a sync is reused, since that one is agreed across ranks.
        dp_size: Data-parallel world size. 1 skips all cross-rank work.
        dp_rank: This rank's index in the DP group.
        max_query_len: Upper bound on per-request query length, for selecting
            varlen decode graphs. None means the graph must not constrain it.
        need_eager: Force `CUDAGraphMode.NONE` instead of dispatching.
        num_active_loras: Active LoRA count for this rank. Does not need
            cross-rank agreement; it never changes a bucket's token count.
        dp_sync: Agreement from a prior dispatch over this same batch, to reuse.
            Must come from a batch with this same padded `num_tokens` and the
            same `uniform_token_count`. An execution contract also supplies the
            rank-identical request capacity used for graph selection. Passing a
            sync from a different batch is a caller error and trips an assert.

    Returns:
        (batch_desc, sync), where `sync` is this batch's agreement for a later
        dispatch to reuse. It is None when `dp_size` is 1 or no rank has work.
    """
    reuse_eager = dp_sync is not None and dp_sync.eager

    dispatch_num_reqs = (
        dp_sync.execution_num_reqs
        if dp_sync is not None and dp_sync.execution_num_reqs is not None
        else num_reqs
    )
    batch_desc = _dispatch_local_batch(
        cudagraph_manager,
        dispatch_num_reqs,
        num_tokens,
        dp_sync.uniform_token_count if dp_sync is not None else uniform_token_count,
        max_query_len,
        need_eager or reuse_eager,
        num_active_loras,
    )

    if dp_size == 1:
        return batch_desc, None

    if dp_sync is not None:
        assert dp_sync.num_tokens_across_dp[dp_rank] == num_tokens, (
            "reusing a DP sync taken over a different batch"
        )
        assert (
            dp_sync.uniform_token_count is None
            or uniform_token_count == dp_sync.uniform_token_count
        ), "reusing a DP sync taken over a different batch"
        if not dp_sync.eager and batch_desc.num_tokens != num_tokens:
            # Capture sizes can differ between managers, so this one may
            # pad further. Every rank pads alike, so report what will run.
            dp_sync = replace(
                dp_sync,
                num_tokens_across_dp=torch.full_like(
                    dp_sync.num_tokens_across_dp, batch_desc.num_tokens
                ),
            )
        return batch_desc, dp_sync

    return sync_cudagraph_and_dp_padding(
        cudagraph_manager,
        batch_desc,
        num_tokens,
        num_reqs,
        uniform_token_count,
        dp_size,
        dp_rank,
        max_query_len=max_query_len,
        num_active_loras=num_active_loras,
    )
