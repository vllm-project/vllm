# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-rank service object shared by a model's MonoKernel ops.

A MonoKernel is a persistent launch whose CTAs wait on each other across TP
ranks, so the resources it waits on (scratch, symmetric peer memory, the mailbox
epoch) and the decision to launch at all are per rank and per step, not per
layer. ``MonoRuntime`` owns both: ops reserve resources under an opaque key and
read the one step decision the runtime took.

The key is ``(width, geometry)``, not a width alone: a model may run several
scratch layouts at one step width (GLM-5.2 runs one for its fused-indexer layers
and one for the rest), and a model with a single layout just passes a constant
geometry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.models.common.mono.spec import MonoSpec

logger = init_logger(__name__)


@dataclass(frozen=True)
class StepDecision:
    """Whether this step takes the mono path, and at which width.

    Falsy when ``reason`` says why it does not, so call sites read as
    ``if not decision: return vllm_path(...)``.
    """

    reason: str = ""
    width: int | None = None

    def __bool__(self) -> bool:
        return not self.reason


NO_STEP = StepDecision("not started")


@dataclass
class _Reserved:
    scratch: torch.Tensor | None = None
    peer: Any = None
    owner: Any = None
    width: int | None = None


class MonoRuntime:
    """One rank's MonoKernel resources and step gate.

    Args:
        spec: The kernel's :class:`MonoSpec`; its widths gate every step.
        vllm_config: The engine's ``VllmConfig``.
        device: The rank's device, defaulting to the current accelerator.
        peer_factory: ``(nbytes, rank, world, cpu_group, device) -> buffer``
            allocating and initialising symmetric peer memory. Collective over
            the TP group, so every rank must reserve the same keys in the same
            order. The runtime owns the buffer's lifetime, not its layout: what
            it returns is handed back to the kernel unexamined.
        epoch_words: Length of the mailbox epoch tensor, 0 for kernels that keep
            their own.
        vote: Take a rank-uniform vote on rank-local state before each step.
            Illegal under FULL CUDA graphs (it syncs and all-reduces on CPU).
        vote_each_step: Vote at every step rather than riding on the last vote.
            Leave off when the caller calls :meth:`revote` at the end of its
            steps, so a disabled rank is picked up there.
        enabled: Initial state, False to build the resources but take vLLM's
            path until something enables it.

    """

    def __init__(
        self,
        spec: MonoSpec,
        vllm_config: VllmConfig,
        device: torch.device | None = None,
        peer_factory: Callable[..., Any] | None = None,
        epoch_words: int = 0,
        vote: bool = False,
        vote_each_step: bool = False,
        enabled: bool = True,
    ) -> None:
        from vllm.distributed import get_tp_group

        group = get_tp_group()
        self.spec = spec
        self.vllm_config = vllm_config
        self.rank = group.rank_in_group
        self.world = group.world_size
        self.cpu_group = group.cpu_group
        self.device = device or torch.device(
            "cuda", torch.accelerator.current_device_index()
        )
        self._peer_factory = peer_factory
        self._voting = vote
        self._vote_each_step = vote_each_step
        self.epoch = (
            torch.zeros(epoch_words, dtype=torch.int32, device=self.device)
            if epoch_words
            else None
        )
        self._reserved: dict[Any, _Reserved] = {}
        self.enabled = enabled
        self._voted: bool | None = None
        self.step_index = 0
        self.step = NO_STEP

    def reserve(
        self,
        key: Any,
        scratch_bytes: int = 0,
        peer_bytes: int = 0,
        owner: Any = None,
        width: int | None = None,
    ) -> _Reserved:
        """Allocate this key's resources once, eagerly.

        Reserving inside a CUDA-graph capture raises: peer memory is exchanged
        collectively and scratch must outlive the capture. vLLM runs every graph
        batch eagerly before capturing it, so a width first reached under
        capture means the model's widths and the capture sizes disagree.

        Args:
            key: ``(width, geometry)`` or any hashable the model keys launches on.
            scratch_bytes: Device scratch for this key, 0 for none.
            peer_bytes: Symmetric peer memory for this key, 0 for none.
            owner: The kernel object owning this key's launch. Objects exposing
                ``advance_step()`` are advanced once per step and objects
                exposing ``poll_error()`` are read by :meth:`health`.
            width: The step width this key's launch is built for, None when it
                serves every width. Only the step width's owners are advanced.

        Returns:
            The key's resources.

        """
        if key in self._reserved:
            return self._reserved[key]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{self.spec.name}: key {key} first reserved inside a CUDA graph "
                "capture, before any eager step at that width"
            )
        res = _Reserved(owner=owner, width=width)
        if scratch_bytes:
            res.scratch = torch.zeros(
                scratch_bytes, dtype=torch.uint8, device=self.device
            )
        if peer_bytes:
            if self._peer_factory is None:
                raise RuntimeError(
                    f"{self.spec.name}: no peer_factory to reserve peers"
                )
            res.peer = self._peer_factory(
                peer_bytes, self.rank, self.world, self.cpu_group, self.device
            )
        self._reserved[key] = res
        return res

    def scratch(self, key: Any) -> torch.Tensor | None:
        return self._reserved[key].scratch

    def peer(self, key: Any) -> Any:
        return self._reserved[key].peer

    def owners(self, width: int | None = None) -> list[Any]:
        """Reserved kernel objects in reservation order.

        Args:
            width: Keep only the objects built for this step width (and those
                built for every width). None keeps all of them.

        """
        return [
            r.owner
            for r in self._reserved.values()
            if r.owner is not None
            and (width is None or r.width is None or r.width == width)
        ]

    def launch_args(self, key: Any, **tags) -> dict[str, Any]:
        """The resources a launch at ``key`` takes, plus the caller's tags.

        Only resources this runtime actually owns are added, so a kernel that
        keeps its own scratch and peers gets its tags back unchanged.
        """
        res = self._reserved[key]
        out = dict(tags)
        if res.scratch is not None:
            out["scratch"] = res.scratch
        if res.peer is not None:
            out["peers"] = res.peer
        if self.epoch is not None:
            out["epoch"] = self.epoch
        return out

    def step_begin(self, rows: int, reason: str = "") -> StepDecision:
        """Take this step's go / no-go decision, once per step.

        Metadata reasons the caller already found need no agreement (every TP
        rank reads the same attention metadata); rank-local state enters only
        through :meth:`vote`.

        Args:
            rows: The step's query rows, padded up to a built width.
            reason: Why the caller already refuses this step, empty to go.

        Returns:
            The decision, also kept as :attr:`step`.

        """
        width = self.spec.width_for(rows) if self.spec.widths else rows
        if not reason and width is None:
            reason = f"{rows} rows exceed the built widths"
        if not reason:
            reason = self._state_reason()
        self.step = StepDecision(reason, None if reason else width)
        if self.step:
            self.step_index += 1
            for owner in self.owners(width):
                advance = getattr(owner, "advance_step", None)
                if advance is not None:
                    advance()
        return self.step

    def _state_reason(self) -> str:
        if not self._voting:
            return "" if self.enabled else "disabled"
        if self._voted is not True or self._vote_each_step:
            self._voted = self.vote(self.enabled)
        if not self._voted:
            return "disabled" if not self.enabled else "peer_no_go"
        return ""

    def vote(self, ok: bool) -> bool:
        """Rank-uniform agreement: device sync, then a MIN over the CPU group.

        Collective: every rank must reach it. :attr:`_voted` is therefore only
        ever set from a vote's result, which keeps the decision to vote at all
        rank-uniform even though ``enabled`` is not.
        """
        import torch.distributed as dist

        torch.accelerator.synchronize()
        flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.cpu_group)
        return bool(flag.item())

    def revote(self) -> bool:
        """Re-take the vote now, standing in for the next step's.

        Callers that already sync at the end of a step (to read poll errors,
        say) fold their vote into that sync, so the next step starts with an
        agreed state and takes no collective of its own.
        """
        self._voted = self.vote(self.enabled)
        return self._voted

    def health(self, width: int | None = None, clear: bool = False) -> tuple:
        """Expired mailbox polls reported by this runtime's kernel objects."""
        return tuple(
            e
            for owner in self.owners(width)
            if (poll := getattr(owner, "poll_error", None)) is not None
            for e in poll(clear=clear)
        )

    def disable(self, why: str) -> None:
        """Stop taking the mono path, from this rank's next vote.

        Rank-local and collective-free: a rank that disables mid-step would
        deadlock the others if it voted here on its own. The next step's vote,
        which every rank reaches, carries the decision to all of them.
        """
        if self.enabled:
            logger.error("%s: %s -> disabling", self.spec.name, why)
        self.enabled = False
