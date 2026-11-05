# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The refusals that need the loaded model or the bound caches, and the
fail-stop watch."""

from collections.abc import Callable

import torch
import torch.distributed as dist

FP8 = torch.float8_e4m3fn


def weight_reasons(layers) -> list[str]:
    """Why the loaded decoder layers cannot run the kernels (empty: they can):
    fp8 per-tensor linears and experts, norms and router in the model dtype
    (float16 or bfloat16). The kernels check every shape and stride again when
    the layers are bound."""
    reasons: list[str] = []
    for mono in layers:
        layer, where = mono.layer, mono.name
        try:
            t = mono.weights()
        except AttributeError as e:
            reasons.append(f"{where}: not a Qwen3.5 MoE decoder layer ({e})")
            continue
        # (norm, norm, router, shared gate, out, scale, w13, scale, w2, scale,
        #  shared gate_up, scale, shared down, scale, mixer ...)
        fp8 = [t[4], t[6], t[8], t[10], t[12], t[14]]
        if any(w.dtype != FP8 for w in fp8):
            reasons.append(f"{where}: linear or expert weights other than fp8 e4m3")
        per_tensor = [t[5], t[11], t[13], t[15]]
        if any(s.numel() != 1 for s in per_tensor):
            reasons.append(f"{where}: fp8 scales other than per-tensor")
        experts = layer.mlp.experts
        routed = getattr(experts, "routed_experts", experts)
        if t[7].numel() != routed.w13_weight.shape[0]:
            reasons.append(f"{where}: expert scales other than one per expert")
        dtypes = {w.dtype for w in (t[0], t[1], t[2])}
        if len(dtypes) != 1 or dtypes - {torch.float16, torch.bfloat16}:
            reasons.append(
                f"{where}: norms or router other than one float16 / bfloat16"
            )
    return reasons


def bind_reasons(bind: Callable[[], None]) -> list[str]:
    """Binds, returning the kernels' refusal (a shape, dtype or stride they do
    not serve) instead of raising."""
    try:
        bind()
    except RuntimeError as e:
        return [str(e).splitlines()[0]]
    return []


class FailStop:
    """Stops every TP rank at the same step once any of them gave up an
    in-kernel wait (its sticky health word, ``_xpu_C.mono_health``). Each step
    queues a D2H copy of this rank's word and keeps the newest copy that has
    landed, without blocking; every CHECK_EVERY steps the ranks agree on those
    over the TP CPU group and all raise together. A rank that raised alone
    would leave the others in the next collective for ever. A failure is seen
    within about CHECK_EVERY steps."""

    CHECK_EVERY = 16

    def __init__(self, health: torch.Tensor, rank: int, world: int, group):
        self.failed = health[0:1]
        self.rank, self.world, self.group = rank, world, group
        self.host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self.copied: torch.Event | None = None
        self.seen = False
        self.steps = 0

    def after_step(self) -> None:
        """Once per executed step on every TP rank (eager, also under graph
        replay)."""
        if torch.xpu.is_current_stream_capturing():
            return
        if self.copied is None or self.copied.query():
            if self.copied is not None:
                self.seen |= bool(self.host[0])
            self.host.copy_(self.failed, non_blocking=True)
            self.copied = torch.xpu.Event()
            self.copied.record()
        self.steps += 1
        if self.steps % self.CHECK_EVERY:
            return
        votes = torch.zeros(self.world, dtype=torch.int32)
        votes[self.rank] = int(self.seen)
        dist.all_reduce(votes, group=self.group)
        if votes.any():
            ranks = [r for r, v in enumerate(votes.tolist()) if v]
            raise RuntimeError(
                f"Qwen3.5 XPU mono decode: TP ranks {ranks} gave up an in-kernel "
                "wait (a stalled peer GPU); stopping the engine"
            )
