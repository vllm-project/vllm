# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental K3 A8W4 TP runtime adapter.

Callers must serialize all GPU work around the persistent kernel, supply identical
inputs/routing on every TP rank, and prepare before graph capture. One pool belongs
to one TP group and model lifetime; layers share it, but own their packed weights.
The fallback must implement the same A8W4 routed MoE with an already reduced output.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class K3TPShape:
    world_size: int = 8
    model_dim: int = 3584
    inter_dim: int = 3072
    experts: int = 896
    topk: int = 16

    def __post_init__(self) -> None:
        if (
            self.world_size,
            self.model_dim,
            self.inter_dim,
            self.experts,
            self.topk,
        ) != (8, 3584, 3072, 896, 16):
            raise ValueError("This experimental adapter only supports K3 TP8")

    @property
    def local_inter_dim(self) -> int:
        return self.inter_dim // self.world_size

    def padded_tokens(self, tokens: int) -> int | None:
        """AR mode requires M divisible by TP; FP8 supports global M <= 256."""
        if tokens < 0:
            raise ValueError("tokens must be nonnegative")
        if tokens > 256:
            return None
        return (tokens + self.world_size - 1) // self.world_size * self.world_size

    @property
    def raw_weight_shapes(self) -> tuple[tuple[int, int, int], ...]:
        e, h, i = self.experts, self.model_dim, self.local_inter_dim
        return (
            (e, 2 * i, h // 2),
            (e, 2 * i, h // 32),
            (e, h, i // 2),
            (e, h, i // 32),
        )


@dataclass(frozen=True)
class K3TPWeights:
    w1: torch.Tensor
    w1_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor

    def as_kwargs(self) -> dict[str, torch.Tensor]:
        return {
            "w1": self.w1,
            "w1_scale": self.w1_scale,
            "w2": self.w2,
            "w2_scale": self.w2_scale,
        }

    @classmethod
    def from_checkpoint(
        cls,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        *,
        gate_up_layout: str,
        shape: K3TPShape | None = None,
    ) -> "K3TPWeights":
        """Convert raw TP-local checkpoint bytes, before any vLLM shuffle.

        W13 must contain separated [gate, up] rows, not interleaved rows or
        already shuffled AITER backend parameters. No dequantization is used.
        The caller owns the original tensors and their replacement lifecycle.
        """
        shape = shape or K3TPShape()
        if gate_up_layout != "separated":
            raise ValueError("MegaMoE TP needs raw separated gate/up checkpoint rows")
        tensors = (w13, w13_scale, w2, w2_scale)
        if any(getattr(tensor, "is_shuffled", False) for tensor in tensors):
            raise ValueError("cannot convert already shuffled backend weights")
        for name, tensor, expected in zip(
            ("w13", "w13_scale", "w2", "w2_scale"),
            tensors,
            shape.raw_weight_shapes,
        ):
            if tensor.dtype != torch.uint8 or tuple(tensor.shape) != expected:
                raise ValueError(f"{name}: expected raw uint8 {expected}")
            if not tensor.is_contiguous() or tensor.device != w13.device:
                raise ValueError("weights must be contiguous on the same device")
        if w13.device.type != "cuda":
            raise ValueError("weight shuffle requires a ROCm device")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("convert weights before capture")
        from aiter.ops.shuffle import shuffle_weight
        from aiter.utility.fp4_utils import e8m0_shuffle

        fp4 = torch.float4_e2m1fn_x2
        e8m0 = torch.float8_e8m0fnu
        return cls(
            shuffle_weight(w13.view(fp4), layout=(16, 16)),
            e8m0_shuffle(w13_scale.view(-1, shape.model_dim // 32).view(e8m0)),
            shuffle_weight(w2.view(fp4), layout=(16, 16)),
            e8m0_shuffle(w2_scale.view(-1, shape.local_inter_dim // 32).view(e8m0)),
        )


class K3MegaMoETPPool:
    """Explicitly owned, shared runtime; never creates a runtime during forward.

    Pass vLLM's TP cpu_group for IPC metadata collectives. Every rank must prepare
    the same capacities in the same order. Prepare one instance before attaching
    it to any layers. Concurrent requests/streams sharing a pool are unsupported.
    """

    model_mode: str

    def __init__(self, *, group: Any, shape: K3TPShape | None = None):
        shape = shape or K3TPShape()
        if not dist.is_initialized() or dist.get_world_size(group) != shape.world_size:
            raise ValueError("an initialized eight-rank TP process group is required")
        self.group = group
        self.rank = dist.get_rank(group)
        self.shape = shape
        self._runtime: Any = None
        self._prepared: frozenset[int] = frozenset()

    def prepare(self, weights: K3TPWeights, tokens: Iterable[int]) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare must run before capture")
        capacities = {self.shape.padded_tokens(m) for m in tokens}
        if None in capacities or 0 in capacities or not capacities:
            raise ValueError("prepare expects nonempty token counts in [1, 256]")
        valid_capacities = cast(set[int], capacities)
        local_tokens = sorted(m // self.shape.world_size for m in valid_capacities)
        if self._runtime is None:
            from aiter import ActivationType
            from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP, MegaMoeTPConfig

            cfg = MegaMoeTPConfig(
                rank=self.rank,
                world_size=self.shape.world_size,
                model_dim=self.shape.model_dim,
                inter_dim=self.shape.local_inter_dim,
                experts=self.shape.experts,
                topk=self.shape.topk,
                max_local_tokens=256 // self.shape.world_size,
                activation=ActivationType.Situv2,
                beta=4.0,
                linear_beta=25.0,
                comm_mode="ar",
                comm_dtype="bf16",
                ar_gather="bf16",
                act_dtype="fp8",
            )
            self._runtime = MegaMoeTP(
                cfg, **weights.as_kwargs(), group=self.group, device=weights.w1.device
            )
        else:
            self._runtime.set_weights(**weights.as_kwargs())
        self._runtime.prepare(local_tokens)
        self._prepared = self._prepared.union(valid_capacities)

    def forward(
        self,
        weights: K3TPWeights,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        fallback: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Return owned, already reduced latent rows; no norm or shared expert.

        Fallback is selected by global M only, identically on every rank. Runtime
        initialization/IPC/watchdog errors propagate, including under capture.
        Call check_errors outside capture before trusting any benchmark result.
        """
        if x.ndim != 2 or x.shape[1] != self.shape.model_dim:
            raise ValueError("expected K3 latent rows [M, 3584]")
        expected = (x.shape[0], self.shape.topk)
        if topk_ids.shape != expected or topk_weights.shape != expected:
            raise ValueError("routing must have shape [M, 16]")
        if x.dtype != torch.bfloat16:
            raise ValueError("expected BF16 input before internal FP8 quantization")
        padded = self.shape.padded_tokens(x.shape[0])
        if padded is None:
            return fallback(x, topk_weights, topk_ids)
        if padded == 0:
            return x.clone()
        if self._runtime is None or padded not in self._prepared:
            raise RuntimeError("prepare this token capacity on every rank first")
        rows = x.shape[0]
        if padded != rows:
            # Zero-weight padding uses valid expert 0 and cannot affect real rows.
            x = torch.cat((x, x.new_zeros(padded - rows, self.shape.model_dim)))
            topk_weights = torch.cat(
                (topk_weights, topk_weights.new_zeros(padded - rows, self.shape.topk))
            )
            topk_ids = torch.cat(
                (topk_ids, topk_ids.new_zeros(padded - rows, self.shape.topk))
            )
        self._runtime.set_weights(**weights.as_kwargs())
        reduced = self._runtime(
            x.contiguous(),
            topk_weights.float().contiguous(),
            topk_ids.to(torch.int32).contiguous(),
        )
        # A later layer reuses the runtime's output buffer; never expose its view.
        return reduced[:rows].clone()

    def check_errors(self) -> None:
        if self._runtime is None:
            raise RuntimeError("pool is not prepared")
        self._runtime.check_errors()


def k3_tp_tail_partial(
    reduced_latent: torch.Tensor,
    shared_partial: torch.Tensor,
    up_proj_shard: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    norm: Callable[[torch.Tensor], torch.Tensor] | None,
) -> torch.Tensor:
    """Merge one rank's hidden shard; caller all-reduces this result exactly once.

    MegaMoE AR output is already reduced: do not reduce it again before RMSNorm.
    Shared output must still be TP partial. Up-projection is BF16, output-row
    sharded, with no gather. Shared work and this tail must run after MegaMoE on
    the same stream (or with explicit cross-stream dependencies).
    """
    if world_size != 8 or not 0 <= rank < world_size:
        raise ValueError("expected a K3 TP8 rank")
    if reduced_latent.ndim != 2 or shared_partial.ndim != 2:
        raise ValueError("expected matrices")
    rows, hidden = shared_partial.shape
    if hidden % world_size or reduced_latent.shape[0] != rows:
        raise ValueError("shared hidden dimension must be divisible by TP")
    shard_size = hidden // world_size
    if up_proj_shard.shape != (shard_size, reduced_latent.shape[1]):
        raise ValueError("expected output-row sharded up-projection weight")
    latent = norm(reduced_latent) if norm is not None else reduced_latent
    output = shared_partial.clone()
    output.narrow(1, rank * shard_size, shard_size).addmm_(latent, up_proj_shard.t())
    return output
