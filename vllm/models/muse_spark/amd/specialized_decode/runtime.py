# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The host side of the gfx942 specialized decode: the weight tables, the
buffers every width shares and the step (``refusals``: what it refuses).

A step runs layer 0 and layer 1's attention stock, then layers 1-60 as one
launch each (``wide.layer``: layer ``i``'s o-path to layer ``i + 1``'s QKVG
output), with vLLM's QK-norm / RoPE / KV insert and attention op between the
launches, then layer 61 stock from the kernels' residual stream. Every rank
must take the same path (the launches wait on their peers), so the step
decision reads rank-invariant data only.
"""

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from vllm.logger import init_logger

from ..mono.decoder import _routed_experts
from ..mono.peer import PeerBuffer
from .refusals import step_refusal
from .wide import layer as wide
from .wide.abi import FAILED, RECORD_WORDS, health_words
from .wide.common.grid import FOLD_BASE, SLOT_WORDS, sync_words
from .wide.layout import (
    BLOCKS,
    E_LOCAL,
    FIRST_LAYER,
    HIDDEN,
    LAST_LAYER,
    MAX_TOKENS,
    POOL_ROWS,
    QKVG,
    TP,
    UNPOOL_ROWS,
)

logger = init_logger(__name__)

HEALTH_POLL_INTERVAL = 64
# widest step the kernels take (the CUDA graph sizes up to it); they build and
# pass their test up to MAX_TOKENS, but from 64 rows on tuned stock MoE is as fast
MAX_ROWS = 64


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------


def _gain(norm: nn.Module) -> torch.Tensor:
    """``weight + gain_center`` in f32, as the fused norm kernels apply it."""
    return (norm.weight.detach().float() + norm.fused_gain_center).contiguous()


def _rows(t: torch.Tensor, rank: int, n: int) -> torch.Tensor:
    return t.detach()[rank * n : (rank + 1) * n]


def router_planes(w: torch.Tensor) -> torch.Tensor:
    """The rank's fp32 router rows ``[32, 8192]`` as ``dense.ROUTER_ROWS``
    bf16 rows: ``hi``, ``mid``, ``lo`` (``w = hi + mid + lo``), zeros."""
    hi = w.to(torch.bfloat16)
    r = w - hi.float()
    mid = r.to(torch.bfloat16)
    lo = (r - mid.float()).to(torch.bfloat16)
    pad = torch.zeros_like(hi)
    return torch.cat([hi, mid, lo, pad]).contiguous()


class LayerTable:
    """Layer ``i``'s ``wide.layer.WEIGHTS`` addresses and the tensors they
    point into (views of vLLM's parameters, and the small f32 / plane copies
    this keeps alive)."""

    def __init__(self, layers: Sequence[nn.Module], i: int, rank: int):
        layer, nxt = layers[i], layers[i + 1]
        moe = layer.feed_forward
        experts = _routed_experts(layer)
        assert experts is not None  # refusals checked every layer
        t = {
            "o_w": layer.self_attn.o_proj.weight,
            "o_ws": layer.self_attn.o_proj.weight_scale,
            "attn_alpha": layer.add_attn.alpha,
            "attn_beta": layer.add_attn.beta,
            "pre_ffn_w1": _gain(layer.pre_ffn_norm),
            "router": router_planes(_rows(moe.router.weight, rank, E_LOCAL)),
            "bias": moe.expert_bias,
            "pooler": _rows(moe.pooler.weight, rank, POOL_ROWS),
            "pre_expert_w1": _gain(moe.pre_expert_norm),
            "w13": experts.w13_weight,
            "w13s": experts.w13_weight_scale,
            "w2": experts.w2_weight,
            "w2s": experts.w2_weight_scale,
            "post_expert_w1": _gain(moe.post_expert_norm),
            "unpooler": _rows(moe.unpooler.weight, rank, UNPOOL_ROWS),
            "post_ffn_w1": _gain(layer.post_ffn_norm),
            "ffn_alpha": layer.add_ffn.alpha,
            "ffn_beta": layer.add_ffn.beta,
            "pre_qkv_w1": _gain(nxt.self_attn.pre_qkv_proj_norm),
            "qkvg_w": nxt.self_attn.qkv_proj.weight,
            "qkvg_ws": nxt.self_attn.qkv_proj.weight_scale,
        }
        assert set(t) == set(wide.WEIGHTS)
        for name in ("pooler", "unpooler"):
            assert t[name].dtype == torch.bfloat16 and t[name].is_contiguous(), name
        self.tensors = t
        self.addresses = [t[n].data_ptr() for n in wide.WEIGHTS]
        self.w13_ld = experts.w13_weight.stride(1)
        self.w2_ld = experts.w2_weight.stride(1)


# --------------------------------------------------------------------------
# widths
# --------------------------------------------------------------------------


class Buffers:
    """The device state every width shares, sized for MAX_TOKENS rows: the
    launches of all widths run in order on one stream, and one epoch keeps
    the mailbox tags increasing across widths."""

    def __init__(self, device, group, tp: int):
        sizes = wide.scratch_bytes(MAX_TOKENS)
        self.scratch = {
            n: torch.zeros(max(sizes[n], 16), dtype=torch.uint8, device=device)
            for n in wide.SCRATCH
        }
        self.stab = torch.tensor(
            [self.scratch[n].data_ptr() for n in wide.SCRATCH],
            dtype=torch.int64,
            device=device,
        )
        self.peers = PeerBuffer(wide.MAILBOX.bytes(), group)
        self.ptab = torch.tensor(self.peers.pointers, dtype=torch.int64, device=device)
        assert len(self.peers.pointers) == tp
        self.sync = torch.zeros(sync_words(), dtype=torch.int32, device=device)
        self.health = torch.zeros(
            health_words(BLOCKS), dtype=torch.int32, device=device
        )
        self.epoch = torch.zeros(4, dtype=torch.int32, device=device)
        self.qkv = [
            torch.zeros(MAX_TOKENS, QKVG, dtype=torch.bfloat16, device=device)
            for _ in range(2)
        ]

    def stream(self, rows: int) -> torch.Tensor:
        """The kernels' f32 residual stream of a ``rows``-row step."""
        flat = self.scratch["stream"][: rows * HIDDEN * 4].view(torch.float32)
        return flat.view(rows, HIDDEN)

    def slots(self, k: int) -> tuple[int, int, int]:
        """(sync, sync_next, folds) of the step's ``k``-th launch: the launches
        alternate barrier slots (60 a step: the pattern repeats a step)."""
        base = self.sync.data_ptr()
        return (
            base + (k % 2) * SLOT_WORDS * 4,
            base + ((k + 1) % 2) * SLOT_WORDS * 4,
            base + FOLD_BASE * 4,
        )

    def close(self) -> None:
        self.peers.close()


def built_widths(vllm_config) -> list[int]:
    """The CUDA graph capture sizes up to ``MAX_ROWS`` (every decode width
    vLLM pads to), or its powers of two when there are none."""
    sizes = vllm_config.compilation_config.cudagraph_capture_sizes or []
    widths = sorted({s for s in sizes if 1 <= s <= MAX_ROWS})
    return widths or [1, 2, 4, 8, 16]


# --------------------------------------------------------------------------
# the step
# --------------------------------------------------------------------------


def _attend(attn: nn.Module, qkv: torch.Tensor, positions: torch.Tensor):
    """Stock QK-norm / RoPE / KV insert and attention of ``qkv``: (o, og)."""
    q, k, v, og = attn._qk_norm_rope_kv_insert(qkv, positions)
    if attn.fused_qk_norm_rope_kv_insert:
        return attn.attn(q, None, None), og
    return attn.attn(q, k, v), og


class SpecializedDecode:
    """Runs decoder layers 1-60 of eligible decode steps through the kernels
    (the ``MuseSpark1xModel.mono`` interface)."""

    def __init__(
        self,
        model: nn.Module,
        rank: int,
        group: Any,
        device,
        widths: Sequence[int],
        route_scale: float,
    ):
        self.rank, self.group, self.device = rank, group, device
        self.refresh_weights(model)
        ld13, ld2 = self.tables[FIRST_LAYER].w13_ld, self.tables[FIRST_LAYER].w2_ld
        assert all(t.w13_ld == ld13 and t.w2_ld == ld2 for t in self.tables.values())
        self.bufs = Buffers(device, group, TP)
        self.widths = {
            rows: wide.build(rows, route_scale=route_scale, w13_ld=ld13, w2_ld=ld2)[0]
            for rows in widths
        }
        self._fallbacks: set[str] = set()
        self._engaged: set[int] = set()
        self._steps = 0
        self._health_host = torch.zeros_like(
            self.bufs.health, device="cpu"
        ).pin_memory()
        self._health_event: torch.cuda.Event | None = None

    def refresh_weights(self, model: nn.Module) -> None:
        """Re-take the views, e.g. after a weight reload replaced parameters."""
        self.tables = {
            i: LayerTable(model.layers, i, self.rank)
            for i in range(FIRST_LAYER, LAST_LAYER + 1)
        }
        self.wtab = torch.tensor(
            [self.tables[i].addresses for i in range(FIRST_LAYER, LAST_LAYER + 1)],
            dtype=torch.int64,
            device=self.device,
        )

    def try_forward(
        self,
        model: nn.Module,
        positions: torch.Tensor,
        x: Any,
        aux_hidden_states: list[torch.Tensor],
    ) -> Any:
        """The residual stream after the last layer, or None for stock."""
        layers = model.layers
        reason = None
        if not isinstance(x, torch.Tensor):
            reason = "input"
        else:
            names = [
                layers[i].self_attn.attn.layer_name
                for i in range(FIRST_LAYER, LAST_LAYER + 2)
            ]
            reason = step_refusal(
                x.shape[0], tuple(self.widths), names, model.aux_hidden_state_layers
            )
        if reason is not None:
            if reason not in self._fallbacks:
                self._fallbacks.add(reason)
                logger.debug("Muse Spark gfx942 decode: stock step (%s)", reason)
            return None
        rows = x.shape[0]
        if rows not in self._engaged:
            self._engaged.add(rows)
            logger.info("Muse Spark gfx942 decode engaged: %d rows", rows)
        bufs, launch = self.bufs, self.widths[rows]
        bufs.epoch.add_(1)

        from ..model import norm_residual_then_norm  # model.py imports this package

        y = layers[0](positions, x)
        attn = layers[FIRST_LAYER].self_attn
        attn_input, stream = norm_residual_then_norm(
            y, attn.pre_qkv_proj_norm, out_dtype=layers[FIRST_LAYER].default_dtype
        )
        bufs.stream(rows).copy_(stream)
        qkv, _ = attn.qkv_proj(attn_input)
        o, og = _attend(attn, qkv, positions)
        qkv_in = qkv
        cur = torch.cuda.current_stream()
        for k, i in enumerate(range(FIRST_LAYER, LAST_LAYER + 1)):
            if i > FIRST_LAYER:
                o, og = _attend(layers[i].self_attn, qkv_in, positions)
            if og.stride(0) != QKVG or not o.is_contiguous():
                raise RuntimeError(
                    f"Muse Spark gfx942 decode: layer {i} attention output "
                    f"{tuple(o.shape)} {o.stride()} / o-gate {og.stride()}"
                )
            qkv_out = bufs.qkv[k % 2][:rows]
            sync, sync_next, folds = bufs.slots(k)
            launch(
                self.wtab[k].data_ptr(),
                bufs.stab.data_ptr(),
                bufs.ptab.data_ptr(),
                o.data_ptr(),
                qkv_in.data_ptr(),
                qkv_out.data_ptr(),
                bufs.epoch.data_ptr(),
                i,
                self.rank,
                sync,
                sync_next,
                folds,
                bufs.health.data_ptr(),
                stream=cur,
            )
            qkv_in = qkv_out

        last = layers[LAST_LAYER + 1]
        o, og = _attend(last.self_attn, qkv_in, positions)
        return last.after_attention(bufs.stream(rows), last.self_attn.back(o, og))

    def poll_health(self) -> None:
        """Raise if a launch timed out waiting for a peer rank or the grid
        (an asynchronous copy every HEALTH_POLL_INTERVAL steps, checked at the
        next poll, so a step never waits for the device)."""
        if torch.cuda.is_current_stream_capturing():
            return
        self._steps += 1
        if self._health_event is not None and self._health_event.query():
            self._health_event = None
            host = self._health_host
            if int(host[FAILED]) != 0:
                records = host[RECORD_WORDS:].view(BLOCKS, RECORD_WORDS)
                stuck = records[records[:, 0] != 0][:8].tolist()
                raise RuntimeError(
                    f"Muse Spark gfx942 decode: rank {self.rank}: "
                    f"a launch timed out {stuck}"
                )
        if self._health_event is None and self._steps % HEALTH_POLL_INTERVAL == 0:
            self._health_host.copy_(self.bufs.health, non_blocking=True)
            self._health_event = torch.cuda.Event()
            self._health_event.record()

    def close(self) -> None:
        self.bufs.close()


__all__ = ["Buffers", "SpecializedDecode", "built_widths", "router_planes"]
