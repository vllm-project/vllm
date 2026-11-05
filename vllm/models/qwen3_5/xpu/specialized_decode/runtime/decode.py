# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen3.5 XPU mono decode executor (layer mode): one routing decision per
step, then a mono kernel per decoder layer inside the stock layer loop.

Built by the model state once the weights are loaded
(``model_state.Qwen3_5XpuModelState``): checks the loaded weights, maps the
peer mailboxes and builds the layer kernels of the target model and of the MTP
drafter, before memory profiling and graph capture. ``bind_caches`` binds them
to every KV cache vLLM allocates. Each step ``plan`` returns a ``Step`` or None;
``Step.layer(i)`` is the kernel replacing decoder layer i. Weight reloads are not
supported."""

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, NamedTuple

import torch
from torch import nn

from vllm.config import CUDAGraphMode
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

from . import (
    DECODE_ROWS,
    MTP_LAYER_ID,
    MTP_PROLOGUE_ID,
    MTP_ROWS,
    SWITCH,
    DecodeConfig,
)
from .guards import FailStop, bind_reasons, weight_reasons
from .layers import MonoLayer, MtpPrologue
from .peer import PeerRegion, tp_agree

logger = init_logger(__name__)


class Step(NamedTuple):
    """A planned step: ``rows`` rows of ``q`` tokens a request (q 1: decode)."""

    rows: int
    q: int
    attn_metadata: Mapping[str, Any]  # by layer name
    kernels: Mapping[int, MonoLayer]  # by decoder layer

    def layer(self, idx: int) -> Callable | None:
        """The kernel replacing decoder layer idx in this step, or None (vLLM's).
        Called like the layer: (positions, hidden_states, residual)."""
        kernel = self.kernels.get(idx)
        return None if kernel is None else partial(_launch, kernel, self)


def _launch(kernel: MonoLayer, step: Step, positions, hidden_states, residual):
    return kernel.launch(step.q, step.attn_metadata, positions, hidden_states, residual)


def agree(reasons: list[str], what: str) -> None:
    """Raises on every TP rank if any rank has ``reasons``: a rank that went on
    alone would leave the other spinning in-kernel."""
    if not tp_agree(not reasons, get_tp_group().cpu_group):
        why = "; ".join(reasons) if reasons else "another TP rank refused"
        raise ValueError(
            f"Qwen3.5 XPU mono decode ({SWITCH}) does not support {what}: "
            f"{why} (without the switch vLLM's decode runs)"
        )


def step_kind(md, gdn_prefix: str, attn_prefix: str, rows: int, q: int) -> int | None:
    """Q if a step's GatedDeltaNet and attention metadata (host fields, the same
    on every TP rank) are a pure verify step of whole requests of q tokens
    filling ``rows`` rows, 1 if they are a pure one-token decode step of
    ``rows`` rows, else None."""
    if not isinstance(md, dict):
        return None
    g: Any = md.get(gdn_prefix)
    a: Any = md.get(attn_prefix)
    if g is None or a is None or g.num_prefills:
        return None
    if q > 1:
        s = g.spec_state_indices_tensor
        if g.num_decodes or not g.num_spec_decodes or s is None or s.size(-1) != q:
            return None
        if g.uniform_spec_sequence_length != q:
            return None
        if rows % q or g.num_spec_decodes * q > rows or a.max_query_len != q:
            return None
        return q
    if g.num_spec_decodes or not 1 <= g.num_decodes <= rows:
        return None
    return 1 if a.max_query_len == 1 else None


def draft_kind(md, attn_prefix: str, rows: int, q: int) -> int | None:
    """The MTP drafter's pass: q for its first pass of a verify step (the
    target's shape, q tokens a request), 1 for the later one-token passes.
    Only the drafter's attention metadata is seen here: a batch of short prompts
    of at most q tokens can look like a verify step. Its drafts may then be
    wrong, which costs acceptance, never correctness (the target verifies them),
    and the kernels stay within the step's rows."""
    if not isinstance(md, dict):
        return None
    a: Any = md.get(attn_prefix)
    if a is None or a.num_prefill_tokens:
        return None
    if a.max_query_len == 1 and a.num_decode_tokens <= rows:
        return 1
    if q > 1 and a.max_query_len == q and rows % q == 0:
        return q
    return None


class SpecializedDecode:
    """The executor: builds the layer kernels, binds them to each KV cache and
    plans every step of the target model and of the MTP drafter."""

    def __init__(
        self,
        backbone: nn.Module,
        config: DecodeConfig,
        drafters: list[nn.Module] | None = None,
    ):
        tp = get_tp_group()
        self.config = config
        self.rank, self.group = tp.rank_in_group, tp.cpu_group
        layers = {
            i: MonoLayer(backbone.layers[i], i, f"layer.{i}")
            for i in range(backbone.start_layer, backbone.end_layer)
        }
        if not layers:
            raise RuntimeError(
                "Qwen3.5 XPU mono decode: no decoder layer to replace in "
                f"{type(backbone).__name__}"
            )
        self.kernels = layers
        # the MTP drafter: its decoder layer and its prologue
        self.draft: dict[int, MonoLayer] = {}
        self.prologue: MtpPrologue | None = None
        for predictor in drafters or ():
            if len(predictor.layers) != 1 or self.prologue is not None:
                raise RuntimeError(
                    "Qwen3.5 XPU mono decode: one MTP drafter of one layer only"
                )
            self.draft[0] = MonoLayer(predictor.layers[0], MTP_LAYER_ID, "mtp.layer")
            self.prologue = MtpPrologue(predictor, MTP_PROLOGUE_ID, "mtp.prologue")
        agree(
            weight_reasons([*layers.values(), *self.draft.values()]),
            "the loaded weights",
        )
        kinds = {k.attn for k in layers.values()}
        if kinds != {True, False}:
            raise RuntimeError(
                "Qwen3.5 XPU mono decode: expected GatedDeltaNet and full-attention "
                "layers"
            )
        self.gdn_prefix = next(k.prefix for k in layers.values() if not k.attn)
        self.attn_prefix = next(k.prefix for k in layers.values() if k.attn)
        names = [k.name for k in layers.values()] + [
            k.name for k in self.draft.values()
        ]
        if self.prologue is not None:
            names.append(self.prologue.name)
        self.region = PeerRegion(names, self.rank, self.group)
        ops = torch.ops._xpu_C
        self.failstop = FailStop(
            ops.mono_health(-1), self.rank, tp.world_size, self.group
        )
        # (data_ptr, shape, stride) of every bound cache, and the binds so far
        self._bound: tuple | None = None
        self.binds = 0
        logger.info(
            "Qwen3.5 XPU mono decode: %d decoder layers%s, %s; other steps run "
            "vLLM's layers",
            len(layers),
            " + MTP drafter" if self.draft else "",
            f"verify of {config.q} tokens a request" if config.q > 1 else "decode",
        )

    def _all(self) -> list[MonoLayer]:
        return [*self.kernels.values(), *self.draft.values()]

    def _key(self) -> tuple | None:
        caches = [c for k in self._all() for c in k.caches()]
        if any(not isinstance(c, torch.Tensor) or c.numel() == 0 for c in caches):
            return None
        return tuple((c.data_ptr(), tuple(c.shape), c.stride()) for c in caches)

    def bind_caches(self) -> None:
        """Checks and binds the caches vLLM just bound (the memory-profiling
        cache, then the real one); called by the model state's
        ``initialize_kv_cache``, never inside graph capture."""
        assert not torch.xpu.is_current_stream_capturing(), (
            "Qwen3.5 XPU mono decode: KV caches bound inside graph capture"
        )
        self._bound = None
        key = self._key()
        if key is None:
            return

        def bind():
            for k in self._all():
                k.bind(self.region.mailbox(k.name))
            if self.prologue is not None:
                self.prologue.bind(self.region.mailbox(self.prologue.name), self.rank)

        agree(bind_reasons(bind), "the KV caches")
        self._bound = key
        self.binds += 1
        logger.info(
            "Qwen3.5 XPU mono decode: caches bound (%d time%s)",
            self.binds,
            "" if self.binds == 1 else "s",
        )

    def _context(self) -> Any:
        """The forward context of a step the kernels may serve, else None:
        before a KV cache is bound (profiling, dummy runs) and in PIECEWISE
        steps (one capture replays for any batch of the same size) vLLM's layers
        run."""
        if self._bound is None or not is_forward_context_available():
            return None
        context = get_forward_context()
        if context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            return None
        if self._key() != self._bound:
            raise RuntimeError(
                "Qwen3.5 XPU mono decode: the KV cache was re-bound without "
                "bind_caches (ModelState.initialize_kv_cache)"
            )
        return context

    def plan(self, positions: torch.Tensor, intermediate_tensors) -> Step | None:
        """This step's plan for the target model, or None to run vLLM's layers:
        a pure decode step of 1 / 2 / 4 / 8 rows, or a pure verify step of
        whole requests with a verify kernel. Every predicate is the same on all
        TP ranks: the kernels' mailboxes need every rank on one path. Under
        FULL graphs this runs at capture only and the plan is baked in."""
        if intermediate_tensors is not None:
            return None
        context = self._context()
        if context is None:
            return None
        rows = positions.size(-1)
        q = self.config.q
        if q == 1 and rows not in DECODE_ROWS:
            return None
        if q > 1 and (rows, q) not in self.config.verify:
            return None
        md = context.attn_metadata
        kind = step_kind(md, self.gdn_prefix, self.attn_prefix, rows, q)
        if kind is None:
            return None
        return Step(rows, kind, md, self.kernels)

    def plan_draft(self, positions: torch.Tensor) -> Step | None:
        """The MTP drafter's plan: its first pass of a verify step (the
        target's rows, q tokens a request) or a later one-token pass of
        1 / 2 / 4 / 8 rows."""
        if not self.draft:
            return None
        context = self._context()
        if context is None:
            return None
        rows, q = positions.size(-1), self.config.q
        kind = draft_kind(context.attn_metadata, self.draft[0].prefix, rows, q)
        if kind is None or rows not in MTP_ROWS:
            return None
        if kind == 1 and rows not in DECODE_ROWS:
            return None
        if kind > 1 and (rows, kind) not in self.config.verify:
            return None
        return Step(rows, kind, context.attn_metadata, self.draft)

    def after_step(self) -> None:
        """Once per executed step on every TP rank: raises on every rank once a
        rank gave up an in-kernel wait."""
        self.failstop.after_step()
