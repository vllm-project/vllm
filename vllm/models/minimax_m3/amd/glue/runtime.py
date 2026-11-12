# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-model runtime of the MiniMax-M3.1 SP (sequence-parallel) prefill MoE
half.

Lifecycle (every phase that can fail on one rank ends in a TP agreement, so all
ranks keep or drop the path together):

* `create` (model __init__): config checks only; registers on the MoE layers.
* `after_load_weights` (end of MiniMaxM3Model.load_weights): marks the weights
  loaded. The setup (the replicated shared-expert weights, all-gathered from
  the TP shards, and the fused combine + reduce-scatter buffers) runs at the
  first eager, uncaptured forward, the profile run, every rank at the same
  point.
* `begin_forward` / `end_forward` (MiniMaxM3Model.forward): one plan a forward,
  from rank-uniform data only (forward context, row count, capture state); kept
  thread-locally for the decoder layers.
* `moe_half` (MiniMaxM3DecoderLayer.forward, after attention): the layer's MoE
  half on the SP path, or None for the stock path."""

from __future__ import annotations

import threading
from typing import Any, NamedTuple

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

SP = "sp"
# Eager steps of MIN_ROWS .. MAX_ROWS tokens take the SP path. Measured on
# MI355X (TP4, the MoE half from the o_proj partial to the gemm1 inputs):
# 1208 rows ~-2%, 2343 -12%, 5857 -24%, 14394 -26%; at 500 rows it is slower
# (the reduce-scatter saves little at that size). Steps of more tokens than the
# measured range keep the stock path (with DBO they are split anyway: DBO
# micro-batches always run the stock path).
MIN_ROWS = 1024
MAX_ROWS = 16384


class StepPlan(NamedTuple):
    mode: str
    rows: int


def _tp_agree(ok: bool, group) -> bool:
    from vllm.models.deepseek_v41.amd.mono.common.peer_memory import tp_agree

    return tp_agree(ok, group.cpu_group)


class M3Glue:
    @classmethod
    def create(cls, model, vllm_config) -> M3Glue | None:
        reasons = cls._refusals(model, vllm_config)
        if reasons:
            logger.debug("MiniMax-M3 SP prefill off: %s", "; ".join(reasons))
            return None
        return cls(model, vllm_config)

    @staticmethod
    def _refusals(model, vllm_config) -> list[str]:
        from vllm.platforms import current_platform

        pc, cfg = vllm_config.parallel_config, model.config
        reasons = []
        if not current_platform.is_rocm():
            return ["requires ROCm"]
        from vllm.platforms.rocm import on_gfx950

        if not on_gfx950():
            reasons.append("requires gfx950")
        if pc.tensor_parallel_size != 4:
            reasons.append("requires TP 4")
        if pc.pipeline_parallel_size != 1 or pc.data_parallel_size != 1:
            reasons.append("requires PP 1 and DP 1")
        if pc.enable_expert_parallel or pc.enable_eplb:
            reasons.append("expert parallelism / EPLB")
        if getattr(vllm_config, "lora_config", None) is not None:
            reasons.append("LoRA")
        aux = getattr(vllm_config, "aux_output_config", None)
        if getattr(aux, "enable_return_routed_experts", False):
            reasons.append("routed-expert capture")
        expect = {
            "hidden_size": 6144,
            "num_local_experts": 128,
            "num_experts_per_tok": 4,
            "routed_scaling_factor": 2.0,
            "rms_norm_eps": 1e-6,
            "scoring_func": "sigmoid",
        }
        for k, v in expect.items():
            if getattr(cfg, k, None) != v:
                reasons.append(f"{k}={getattr(cfg, k, None)} (built for {v})")
        if not getattr(cfg, "use_routing_bias", False):
            reasons.append("no routing bias")
        layers = [
            layer for layer in model.layers if getattr(layer, "is_moe_layer", False)
        ]
        if not layers or any(
            layer.block_sparse_moe.shared_experts is None for layer in layers
        ):
            reasons.append("no MoE layers or fused shared experts")
        return reasons

    def __init__(self, model, vllm_config) -> None:
        from vllm.distributed import get_tp_group

        self.group = get_tp_group()
        self.model = model
        self.layers = [
            layer for layer in model.layers if getattr(layer, "is_moe_layer", False)
        ]
        for layer in self.layers:
            layer._m3_glue = self
        self.sp: Any = None  # SpPrefill, once set up
        self.shared: dict[int, object] = {}  # layer id -> SharedWeights
        self._tls = threading.local()
        self._moe: dict[int, object] = {}  # layer id -> AiterMoe (lazy, host only)
        # padded M -> True, or None when refused
        self._sp_ok: dict[int, bool | None] = {}
        self._seen: set = set()
        self._loaded = False  # load_weights finished at least once
        self._ready = False  # the setup ran (first eager forward)

    # ------------------------------------------------------------ setup
    def after_load_weights(self) -> None:
        """End of a load_weights call. The loader can call load_weights several
        times and post-processes the parameters afterwards, so nothing is
        derived here: the setup runs at the first eager, uncaptured forward."""
        if self._ready:
            logger.warning(
                "MiniMax-M3 SP prefill: weights reloaded after setup; derived "
                "weights not rebuilt"
            )
        self._loaded = True

    def _maybe_setup(self) -> None:
        """Collective, every TP rank: the first forward is the same on all of
        them (the profile run)."""
        if self._ready or not self._loaded or torch.cuda.is_current_stream_capturing():
            return
        self._ready = True
        self._setup()

    def _setup(self) -> None:
        from .shared import build_shared_weights
        from .sp import SpPrefill

        ok, err = True, None
        try:
            for layer in self.layers:
                mlp, lid = layer.block_sparse_moe.shared_experts, layer.layer_id
                for lin in (mlp.gate_up_proj, mlp.down_proj):
                    if lin.weight.element_size() != 1 or not hasattr(
                        lin, "weight_scale"
                    ):
                        raise ValueError(f"layer {lid}: shared expert is not MXFP8")
        except Exception as e:  # noqa: BLE001
            ok, err = False, e
        if not _tp_agree(ok, self.group):
            logger.warning(
                "MiniMax-M3 SP prefill off: %s", err or "refused on another rank"
            )
            return
        for layer in self.layers:  # all-gathers in layer order on every rank
            self.shared[layer.layer_id] = build_shared_weights(
                layer.block_sparse_moe.shared_experts, False, True, self.group
            )
        self.sp = SpPrefill(self.group, MIN_ROWS, MAX_ROWS)
        torch.accelerator.synchronize()

    # ------------------------------------------------------------ per forward
    def begin_forward(
        self, positions: torch.Tensor, intermediate_tensors
    ) -> StepPlan | None:
        setup_now = not self._ready
        self._maybe_setup()
        if setup_now and self._ready:
            # The setup forward is vLLM's profile run (its largest step): leave
            # it to the stock path, so the MoE workspace is sized (and later
            # locked) for the largest step the stock path can still see, e.g.
            # eager steps below SP's minimum. The glue's own buffers were just
            # allocated, so the profile counts them too.
            self._tls.plan = None
            return None
        plan = self._decide(positions.shape[-1], intermediate_tensors)
        self._tls.plan = plan
        return plan

    def end_forward(self, plan: StepPlan | None) -> None:
        self._tls.plan = None
        if plan is not None:
            self.sp.assert_healthy()

    def _decide(self, rows: int, intermediate_tensors) -> StepPlan | None:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
        from vllm.v1.worker.ubatching import dbo_enabled

        if (
            self.sp is None
            or not self.sp.min_rows <= rows <= self.sp.max_rows
            or intermediate_tensors is not None
            or not is_forward_context_available()
            or dbo_enabled()
            or torch.cuda.is_current_stream_capturing()
        ):
            return None
        ctx = get_forward_context()
        if (
            ctx.ubatch_slices is not None
            or ctx.cudagraph_runtime_mode != CUDAGraphMode.NONE
        ):
            return None
        if self._sp_row(rows) is None:
            return None
        return StepPlan(SP, rows)

    # ------------------------------------------------------------ tuned rows
    def aiter_moe(self, layer):
        moe = self._moe.get(layer.layer_id)
        if moe is None:
            from .aiter_moe import AiterMoe

            mod = layer.block_sparse_moe
            moe = getattr(mod, "_m3_aiter_moe", None) or AiterMoe.from_moe_module(mod)
            self._moe[layer.layer_id] = moe
        return moe

    def _sp_row(self, rows: int):
        from aiter.fused_moe import get_padded_M

        key = get_padded_M(rows)
        if key not in self._sp_ok:
            try:
                moe = [self.aiter_moe(layer) for layer in self.layers][0]
                why = moe.sp_refusal(rows)
                info = why or moe.describe(rows)
            except Exception as e:  # noqa: BLE001 - a refusal, not a crash
                why = info = f"{type(e).__name__}: {e}"
            self._sp_ok[key] = None if why else True
            if self.group.rank_in_group == 0 and key not in self._seen:
                self._seen.add(key)
                logger.info(
                    "MiniMax-M3 SP prefill rows %d (tuned row): %s",
                    key,
                    ("refused: " if why else "") + info,
                )
        return self._sp_ok[key]

    # ------------------------------------------------------------ per layer
    def current(self) -> StepPlan | None:
        return getattr(self._tls, "plan", None)

    def moe_half(self, layer, hidden_states: torch.Tensor, residual: torch.Tensor):
        """The layer's MoE half on this forward's path: (hidden, residual), or
        None for the stock path."""
        plan = self.current()
        if plan is None or plan.rows != hidden_states.shape[0]:
            return None
        return self.sp.run(
            layer,
            hidden_states,
            residual,
            self.shared[layer.layer_id],
            self.aiter_moe(layer),
            self._next_norm(layer),
        )

    def _next_norm(self, layer):
        """The next layer's input norm when the all-reduce after this MoE can be
        an SP reduce-scatter + MXFP8 gather: the next layer is a MoE layer taking
        MXFP8 attention input, and no drafter hidden state is taken between them
        (those need full rows); else None (all-reduce)."""
        nid = layer.layer_id + 1
        layers = self.model.layers
        if nid >= len(layers) or nid in getattr(
            self.model, "aux_hidden_state_layers", ()
        ):
            return None
        nxt = layers[nid]
        if not getattr(nxt, "is_moe_layer", False) or not getattr(
            nxt, "fuse_input_norm_mxfp8", False
        ):
            return None
        return nxt.input_layernorm
