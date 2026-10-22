# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen3.8-Flash-Next specialized decode executor (step mode): one routing
decision per step, then the whole forward on the kernels.

Built by the model state once the weights are loaded (``Qwen4ExpModelState``):
checks the loaded model, builds the per-rank launch state (``Launcher``) and
compiles every kernel build of every width, before memory profiling and graph
capture. ``bind_caches`` checks the GDN states vLLM binds (the CUDA-graph
memory-profiling cache, then the real one). Each step ``plan`` returns a
``Step`` or None, and ``run`` runs a planned step's forward. Weight reloads are
not supported.

A planned step runs vLLM's decoder layer loop with three of each layer's
modules each one persistent launch (``mono.blocks``), and each hyper-
connection's mix as one more (``mono.front``):

    attention HC mix  -> front.hc_front
    linear_attn       -> blocks.gdn_block (up to out_proj's input)
    self_attn (QSA)   -> vLLM's, up to o_proj's input (``qsa_core``)
    out_proj / o_proj -> blocks.oproj_block (+ its all-reduce + the attention
                         HC's combine)
    MLP HC mix        -> front.hc_front
    mlp               -> blocks.moe_block (+ its all-reduce + the MLP HC's
                         combine)

The PLE layers, the final mixer and the embedding stay vLLM's. Stock defers
each HC combine to the next mix; the blocks materialize it with the same bf16
rounding, so the PLE layers and the final mixer read a plain state.

With MTP every planned step is a verify step: a request's rows are its
consecutive tokens, so a GDN layer runs vLLM's in_proj, GDN core (on the
spec-decode metadata) and gated norm in place of the GDN block
(``gdn_verify_core``); the rest is as above. The drafter's input, the state
before the final mixer, goes to the buffer the stock forward fills."""

import os
from collections.abc import Callable
from typing import Any, NamedTuple

import torch

from vllm.distributed import get_tp_group
from vllm.forward_context import (
    get_forward_context,
    in_piecewise_cudagraph,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.models.qwen4_exp.amd.specialized_decode.mono import layout as L
from vllm.models.qwen4_exp.amd.specialized_decode.mono.abi import FAILED
from vllm.models.qwen4_exp.amd.specialized_decode.runtime import NAME, DecodeConfig
from vllm.models.qwen4_exp.amd.specialized_decode.runtime.guards import (
    FailStop,
    cache_reasons,
    model_reasons,
)
from vllm.models.qwen4_exp.amd.specialized_decode.runtime.layers import (
    Launcher,
    gdn_verify_core,
    qsa_core,
)
from vllm.models.qwen4_exp.amd.specialized_decode.runtime.routing import step_reason

logger = init_logger(__name__)


class Step(NamedTuple):
    """A planned step: ``width`` rows, one decode token a request, or (MTP,
    ``verify``) whole requests of num_speculative_tokens + 1 rows."""

    width: int
    attn_metadata: dict[str, Any]  # by layer name
    verify: bool


def agree(reasons: list[str], group, what: str) -> None:
    """Raises on every TP rank if any rank has ``reasons``: a rank that refused
    alone would leave the others spinning in-kernel."""
    from vllm.models.qwen4_exp.amd.specialized_decode.runtime.peer import tp_agree

    if not tp_agree(not reasons, group):
        why = "; ".join(reasons) if reasons else "another TP rank refused"
        raise ValueError(
            f"{NAME} does not support {what}: {why} (without --monokernel, "
            "vLLM's decode runs)"
        )


def compile_build(build, key) -> None:
    """Compiles one launch's kernel without running it: FlyDSL's compile-only
    mode returns before the launch. ``build``: the module's ``build``."""
    launch = build(key)
    # FlyDSL's compile-only switch has no FLYDSL_ prefix
    old = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"
    try:
        launch(*_zeros(key), stream=torch.cuda.current_stream())
    finally:
        if old is None:
            del os.environ["COMPILE_ONLY"]
        else:
            os.environ["COMPILE_ONLY"] = old


def _zeros(key) -> list[int]:
    from vllm.models.qwen4_exp.amd.specialized_decode.mono import blocks, front

    abi = {
        front.FrontBuild: front.ABI,
        blocks.GdnBuild: blocks.GDN_ABI,
        blocks.OprojBuild: blocks.OPROJ_ABI,
        blocks.MoeBuild: blocks.MOE_ABI,
    }[type(key)]
    return [0] * len(abi.names)


class SpecializedDecode:
    """The executor over one ``Qwen4ExpModel``: builds the launch state,
    binds it to each KV cache and plans and runs every decode step."""

    def __init__(self, model, config: DecodeConfig):
        tp = get_tp_group()
        agree(model_reasons(model, tp.world_size), tp.cpu_group, "the loaded model")
        self.model, self.config = model, config
        self.gdn_prefix = next(
            ly.linear_attn.prefix
            for ly in model.layers
            if ly.layer_type == "linear_attention"
        )
        dev = model.embed_tokens.weight.device
        self.launch = Launcher(
            config.widths, tp.world_size, tp.rank_in_group, tp.cpu_group, dev
        )
        builds = self._builds()
        for build, key in builds:
            compile_build(build, key)
        self.failstop = FailStop(
            self.launch.health[FAILED : FAILED + 1],
            tp.rank_in_group,
            tp.world_size,
            tp.cpu_group,
        )
        self._kv: tuple[int, int] | None = None
        logger.info(
            "%s: %d layers, widths %d-%d step %d, %d builds; other steps run "
            "vLLM's decode",
            NAME,
            len(model.layers),
            config.widths[0],
            config.widths[-1],
            config.q,
            len(builds),
        )

    def _builds(self) -> list:
        from vllm.models.qwen4_exp.amd.specialized_decode.mono import blocks, front

        la = self.launch
        keys: set[tuple[Callable[[Any], Any], Any]] = set()
        for s in self.config.widths:
            keys.add((blocks.build_moe, blocks.MoeBuild(tokens=s, tp=la.tp)))
            for layer in self.model.layers:
                for hc in (layer.attn_hyper_connection, layer.mlp_hyper_connection):
                    for m in (False, True):
                        keys.add((front.build, la.front_key(hc, s, m)))
                if layer.layer_type == "linear_attention" and self.config.q == 1:
                    keys.add((blocks.build_gdn, la.gdn_key(layer, s)))
                keys.add((blocks.build_oproj, la.oproj_key(layer, s)))
        return sorted(keys, key=lambda bk: repr(bk[1]))

    def _kv_key(self) -> tuple[int, int] | None:
        layer = next(
            ly for ly in self.model.layers if ly.layer_type == "linear_attention"
        )
        state = layer.linear_attn.kv_cache
        if not isinstance(state, (list, tuple)) or len(state) != 2:
            return None
        if state[1].numel() == 0:
            return None
        return state[1].data_ptr(), state[1].numel()

    def bind_caches(self) -> None:
        """Checks the GDN states vLLM just bound (the CUDA-graph memory
        profiling cache, then the real one)."""
        self._kv = None
        if self._kv_key() is None:
            return
        tp = get_tp_group()
        agree(
            cache_reasons(self.model, tp.world_size, self.config.q),
            tp.cpu_group,
            "the KV caches",
        )
        self._kv = self._kv_key()

    def plan(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors,
    ) -> Step | None:
        """This step's plan, or None to run vLLM's forward.

        Steps that run vLLM's forward:
        - pipeline-parallel stage inputs, and steps before a KV cache is bound
          (profiling, dummy runs);
        - PIECEWISE steps: they run (or capture) the compiled stock model's
          graphs, and taking one would leave them uncaptured for the prefills
          of that size;
        - more than ``max_tokens`` rows;
        - steps that are not pure decode steps, with MTP not pure verify steps
          of whole uniform requests (``step_reason``).

        Eager steps run the kernels too. Every predicate is the same on all TP
        ranks: the kernels' mailboxes need every rank on one path. Under FULL
        graphs this runs at capture only and the plan is baked in; a replay
        with fewer requests than its graph pads rows with NULL_BLOCK_ID state
        indices, which the GDN block skips (with MTP, whole requests with the
        null state slot and one accepted token, as vLLM's GDN core serves
        them)."""
        if intermediate_tensors is not None or self._kv is None or input_ids is None:
            return None
        if not is_forward_context_available() or in_piecewise_cudagraph():
            return None
        width = input_ids.size(0)
        if not 1 <= width <= self.config.max_tokens:
            return None
        md = get_forward_context().attn_metadata
        q = self.config.q
        if step_reason(md, self.gdn_prefix, width, q) is not None:
            return None
        assert isinstance(md, dict)
        if self._kv_key() != self._kv:
            raise RuntimeError(
                f"{NAME}: the KV cache was re-bound without bind_caches "
                "(ModelState.initialize_kv_cache)"
            )
        logger.info_once("%s: first specialized step (%d rows)", NAME, width)
        return Step(width, md, q > 1)

    def run(
        self,
        step: Step,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        ngram_context: torch.Tensor | None,
    ) -> torch.Tensor:
        """``Qwen4ExpModel.forward`` of a planned step -> the final mixer's
        hidden states [S, H]. ``inputs_embeds``: the multimodal wrapper's
        embeddings of the step's tokens (decode rows are sampled text tokens,
        so the token embeddings), else None."""
        model, la, md = self.model, self.launch, step.attn_metadata
        la.epoch.add_(1)
        if inputs_embeds is None:
            inputs_embeds = model.embed_input_ids(input_ids)
        state = inputs_embeds.repeat(1, L.HC)
        msq: torch.Tensor | None = None
        for layer in model.layers:
            if layer.ple is not None:
                state = layer.ple(state, input_ids, query_start_loc, ngram_context)
                msq = None
            x, inj = la.hc_front(
                layer, layer.attn_hyper_connection, state, msq, la.attn_scratch
            )
            gate = None
            if layer.layer_type == "linear_attention" and step.verify:
                core = gdn_verify_core(layer.linear_attn, x)
            elif layer.layer_type == "linear_attention":
                core = la.gdn(layer, md, x)
            else:
                core, gate = qsa_core(layer.self_attn, x, positions)
            state, msq = la.oproj(layer, core, gate, state, inj)
            x, inj = la.hc_front(
                layer, layer.mlp_hyper_connection, state, msq, la.mlp_scratch
            )
            state, msq = la.moe(layer, x, state, inj)
        if model._mtp_hidden_buffer is not None:
            model._mtp_hidden_buffer[: state.size(0)].copy_(state)
        _, h, _ = model.hyper_connection_mixer.mix(state)
        return h

    def after_step(self) -> None:
        """Once per executed step, from compute_logits (eager on every rank,
        also under graph replay). Raises on every rank once a rank gave up an
        in-kernel wait."""
        self.failstop.after_step()
