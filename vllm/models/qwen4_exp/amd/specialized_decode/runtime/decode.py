# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen3.8-Flash-Next specialized decode executor (step mode): one routing
decision per step, then the whole forward on the wide kernels.

Built by the model state once the weights are loaded (``Qwen4ExpModelState``):
checks the loaded model, builds the per-rank launch state (``Launcher``) and
compiles every kernel build of every width, before memory profiling and graph
capture. ``bind_caches`` checks the GDN states vLLM binds (the CUDA-graph
memory-profiling cache, then the real one). Each step ``plan`` returns a
``Step`` or None, and ``run`` runs a planned step's forward. Weight reloads are
not supported.

A planned step runs one persistent launch per GDN layer and two per QSA layer
around vLLM's attention (``wide.layer``):

    GDN layer   gdn        attn mix, GDN, out_proj + all-reduce + combine,
                           MLP mix, MoE + all-reduce + combine
    QSA layer   qsa_front  attn mix -> x, qkv_proj + index_qk_proj
                (vLLM)     QSA attention -> attention output, gate
                qsa_back   o_proj + all-reduce + combine, MLP half

The PLE layer, the final mixer and the embedding stay vLLM's. The kernels keep
the multi-stream state materialized (stock defers each combine to the next
mix, with the same bf16 rounding), so the PLE layer and the final mixer read a
plain state."""

from typing import Any, NamedTuple

import torch

from vllm.distributed import get_tp_group
from vllm.forward_context import (
    get_forward_context,
    in_piecewise_cudagraph,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.models.qwen4_exp.amd.specialized_decode.runtime import NAME, DecodeConfig
from vllm.models.qwen4_exp.amd.specialized_decode.runtime.guards import (
    FailStop,
    cache_reasons,
    model_reasons,
)
from vllm.models.qwen4_exp.amd.specialized_decode.runtime.layers import (
    Launcher,
    qsa_core,
)
from vllm.models.qwen4_exp.amd.specialized_decode.runtime.routing import step_reason
from vllm.models.qwen4_exp.amd.specialized_decode.wide import layer as K
from vllm.models.qwen4_exp.amd.specialized_decode.wide import layout as L
from vllm.models.qwen4_exp.amd.specialized_decode.wide.abi import FAILED

logger = init_logger(__name__)


class Step(NamedTuple):
    """A planned step: ``width`` rows, one decode token a request."""

    width: int
    attn_metadata: dict[str, Any]  # by layer name


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


class SpecializedDecode:
    """The executor over one ``Qwen4ExpModel``: builds the launch state, binds
    it to each KV cache and plans and runs every decode step."""

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
            model, config.widths, tp.world_size, tp.rank_in_group, tp.cpu_group, dev
        )
        builds = sorted({k for s in config.widths for k in self._keys(s)}, key=repr)
        for key in builds:
            K.compile_only(key)
        self.failstop = FailStop(
            self.launch.health[FAILED : FAILED + 1],
            tp.rank_in_group,
            tp.world_size,
            tp.cpu_group,
        )
        self._kv: tuple[int, int] | None = None
        self.steps = 0
        # the epoch at the last LOG_EVERY step, copied without blocking
        self.counted = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self.counted_event: torch.cuda.Event | None = None
        self.counted_at = 0
        logger.info(
            "%s: %d layers, widths %s, %d builds, %d launches a step, %s TP "
            "all-reduce lines; other steps run vLLM's decode",
            NAME,
            len(model.layers),
            ",".join(map(str, config.widths)),
            len(builds),
            self.launch.launches,
            self.launch.allreduce,
        )

    def _keys(self, rows: int) -> list:
        """Every layer's builds at ``rows``, in step order."""
        out, stats = [], True
        for layer in self.model.layers:
            stats |= layer.ple is not None
            out += self.launch.keys(layer, rows, stats)
            stats = False
        return out

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
        profiling cache, then the real one) and binds them."""
        self._kv = None
        if self._kv_key() is None:
            return
        tp = get_tp_group()
        agree(cache_reasons(self.model, tp.world_size), tp.cpu_group, "the KV caches")
        self.launch.bind_caches()
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
        - widths without a build (``DecodeConfig.widths``);
        - steps that are not pure decode steps (``step_reason``).

        Eager steps of a built width run the kernels too. Every predicate is the
        same on all TP ranks: the kernels' exchanges need every rank on one
        path. Under FULL graphs this runs at capture only and the plan is baked
        in; a replay with fewer requests than its graph pads rows with
        NULL_BLOCK_ID state slots, whose GDN state the kernels leave alone."""
        if intermediate_tensors is not None or self._kv is None or input_ids is None:
            return None
        if not is_forward_context_available() or in_piecewise_cudagraph():
            return None
        width = input_ids.size(0)
        if width not in self.config.widths:
            return None
        md = get_forward_context().attn_metadata
        if step_reason(md, self.gdn_prefix, width) is not None:
            return None
        assert isinstance(md, dict)
        if self._kv_key() != self._kv:
            raise RuntimeError(
                f"{NAME}: the KV cache was re-bound without bind_caches "
                "(ModelState.initialize_kv_cache)"
            )
        logger.info_once("%s: first specialized step (%d rows)", NAME, width)
        return Step(width, md)

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
        model, la, md, rows = self.model, self.launch, step.attn_metadata, step.width
        la.epoch.add_(1)
        if inputs_embeds is None:
            inputs_embeds = model.embed_input_ids(input_ids)
        state = inputs_embeds.repeat(1, L.HC)
        x = la.x[:rows]
        seq, stats = 0, True
        for i, layer in enumerate(model.layers):
            if layer.ple is not None:
                # every row on every rank, also when --enable-hc-sp built it
                state = layer.ple(
                    state, input_ids, query_start_loc, ngram_context, full_rows=True
                )
                stats = True
            keys = la.keys(layer, rows, stats)
            dyn = {"state": state, "x": x}
            if layer.layer_type == "linear_attention":
                dyn["st_idx"] = md[
                    layer.linear_attn.prefix
                ].non_spec_state_indices_tensor
                seq = la.launch(keys[0], i, seq, dyn)
            else:
                seq = la.launch(keys[0], i, seq, dyn)
                attn, gate = qsa_core(
                    layer.self_attn, x, la.qkv[:rows], la.ixqk[:rows], positions
                )
                seq = la.launch(keys[1], i, seq, dyn | {"attn": attn, "gate": gate})
            stats = False
        _, h, _ = model.hyper_connection_mixer.mix(state)
        return h

    LOG_EVERY = 1024

    def after_step(self) -> None:
        """Once per executed step, from compute_logits (eager on every rank,
        also under graph replay). Raises on every rank once a rank gave up an
        in-kernel wait; every LOG_EVERY steps logs how many of the steps up to
        the previous LOG_EVERY ran on the kernels (the epoch counts
        specialized steps, graph replays included; copied without a host
        sync, so the count trails by LOG_EVERY steps)."""
        self.failstop.after_step()
        if torch.cuda.is_current_stream_capturing():
            return
        self.steps += 1
        if self.steps % self.LOG_EVERY:
            return
        if self.counted_event is not None and self.counted_event.query():
            logger.info(
                "%s: %d specialized steps of %d",
                NAME,
                int(self.counted[0]),
                self.counted_at,
            )
        self.counted.copy_(self.launch.epoch[:1], non_blocking=True)
        self.counted_event = torch.cuda.Event()
        self.counted_event.record()
        self.counted_at = self.steps
