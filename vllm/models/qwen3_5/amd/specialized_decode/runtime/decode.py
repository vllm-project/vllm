# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen3.8-2.4T specialized decode executor (step mode): one routing
decision per step, then the whole forward on the wide kernels.

Built by the model state once the weights are loaded (`Qwen3_5ModelState`):
checks the loaded model, builds the per-rank launch state (`Launcher`) and
compiles every kernel build of every width, before memory profiling and graph
capture. `bind_caches` checks the GDN states vLLM binds (the CUDA-graph
memory-profiling cache, then the real one). Each step `plan` returns a
`Step` or None, and `run` runs a planned step's forward. Weight reloads are
not supported.

A planned step runs one persistent launch per decoder layer (`wide.layer`),
a full-attention layer's after vLLM's attention:

    GDN layer   gdn   [embedding -> input norm] GDN, out_proj + all-reduce
                      + post-attention norm, MoE + all-reduce + the next
                      layer's input norm [-> the next layer's qkv_proj]
    attention   (vLLM) q / k norms, RoPE, attention -> output, gate
                attn  o_proj + all-reduce + post-attention norm, MoE as above

The embedding stays vLLM's; the last launch's closing norm is the final
norm.

A step of up to `L.SPLIT_MAX` rows runs each layer's MLP half on
`mono.blocks.layer_block` instead:

    GDN layer   gdn   [embedding or] residual add + input norm, GDN -> core
                layer_block   out_proj + all-reduce + post-attention norm,
                              MoE + all-reduce
    attention   attn  residual add + input norm, qkv_proj
                (vLLM) q / k norms, RoPE, attention -> output, gate
                layer_block   o_proj (gated) + ... as above

and the final norm is vLLM's."""

from typing import Any, NamedTuple

import torch

from vllm.distributed import get_tp_group
from vllm.forward_context import (
    get_forward_context,
    in_piecewise_cudagraph,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.models.qwen3_5.amd.specialized_decode import layout as L
from vllm.models.qwen3_5.amd.specialized_decode.abi import FAILED
from vllm.models.qwen3_5.amd.specialized_decode.runtime import NAME, DecodeConfig
from vllm.models.qwen3_5.amd.specialized_decode.runtime.guards import (
    FailStop,
    cache_reasons,
    model_reasons,
)
from vllm.models.qwen3_5.amd.specialized_decode.runtime.layers import (
    Launcher,
    attention_core,
    gemma_norm,
)
from vllm.models.qwen3_5.amd.specialized_decode.runtime.routing import step_reason
from vllm.models.qwen3_5.amd.specialized_decode.wide import layer as K

logger = init_logger(__name__)


class Step(NamedTuple):
    """A planned step: `width` rows, one decode token a request."""

    width: int
    attn_metadata: dict[str, Any]  # by layer name


def agree(reasons: list[str], group, what: str) -> None:
    """Raises on every TP rank if any rank has `reasons`: a rank that refused
    alone would leave the others spinning in-kernel."""
    from vllm.models.qwen3_5.amd.specialized_decode.runtime.peer import tp_agree

    if not tp_agree(not reasons, group):
        why = "; ".join(reasons) if reasons else "another TP rank refused"
        raise ValueError(
            f"{NAME} does not support {what}: {why} (without --monokernel, "
            "vLLM's decode runs)"
        )


def _compile_mono(launch, abi) -> None:
    """Compiles a `mono` launch without running it (FlyDSL's compile-only
    mode, which has no FLYDSL_ prefix, returns before the launch)."""
    import os

    old = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"
    try:
        launch(*abi.zeros(), stream=torch.cuda.current_stream())
    finally:
        if old is None:
            del os.environ["COMPILE_ONLY"]
        else:
            os.environ["COMPILE_ONLY"] = old


class SpecializedDecode:
    """The executor over one `Qwen3_5Model`: builds the launch state, binds
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
        self.keys = {s: self._keys(s) for s in config.widths}
        builds = sorted({k for ks in self.keys.values() for k in ks}, key=repr)
        for key in builds:
            K.compile_only(key)
        failed = [self.launch.health[FAILED : FAILED + 1]]
        if self.launch.split_widths:
            builds += self._compile_split(dev)
            failed.append(self.launch.mono_health[FAILED : FAILED + 1])
        self.failstop = FailStop(failed, tp.rank_in_group, tp.world_size, tp.cpu_group)
        self._kv: tuple[int, int] | None = None
        self.steps = 0
        # the epoch at the last LOG_EVERY step, copied without blocking
        self.counted = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self.counted_event: torch.cuda.Event | None = None
        self.counted_at = 0
        k0 = self.keys[config.widths[0]][0]
        logger.info(
            "%s: %d layers, widths %s, %d builds, %d launches a step, GDN "
            "projections %s / %s; other steps run vLLM's decode",
            NAME,
            len(model.layers),
            ",".join(map(str, config.widths)),
            len(builds),
            self.launch.launches,
            k0.qkvz,
            k0.out,
        )

    def _compile_split(self, dev) -> list:
        """Compiles the split widths' `layer_block` and `gdn_block` builds
        and warms up the residual add + norm vLLM's parts of their steps run,
        before graph capture."""
        from vllm.models.qwen3_5.amd.specialized_decode.mono import blocks

        la = self.launch
        keys = {
            la.mono_key(i, s)
            for s in la.split_widths
            for i in range(len(self.model.layers))
        }
        gdn_keys = {
            la.gdn_key(i, s, i == 0)
            for s in la.split_widths
            for i in range(len(self.model.layers))
            if la.mono_gdn_layer(i, s)
        }
        for key in sorted(keys, key=repr):
            _compile_mono(blocks.build_layer(key), blocks.LAYER_ABI)
        for key in sorted(gdn_keys, key=repr):
            _compile_mono(blocks.build_gdn(key), blocks.GDN_ABI)
        with torch.inference_mode():
            for s in (1, 2):
                x = torch.zeros(s, L.HIDDEN, dtype=torch.bfloat16, device=dev)
                gemma_norm(self.model.norm, x, x)
        return sorted(keys, key=repr) + sorted(gdn_keys, key=repr)

    def _keys(self, rows: int) -> list:
        """Every layer's build at `rows`, in step order."""
        return [self.launch.key(i, rows) for i in range(len(self.model.layers))]

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
        - widths without a build (`DecodeConfig.widths`);
        - steps that are not pure decode steps (`step_reason`).

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
    ) -> torch.Tensor:
        """`Qwen3_5Model.forward` of a planned step -> the final norm's
        hidden states [S, H]. `inputs_embeds`: the multimodal wrapper's
        embeddings of the step's tokens (decode rows are sampled text tokens,
        so the token embeddings), else None."""
        model, la, md, rows = self.model, self.launch, step.attn_metadata, step.width
        la.epoch.add_(1)
        if inputs_embeds is None:
            inputs_embeds = model.embed_input_ids(input_ids)
        if rows <= L.SPLIT_MAX:
            return self._run_split(step, inputs_embeds, positions)
        keys = self.keys[rows]
        seq = 0
        for i, layer in enumerate(model.layers):
            if layer.layer_type == "linear_attention":
                dyn = {
                    "emb": inputs_embeds if i == 0 else 0,
                    "st_idx": md[
                        layer.linear_attn.prefix
                    ].non_spec_state_indices_tensor,
                }
            else:
                qkv = la.view("qkv_out", rows, L.QKV)
                attn, gate = attention_core(layer.self_attn, qkv, positions)
                dyn = {"attn": attn, "gate": gate}
            seq = la.launch(keys[i], i, seq, dyn)
        out = la.out[:rows]
        out.copy_(la.view("x", rows, L.HIDDEN))
        return out

    def _run_split(self, step: Step, emb: torch.Tensor, positions) -> torch.Tensor:
        """A split width's forward (`L.SPLIT_MAX`): each layer's split
        launch (or `gdn_block`: `Launcher.mono_gdn_layer`), then its
        `layer_block`."""
        model, la, md, rows = self.model, self.launch, step.attn_metadata, step.width
        keys = self.keys[rows]
        h, res = emb, None
        seq = 0
        for i, layer in enumerate(model.layers):
            gate = None
            dyn = {"emb": h, "res_in": 0 if res is None else res}
            if la.mono_gdn_layer(i, rows):
                st_idx = md[layer.linear_attn.prefix].non_spec_state_indices_tensor
                core, residual = la.mono_gdn(i, h, res, st_idx)
                h, res = la.mono_layer(i, core, gate, residual)
                continue
            if layer.layer_type == "linear_attention":
                core = la.core[:rows]
                st_idx = md[layer.linear_attn.prefix].non_spec_state_indices_tensor
                dyn |= {"core_out": core, "st_idx": st_idx}
                seq = la.launch(keys[i], i, seq, dyn)
            else:
                dyn["w_qkv"] = layer.self_attn.qkv_proj.weight
                seq = la.launch(keys[i], i, seq, dyn)
                qkv = la.view("qkv_out", rows, L.QKV)
                core, gate = attention_core(layer.self_attn, qkv, positions)
            residual = la.view("res", rows, L.HIDDEN)
            h, res = la.mono_layer(i, core, gate, residual)
        out = la.out[:rows]
        out.copy_(gemma_norm(model.norm, h, res)[0])
        return out

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
