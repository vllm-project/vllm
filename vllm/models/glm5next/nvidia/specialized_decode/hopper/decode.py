# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode executor on Hopper (SM90).

`HopperSpecializedDecode` is `SpecializedDecode` (its plan, layer loop and mHC
boundaries) over the Hopper stages, and `HopperSpecializedDraft` is
`SpecializedDraft` over the same stages on the MTP draft layer:

- FULL-graph captures of pure decode and uniform verify steps of up to
  `MAX_TOKENS` tokens (32: plain decode of 1-32 requests, MTP5 verify steps of
  1-5 requests) run the executor; every other forward runs the stock layers.
- KDA layers run `kda.HopperKdaLayer`, DSA layers `dsa.HopperDsaLayer`, MoE
  layers `moe.HopperMoe` and dense layers `dense.HopperDenseMlp`. Each returns
  its sublayer output TP-partial.
- Every backbone boundary runs `boundary.HopperBoundary` (all-reduce, mHC post,
  mHC pre and RMSNorm), the draft's boundaries `boundary.HopperDraftAllReduceNorm`.
- The LM head shard GEMM and the logits all-gather run `logits.HopperLogitsHead`
  and `logits.HopperLogitsGather` (eager, after the graph replay).

`policy.HOPPER` switches each stage per token count; "" runs it stock and "auto"
takes the stage module's own choice. A stage also runs stock where its module
is not part of the build, where it declines its layer (weights or caches its
kernels do not cover) and at token counts it does not support. As on GB300,
stock sublayers leave their output TP-partial when the Hopper boundary reduces
it, and Hopper sublayers are all-reduced when the boundary runs stock.
"""

from collections import Counter
from typing import Any

import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger

from ..policy import HOPPER, Policy
from ..runtime.decode import LayerStages, SpecializedDecode, Step, _override
from ..runtime.draft import SpecializedDraft
from . import steps
from .steps import Stage

logger = init_logger(__name__)

# Token counts (requests x tokens per request) with a planned Hopper step: the
# row cap of the stage kernels.
MAX_TOKENS = 32
# The policy choice that takes the stage module's default for the token count.
AUTO = "auto"
# The sparse MLA backend whose metadata and BF16 KV cache the Hopper DSA layer
# reads (the SM90 default for GLM-5.3-Flash).
DSA_BACKEND = "FLASHINFER_MLA_SPARSE_SM90"


def mla_backend(attention) -> str | None:
    """The attention backend of a DSA layer's MLA attention."""
    backend = getattr(attention.mla_attn.mla_attn, "attn_backend", None)
    return backend.get_name() if backend is not None else None


class HopperLayerStages(LayerStages):
    """`LayerStages.attention` and `mlp` on the Hopper stages, shared by the
    backbone and the MTP draft layer."""

    # By layer index: the attention stages (one per tokens-per-request count it
    # was built for) and the MLP stage.
    attn_stages: dict[int, list[Stage]]
    ffn_stages: dict[int, Stage]
    # (stage, ran on Hopper) -> calls in the step being captured.
    calls: Counter
    # The policy entry of each stage, where it is not the stage's name.
    entries: dict[str, str] = {}

    def tokens(self, stage: str) -> list[int]:
        """The planned token counts the policy gives `stage`."""
        entry = self.entries.get(stage, stage)
        return [b for b in self.batches if self.policy(entry, b)]

    def _attention_stages(self, layers, qs: tuple[int, ...]) -> dict[int, list[Stage]]:
        built: dict[int, list[Stage]] = {}
        for layer in layers:
            attn = layer.self_attn
            if layer.layer_kind == "kda":
                stage, module, name = "kda", "kda", "HopperKdaLayer"
            else:
                stage, module, name = "dsa", "dsa", "HopperDsaLayer"
                backend = mla_backend(attn)
                if backend != DSA_BACKEND:
                    steps.declined(stage, f"{backend} attention (needs {DSA_BACKEND})")
                    continue
            tokens = self.tokens(stage)
            found = []
            for q in qs:
                planned = [b for b in tokens if b % q == 0]
                built_q = steps.per_attention(stage, module, name, attn, planned, q)
                if built_q is not None:
                    found.append(built_q)
            if found:
                built[layer.layer_idx] = found
        return built

    def _ffn_stages(self, layers) -> dict[int, Stage]:
        built: dict[int, Stage] = {}
        for layer in layers:
            if layer._mlp_is_moe:
                stage = steps.per_layer(
                    "moe",
                    "moe",
                    "HopperMoe",
                    self.tokens("moe"),
                    layer.mlp,
                    self.text_config,
                )
            else:
                stage = steps.per_layer(
                    "dense_mlp",
                    "dense",
                    "HopperDenseMlp",
                    self.tokens("dense_mlp"),
                    layer.mlp,
                )
            if stage is not None:
                built[layer.layer_idx] = stage
        return built

    def _attention_stage(self, layer, tokens: int, q: int) -> Stage | None:
        for stage in self.attn_stages.get(layer.layer_idx, ()):
            if stage.serves(tokens) and stage.kernel.supports(tokens, q):
                return stage
        return None

    def attention(self, layer, x, step: Step):
        """The attention output and whether it is still TP-partial."""
        tokens = x.shape[0]
        partial = self.partial(tokens)
        stage = self._attention_stage(layer, tokens, step.q)
        kind = "kda" if layer.layer_kind == "kda" else "dsa"
        self.calls[kind, stage is not None] += 1
        if stage is not None:
            out = self._run_attention(stage, layer, x, step, not partial)
            return out, partial
        o_proj = layer.self_attn.o_proj
        with _override(o_proj, "reduce_results", False, partial):
            out = layer.self_attn(hidden_states=x, positions=step.positions)
        return out, partial

    def _run_attention(self, stage: Stage, layer, x, step: Step, reduce: bool):
        if stage.name == "kda":
            md = step.metadata[layer.self_attn.prefix]
            return stage.kernel.run(x, md, step.q, reduce=reduce)
        # The DSA layer reads its MLA, indexer and kpool tail metadata.
        out = stage.kernel.run(x, step.positions, step.metadata, step.q)
        return tensor_model_parallel_all_reduce(out) if reduce else out

    def mlp(self, layer, x, moe_policy_tokens: int | None = None):
        """The FFN output and whether it is still TP-partial. The Hopper stages
        pick by the rows they get, so the draft's pruned MTP rows need no
        `moe_policy_tokens` (the GB300 policy key)."""
        del moe_policy_tokens
        tokens = x.shape[0]
        partial = self.partial(tokens)
        stage = self.ffn_stages.get(layer.layer_idx)
        hopper = stage is not None and stage.serves(tokens)
        self.calls["moe" if layer._mlp_is_moe else "dense_mlp", hopper] += 1
        if hopper:
            assert stage is not None
            out = stage.kernel(x)
            return (out if partial else tensor_model_parallel_all_reduce(out)), partial
        if layer._mlp_is_moe:
            config = layer.mlp.experts.moe_config
            unreduced = _override(config, "skip_final_all_reduce", True, partial)
        else:
            unreduced = _override(layer.mlp.down_proj, "reduce_results", False, partial)
        with unreduced:
            out = layer.mlp(x)
        return out, partial

    def _prepare_caches(self) -> None:
        """Compiles the attention stages for the bound KV caches (the KDA
        layer takes its state caches; the DSA layer finds its own)."""
        layers = {layer.layer_idx: layer for layer in self.layers}
        for index, stages in self.attn_stages.items():
            for stage in stages:
                prepare = getattr(stage.kernel, "prepare_caches", None)
                if prepare is not None:
                    prepare(getattr(layers[index].self_attn, "kv_cache", None))

    def _serves(self, tokens: int, q: int) -> bool:
        """Whether a step of `tokens` tokens at `q` per request reaches any
        Hopper stage (otherwise the stock forward runs it)."""
        return (
            self.partial(tokens)
            or any(
                self._attention_stage(layer, tokens, q) is not None
                for layer in self.layers
            )
            or any(
                stage.serves(tokens) or (q > 1 and stage.serves(tokens // q))
                for stage in self.ffn_stages.values()
            )
        )

    def summary(self) -> str:
        """Hopper / all calls per stage in the step just captured."""
        names = sorted({name for name, _ in self.calls})
        return ", ".join(
            f"{name} {self.calls[name, True]}/"
            f"{self.calls[name, True] + self.calls[name, False]}"
            for name in names
        )

    def describe(self) -> str:
        """The layers each stage was built for."""
        kinds = Counter(
            "kda" if layer.layer_kind == "kda" else "dsa" for layer in self.layers
        )
        mlps = Counter(
            "moe" if layer._mlp_is_moe else "dense_mlp" for layer in self.layers
        )
        attn = Counter(stages[0].name for stages in self.attn_stages.values())
        ffn = Counter(stage.name for stage in self.ffn_stages.values())
        parts = [f"{k} {attn[k]}/{kinds[k]}" for k in sorted(kinds)]
        parts += [f"{k} {ffn[k]}/{mlps[k]}" for k in sorted(mlps)]
        return ", ".join(parts) + " layers"


class HopperSpecializedDecode(HopperLayerStages, SpecializedDecode):
    """`SpecializedDecode` on Hopper. Built once the weights are loaded; reads
    the backbone, never modifies it."""

    def __init__(self, backbone, config):
        self.backbone = backbone
        self.config = config
        self.policy = Policy(config.policy, HOPPER)
        self.q = config.q
        self.layers = list(backbone.layers)
        self.text_config = backbone.config
        self.batches = sorted(
            b for b in config.capture_sizes if b <= MAX_TOKENS and b % self.q == 0
        )
        self.calls = Counter()
        self.attn_stages = self._attention_stages(self.layers, (self.q,))
        self.ffn_stages = self._ffn_stages(self.layers)
        # Collective: every TP rank builds the boundary in lockstep.
        self.boundary_modes: dict[int, Any] = {}
        self.ar_mhc: Any = self._build_boundary()
        if self.ar_mhc is not None:
            # hc_fn is BF16 in the checkpoint, so the BF16 copy is exact.
            self.hc_fn = {
                id(fn): fn.detach().bfloat16()
                for layer in self.layers
                for fn in (layer.hc_attn_fn, layer.hc_ffn_fn)
            }
        self.active = bool(
            self.attn_stages or self.ffn_stages or self.ar_mhc is not None
        )
        self.logits_gather: Any = None
        self.logits_tokens: set[int] = set()
        self.head: Any = None
        self.head_tokens: set[int] = set()
        logger.info(
            "Hopper specialized decode at %s tokens (q=%d): %s; boundary at %s",
            self.batches,
            self.q,
            self.describe(),
            sorted(self.boundary_modes) or "none",
        )

    def _build_boundary(self):
        cls = steps.stage_class("boundary", "HopperBoundary")
        tokens = self.tokens("boundary")
        if cls is None or not tokens:
            return None
        modes = boundary_modes(self.policy, tokens)
        if not modes:
            return None
        first = self.layers[0]
        norms = {
            n.variance_epsilon
            for layer in self.layers
            for n in (layer.input_layernorm, layer.post_attention_layernorm)
        }
        if len(norms) != 1:
            logger.warning("Hopper specialized decode: mixed norm eps, stock boundary")
            return None
        self.boundary_modes = modes
        return cls(
            hidden_size=first.hidden_size,
            max_num_tokens=max(modes),
            rms_eps=first.rms_norm_eps,
            hc_eps=first.hc_eps,
            post_mult=first.mhc_post_mult_value,
            sinkhorn_repeat=first.mhc_sinkhorn_iterations,
            norm_eps=norms.pop(),
            device=first.hc_attn_fn.device,
            modes=modes,
        )

    def partial(self, tokens: int) -> bool:
        return self.ar_mhc is not None and self.ar_mhc.supports(tokens)

    def boundary(self, layer, x, residual, post, comb, fn, scale, base, norm):
        self.calls["boundary", False] += 1
        out = super().boundary(layer, x, residual, post, comb, fn, scale, base, norm)
        return out

    def reduced_boundary(self, layer, x, residual, post, comb, fn, scale, base, norm):
        self.calls["boundary", True] += 1
        out = super().reduced_boundary(
            layer, x, residual, post, comb, fn, scale, base, norm
        )
        return out

    def prepare_kv(self) -> None:
        """Compiles the layer stages for the KV cache tensors the runner bound."""
        self._prepare_caches()

    def plan(self, positions, intermediate_tensors) -> Step | None:
        """`SpecializedDecode.plan` up to `MAX_TOKENS` tokens, for steps that
        reach at least one Hopper stage."""
        if not self.active:
            return None
        step = super().plan(positions, intermediate_tensors)
        if step is None:
            return None
        tokens = positions.shape[-1]
        if not self._serves(tokens, step.q):
            logger.info_once(
                "Hopper specialized decode: no Hopper stage serves a %d-token "
                "step (q=%d), it runs stock",
                tokens,
                step.q,
            )
            return None
        return step

    def run(self, step: Step, input_ids, inputs_embeds) -> torch.Tensor:
        self.calls.clear()
        out = super().run(step, input_ids, inputs_embeds)
        # One line per captured size: which stages the graph holds.
        logger.info(
            "Hopper specialized decode captured a %d-token step (q=%d): %s",
            step.positions.shape[-1],
            step.q,
            self.summary(),
        )
        return out

    def build_logits(self, logits_processor, lm_head) -> None:
        """The LM head GEMM and logits all-gather at the row counts the policy
        gives them."""
        weight = lm_head.weight
        if (
            weight.dtype != torch.bfloat16
            or logits_processor.logits_as_input
            or logits_processor.head_dtype not in (None, torch.bfloat16)
        ):
            return
        tokens = {m for m in self.policy.tables["logits_gather"] if m <= MAX_TOKENS}
        gather = steps.stage_class("logits", "HopperLogitsGather")
        if gather is not None and tokens:
            # Collective: every TP rank builds the gather in lockstep.
            self.logits_gather = gather(
                columns=weight.shape[0], capacity=max(tokens), device=weight.device
            )
            self.logits_tokens = tokens
        tokens = {m for m in self.policy.tables["logits_head"] if m <= MAX_TOKENS}
        head = steps.stage_class("logits", "HopperLogitsHead")
        if (
            head is not None
            and tokens
            and logits_processor.soft_cap is None
            and logits_processor.scale == 1.0
        ):
            self.head = head(weight, sorted(tokens))
            self.head_tokens = {m for m in tokens if self.head.supports(m)}

    def compute_logits(self, logits_processor, lm_head, hidden_states):
        """`logits_processor(lm_head, hidden_states)` on the Hopper head GEMM
        and logits gather where the policy gives them, or None for stock."""
        rows = hidden_states.shape[0]
        head = (
            self.head is not None
            and rows in self.head_tokens
            and hidden_states.is_contiguous()
        )
        gather = self.logits_gather is not None and rows in self.logits_tokens
        if (
            not (head or gather)
            or hidden_states.dtype != torch.bfloat16
            or torch.cuda.is_current_stream_capturing()
        ):
            return None
        if head:
            logits = self.head(hidden_states)
        else:
            logits = logits_processor(lm_head, hidden_states, skip_gather=True)
        if gather:
            logits = self.logits_gather(logits)
        else:
            logits = logits_processor._gather_logits(logits)
        return logits[:, : logits_processor.org_vocab_size]


def boundary_modes(policy: Policy, tokens: list[int]) -> dict[int, Any]:
    """The Hopper boundary's mode per token count: the policy's choice, or the
    boundary module's default for "auto"."""
    from .boundary import DEFAULT_MODES, parse_mode

    modes = {}
    for m in tokens:
        choice = policy("boundary", m)
        if not choice:
            continue
        mode = DEFAULT_MODES.get(m) if choice == AUTO else parse_mode(choice)
        if mode is not None:
            modes[m] = mode
    return modes


class HopperSpecializedDraft(HopperLayerStages, SpecializedDraft):
    """`SpecializedDraft` on Hopper: the backbone's Hopper stages on the MTP
    draft layer (its DSA attention and MoE have the backbone's shapes) and the
    draft boundaries as `boundary.HopperDraftAllReduceNorm`. The draft's
    embedding and eh_proj run stock unless the boundary fuses the embedding's
    all-reduce. Built on the first eager draft forward; reads the MTP layers,
    never modifies them."""

    entries = {"dsa": "draft_dsa", "moe": "draft_moe"}

    def __init__(self, predictor, config):
        self.config = config
        self.policy = Policy(config.policy, HOPPER)
        self.q = config.q
        self.mtp_layers = list(predictor._mtp_layers)
        self.layers = [layer.mtp_block for layer in self.mtp_layers]
        self.text_config = self.mtp_layers[0].config
        # Draft decode steps run one token per request; draft prefills run q.
        self.batches = sorted(b for b in config.capture_sizes if b <= MAX_TOKENS)
        self.calls = Counter()
        self.kda: dict[int, Any] = {}
        self.eh_proj: dict[int, Any] = {}
        self.attn_stages = self._attention_stages(
            self.layers, tuple(dict.fromkeys((self.q, 1)))
        )
        self.ffn_stages = self._ffn_stages(self.layers)
        self.prepared_for: tuple | None = None
        # Collective: every TP rank builds the boundary in lockstep.
        self.ar_norm = self._build_ar_norm()
        self.embed_tokens = predictor.embed_tokens
        self.fused_embedding = self.ar_norm is not None and self._build_embedding()
        self.active = bool(
            self.attn_stages or self.ffn_stages or self.ar_norm is not None
        )
        logger.info(
            "Hopper specialized draft at %s tokens (q=%d): %s; boundary %s, "
            "fused embedding %s",
            self.batches,
            self.q,
            self.describe(),
            "on" if self.ar_norm is not None else "off",
            self.fused_embedding,
        )

    def _build_ar_norm(self):
        cls = steps.stage_class("boundary", "HopperDraftAllReduceNorm")
        tokens = self.tokens("draft_boundary")
        if cls is None or not tokens:
            return None
        norms = {
            n.variance_epsilon
            for mtp, layer in zip(self.mtp_layers, self.layers)
            for n in (layer.post_attention_layernorm, mtp.shared_head.norm)
        }
        if len(norms) != 1:
            logger.warning("Hopper specialized draft: mixed norm eps, stock boundary")
            return None
        return cls(
            hidden_size=self.layers[0].hidden_size,
            max_num_tokens=max(tokens),
            top_k=self.text_config.num_experts_per_tok,
            eps=norms.pop(),
            device=self.mtp_layers[0].eh_proj.weight.device,
        )

    def partial(self, tokens: int) -> bool:
        return (
            self.ar_norm is not None
            and bool(self.policy("draft_boundary", tokens))
            and self.ar_norm.supports(tokens)
        )

    def reduced_norm(self, x, partial, residual, norm):
        """`norm(x + residual)` and the new residual of a sublayer output; a
        TP-partial `x` at a token count the boundary does not serve (a draft
        prefill's sampled rows) is all-reduced first."""
        if partial:
            assert self.ar_norm is not None
            if not self.ar_norm.supports(x.shape[0]):
                x, partial = tensor_model_parallel_all_reduce(x), False
        self.calls["draft_boundary", partial] += 1
        out = super().reduced_norm(x, partial, residual, norm)
        return out

    def _bound_caches(self) -> tuple | None:
        """The draft MLA KV cache's address and layout once the runner has
        bound it."""
        cache = self.layers[0].self_attn.mla_attn.mla_attn.kv_cache
        if not isinstance(cache, torch.Tensor) or cache.numel() == 0:
            return None
        return cache.data_ptr(), tuple(cache.shape), cache.stride()

    def plan(self, positions) -> Step | None:
        """`SpecializedDraft.plan` up to `MAX_TOKENS` tokens, for steps that
        reach at least one Hopper stage.

        The draft is built on its first eager forward, during memory profiling
        and before the KV caches are bound; its attention stages compile for
        the bound caches on the first eager forward after each binding (the
        warmup run before the first capture)."""
        if not self.active:
            return None
        if self.attn_stages and not torch.cuda.is_current_stream_capturing():
            bound = self._bound_caches()
            if bound is not None and bound != self.prepared_for:
                self._prepare_caches()
                self.prepared_for = bound
        tokens = positions.shape[-1]
        if tokens > MAX_TOKENS:
            return None
        step = super().plan(positions)
        if step is None or not self._serves(tokens, step.q):
            return None
        return step

    def run(
        self, mtp, step: Step, input_ids, inputs_embeds, previous_hidden_states, rows
    ):
        self.calls.clear()
        out = super().run(
            mtp, step, input_ids, inputs_embeds, previous_hidden_states, rows
        )
        logger.info(
            "Hopper specialized draft captured a %d-token step (q=%d): %s",
            step.positions.shape[-1],
            step.q,
            self.summary(),
        )
        return out
