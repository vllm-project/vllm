# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1's decode kernels: one launch a layer over the stock backbone,
and the same over the DSpark draft head's layers."""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.common.specialized_decode.spec import Step

from ..spec import (
    DRAFT_UG_GROUPS,
    EXPERTS,
    MAX_ROWS,
    RECORD,
    SWA_WIDTH,
    TOPK,
    UG_GROUPS,
    draft_routing,
    draft_swa_width,
    on_gfx942,
)
from .weights942 import scale_bytes as _scale_bytes

if TYPE_CHECKING:
    from vllm.models.deepseek_v41.amd.model import DeepseekV4DecoderLayer

logger = init_logger(__name__)


def _record_view(kv: torch.Tensor, block: int) -> torch.Tensor:
    """A layer's fp8_ds_mla cache as [blocks, block, RECORD] bytes (a block's
    data rows, then its scale words: the kernels index both from it)."""
    kv = kv if kv.dtype == torch.uint8 else kv.view(torch.uint8)
    return torch.as_strided(kv, (kv.shape[0], block, RECORD), (kv.stride(0), RECORD, 1))


def _select(
    layer: "DeepseekV4DecoderLayer",
    positions: torch.Tensor,
    normed: torch.Tensor,
    qr: torch.Tensor,
) -> None:
    """An index source's compressor and top-k selection, from K1's normed input
    and bf16 qr: ``_forward_csa2_full`` / ``_forward_csa2_reindex`` and the
    indexer op, serial."""
    attn = layer.attn
    comp, indexer = attn.compressor, attn.indexer
    assert indexer is not None
    if comp is not None:
        kv_score = torch.mm(
            normed, comp.fused_wkv_wgate.weight.T, out_dtype=torch.float32
        )
        latent = comp(kv_score, positions)
        comp.insert_cache(latent, positions, attn.rotary_emb)
        indexer._produce_k(latent, positions, attn.indexer_rotary_emb)
    weights, _ = indexer.weights_proj(normed)
    q, q_scale, weights = indexer.forward_q(
        qr, None, weights, positions, attn.indexer_rotary_emb
    )
    assert weights is not None
    indexer.indexer_op(
        normed, (q, q_scale) if q_scale is not None else q, None, weights
    )


@dataclass(frozen=True)
class LayerPlan:
    """One decoder layer's launch, settled at load.

    An ``ffn_only`` layer keeps vLLM's attention and its launch takes wo_b's
    unreduced output, so its weights are all it has; the rest is what a
    whole-layer launch reads off the layer.
    """

    layer: "DeepseekV4DecoderLayer"
    weights: Any
    ffn_only: bool
    ratio: int = 0
    index: bool = False
    swa_key: str | None = None
    comp_key: str | None = None
    folds: tuple[bool, ...] = ()


class LayerLaunches:
    """The launches of one stack of DSv4.1 decoder layers: the backbone's, or
    the DSpark draft's, which has its own routing, SWA rows and runner.

    A subclass owns the layer loop in ``run``; everything the launches need is
    settled here, at load.
    """

    draft = False

    def __init__(
        self,
        vllm_config,
        model,
        widths: tuple[int, ...],
        experts: int,
        topk: int,
        ug_groups: int,
        swa_width: int,
    ):
        from vllm.models.common.specialized_decode.guards import tp_group
        from vllm.models.deepseek_v41.amd.mono.runner import (
            MAX_TOKENS,
            DSV41MonoLayer,
        )

        assert MAX_TOKENS >= MAX_ROWS, MAX_TOKENS
        rank, world, cpu_group = tp_group()
        device = next(model.parameters()).device
        self.model = model
        self.experts, self.topk, self.swa_width = experts, topk, swa_width
        self.runner = DSV41MonoLayer(
            world,
            rank,
            cpu_group,
            device,
            experts=experts,
            topk=topk,
            ug_groups=ug_groups,
            swa_width=swa_width,
        )
        self.embed_layer = getattr(model, "start_layer", 0)
        # A decode step is never trimmed, so ``run`` owns every layer, the ones
        # past the replay cut included.
        self.layer_end = getattr(model, "end_layer", len(model.layers))
        self.widths = widths
        self.swa_keys: list[str] = []
        self.comp_keys: list[str] = []
        self.plans: dict[int, LayerPlan] = {}

        tp_size = vllm_config.parallel_config.tensor_parallel_size
        for idx in range(self.embed_layer, self.layer_end):
            layer = model.layers[idx]
            plan = self._plan_layer(layer, tp_size, first=idx == self.embed_layer)
            if plan is None:
                continue
            self.plans[layer.attn.layer_id] = plan
            if plan.ffn_only:
                continue
            for key, keys in (
                (plan.swa_key, self.swa_keys),
                (plan.comp_key, self.comp_keys),
            ):
                if key is not None and key not in keys:
                    keys.append(key)
        self.cache_layers = tuple(sorted(set(self.swa_keys) | set(self.comp_keys)))

    def _plan_layer(self, layer, tp_size: int, first: bool) -> LayerPlan | None:
        """Info needed to run the specialized decode kernel for this layer."""
        attn = layer.attn
        why = _skip_layer(layer, self.experts, self.topk, tp_size)
        if why is not None:
            logger.info_once(
                "DSv4.1 specialized decode: layer %d on vLLM's path (%s)",
                attn.layer_id,
                why,
            )
            return None
        if self.draft and on_gfx942():
            # On gfx942 the draft layers take the FFN launch after vLLM's
            # attention. vLLM's draft MoE multiplies bf16 activations with the
            # FP4 experts, and the FFN launch quantizes them to MXFP8 as it does
            # for the target. That changes only how many drafts the target
            # accepts, not the target's output.
            logger.info_once(
                "DSv4.1 specialized decode: the FFN launch for the draft layers "
                "(%d / %d routing)",
                self.experts,
                self.topk,
            )
            return self._ffn_plan(layer, tp_size)
        why = _ffn_only(layer)
        if why is None and self.draft and (attn.compress_ratio or attn.indexer):
            why = "compressed keys"
        if why is not None and self.draft:
            # The draft has no FFN launch: its layers run whole or not at all.
            logger.info_once(
                "DSv4.1 specialized decode: the draft layers on vLLM's path (%s)", why
            )
            return None
        if why is not None:
            logger.info_once(
                "DSv4.1 specialized decode: the FFN launch for layers with %s", why
            )
            return self._ffn_plan(layer, tp_size)
        weights = _weights(layer, False, tp_size, self.experts)
        # gfx942: the entry seam of a whole first layer projects the expanded
        # residual with the unfolded hc_attn_fn, as K1 does, and not the
        # embedding with the folded copy, which has no AITER kernel there.
        layer.unfolded_entry = on_gfx942()
        comp_key = attn.compressed_cache_prefix if attn.compress_ratio else None
        # K1's seam folds the owed post into the residual, except where the
        # residual arrives settled: the embedding, or after the Engram write.
        settled = first or layer.engram is not None
        return LayerPlan(
            layer,
            weights,
            ffn_only=False,
            ratio=attn.compress_ratio,
            index=attn.indexer is not None,
            swa_key=attn.swa_cache_layer.prefix,
            comp_key=comp_key,
            folds=(True, False) if settled else (True,),
        )

    def _ffn_plan(self, layer, tp_size: int) -> LayerPlan:
        # A decode step hands wo_b's TP partial to the FFN launch; the layer
        # reduces it on the steps that keep vLLM's path.
        layer.attn.wo_b.reduce_results = False
        return LayerPlan(
            layer, _weights(layer, True, tp_size, self.experts), ffn_only=True
        )

    def bind_caches(self, caches: Mapping[str, Any]) -> None:
        """Nothing to take: a launch reads the cache off its attention layer."""

    def warmup_keys(self) -> list[Any]:
        keys: dict[Any, None] = {}
        for plan in self.plans.values():
            for width in self.widths:
                if plan.ffn_only:
                    keys[width, "ffn", self.experts, self.topk] = None
                else:
                    for fold in plan.folds:
                        keys[width, plan.ratio, fold, plan.index] = None
        return list(keys)

    def compile(self, key: Any) -> None:
        self.runner.scratch(key[0])
        if key[1] == "ffn":
            self.runner.ffn_kernel(key[0], key[2], key[3])
        else:
            width, ratio, fold, index = key
            self.runner.kernels(width, ratio, fold=fold, index=index)

    def launch_whole(
        self,
        plan: LayerPlan,
        x: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor | None,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        pre_mix: torch.Tensor | None,
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...] | None:
        """Run a decode layer as one logical kernel."""
        layer, runner = plan.layer, self.runner
        attn = layer.attn
        first = residual is None
        # The draft's first layer takes the embedding already broadcast to the
        # hc copies ([rows, hc, dim]), the backbone's the 2-D embedding.
        if first and x.dim() != (3 if self.draft else 2):
            return None
        tensors = (x,) if first else (x, residual, post_mix, res_mix, pre_mix)
        if not all(t is not None and t.is_contiguous() for t in tensors):
            return None
        fold = True
        if first:
            if self.draft:
                residual, x = x, x[:, 0].contiguous()
            else:
                residual = x.unsqueeze(1).expand(-1, layer.hc_mult, -1).contiguous()
            pre_mix = runner.first_pre_mix(x.shape[0])
            post_mix = res_mix = pre_mix  # unread without the fold
            fold = False
        elif layer.engram is not None and engram_hashes is not None:
            residual = layer.mhc_post(x, residual, post_mix, res_mix)
            hashes = engram_hashes[:, layer.engram.layer_hash_index]
            residual = layer.engram(residual, hashes, engram_mask)
            fold = False
        md = cast("dict[str, Any]", get_forward_context().attn_metadata)
        swa = md[cast("str", plan.swa_key)]
        kw: dict[str, Any] = {}
        if plan.index:
            kw["select"] = lambda normed, qr: _select(layer, positions, normed, qr)
        if plan.ratio:
            comp = md[cast("str", plan.comp_key)]
            kw.update(
                topk_indices=attn.topk_indices_buffer,
                comp_cache=_record_view(
                    attn._compressed_kv_cache(), comp.block_size // plan.ratio
                ),
                comp_block_table=comp.block_table,
            )
        return runner.forward(
            plan.weights,
            x,
            residual,
            post_mix,
            res_mix,
            pre_mix,
            positions,
            swa.slot_mapping,
            _record_view(attn.swa_cache_layer.kv_cache, swa.block_size),
            swa.decode_swa_indices,
            swa.decode_swa_lens,
            swa.token_to_req_indices,
            fold=fold,
            **kw,
        )

    def launch_ffn(
        self,
        plan: LayerPlan,
        part: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        pre_mix: torch.Tensor,
    ) -> tuple[torch.Tensor, ...] | None:
        """The FFN seam, MoE and both reductions as one launch, from wo_b's
        unreduced output after vLLM's attention."""
        tensors = (part, residual, post_mix, res_mix, pre_mix)
        if not all(t.is_contiguous() for t in tensors):
            return None
        return self.runner.ffn(plan.weights, *tensors, topk=self.topk)


class DSV41Kernel(LayerLaunches):
    """The backbone's decode: 384 experts, top-6, causal SWA windows."""

    def __init__(self, vllm_config, model, widths: tuple[int, ...]):
        super().__init__(
            vllm_config,
            model,
            widths,
            experts=EXPERTS,
            topk=TOPK,
            ug_groups=UG_GROUPS,
            swa_width=SWA_WIDTH,
        )

    def run(self, step: Step, input_ids, inputs_embeds, **kwargs) -> Any:
        model, positions = self.model, step.positions
        (
            hidden,
            input_ids,
            pre_mix,
            engram_hashes,
            engram_mask,
            full_num_tokens,
        ) = model.prologue(
            input_ids, positions, None, inputs_embeds, kwargs.get("lookback_token_ids")
        )
        residual = post_mix = res_mix = None
        aux_hidden_states: list[torch.Tensor] = []
        final_aux_recon = None
        for idx in range(self.embed_layer, self.layer_end):
            layer = model.layers[idx]
            plan = self.plans.get(layer.attn.layer_id)
            ffn = out = None
            if plan is None:
                # TODO: check if we should crash here
                pass
            elif plan.ffn_only:
                ffn = partial(self.launch_ffn, plan)
            else:
                # Run entire layer
                out = self.launch_whole(
                    plan,
                    hidden,
                    positions,
                    residual,
                    post_mix,
                    res_mix,
                    pre_mix,
                    engram_hashes,
                    engram_mask,
                )
            if out is None:
                # Run standard model implementation, with injected specialized ffn
                out = layer(
                    hidden,
                    positions,
                    input_ids,
                    pre_mix,
                    post_mix,
                    res_mix,
                    residual,
                    engram_hashes,
                    engram_mask,
                    decode_ffn=ffn,
                )
            hidden, residual, post_mix, res_mix, pre_mix = out
            aux_recon = model.after_layer(
                idx,
                hidden,
                residual,
                post_mix,
                res_mix,
                full_num_tokens,
                aux_hidden_states,
                replay=False,
            )
            if aux_recon is not None:
                final_aux_recon = aux_recon
        return model.epilogue(
            hidden,
            positions,
            input_ids,
            pre_mix,
            post_mix,
            res_mix,
            residual,
            aux_hidden_states,
            final_aux_recon,
            self.layer_end > self.embed_layer,
            full_num_tokens,
            replay=False,
        )


class DSV41DraftKernel(LayerLaunches):
    """The DSpark draft's decode: the draft's own routing and non-causal SWA
    rows, over the draft head's layers."""

    draft = True

    def __init__(self, vllm_config, model, widths: tuple[int, ...]):
        experts, topk = draft_routing(vllm_config)
        super().__init__(
            vllm_config,
            model,
            widths,
            experts=experts,
            topk=topk,
            # the draft's unions are small on gfx950, where it runs whole; on
            # gfx942 it takes the FFN launch, validated at the target's groups
            ug_groups=UG_GROUPS if on_gfx942() else DRAFT_UG_GROUPS,
            swa_width=draft_swa_width(vllm_config),
        )

    def run(self, step: Step, input_ids, inputs_embeds, **kwargs) -> Any:
        model, positions = self.model, step.positions
        hidden, input_ids, full_num_tokens = model.prologue(
            input_ids, positions, inputs_embeds
        )
        residual = post_mix = res_mix = pre_mix = None
        for idx in range(self.embed_layer, self.layer_end):
            layer = model.layers[idx]
            plan = self.plans.get(layer.attn.layer_id)
            ffn = out = None
            if plan is None:
                pass
            elif plan.ffn_only:
                ffn = partial(self.launch_ffn, plan)
            else:
                out = self.launch_whole(
                    plan, hidden, positions, residual, post_mix, res_mix, pre_mix
                )
            if out is None:
                out = layer(
                    hidden,
                    positions,
                    input_ids,
                    pre_mix,
                    post_mix,
                    res_mix,
                    residual,
                    decode_ffn=ffn,
                )
            hidden, residual, post_mix, res_mix, pre_mix = out
        return model.epilogue(
            hidden, residual, post_mix, res_mix, pre_mix, full_num_tokens
        )


def prepare_weights(vllm_config, model, draft: bool = False) -> None:
    """Make ready, at model creation, whatever the launches need from the
    weights before vLLM loads them: the flag the gfx942 decode helpers read, the
    top-k hooks they call, and the gfx942 kernels' own copies of the weights,
    which vLLM then makes as it processes each layer's at load.

    Args:
        vllm_config: the config of the model being created.
        model: the backbone, or the DSpark draft head's.
        draft: whether ``model`` is the draft head's backbone.

    """
    from vllm.models.deepseek_v41.amd import ops

    # The gfx942 decode helpers run in forward passes, where vLLM's current
    # config is not set, so they read the flag recorded here.
    ops.record_specialized_decode(vllm_config)
    if not vllm_config.kernel_config.specialized_decode or not on_gfx942():
        return
    from vllm.models.deepseek_v41.amd.ops import topk as topk_ops

    from .weights942 import copy_at_load

    # A failed build shows at startup, and the hooks then leave the top-k to
    # vLLM from the first step on.
    topk_ops.build()
    experts, topk = draft_routing(vllm_config) if draft else (EXPERTS, TOPK)
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    for layer in model.layers:
        if _skip_layer(layer, experts, topk, tp_size) is None:
            copy_at_load(layer, ffn_only=draft or _ffn_only(layer) is not None)


def _ffn_only(layer) -> str | None:
    """Why this layer's attention seam and attention stay on vLLM's path, which
    leaves it the FFN launch, or None when it runs whole."""
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV41RocmMxfp4Indexer
    from vllm.models.deepseek_v41.attention import DeepseekV4Indexer

    # The MXFP4 indexer cache needs CDNA4, so gfx942 drives vLLM's FP8 indexer,
    # whose wq_b also reads K1's bf16 qr.
    indexer_cls = DeepseekV4Indexer if on_gfx942() else DeepseekV41RocmMxfp4Indexer
    attn = layer.attn
    if attn.compressor is not None and attn.indexer is None:
        return "a compressor without an indexer"
    if attn.indexer is not None and not isinstance(attn.indexer, indexer_cls):
        return f"a {type(attn.indexer).__name__}"
    if attn.compress_ratio not in (0, 1, 2):
        return f"compress ratio {attn.compress_ratio}"
    if attn.kv_mxfp8 or attn.kv_cache_dtype != "fp8_ds_mla":
        return f"a {attn.kv_cache_dtype} KV cache"
    return None


def _skip_layer(layer, experts: int, topk: int, tp_size: int) -> str | None:
    """Why this layer keeps vLLM's path for every step, or None when it takes a
    launch (whole or FFN only: ``_plan_layer`` picks which)."""
    ffn = layer.ffn
    gfx942 = on_gfx942()
    if layer.use_sequence_parallel:
        return "sequence-parallel layer"
    if not layer.fuse_seam_norm and not gfx942:
        # gfx942 has no fused seam kernel. vLLM rounds the collapsed stream to
        # bf16 before its RMSNorm there, which the kernels' fused seam does not,
        # so the two paths differ by that rounding.
        return "needs aiter's fused mHC seam"
    if gfx942 and tp_size != 4:
        return "gfx942 builds for tensor parallel size 4 only"
    if (
        ffn.shared_experts is None
        or ffn.gate.tid2eid is not None
        or ffn.n_routed_experts != experts
        or ffn.n_activated_experts != topk
        or ffn.routed_scaling_factor != 1.5
        or ffn.swiglu_limit != 10.0
        or ffn.scoring_func != "sqrtsoftplus"
        or not ffn.renormalize
    ):
        return (
            f"MoE routing outside the kernels' ({experts} / {topk}, sqrtsoftplus, x1.5)"
        )
    return None


def _weights(layer, ffn_only: bool, world: int, experts: int):
    from vllm.models.deepseek_v41.amd.mono.runner import (
        MonoLayerWeights,
    )

    if on_gfx942():
        from .weights942 import layer_weights

        return layer_weights(layer, ffn_only, world)
    a, f = layer.attn, layer.ffn
    e, sh = f.experts.routed_experts, f.shared_experts
    weights = MonoLayerWeights(
        attn=None if ffn_only else _attn_weights(a, world),
        hc_attn_fn=layer.hc_attn_fn,
        hc_attn_scale=layer.hc_attn_scale,
        hc_attn_base=layer.hc_attn_base,
        attn_norm=layer.attn_norm.weight,
        hc_ffn_fn=layer.hc_ffn_fn,
        hc_ffn_scale=layer.hc_ffn_scale,
        hc_ffn_base=layer.hc_ffn_base,
        ffn_norm=layer.ffn_norm.weight,
        gate_w=f.gate.weight,
        bias=f.gate.e_score_correction_bias,
        w13=e.w13_weight,
        w13_s=e.w13_weight_scale,
        w2=e.w2_weight,
        w2_s=e.w2_weight_scale,
        sgu=sh.gate_up_proj.weight,
        sgu_s=_scale_bytes(sh.gate_up_proj.weight_scale),
        sw2=sh.down_proj.weight,
        sw2_s=_scale_bytes(sh.down_proj.weight_scale),
    )
    weights.check(world, experts)
    return weights


def _attn_weights(a, world: int):
    from vllm.models.deepseek_v41.amd.mono.attention.plan import Dims
    from vllm.models.deepseek_v41.amd.mono.runner import AttnWeights

    attn = AttnWeights(
        layer_id=a.layer_id,
        wqkv=a.fused_wqa_wkv.weight,
        wqkv_scale=_scale_bytes(a.fused_wqa_wkv.weight_scale),
        q_norm=a.q_norm.weight,
        kv_norm=a.kv_norm.weight,
        wq_b=a.wq_b.weight,
        wq_b_scale=_scale_bytes(a.wq_b.weight_scale),
        wo_a=a.wo_a.weight,
        wo_a_scale=_scale_bytes(a.wo_a.weight_scale),
        wo_b=a.wo_b.weight,
        wo_b_scale=_scale_bytes(a.wo_b.weight_scale),
        attn_sink=a.attn_sink,
        cos_sin=a.rotary_emb.cos_sin_cache,
        ratio=a.compress_ratio,
    )
    attn.check(Dims(world))
    return attn
