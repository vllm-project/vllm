# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1's specialized decode."""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import torch

import vllm.envs as envs
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.common.specialized_decode.spec import DecodeSpec, Feature, Step
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.models.deepseek_v41.amd.model import DeepseekV4DecoderLayer
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterSparseSWAMetadata

logger = init_logger(__name__)

RECORD = 584  # an fp8_ds_mla KV record: 576 data bytes, 8 scale bytes
SWA_WIDTH = 128  # a causal decode token's window slots
MAX_ROWS = 48  # the kernels' step rows at most: 8 requests x (1 + 5 DSpark drafts)
# the kernels' ISA: CDNA4 (gfx950) scaled MFMA and MX formats
CDNA_VERSIONS = (4,)


def _record_view(kv: torch.Tensor, block: int) -> torch.Tensor:
    """A layer's fp8_ds_mla cache as [blocks, block, RECORD] bytes (a block's
    data rows, then its scale words: the kernels index both from it)."""
    kv = kv if kv.dtype == torch.uint8 else kv.view(torch.uint8)
    return torch.as_strided(kv, (kv.shape[0], block, RECORD), (kv.stride(0), RECORD, 1))


def _scale_bytes(scale: torch.Tensor) -> torch.Tensor:
    return scale if scale.dtype == torch.uint8 else scale.view(torch.uint8)


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


class DSV41Kernel:
    def __init__(self, vllm_config, model, widths: tuple[int, ...]):
        from vllm.models.common.specialized_decode.guards import tp_group
        from vllm.models.deepseek_v41.amd.specialized_decode.mono.runner import (
            MAX_TOKENS,
            DSV41MonoLayer,
        )

        assert MAX_TOKENS >= MAX_ROWS, MAX_TOKENS
        rank, world, cpu_group = tp_group()
        device = next(model.parameters()).device
        self.model = model
        self.runner = DSV41MonoLayer(world, rank, cpu_group, device)
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
            plan = self._plan_layer(layer, tp_size)
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

    def _plan_layer(self, layer, tp_size: int) -> LayerPlan | None:
        """Info needed to run the specialized decode kernel for this layer."""
        attn = layer.attn
        why = _skip_layer(layer)
        if why is not None:
            logger.info_once(
                "DSv4.1 specialized decode: layer %d on vLLM's path (%s)",
                attn.layer_id,
                why,
            )
            return None
        why = _ffn_only(layer)
        weights = _weights(layer, why is not None, tp_size)
        if why is not None:
            logger.info_once(
                "DSv4.1 specialized decode: the FFN launch for layers with %s", why
            )
            # A decode step hands wo_b's TP partial to the FFN launch; the layer
            # reduces it on the steps that keep vLLM's path.
            attn.wo_b.reduce_results = False
            return LayerPlan(layer, weights, ffn_only=True)
        comp_key = attn.compressed_cache_prefix
        # K1's seam folds the owed post into the residual, except where the
        # residual arrives settled: the embedding, or after the Engram write.
        settled = attn.layer_id == self.embed_layer or layer.engram is not None
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

    def bind_caches(self, caches: Mapping[str, Any]) -> None:
        """Nothing to take: a launch reads the cache off its attention layer."""

    def warmup_keys(self) -> list[Any]:
        keys: dict[Any, None] = {}
        for plan in self.plans.values():
            for width in self.widths:
                if plan.ffn_only:
                    keys[width, "ffn"] = None
                else:
                    for fold in plan.folds:
                        keys[width, plan.ratio, fold, plan.index] = None
        return list(keys)

    def compile(self, key: Any) -> None:
        self.runner.scratch(key[0])
        if key[1] == "ffn":
            self.runner.ffn_kernel(key[0])
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
        if first and x.dim() != 2:
            return None
        tensors = (x,) if first else (x, residual, post_mix, res_mix, pre_mix)
        if not all(t is not None and t.is_contiguous() for t in tensors):
            return None
        fold = True
        if first:
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
        return self.runner.ffn(plan.weights, *tensors)

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


def _ffn_only(layer) -> str | None:
    """Why this layer's attention seam and attention stay on vLLM's path, which
    leaves it the FFN launch, or None when it runs whole."""
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV41RocmMxfp4Indexer

    attn = layer.attn
    if attn.compressor is not None and attn.indexer is None:
        return "a compressor without an indexer"
    if attn.indexer is not None and not isinstance(
        attn.indexer, DeepseekV41RocmMxfp4Indexer
    ):
        return f"a {type(attn.indexer).__name__}"
    if attn.compress_ratio not in (0, 1, 2):
        return f"compress ratio {attn.compress_ratio}"
    if attn.kv_mxfp8 or attn.kv_cache_dtype != "fp8_ds_mla":
        return f"a {attn.kv_cache_dtype} KV cache"
    return None


def _widths(vllm_config) -> tuple[int, ...]:
    """The configured capture sizes the kernels can serve, so a deployment
    that asked for fewer graphs keeps them; every size at all when it asked for
    none."""
    sizes = vllm_config.compilation_config.cudagraph_capture_sizes or []
    served = tuple(s for s in sorted(sizes) if 1 <= s <= MAX_ROWS)
    return served or tuple(range(1, MAX_ROWS + 1))


def _refuse_config(vllm_config) -> list[str]:
    """Every reason this config cannot run the kernels, before load."""
    reasons = []
    if vllm_config.parallel_config.tensor_parallel_size not in (2, 4):
        reasons.append("needs tensor parallel size 2 or 4")
    if envs.VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4:
        reasons.append(
            "needs AITER's A8W4 MoE (VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4 unset)"
        )
    try:
        import vllm.models.deepseek_v41.amd.specialized_decode.mono.runner  # noqa: F401
    except ImportError as err:
        reasons.append(f"needs FlyDSL and AITER's FlyDSL helpers ({err})")
    return reasons


def _refuse_model(model) -> list[str]:
    """Every reason the loaded model and this GPU cannot run the kernels. The
    kernels read every one of the 384 experts on each rank, TP-sharded, in
    AITER's A8W4 MXFP4 layout, from CTAs that all stay resident."""
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
    from vllm.models.deepseek_v41.amd.specialized_decode.mono.runner import BLOCKS
    from vllm.platforms.rocm import get_cdna_version

    reasons = []
    if get_cdna_version() not in CDNA_VERSIONS:
        reasons.append(f"needs CDNA{'/'.join(map(str, CDNA_VERSIONS))}")
    cus = current_platform.num_compute_units(torch.accelerator.current_device_index())
    if cus < BLOCKS:
        reasons.append(f"needs {BLOCKS} compute units, the GPU has {cus}")
    backends = set()
    for layer in model.layers:
        experts = getattr(getattr(layer, "ffn", None), "experts", None)
        routed = getattr(experts, "routed_experts", None)
        if routed is not None:
            backends.add(getattr(routed.quant_method, "mxfp4_backend", None))
    wrong = backends - {Mxfp4MoeBackend.AITER_MXFP4_BF16}
    if wrong:
        reasons.append(f"needs the AITER_MXFP4_BF16 MoE backend, not {wrong}")
    return reasons


def _skip_layer(layer) -> str | None:
    """Why this layer keeps vLLM's path for every step, or None when it takes a
    launch (whole or FFN only: ``_plan_layer`` picks which)."""
    ffn = layer.ffn
    if layer.use_sequence_parallel:
        return "sequence-parallel layer"
    if not layer.fuse_seam_norm:
        return "needs aiter's fused mHC seam"
    if (
        ffn.shared_experts is None
        or ffn.gate.tid2eid is not None
        or ffn.n_routed_experts != 384
        or ffn.n_activated_experts != 6
        or ffn.routed_scaling_factor != 1.5
        or ffn.swiglu_limit != 10.0
        or ffn.scoring_func != "sqrtsoftplus"
        or not ffn.renormalize
    ):
        return "MoE routing outside the kernels' (384 / 6, sqrtsoftplus, x1.5)"
    return None


def _skip_step(vllm_config, decode: DSV41Kernel, width: int) -> str | None:
    """Why this step keeps vLLM's path at every layer, or None when the kernels
    take it. One verdict for every kernel layer, from the batch the whole TP
    group sees alike."""
    md = get_forward_context().attn_metadata
    if not decode.plans:
        return "no layer takes a launch"
    if not isinstance(md, dict):  # a profile / dummy run, or DBO ubatches
        return "no attention metadata"
    replay = decode.model.decoder_replay_layers
    if replay is not None and replay.uses_replay_batch():
        return "replay batch step"
    if any(md.get(k) is None for k in decode.comp_keys):
        return "layer metadata missing"
    for key in decode.swa_keys:
        swa = cast("DeepseekV41ROCMAiterSparseSWAMetadata | None", md.get(key))
        if swa is None:
            return "layer metadata missing"
        if swa.num_prefills != 0 or swa.num_decodes == 0:
            return "not a decode-only step"
        if swa.num_decode_tokens > width:
            return "more decode tokens than rows"
        if swa.decode_swa_width != SWA_WIDTH:
            return f"SWA width {swa.decode_swa_width}"
        if (
            swa.decode_swa_indices is None
            or swa.decode_swa_lens is None
            or swa.token_to_req_indices is None
        ):
            return "SWA decode metadata missing"
    return None


def _weights(layer, ffn_only: bool, world: int):
    from vllm.models.deepseek_v41.amd.specialized_decode.mono.runner import (
        MonoLayerWeights,
    )

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
    weights.check(world)
    return weights


def _attn_weights(a, world: int):
    from vllm.models.deepseek_v41.amd.specialized_decode.mono.attention.plan import Dims
    from vllm.models.deepseek_v41.amd.specialized_decode.mono.runner import AttnWeights

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


def _refuse_caches(vllm_config, decode: DSV41Kernel, caches) -> list[str]:
    """Every reason the kernels cannot read these caches: they index each as
    [blocks, block, RECORD] bytes."""
    reasons = []
    for name, cache in caches.items():
        if not isinstance(cache, torch.Tensor):
            reasons.append(f"{name}: no cache tensor")
            continue
        if cache.element_size() != 1:
            reasons.append(f"{name}: {cache.dtype} records, needs bytes")
        if cache.dim() < 2:
            reasons.append(f"{name}: {cache.dim()} dims, needs [blocks, block, bytes]")
        elif cache.stride(0) < RECORD:
            reasons.append(f"{name}: block stride {cache.stride(0)} under {RECORD}")
    return reasons


DECODE_SPEC = DecodeSpec[DSV41Kernel](
    name="dsv41-cdna4",
    supports=frozenset({Feature.PIPELINE_PARALLEL, Feature.SPECULATIVE_DECODING}),
    widths=_widths,
    refuse_config=_refuse_config,
    refuse_model=_refuse_model,
    refuse_caches=_refuse_caches,
    skip_step=_skip_step,
    build=DSV41Kernel,
)
