# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit Nemotron W4A16 routed-expert backend.

Batch invariant with one tile tactic for every token count: each output
element is one CTA's K_TILE=256 FP32 accumulation along the full K, with no
split-K, and the top-k combine is a per-token FP32 sum in slot order, so a
token's result does not depend on the batch it is in.

Batch invariance also fixes the combine across layouts: without EP the routed
output is reduced as the EP4 one-sided all-to-all reduces it (BF16 per-rank
partials, FP32 tree over top-k slots), so EP1 and EP4 deployments and the
trainer produce the same bits.
"""

import weakref

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_moe import (
    FlashInferCuteDSLExperts,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static

# Under batch invariance every token count uses this tactic.
BATCH_INVARIANT_TACTIC = ((256, 128, 256), (2, 1), True)
# Expert-parallel layout whose combine order batch invariance reproduces.
BATCH_INVARIANT_EP_SIZE = 4


def w4a16_tactic(num_tokens, batch_invariant):
    """GEMM tactic for ``num_tokens`` routed source tokens."""
    if batch_invariant or num_tokens > 64:
        return BATCH_INVARIANT_TACTIC
    return ((128, 8, 256), (1, 1), True)


def nemotron_w4a16_support(config, model_type):
    """Limit explicit backend selection to the supported Nemotron recipe."""
    parallel = config.moe_parallel_config
    if model_type != "nemotron_h" or config.moe_backend != "flashinfer_cutedsl":
        return False, "requires explicit Nemotron flashinfer_cutedsl selection"
    if config.in_dtype != torch.bfloat16:
        return False, "requires BF16 activations"
    if config.activation != MoEActivation.RELU2_NO_MUL:
        return False, "requires non-gated ReLU2"
    if (
        config.hidden_dim,
        config.intermediate_size_per_partition,
        config.num_experts,
        config.experts_per_token,
    ) != (2688, 1856, 128, 6):
        return False, "requires Lightning H2688/I1856/E128/top6"
    if parallel.tp_size != 1 or parallel.ep_size not in (1, 4):
        return False, "requires TP1 and EP1 or EP4"
    if parallel.enable_eplb or parallel.use_batched_activation_format:
        return False, "EPLB and batched-expert layouts are unsupported"
    if config.num_local_experts != 128 // parallel.ep_size:
        return False, "requires contiguous uniform expert ownership"
    return True, None


def prepare_w4a16_scales(scale):
    """Keep padded SF storage registered; the runner consumes its raw pointer."""
    from flashinfer.quantization.fp4_quantization import block_scale_interleave

    if scale.dtype != torch.float8_e4m3fn or scale.ndim != 3:
        raise ValueError("Expected expert-major E4M3 block scales")
    experts, rows, cols = scale.shape
    swizzled = block_scale_interleave(scale.contiguous().view(torch.uint8))
    return swizzled.view(torch.float8_e4m3fn).reshape(
        experts, (rows + 127) // 128 * 128, (cols + 3) // 4 * 4
    )


class FlashInferCuteDSLW4A16Experts(FlashInferCuteDSLExperts):
    @staticmethod
    def is_supported_config(cls, config, weight_key, activation_key, activation_format):
        current = get_current_vllm_config_or_none()
        model = current.model_config if current is not None else None
        model_type = getattr(getattr(model, "hf_config", None), "model_type", None)
        supported, reason = nemotron_w4a16_support(config, model_type)
        if not supported:
            return supported, reason
        try:
            from flashinfer.fused_moe.cute_dsl.tuner import (
                CuteDslFusedMoEW4A16Runner,  # noqa: F401
            )
        except ImportError:
            return False, "installed FlashInfer lacks the W4A16 runner"
        return mk.FusedMoEExperts.is_supported_config(
            cls, config, weight_key, activation_key, activation_format
        )

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key):
        return weight_key == kNvfp4Static and activation_key is None

    @staticmethod
    def _supports_activation(activation):
        return activation == MoEActivation.RELU2_NO_MUL

    @staticmethod
    def _supports_batch_invariance():
        return True

    def __init__(self, moe_config, quant_config):
        from flashinfer.fused_moe.cute_dsl.tuner import CuteDslFusedMoEW4A16Runner
        from flashinfer.tllm_enums import ActivationType

        mk.FusedMoEExpertsModular.__init__(self, moe_config, quant_config)
        # The parent's per-token NVFP4 activation mode does not apply here.
        self.per_token_activation = False
        self.hidden_dim = moe_config.hidden_dim
        self.local_num_experts = moe_config.num_local_experts
        self.global_num_experts = moe_config.num_experts
        self.local_expert_offset = (
            moe_config.moe_parallel_config.ep_rank * self.local_num_experts
        )
        self.runner = CuteDslFusedMoEW4A16Runner(
            num_experts=self.global_num_experts,
            top_k=moe_config.experts_per_token,
            num_local_experts=self.local_num_experts,
            local_expert_offset=self.local_expert_offset,
            use_fused_finalize=False,
            output_dtype=torch.bfloat16,
            activation_type=ActivationType.Relu2.value,
        )

    def process_weights_after_loading(self, layer):
        if getattr(layer, "w13_input_scale", None) is not None:
            raise ValueError("Unexpected activation scale after W4A16 conversion")
        if layer.expert_map is not None:
            expected = torch.full_like(layer.expert_map, -1)
            start = self.local_expert_offset
            expected[start : start + self.local_num_experts] = torch.arange(
                self.local_num_experts, device=expected.device, dtype=expected.dtype
            )
            if not torch.equal(layer.expert_map, expected):
                raise ValueError("Non-contiguous expert maps are unsupported")
        self._layer_ref = weakref.ref(layer)
        # Scales are read through the layer; a layerwise reload would otherwise
        # keep its processed copies alive here.
        for desc in (self.quant_config._w1, self.quant_config._w2):
            desc.scale = desc.alpha_or_gscale = None

    def _registered_scale(self, name, initial):
        ref = getattr(self, "_layer_ref", None)
        if ref is None:
            return initial
        layer = ref()
        if layer is None:
            raise RuntimeError("W4A16 expert layer was released")
        # Layerwise reload restores registered storage after building the kernel.
        return getattr(layer, name)

    @property
    def w1_scale(self):
        return self._registered_scale("w13_weight_scale", super().w1_scale)

    @property
    def w2_scale(self):
        return self._registered_scale("w2_weight_scale", super().w2_scale)

    # The alphas are the checkpoint's weight_scale_2 as is, without the
    # 1/(1/s) round trip of the shared experts (NemotronSharedNvFp4LinearKernel);
    # the trainer passes the same raw alphas to this kernel.
    @property
    def g1_alphas(self):
        return self._registered_scale("w13_weight_scale_2", super().g1_alphas)

    @property
    def g2_alphas(self):
        return self._registered_scale("w2_weight_scale_2", super().g2_alphas)

    def workspace_shapes(
        self,
        M,
        N,
        K,
        topk,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta,
        activation,
    ):
        # The modular caller derives K from unquantized a1, not packed weights.
        if self.hidden_dim != K:
            raise ValueError("W4A16 workspace requires the full BF16 hidden dimension")
        return (0,), (0,), (M, self.hidden_dim)

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ):
        if a1q_scale is not None or apply_router_weight_on_input:
            raise ValueError("Expected unquantized activations, routing on output")
        inputs = [
            hidden_states,
            topk_ids.to(torch.int32),
            topk_weights.float(),
            w1,
            self.w1_scale,
            self.g1_alphas.reshape(-1),
            w2,
            self.w2_scale,
            self.g2_alphas.reshape(-1),
            output,
        ]
        # Screened tiles chosen from host shapes: no autotuning in graph replay.
        tile = w4a16_tactic(hidden_states.shape[0], envs.VLLM_BATCH_INVARIANT)
        if (
            envs.VLLM_BATCH_INVARIANT
            and self.local_num_experts == self.global_num_experts
        ):
            self._apply_ep_combine(inputs, tile)
        else:
            self.runner.forward(inputs, tactic=(tile, tile))

    def _apply_ep_combine(self, inputs, tile):
        """Run every expert locally and reduce as the EP4 combine does.

        Each EP rank reduces its own slots with ``moe_unpermute`` into a BF16
        partial; the one-sided combine places that partial at the rank's first
        top-k slot, zero elsewhere, and sums ``((s0+s1)+(s2+s3))+(s4+s5)`` in
        FP32 (FlashInfer ``moeAlltoAllKernels.cu``).
        """
        from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16 import (
            _get_workspace,
            _run_grouped_gemm,
        )
        from flashinfer.fused_moe.cute_dsl.moe_utils import (
            get_max_num_permuted_tokens,
            moe_permute,
            moe_sort,
            moe_unpermute,
        )
        from flashinfer.tllm_enums import (
            DEFAULT_SWIGLU_ALPHA,
            DEFAULT_SWIGLU_BETA,
            DEFAULT_SWIGLU_LIMIT,
        )

        x, ids, scales, w1, w1_sf, w1_alpha, w2, w2_sf, w2_alpha, output = inputs
        num_tokens, top_k = ids.shape
        if num_tokens == 0:
            return
        if top_k != 6:
            raise ValueError("EP combine order is defined for top-6 routing")
        runner = self.runner
        route_tile = tile[0][1]
        workspace = _get_workspace(
            x,
            top_k,
            self.global_num_experts,
            self.local_num_experts,
            w2.size(2) * 2,
            route_tile,
        )
        t2e, t2lim, e2p, p2e, _, live = moe_sort(
            token_selected_experts=ids,
            token_final_scales=scales,
            num_experts=self.global_num_experts,
            top_k=top_k,
            local_expert_offset=0,
            num_local_experts=self.local_num_experts,
            tile_tokens_dim=route_tile,
            enable_pdl=runner.enable_pdl,
            **workspace.moe_sort_buffers,
        )
        route_slots = get_max_num_permuted_tokens(
            num_tokens, top_k, self.local_num_experts, route_tile
        )
        t2e = t2e[: route_slots // route_tile]
        t2lim = t2lim[: route_slots // route_tile]
        e2p = e2p[:num_tokens]
        hidden = workspace.hidden_workspace[:route_slots]
        intermediate = workspace.intermediate[:route_slots]
        moe_permute(
            input=x,
            permuted_output=hidden,
            tile_idx_to_mn_limit=t2lim,
            permuted_idx_to_expanded_idx=p2e[:route_slots],
            num_non_exiting_tiles=live,
            max_num_permuted_tokens=route_slots,
            top_k=top_k,
            tile_size=route_tile,
            enable_pdl=runner.enable_pdl,
        )
        epilogue = dict(
            activation_type=runner.activation_type,
            swiglu_alpha=runner.swiglu_alpha,
            swiglu_beta=runner.swiglu_beta,
            swiglu_limit=runner.swiglu_limit,
            situ_beta=runner.situ_beta,
            situ_linear_beta=runner.situ_linear_beta,
        )
        identity = dict(
            activation_type=None,
            swiglu_alpha=DEFAULT_SWIGLU_ALPHA,
            swiglu_beta=DEFAULT_SWIGLU_BETA,
            swiglu_limit=DEFAULT_SWIGLU_LIMIT,
            situ_beta=None,
            situ_linear_beta=None,
        )
        for weight, weight_sf, alpha, src, dst, act in (
            (w1, w1_sf, w1_alpha, hidden, intermediate, epilogue),
            (w2, w2_sf, w2_alpha, intermediate, hidden, identity),
        ):
            _run_grouped_gemm(
                weight=weight,
                weight_sf=weight_sf,
                activations=src,
                tile_idx_to_expert_idx=t2e,
                tile_idx_to_mn_limit=t2lim,
                num_non_exiting_tiles=live,
                alpha=alpha,
                output=dst,
                num_local_experts=self.local_num_experts,
                **act,
                use_fused_finalize=False,
                permuted_idx_to_expanded_idx=None,
                token_final_scales=None,
                enable_pdl=runner.enable_pdl,
                tactic=tile,
            )

        slot = getattr(self, "_slot", None)
        if slot is None or slot.device != ids.device:
            slot = self._slot = torch.arange(top_k, device=ids.device, dtype=ids.dtype)
        owner = ids // (self.global_num_experts // BATCH_INVARIANT_EP_SIZE)
        # Route j joins the partial at the first slot routed to its EP rank.
        first = torch.where(owner[:, :, None] == owner[:, None, :], slot, top_k)
        first = first.amin(dim=-1)
        # One unpermute over (token, slot) rows gives every slot's partial.
        slot_e2p = torch.where(
            first[:, None, :] == slot[None, :, None], e2p[:, None, :], -1
        )
        slot_scales = scales[:, None, :].expand(-1, top_k, -1).contiguous()
        partials = output.new_empty((num_tokens * top_k, output.size(1)))
        moe_unpermute(
            permuted_input=hidden,
            output=partials,
            expanded_idx_to_permuted_idx=slot_e2p,
            topk_scales=slot_scales,
            num_tokens=num_tokens * top_k,
            top_k=top_k,
            enable_pdl=runner.enable_pdl,
        )
        pairs = partials.view(num_tokens, top_k // 2, 2, -1)
        pair_sums = pairs[:, :, 0].float()
        pair_sums += pairs[:, :, 1]
        torch.add(pair_sums[:, 0] + pair_sums[:, 1], pair_sums[:, 2], out=output)
