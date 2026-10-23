# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in K3 TP8 A8W4 model integration and matched split-MoE baseline."""

from typing import Any, cast

import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.models.kimi_k3.amd.latent_moe_runner import ROCmLatentMoERunner
from vllm.models.kimi_k3.amd.mega_moe_tp import (
    K3MegaMoETPPool,
    K3TPWeights,
    k3_tp_tail_partial,
)


def model_pool(vllm_config, config):
    """Create one runtime owner per model; reject unsupported deployments early."""
    extra = vllm_config.additional_config
    mode = extra.get("k3_megamoe_tp", "off") if isinstance(extra, dict) else "off"
    if mode == "off":
        return None
    if mode not in ("mega", "split"):
        raise ValueError("k3_megamoe_tp must be off, mega, or split")
    if (
        mode == "mega"
        and max(vllm_config.compilation_config.cudagraph_capture_sizes or [], default=0)
        > 256
    ):
        raise ValueError("K3 MegaMoE graph capacities must not exceed 256")
    p = vllm_config.parallel_config
    if (
        p.tensor_parallel_size != 8
        or p.pipeline_parallel_size != 1
        or p.data_parallel_size != 1
        or p.enable_expert_parallel
        or p.enable_dbo
        or p.ubatch_size > 1
        or p.prefill_context_parallel_size != 1
        or p.decode_context_parallel_size != 1
        or p.use_sequence_parallel_moe
        or vllm_config.speculative_config is not None
        or vllm_config.lora_config is not None
        or extra.get("decode_backend", "auto") != "auto"
    ):
        raise ValueError("K3 MegaMoE requires TP8/PP1/DP1, no EP/DBO/MTP/LoRA")
    if not vllm_config.model_config.enforce_eager:
        raise ValueError(
            "K3 MegaMoE model integration currently requires enforce_eager=True; "
            "whole-model graph execution failed quality validation"
        )
    if (
        vllm_config.model_config.dtype != torch.bfloat16
        or config.hidden_size != 7168
        or config.routed_expert_hidden_size != 3584
        or config.moe_intermediate_size != 3072
        or config.num_experts != 896
        or config.num_experts_per_token != 16
        or config.hidden_act != "situ"
        or config.activation_situ_beta != 4.0
        or config.activation_situ_linear_beta != 25.0
        or config.routed_scaling_factor != 1.0
        or not config.num_shared_experts
    ):
        raise ValueError("unsupported K3 shape, activation, shared or routing scale")
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        raise ValueError("K3 MegaMoE requires gfx950")
    pool = K3MegaMoETPPool(group=get_tp_group().cpu_group)
    pool.model_mode = mode
    return pool


def pack_split_weights(w1, s1, w2, s2):
    """Pack canonical raw MXFP4 bytes for the FP8 split kernels' GUGU layout."""
    from aiter.ops.shuffle import shuffle_scale, shuffle_weight

    e, _, half_h = w1.shape
    i = w2.shape[-1] * 2
    fp4, e8m0 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu
    return K3TPWeights(
        shuffle_weight(w1.view(fp4), is_guinterleave=True, gate_up=True),
        shuffle_scale(
            s1.view(-1, half_h // 16).view(e8m0),
            experts_cnt=e,
            is_guinterleave=True,
            gate_up=True,
        ),
        shuffle_weight(w2.view(fp4), is_guinterleave=True, gate_up=False),
        shuffle_scale(
            s2.view(-1, i // 32).view(e8m0),
            experts_cnt=e,
            is_guinterleave=True,
            gate_up=False,
        ),
    )


def unpack_tp_weights(weights):
    """Recover canonical bytes from TP's standard 16x16 and E8M0 packing."""

    def weight(t):
        e, n, k = t.shape
        return (
            t.view(torch.uint8)
            .view(e, n // 16, k // 32, 2, 16, 16)
            .permute(0, 1, 4, 2, 3, 5)
            .contiguous()
            .view(e, n, k)
        )

    def scale(t, rows, cols):
        sm, sn = (rows + 255) // 256 * 256, (cols + 7) // 8 * 8
        return (
            t.view(torch.uint8)
            .view(sm // 32, sn // 8, 4, 16, 2, 2)
            .permute(0, 5, 3, 1, 4, 2)
            .contiguous()
            .view(sm, sn)[:rows, :cols]
            .contiguous()
        )

    e, n, h2 = weights.w1.shape
    i2 = weights.w2.shape[-1]
    return (
        weight(weights.w1),
        scale(weights.w1_scale, e * n, h2 // 16),
        weight(weights.w2),
        scale(weights.w2_scale, e * h2 * 2, i2 // 16),
    )


def split_a8w4(weights, x, topk_weights, topk_ids):
    """Split A8W4; TP fallback repacks temporarily, split mode stores GUGU once."""
    import aiter
    from aiter import dtypes
    from aiter.fused_moe import fused_moe

    if x.shape[0] == 0:
        return x.clone()
    if not getattr(weights.w1, "gate_up_interleave", False):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("TP-to-split repacking must run outside graph capture")
        weights = pack_split_weights(*unpack_tp_weights(weights))
    result = fused_moe(
        x.contiguous(),
        weights.w1,
        weights.w2,
        topk_weights.float().contiguous(),
        topk_ids.int().contiguous(),
        activation=aiter.ActivationType.Situv2,
        quant_type=aiter.QuantType.per_1x32,
        w1_scale=weights.w1_scale,
        w2_scale=weights.w2_scale,
        dtype=torch.bfloat16,
        beta=4.0,
        linear_beta=25.0,
        gate_mode="interleave",
        quant_dtype_a=dtypes.fp8,
        quant_dtype_a2=dtypes.fp8,
    )
    return tensor_model_parallel_all_reduce(result)


class K3TPMoEMethod(Mxfp4MoEMethod):
    """Reuse the canonical checkpoint loader, replace only packing/execution."""

    def __init__(self, moe, pool):
        FusedMoEMethodBase.__init__(self, moe)
        self.weight_dtype = "mxfp4"
        self.pool = pool
        self.weights = None

    @property
    def is_monolithic(self):
        return False

    @property
    def supports_eplb(self):
        return False

    @property
    def skip_forward_padding(self):
        return True

    @property
    def has_unpadded_output(self):
        return False

    def maybe_roundup_sizes(
        self,
        hidden_size,
        intermediate_size_per_partition,
        act_dtype,
        moe_parallel_config,
    ):
        p = moe_parallel_config
        if (hidden_size, intermediate_size_per_partition) != (3584, 384) or (
            p.use_ep or p.dp_size != 1 or p.pcp_size != 1 or p.sp_size != 1
        ):
            raise ValueError("K3 MegaMoE requires unpadded TP8 latent experts")
        return hidden_size, intermediate_size_per_partition

    def get_fused_moe_quant_config(self, layer):
        return None

    def process_weights_after_loading(self, layer):
        if self.weights is not None:
            raise RuntimeError("K3 MegaMoE weight reload is not supported")
        raw = (
            layer.w13_weight.data,
            layer.w13_weight_scale.data,
            layer.w2_weight.data,
            layer.w2_weight_scale.data,
        )
        packed = (
            pack_split_weights(*raw)
            if self.pool.model_mode == "split"
            else K3TPWeights.from_checkpoint(*raw, gate_up_layout="separated")
        )
        # Replace raw parameters; split fallback consumes exactly this layout.
        for name, value in zip(
            ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"),
            (packed.w1, packed.w1_scale, packed.w2, packed.w2_scale),
        ):
            setattr(layer, name, nn.Parameter(value, requires_grad=False))
        self.weights = packed
        if self.pool.model_mode == "mega" and self.pool._runtime is None:
            self.pool.prepare(packed, range(1, 257))

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        if self.weights is None:
            raise RuntimeError("K3 MegaMoE weights are not loaded")

        def fallback(x, w, ids):
            return split_a8w4(self.weights, x, w, ids)

        if self.pool.model_mode == "split":
            return fallback(x, topk_weights, topk_ids)
        return self.pool.forward(
            self.weights, x, topk_weights, topk_ids, fallback=fallback
        )


class K3TPRoutedExperts(RoutedExperts):
    def __init__(self, *args, mega_moe_pool, **kwargs):
        self.mega_moe_pool = mega_moe_pool
        super().__init__(*args, **kwargs)

    def _get_quant_method(self, prefix, quant_config, moe_config):
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
            CompressedTensorsConfig,
        )
        from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config

        if isinstance(quant_config, CompressedTensorsConfig):
            quant_config._add_fused_moe_to_target_scheme_map()
            schemes = [
                quant_config.get_scheme_dict(self, f"{prefix}.0.{projection}")
                for projection in (
                    self.ckpt_gate_proj_name,
                    self.ckpt_up_proj_name,
                    self.ckpt_down_proj_name,
                )
            ]
            scheme = schemes[0]
            if scheme is None or any(s != scheme for s in schemes):
                raise ValueError("K3 MegaMoE requires matching MXFP4 expert schemes")
            activation = scheme.get("input_activations")
            if not quant_config._is_mxfp4(scheme.get("weights")) or (
                activation is not None
                and not quant_config._is_mxfp4_w4a8_fp8(
                    scheme.get("weights"), activation
                )
            ):
                raise ValueError(
                    "K3 MegaMoE requires MXFP4 weights and FP8 activations"
                )
        elif not isinstance(quant_config, Mxfp4Config):
            raise ValueError("K3 MegaMoE requires the canonical MXFP4 weight loader")
        return K3TPMoEMethod(moe_config, self.mega_moe_pool)


class K3TPMoERunner(ROCmLatentMoERunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._tail_shardable or self._shared_experts is None:
            raise ValueError("K3 MegaMoE requires a sharded latent tail with shared")
        # Serialized shared work is required by the persistent whole-GPU kernel.
        self._shared_experts._stream = None
        if get_current_vllm_config().parallel_config.enable_dbo:
            raise ValueError("K3 MegaMoE cannot overlap ubatches")

    @property
    def _fused_output_is_reduced(self):
        return True

    def forward(
        self, hidden_states, router_logits, input_ids=None, shared_experts_input=None
    ):
        return self._fused_forward(
            hidden_states, router_logits, input_ids, shared_experts_input
        )

    def _shard_up_proj_tail(self, fused_output, shared_output, trunc_size):
        transform = cast(Any, self.routed_output_transform)
        weight = transform.up_proj.weight
        rank = get_tensor_model_parallel_rank()
        if not self._up_proj_preshard:
            weight = weight.chunk(8)[rank]
        output = k3_tp_tail_partial(
            fused_output,
            shared_output,
            weight,
            rank=rank,
            world_size=8,
            norm=transform.norm,
        )
        return self._maybe_reduce_final_output(
            output, trunc_size, output_is_reduced=False
        )
