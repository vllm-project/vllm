# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in gfx950 TP MegaMoE for MiMo-V2.6 routed experts.

All experts are present on each TP rank; only the intermediate dimension is
sharded. Both execution branches return globally reduced routed outputs. The
ordinary MoERunner then reduces the native shared-MLP partial separately, once.
"""

import os
import weakref
from dataclasses import replace

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.utils import replace_parameter
from vllm.platforms.rocm import on_gfx950

logger = init_logger(__name__)
_RUNTIMES: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def validate_mimo_tp_mega_moe(vllm_config: VllmConfig) -> None:
    p = vllm_config.parallel_config
    c = vllm_config.model_config.hf_config
    if not on_gfx950() or not rocm_aiter_ops.is_fused_moe_enabled():
        raise ValueError(
            "aiter_mimo_mega_moe_tp requires gfx950 and AITER MoE enabled."
        )
    if (
        p.tensor_parallel_size != 8
        or p.data_parallel_size != 1
        or p.pipeline_parallel_size != 1
        or p.prefill_context_parallel_size != 1
        or p.decode_context_parallel_size != 1
        or p.enable_expert_parallel
        or p.enable_eplb
        or p.use_ubatching
        or p.use_sequence_parallel_moe
    ):
        raise ValueError(
            "aiter_mimo_mega_moe_tp requires TP8/DP1/PP1, no EP/EPLB/CP/SP/DBO."
        )
    if (
        c.hidden_size != 6144
        or c.moe_intermediate_size != 2048
        or c.n_routed_experts != 384
        or c.num_experts_per_tok != 8
        or getattr(c, "expert_dtype", "fp4") != "fp4"
        or vllm_config.model_config.dtype != torch.bfloat16
    ):
        raise ValueError(
            "aiter_mimo_mega_moe_tp supports MiMo-V2.6-Pro BF16/MXFP4 "
            "routed experts only."
        )
    quant = vllm_config.quant_config
    if quant is None or getattr(quant, "weight_block_size", None) != [128, 128]:
        raise ValueError(
            "aiter_mimo_mega_moe_tp requires block-128 checkpoint weights."
        )
    if getattr(c, "n_shared_experts", None) is not None:
        raise ValueError(
            "aiter_mimo_mega_moe_tp supports MiMo-V2.6 routed experts "
            "without shared experts only."
        )
    if os.getenv("AITER_MEGAMOE_TP_CHECK", "0") == "1":
        raise ValueError(
            "Unset AITER_MEGAMOE_TP_CHECK: its per-layer host synchronization "
            "is incompatible with serving. The adapter asserts watchdog state "
            "on device after every launch, including graph replay."
        )


@torch.no_grad()
def pack_tp_weights(layer: RoutedExperts) -> None:
    """Lossless checkpoint -> separated (16,16) / E8M0-shuffled layout.

    The default V4 AITER layout interleaves gate/up and is NOT MegaTP's layout.
    Load with the existing TP loader, then permute bytes in 8-expert chunks,
    replacing the original storage in place. The distinct interleaved W13
    cache below is retained for fallback, while W2 is shared. GEMM2's 12
    scale columns are padded to 16 with E8M0 1.0, not uninitialized bytes.
    """
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    for weight_name, scale_name in (
        ("w13_weight", "w13_weight_scale"),
        ("w2_weight", "w2_weight_scale"),
    ):
        weight = getattr(layer, weight_name)
        raw = weight.data.view(torch.float4_e2m1fn_x2)
        for start in range(0, raw.shape[0], 8):
            part = raw[start : start + 8]
            part.copy_(shuffle_weight(part, layout=(16, 16)))
        weight.data = raw
        weight.is_shuffled = True
        # Explicitly advertise separated layout to the normal AITER dispatcher.
        weight.gate_up_interleave = False
        scale = getattr(layer, scale_name)
        packed_scale = e8m0_shuffle(
            scale.data.view(torch.float8_e8m0fnu).reshape(-1, scale.shape[-1])
        )
        replace_parameter(layer, scale_name, packed_scale)


class TPFallbackWeights:
    """One-time per-layer GUGU W13 cache; W2/scales alias the MegaTP weights.

    The real gate/up layout difference requires 1,123,024,896 extra bytes per
    original V4-Pro layer per TP rank (W13=1,056,964,608, S1=66,060,288).
    Across 61 layers this is 68,504,518,656 bytes/rank, excluding MTP layers.
    Pay that storage cost instead of permuting ~1 GiB on every fallback call.
    """

    def __init__(self, layer):
        self.w1 = torch.empty_like(layer.w13_weight)
        self.w2 = layer.w2_weight
        self.s1 = torch.empty_like(layer.w13_weight_scale)
        self.s2 = layer.w2_weight_scale
        self.w1.is_shuffled = True
        self.w1.gate_up_interleave = True
        self.storage_bytes = (
            self.w1.numel() * self.w1.element_size()
            + self.s1.numel() * self.s1.element_size()
        )
        self._pack(layer)

    def _pack(self, layer):
        source, scales = layer.w13_weight, layer.w13_weight_scale
        experts, n, k = source.shape
        cols = scales.shape[-1]
        for lo in range(0, experts, 8):
            hi = min(lo + 8, experts)
            # Both layouts already tile K identically when flattened. Move
            # the separated gate/up axis inside the N-tile axis directly;
            # no unshuffle -> shuffle roundtrip or numerical conversion.
            packed = (
                source[lo:hi]
                .view(torch.uint8)
                .view(hi - lo, 2, n // 32, k // 16, 16, 16)
                .permute(0, 2, 1, 3, 4, 5)
                .contiguous()
                .view(hi - lo, n, k)
                .view(torch.float4_e2m1fn_x2)
            )
            self.w1[lo:hi].copy_(packed)
            # E8M0's final axis packs two N halves. Move it into N, and
            # replace it with the gate/up axis, exactly shuffle_scale(GUGU).
            packed_scale = (
                scales[lo * n : hi * n]
                .view(torch.uint8)
                .view(hi - lo, 2, n // 64, cols // 8, 4, 16, 2, 2)
                .permute(0, 2, 7, 3, 4, 5, 6, 1)
                .contiguous()
                .view((hi - lo) * n, cols)
                .view(scales.dtype)
            )
            self.s1[lo * n : hi * n].copy_(packed_scale)


def aiter_tp_partial(layer, weights, x, topk_weights, topk_ids):
    """Complete normal V4 AITER A8W4 path, without shared-expert fusion."""
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.moe_common import GateMode

    return fused_moe(
        x,
        weights.w1,
        weights.w2,
        topk_weights.float(),
        topk_ids.int(),
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        w1_scale=weights.s1,
        w2_scale=weights.s2,
        gate_mode=GateMode.INTERLEAVE.value,
        quant_dtype_a=dtypes.fp8,
        dtype=torch.bfloat16,
        swiglu_limit=getattr(layer, "swiglu_limit", None),
        hidden_pad=0,
        intermediate_pad=0,
        moe_sorting_dispatch_policy=envs.VLLM_ROCM_AITER_MOE_DISPATCH_POLICY,
    )


class MiMoV2TPMegaMoEMethod(Mxfp4MoEMethod):
    supports_pre_processed_weights = False

    def __init__(self, moe):
        # Reuse the complete MXFP4 parameter creation/TP loading contract. Only
        # weight finalization and expert execution differ from normal AITER.
        super().__init__(replace(moe, moe_backend="aiter"))
        self.runtime = None
        self.fallback_weights = None
        self.fallback_cache_bytes = 0
        self.act_dtype = envs.VLLM_ROCM_TP_MEGA_MOE_ACT_DTYPE
        self.comm_dtype = envs.VLLM_ROCM_TP_MEGA_MOE_COMM_DTYPE
        self.ar_gather = envs.VLLM_ROCM_TP_MEGA_MOE_AR_GATHER
        self.capacity_rows = ((moe.max_num_tokens + 7) // 8) * 8
        self.max_rows = min(
            256 if self.act_dtype == "fp8" else 8192, self.capacity_rows
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        from aiter import ActivationType
        from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP, MegaMoeTPConfig

        pack_tp_weights(layer)
        # FP4 at the original 8192-token budget has no reachable fallback:
        # do not retain ~64 GiB/rank of dead W13/S1 caches in that experiment.
        if self.max_rows < self.capacity_rows:
            self.fallback_weights = TPFallbackWeights(layer)
            self.fallback_cache_bytes = self.fallback_weights.storage_bytes
        group = get_tp_group()
        key = (
            id(group.device_group),
            layer.w13_weight.device,
            self.moe.hidden_dim,
            self.moe.intermediate_size_per_partition,
            layer.global_num_experts,
            self.moe.experts_per_token,
            self.max_rows,
            getattr(layer, "swiglu_limit", None),
            self.act_dtype,
            self.comm_dtype,
            self.ar_gather,
        )
        runtime = _RUNTIMES.get(key)
        if runtime is None:
            cfg = MegaMoeTPConfig(
                rank=group.rank_in_group,
                world_size=group.world_size,
                model_dim=self.moe.hidden_dim,
                inter_dim=self.moe.intermediate_size_per_partition,
                experts=layer.global_num_experts,
                topk=self.moe.experts_per_token,
                max_local_tokens=self.max_rows // group.world_size,
                activation=ActivationType.Silu,
                swiglu_limit=getattr(layer, "swiglu_limit", None),
                comm_mode="ar",
                act_dtype=self.act_dtype,
                comm_dtype=self.comm_dtype,
                ar_gather=self.ar_gather,
            )
            runtime = MegaMoeTP(
                cfg,
                group=group.device_group,
                device=layer.w13_weight.device,
                **self._weights(layer),
            )
            # Cover ALL padded/draining/capture shapes, not only nominal batch
            # sizes. One warmup per distinct upstream launch configuration is
            # sufficient: local m is a runtime argument, not a compile constant.
            representatives = {}
            for local in range(1, runtime.max_local_tokens + 1):
                representatives[runtime.launch_config(local)] = local
            runtime.prepare(sorted(representatives.values()))
            _RUNTIMES[key] = runtime
            logger.info(
                "MiMo MegaMoeTP ar: I=%d E=%d K=%d max_rows=%d act=%s comm=%s "
                "gather=%s; fallback=AITER interleaved A8W4 "
                "(cached W13/S1=%d bytes/layer)",
                cfg.inter_dim,
                cfg.experts,
                cfg.topk,
                self.max_rows,
                self.act_dtype,
                self.comm_dtype,
                self.ar_gather,
                self.fallback_cache_bytes,
            )
        self.runtime = runtime
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

    @staticmethod
    def _weights(layer):
        return dict(
            w1=layer.w13_weight,
            w1_scale=layer.w13_weight_scale,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
        )

    def execution_path(self, rows: int) -> str:
        if not 0 <= rows <= self.capacity_rows:
            raise ValueError(
                f"MegaMoeTP received {rows} rows, outside configured padded "
                f"capacity [0, {self.capacity_rows}]."
            )
        if rows == 0:
            return "empty"
        return "mega_tp_ar" if rows <= self.max_rows else "aiter_a8w4"

    def apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    ):
        rows = x.shape[0]
        path = self.execution_path(rows)
        if rows == 0:
            return torch.empty_like(x)
        runtime = self.runtime
        if runtime is None:
            raise RuntimeError("MegaMoeTP weights were not finalized before forward.")
        if path == "aiter_a8w4":
            if self.fallback_weights is None:
                raise RuntimeError(
                    "Reachable AITER fallback weights were not finalized."
                )
            # Normalize reduction ownership across both branches. Runner must
            # NEVER all-reduce these already-global routed outputs again.
            return tensor_model_parallel_all_reduce(
                aiter_tp_partial(
                    layer, self.fallback_weights, x, topk_weights, topk_ids
                )
            )
        runtime.set_weights(**self._weights(layer))
        padded_rows = ((rows + 7) // 8) * 8
        if padded_rows != rows:
            padded_x = x.new_zeros((padded_rows, x.shape[1]))
            padded_ids = topk_ids.new_zeros((padded_rows, topk_ids.shape[1]))
            padded_weights = topk_weights.new_zeros(padded_ids.shape)
            padded_x[:rows].copy_(x)
            padded_ids[:rows].copy_(topk_ids)
            padded_weights[:rows].copy_(topk_weights)
            x, topk_ids, topk_weights = padded_x, padded_ids, padded_weights
        # Own the result: upstream AR may otherwise return a shared IPC view
        # overwritten by the next layer or graph replay of this same runtime.
        out = torch.empty_like(x)
        runtime(x, topk_weights, topk_ids, out=out)
        # Capturable, no host synchronization. A sticky watchdog error makes
        # this device context fail; never continue with invalid outputs or
        # attempt a normal-MoE fallback after a failed collective launch.
        torch._assert_async(
            runtime.error_flag().eq(0), "MegaMoeTP watchdog timeout: output invalid"
        )
        return out[:rows]


class MiMoV2TPMegaMoEExperts(RoutedExperts):
    def _get_quant_method(self, prefix, quant_config, moe_config):
        return MiMoV2TPMegaMoEMethod(moe_config)


class MiMoV2TPMegaMoERunner(MoERunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _fused_output_is_reduced(self) -> bool:
        return True
