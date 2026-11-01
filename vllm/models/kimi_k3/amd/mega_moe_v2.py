# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 latent experts using AITER MegaMoEV2 with TP attention and EP."""

from typing import Any, cast

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.distributed import get_ep_group
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.models.deepseek_v4.amd.mega_moe import DeepseekV4AiterMegaMoEExperts


def validate_config(vllm_config: VllmConfig, config) -> None:
    """Reject layouts that cannot preserve the K3 latent/shared contract."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.platforms.rocm import on_gfx950
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    p = vllm_config.parallel_config
    extra = cast(dict[str, Any], vllm_config.additional_config)
    if (
        not on_gfx950()
        or not rocm_aiter_ops.is_fused_moe_enabled()
        or not p.enable_expert_parallel
        or p.tensor_parallel_size != 8
        or p.pipeline_parallel_size != 1
        or p.data_parallel_size != 1
        or p.prefill_context_parallel_size != 1
        or p.decode_context_parallel_size != 1
        or p.enable_eplb
        or p.enable_dbo
        or p.ubatch_size > 1
        or p.use_sequence_parallel_moe
        or vllm_config.lora_config is not None
        or p.all2all_backend not in ("mori_high_throughput", "mori_low_latency")
        or extra.get("k3_megamoe_tp", "off") != "off"
        or extra.get("decode_backend", "auto") != "auto"
    ):
        raise ValueError(
            "K3 MegaMoEV2 requires gfx950, AITER, DP1/EP8/TP8/PP1 and MoRI; "
            "no TP MegaMoE, specialized decode, EPLB, DBO, CP, SP or LoRA"
        )
    if (
        vllm_config.model_config.dtype != torch.bfloat16
        or (
            config.hidden_size,
            config.routed_expert_hidden_size,
            config.moe_intermediate_size,
            config.num_experts,
            config.num_experts_per_token,
        )
        != (7168, 3584, 3072, 896, 16)
        or config.hidden_act != "situ"
        or config.activation_situ_beta != 4.0
        or config.activation_situ_linear_beta != 25.0
        or config.routed_scaling_factor != 1.0
    ):
        raise ValueError("K3 MegaMoEV2 requires the K3 BF16/MXFP4 SiTUv2 shape")
    if not vllm_config.use_v2_model_runner:
        raise ValueError("K3 MegaMoEV2 requires Model Runner V2")
    cache = vllm_config.cache_config
    scheduler = vllm_config.scheduler_config
    if (
        vllm_config.model_config.max_model_len > 4096
        or scheduler.max_num_seqs > 32
        or scheduler.max_num_batched_tokens > 1024
        or cache.enable_prefix_caching
        or cache.kv_cache_memory_bytes != 8 * 1024**3
        or cache.cache_dtype != "auto"
    ):
        raise ValueError(
            "K3 MegaMoEV2 requires <=4K context/C32/batch1024, 8GiB KV, "
            "auto KV dtype, no prefix caching"
        )
    if not vllm_config.model_config.enforce_eager:
        c = vllm_config.compilation_config
        from vllm import envs

        if (
            c.mode != CompilationMode.NONE
            or vllm_config.attention_config.backend
            not in (None, AttentionBackendEnum.ROCM_AITER_MLA)
            or c.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
            or max(c.cudagraph_capture_sizes or [], default=0) > 128
            or envs.VLLM_ROCM_AITER_MLA_ASM_PADDING != "gluon"
        ):
            raise ValueError(
                "K3 MegaMoEV2 graph requires bounded FULL_DECODE_ONLY, "
                "no Inductor and Gluon MLA"
            )
    spec = vllm_config.speculative_config
    if spec is not None and (
        spec.method != "dspark"
        or spec.attention_backend != AttentionBackendEnum.TRITON_MLA
        or spec.num_speculative_tokens != 3
        or spec.draft_parallel_config.tensor_parallel_size != 8
        or spec.draft_model_config.hf_config.architectures != ["K3DSparkModel"]
    ):
        raise ValueError(
            "K3 MegaMoEV2 supports only TP8 K3 DSpark with 3 draft tokens "
            "and TRITON_MLA draft attention"
        )


class K3MegaMoEV2Experts(DeepseekV4AiterMegaMoEExperts):
    """Reuse EP checkpoint packing/runtime ownership with the K3 SiTUv2 kernel."""

    def __init__(self, vllm_config, config, prefix):
        from torch import nn

        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
            CompressedTensorsConfig,
        )
        from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config

        quant = vllm_config.quant_config
        if isinstance(quant, CompressedTensorsConfig):
            linear = nn.Linear(1, 1, bias=False, device="meta")
            schemes = [
                quant.get_scheme_dict(linear, f"{prefix}.0.{name}")
                for name in ("w1", "w2", "w3")
            ]
            scheme = schemes[0]
            if scheme is None or any(s != scheme for s in schemes):
                raise ValueError("K3 MegaMoEV2 requires matching expert schemes")
            activation = scheme.get("input_activations")
            if not quant._is_mxfp4(scheme.get("weights")) or (
                activation is not None
                and not quant._is_mxfp4_w4a8_fp8(scheme.get("weights"), activation)
            ):
                raise ValueError(
                    "K3 MegaMoEV2 requires MXFP4 weights and FP8 activations"
                )
        elif not isinstance(quant, Mxfp4Config):
            raise ValueError(
                "K3 MegaMoEV2 requires the canonical MXFP4 checkpoint loader"
            )
        super().__init__(
            vllm_config,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_token,
            hidden_size=config.routed_expert_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            swiglu_limit=None,
            fuse_shared_expert=False,
            prefix=prefix,
        )

    def get_runtime(self, max_tok_per_rank):
        # DP1 does not construct an all-to-all manager automatically. EP V2
        # still requires the same MoRI heap initialization used by DP+EP.
        group = get_ep_group()
        communicator = cast(DeviceCommunicatorBase, group.device_communicator)
        if communicator.all2all_manager is None:
            from vllm.config import get_current_vllm_config
            from vllm.distributed.device_communicators.all2all import MoriAll2AllManager

            communicator.all2all_manager = MoriAll2AllManager(
                group.cpu_group,
                get_current_vllm_config().parallel_config.all2all_backend,
            )
        # Keep the shared FP8 MLP separate: it has the full hidden dimension,
        # whereas routed experts operate in K3's latent dimension.
        key = (
            "k3_situv2_a8w4",
            id(get_ep_group().cpu_group),
            torch.accelerator.current_device_index(),
            self.num_experts,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            max_tok_per_rank,
        )
        runtime = self._runtime_cache.get(key)
        if runtime is None:
            from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

            runtime = MegaMoEV2(
                rank=self.ep_rank,
                world_size=self.ep_size,
                model_dim=self.hidden_size,
                inter_dim=self.intermediate_size,
                experts=self.num_experts,
                topk=self.top_k,
                quant="a8w4",
                w1=self.w13_weight,
                w1_scale=self.w13_weight_scale,
                w2=self.w2_weight,
                w2_scale=self.w2_weight_scale,
                max_tok_per_rank=max_tok_per_rank,
                act="situv2",
                situ_beta=4.0,
                situ_linear_beta=25.0,
                stage2_p2p_quant="none",
            )
            self._runtime_cache[key] = runtime
        return runtime
