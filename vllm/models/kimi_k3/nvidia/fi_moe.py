# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer NVFP4 MegaMoE experts for Kimi K3."""

from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v4.nvidia.fi_moe import (
    apply_mega_moe_routing_preprocess,
    resolve_mega_moe_is_padding,
)
from vllm.utils.flashinfer_moe_ep import (
    ensure_fi_moe_ep_runtime,
    make_fi_moe_ep_bootstrap,
)
from vllm.utils.math_utils import cdiv

from .model import KimiK3MegaMoEExperts


class KimiK3FlashInferMegaMoEExperts(KimiK3MegaMoEExperts):
    """Consume ModelOpt NVFP4 routed weights without requantization."""

    def __init__(self, vllm_config: VllmConfig, **kwargs: Any) -> None:
        super().__init__(vllm_config, **kwargs)
        self._vllm_config = vllm_config
        parallel = vllm_config.parallel_config
        if parallel.enable_eplb:
            raise NotImplementedError("FlashInfer MegaMoE does not support EPLB.")
        if (
            parallel.pipeline_parallel_size == 1
            and parallel.enable_expert_parallel
            and parallel.tensor_parallel_size > 1
        ):
            self.max_num_tokens = cdiv(
                self.max_num_tokens, parallel.tensor_parallel_size
            )

        def param(shape: tuple[int, ...], dtype: torch.dtype) -> nn.Parameter:
            value = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
            set_weight_attrs(value, {"weight_loader": self.weight_loader})
            return value

        ne, hidden, inter = (
            self.num_local_experts,
            self.hidden_size,
            self.intermediate_size,
        )
        self.w13_weight_scale = param(
            (ne, 2 * inter, hidden // 16), torch.float8_e4m3fn
        )
        self.w2_weight_scale = param((ne, hidden, inter // 16), torch.float8_e4m3fn)
        self.w13_weight_scale.quant_method = "block"
        self.w2_weight_scale.quant_method = "block"
        self.w13_weight_scale_2 = param((ne, 2), torch.float32)
        self.w2_weight_scale_2 = param((ne,), torch.float32)
        self.w13_input_scale = param((ne, 2), torch.float32)
        self.w2_input_scale = param((ne,), torch.float32)
        self.register_buffer("_fc1_alpha", None, persistent=False)
        self.register_buffer("_fc2_alpha", None, persistent=False)
        self.register_buffer("_fc1_norm_const", None, persistent=False)
        self._flashinfer_layer: nn.Module | None = None

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        if (
            "w13_weight_scale_2" not in weight_name
            and "w13_input_scale" not in weight_name
        ):
            return super().weight_loader(
                param, loaded_weight, weight_name, shard_id, expert_id, return_success
            )
        if shard_id not in ("w1", "w3"):
            raise ValueError(f"Unsupported FC1 metadata shard id: {shard_id}")
        local_ids = self._map_global_expert_id(expert_id)
        for local_id in local_ids:
            param.data[local_id, 0 if shard_id == "w1" else 1].copy_(loaded_weight)
        return bool(local_ids) if return_success else None

    def _check_runtime_supported(self) -> None:
        if torch.cuda.get_device_capability(self.w13_weight.device)[0] != 10:
            raise NotImplementedError("FlashInfer NVFP4 MegaMoE requires SM100/SM103.")
        if self.hidden_size % 64 or self.intermediate_size % 64:
            raise ValueError(
                "FlashInfer NVFP4 MegaMoE dimensions must be multiples of 64."
            )

    def finalize_weights(self) -> None:
        if self._flashinfer_layer is not None:
            return
        self._check_runtime_supported()
        if self._vllm_config.load_config.load_format == "dummy":
            self.w13_weight_scale_2[:, 1].copy_(self.w13_weight_scale_2[:, 0])
        if not torch.equal(
            self.w13_weight_scale_2[:, 0], self.w13_weight_scale_2[:, 1]
        ):
            raise ValueError(
                "FlashInfer NVFP4 MegaMoE requires identical gate/up global scales."
            )

        from flashinfer.moe_ep import (
            FleetParams,
            MegaConfig,
            MoEEpLayer,
            MoEWeightPack,
            Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
        )

        if (
            "situ_beta"
            not in Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig.__dataclass_fields__
        ):
            raise RuntimeError("Kimi K3 requires FlashInfer MegaMoE with SiTU support.")
        ensure_fi_moe_ep_runtime(self._vllm_config)
        ep_group = get_ep_group()
        a13_scale = self.w13_input_scale.max().float()
        a2_scale = self.w2_input_scale.max().float()
        if ep_group.world_size > 1:
            for scale in (a13_scale, a2_scale):
                torch.distributed.all_reduce(
                    scale,
                    op=torch.distributed.ReduceOp.MAX,
                    group=ep_group.device_group,
                )
        if not all(
            torch.isfinite(s).all() and (s > 0).all() for s in (a13_scale, a2_scale)
        ):
            raise ValueError("NVFP4 activation scales must be positive and finite.")

        self._fc1_alpha = (
            self.w13_weight_scale_2[:, 0].float() * a13_scale
        ).contiguous()
        self._fc2_alpha = (self.w2_weight_scale_2.float() * a2_scale).contiguous()
        self._fc1_norm_const = torch.ones_like(self._fc1_alpha) / a2_scale
        config = Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=self.intermediate_size,
            top_k=self.top_k,
            activation=self.activation,
            situ_beta=self.activation_beta,
            situ_linear_beta=self.activation_linear_beta,
            input_norm_const=float(a13_scale.reciprocal().item()),
            enable_in_kernel_fc2_reduce=True,
            knobs={
                "cluster_shape_mnk": (2, 1, 1),
                "group_hint": 512,
                "epi_flag_batch": (2, 4),
                "load_balance_mode": "atomic_counter",
                "mma_tiler_mnk": (256, 128, 256),
                "flag_batch": 4,
                "token_back_mode": "epi_warps",
                "in_kernel_fc2_reduce": True,
            },
        )
        self._flashinfer_layer = MoEEpLayer(
            bootstrap=make_fi_moe_ep_bootstrap(),
            fleet_params=FleetParams(
                num_experts=self.num_experts,
                max_tokens_per_rank=self.max_num_tokens,
                token_hidden_size=self.hidden_size,
            ),
            weights=MoEWeightPack(
                w13=self.w13_weight.data,
                w2=self.w2_weight.data,
                w13_scale=self.w13_weight_scale.data,
                w2_scale=self.w2_weight_scale.data,
            ),
            backend=MegaConfig(megakernel=config),
        )
        self._flashinfer_layer._ensure_workspace()
        self._drop_raw_mega_weights()
        self.w13_weight_scale_2 = None
        self.w2_weight_scale_2 = None
        self.w13_input_scale = None
        self.w2_input_scale = None

    def get_expert_weights(self) -> list[torch.Tensor]:
        raise NotImplementedError("FlashInfer MegaMoE does not support EPLB.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        activation_clamp: float | None,
        fast_math: bool = True,
    ) -> torch.Tensor:
        from flashinfer.moe_ep import MoEEpTensors

        if activation_clamp is not None:
            raise ValueError("SiTU does not support activation_clamp.")
        self.synchronize_first_launch()
        if hidden_states.shape[0] > self.max_num_tokens:
            raise ValueError("Kimi K3 MegaMoE input exceeds its token capacity.")
        topk_ids = apply_mega_moe_routing_preprocess(
            topk_ids,
            is_padding=resolve_mega_moe_is_padding(hidden_states.shape[0]),
        )
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)
        self.finalize_weights()
        assert self._flashinfer_layer is not None
        return self._flashinfer_layer(
            MoEEpTensors(
                hidden_states=hidden_states,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                fc1_alpha=self._fc1_alpha,
                fc2_alpha=self._fc2_alpha,
                fc1_norm_const=self._fc1_norm_const,
            )
        )


KimiK3FlashInferMegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
