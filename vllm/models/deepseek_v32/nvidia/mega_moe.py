# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 x NVFP4 DeepGEMM MegaMoE for DSA models (DeepSeek V3.2 / GLM-5.2).

A single DeepGEMM kernel fuses EP dispatch, both expert GEMMs, SwiGLU and EP
combine, consuming NVFP4 (packed E2M1 + per-16-element E4M3 scales) activations
and weights from a PyTorch symmetric-memory buffer. Targets ModelOpt NVFP4
checkpoints such as ``nvidia/GLM-5.2-NVFP4``: the per-(expert, projection)
``weight_scale_2`` tensors are passed to the kernel as per-expert alphas, and
activations are quantized dynamically (``input_scale`` is not needed).

Enable with ``--kernel-config '{"moe_backend": "deep_gemm_mega_moe"}'`` plus
``--enable-expert-parallel``. The kernel's combine already sums expert outputs
across the EP group, so tensor parallelism must be 1 (deploy with DP x EP).
"""

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    fused_grouped_topk,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2MLP,
    _get_moe_router_dtype,
)
from vllm.model_executor.utils import set_weight_attrs

from .ops.prepare_megamoe_nvfp4 import prepare_megamoe_nvfp4_inputs


class DeepseekV32MegaMoEExperts(nn.Module):
    """Routed experts backed by the DeepGEMM NVFP4 x NVFP4 Mega MoE kernel.

    Owns the raw ModelOpt NVFP4 checkpoint parameters, transforms them into
    the MegaMoE layout after loading, and stages quantized activations plus
    routing metadata into a cached symmetric-memory buffer per forward.
    """

    _symm_buffer_cache: dict[tuple[int, int, int, int, int, int, int], object] = {}

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        num_experts: int,
        num_local_experts: int,
        experts_start_idx: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        prefix: str = "",
    ):
        super().__init__()
        self.prefix = prefix
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_start_idx = experts_start_idx
        self.experts_end_idx = experts_start_idx + num_local_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        if hidden_size % 256 != 0 or intermediate_size % 256 != 0:
            raise ValueError(
                "DeepGEMM NVFP4 MegaMoE requires hidden and intermediate "
                "sizes to be multiples of 256."
            )

        # ModelOpt NVFP4 checkpoint parameters (per local expert):
        # packed E2M1 weights, per-16-element E4M3 block scales, and
        # per-(expert, projection) FP32 ``weight_scale_2`` / ``input_scale``.
        weight_attrs = {"weight_loader": self.weight_loader}
        self.w13_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_size,
                hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight, weight_attrs)

        self.w13_weight_scale = nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_size,
                hidden_size // 16,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight_scale, weight_attrs)

        self.w2_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                hidden_size,
                intermediate_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight, weight_attrs)

        self.w2_weight_scale = nn.Parameter(
            torch.zeros(
                num_local_experts,
                hidden_size,
                intermediate_size // 16,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight_scale, weight_attrs)

        # Column 0 holds the gate projection's scale, column 1 the up projection's
        self.w13_weight_scale_2 = nn.Parameter(
            torch.ones(num_local_experts, 2, dtype=torch.float32),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight_scale_2, weight_attrs)
        self.w2_weight_scale_2 = nn.Parameter(
            torch.ones(num_local_experts, dtype=torch.float32),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight_scale_2, weight_attrs)

        # Calibrated per-tensor activation scales. Unused: activations are
        # quantized dynamically (per-16-element amax), but the parameters must
        # exist to absorb the checkpoint entries.
        self.w13_input_scale = nn.Parameter(
            torch.ones(num_local_experts, 2, dtype=torch.float32),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_input_scale, weight_attrs)
        self.w2_input_scale = nn.Parameter(
            torch.ones(num_local_experts, dtype=torch.float32),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_input_scale, weight_attrs)

        self._transformed_l1_weights: tuple[torch.Tensor, torch.Tensor] | None = None
        self._transformed_l2_weights: tuple[torch.Tensor, torch.Tensor] | None = None
        self._l1_alphas: torch.Tensor | None = None
        self._l2_alphas: torch.Tensor | None = None

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        if not (self.experts_start_idx <= expert_id < self.experts_end_idx):
            return False if return_success else None
        local_expert_id = expert_id - self.experts_start_idx

        if shard_id in ("w1", "w3"):
            if "w13_" not in weight_name:
                return False if return_success else None
            proj_idx = 0 if shard_id == "w1" else 1
            if param.data.dim() == 2:
                # Scalar per (expert, projection): weight_scale_2 / input_scale
                param.data[local_expert_id, proj_idx].copy_(loaded_weight.reshape(()))
            else:
                shard_offset = proj_idx * self.intermediate_size
                expert_data = param.data[local_expert_id].narrow(
                    0, shard_offset, self.intermediate_size
                )
                if expert_data.shape != loaded_weight.shape:
                    raise ValueError(
                        f"NVFP4 MegaMoE expert weight shape mismatch for "
                        f"{weight_name}: parameter shard {tuple(expert_data.shape)} "
                        f"vs checkpoint {tuple(loaded_weight.shape)}"
                    )
                expert_data.copy_(loaded_weight)
        elif shard_id == "w2":
            if "w2_" not in weight_name:
                return False if return_success else None
            if param.data.dim() == 1:
                param.data[local_expert_id].copy_(loaded_weight.reshape(()))
            else:
                expert_data = param.data[local_expert_id]
                if expert_data.shape != loaded_weight.shape:
                    raise ValueError(
                        f"NVFP4 MegaMoE expert weight shape mismatch for "
                        f"{weight_name}: parameter shard {tuple(expert_data.shape)} "
                        f"vs checkpoint {tuple(loaded_weight.shape)}"
                    )
                expert_data.copy_(loaded_weight)
        else:
            raise ValueError(f"Unsupported expert shard id: {shard_id}")

        return True if return_success else None

    def _check_runtime_supported(self) -> None:
        device = self.w13_weight.device if self.w13_weight is not None else "cuda"
        if torch.cuda.get_device_capability(device)[0] != 10:
            raise NotImplementedError("DeepGEMM MegaMoE requires SM100 GPUs.")

    @staticmethod
    def _pack_e4m3_sf(sf: torch.Tensor) -> torch.Tensor:
        # (E, n, k/16) E4M3 -> (E, n, k/64) int32 (4 scales per int32 along K),
        # then TMA-aligned MN-major as the kernel expects
        packed = sf.data.view(torch.uint8).contiguous().view(torch.int32)
        return packed.transpose(-1, -2).contiguous().transpose(-1, -2)

    def finalize_weights(self) -> None:
        if self._transformed_l1_weights is not None:
            return

        self._check_runtime_supported()
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()

        w13_scale = self._pack_e4m3_sf(self.w13_weight_scale)
        w2_scale = self._pack_e4m3_sf(self.w2_weight_scale)
        self._transformed_l1_weights, self._transformed_l2_weights = (
            deep_gemm.transform_weights_for_mega_moe(
                (self.w13_weight.data.view(torch.int8).contiguous(), w13_scale),
                (self.w2_weight.data.view(torch.int8).contiguous(), w2_scale),
            )
        )

        # Per-expert kernel alphas: the checkpoint's second-level weight scales.
        # L1 gets separate gate/up factors (applied before SwiGLU); L2's factor
        # is applied before the combine write-back. Activation quantization is
        # dynamic on both GEMM inputs, so no activation scale is folded in.
        self._l1_alphas = self.w13_weight_scale_2.data.contiguous()
        self._l2_alphas = self.w2_weight_scale_2.data.contiguous()

        # Drop the original loader-side parameters; the kernel only consumes
        # the transformed views above.
        self.w13_weight = None
        self.w13_weight_scale = None
        self.w2_weight = None
        self.w2_weight_scale = None
        self.w13_weight_scale_2 = None
        self.w2_weight_scale_2 = None
        self.w13_input_scale = None
        self.w2_input_scale = None

    def get_symm_buffer(self):
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()

        group = get_ep_group().device_group
        device = torch.accelerator.current_device_index()
        key = (
            id(group),
            device,
            self.num_experts,
            self.max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
        )
        symm_buffer = self._symm_buffer_cache.get(key)
        if symm_buffer is None:
            symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
                group,
                self.num_experts,
                self.max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                mma_type="fp4xfp4",
            )
            self._symm_buffer_cache[key] = symm_buffer
        return symm_buffer

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.shape[0] > self.max_num_tokens:
            raise ValueError(
                f"NVFP4 MegaMoE got {hidden_states.shape[0]} tokens, but the "
                f"symmetric buffer was sized for {self.max_num_tokens}."
            )
        y = torch.empty_like(hidden_states, dtype=torch.bfloat16)

        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()

        symm_buffer = self.get_symm_buffer()
        num_tokens = hidden_states.shape[0]
        is_padding = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            is_padding = get_forward_context().is_padding
            if is_padding is not None:
                is_padding = is_padding[:num_tokens]

        prepare_megamoe_nvfp4_inputs(
            hidden_states,
            topk_weights,
            topk_ids,
            symm_buffer.x[:num_tokens],
            symm_buffer.x_sf[:num_tokens],
            symm_buffer.topk_idx[:num_tokens],
            symm_buffer.topk_weights[:num_tokens],
            is_padding=is_padding,
        )

        # This method must have been already called during the weight loading phase.
        # We call it again here to cover the dummy weight loading case.
        self.finalize_weights()

        assert self._transformed_l1_weights is not None
        assert self._transformed_l2_weights is not None
        deep_gemm.fp4_fp4_mega_moe(
            y,
            self._transformed_l1_weights,
            self._transformed_l2_weights,
            symm_buffer,
            l1_alphas=self._l1_alphas,
            l2_alphas=self._l2_alphas,
        )
        return y


DeepseekV32MegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]


class DeepseekV32MegaMoE(nn.Module):
    """Drop-in MoE block for DSA layers using the NVFP4 MegaMoE kernel.

    Mirrors ``DeepseekV2MoE``'s interface (gate + shared experts + routed
    experts) but computes routing explicitly and feeds the fused kernel.
    The kernel output is already combined across the EP group, so no further
    all-reduce is required (TP must be 1; deploy with DP x EP).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        prefix: str = "",
    ):
        super().__init__()
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError(
                "DeepGEMM NVFP4 MegaMoE requires tensor_parallel_size == 1; "
                "deploy with data parallelism + expert parallelism instead."
            )
        if not parallel_config.enable_expert_parallel:
            raise NotImplementedError(
                "DeepGEMM NVFP4 MegaMoE requires expert parallel. Enable it "
                "with --enable-expert-parallel, or pick a different moe backend."
            )
        if parallel_config.enable_eplb:
            raise NotImplementedError(
                "DeepGEMM NVFP4 MegaMoE does not support EPLB yet."
            )
        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}.")

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts
        self.top_k = config.num_experts_per_tok
        self.num_expert_group = getattr(config, "n_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)
        self.renormalize = config.norm_topk_prob
        self.scoring_func = getattr(config, "scoring_func", "softmax")
        if getattr(config, "topk_method", None) != "noaux_tc":
            raise NotImplementedError(
                "DeepGEMM NVFP4 MegaMoE currently supports noaux_tc routing only."
            )

        self.router_dtype = _get_moe_router_dtype(config)
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            params_dtype=self.router_dtype,
            out_dtype=self.router_dtype,
            force_fp32_compute=self.router_dtype == torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32)
        )

        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        ep_group = get_ep_group()
        ep_size = ep_group.world_size
        ep_rank = ep_group.rank_in_group
        assert self.n_routed_experts % ep_size == 0, (
            f"n_routed_experts={self.n_routed_experts} must be divisible by "
            f"ep_size={ep_size}."
        )
        num_local_experts = self.n_routed_experts // ep_size
        self.experts = DeepseekV32MegaMoEExperts(
            vllm_config,
            num_experts=self.n_routed_experts,
            num_local_experts=num_local_experts,
            experts_start_idx=ep_rank * num_local_experts,
            top_k=self.top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            prefix=f"{prefix}.experts",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        org_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, org_shape[-1])

        router_logits, _ = self.gate(hidden_states)
        topk_weights, topk_ids = fused_grouped_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            e_score_correction_bias=self.gate.e_score_correction_bias.data,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
        )

        final_hidden_states = self.experts(hidden_states, topk_weights, topk_ids)
        if self.shared_experts is not None:
            final_hidden_states = final_hidden_states + self.shared_experts(
                hidden_states
            )
        return final_hidden_states.view(org_shape)

    def finalize_mega_moe_weights(self) -> None:
        self.experts.finalize_weights()
