# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Sonic MoE integration for Hopper GPUs.

Sonic MoE uses swiglu format: x[::2] * silu(x[1::2]) (even * silu(odd))
vLLM uses silu_and_mul: silu(x[:d]) * x[d:] (first_half * second_half)

Weight permutation is required during loading to convert between formats.
See: https://github.com/Dao-AILab/sonic-moe/issues/12
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

_sonicmoe_available: bool | None = None


def _check_sonicmoe_available() -> bool:
    global _sonicmoe_available
    if _sonicmoe_available is not None:
        return _sonicmoe_available

    try:
        import sonicmoe  # noqa: F401

        _sonicmoe_available = True
        logger.info("Sonic MoE is available")
    except ImportError:
        _sonicmoe_available = False
        logger.debug("Sonic MoE not available: sonicmoe package not installed")

    return _sonicmoe_available


def _is_hopper_gpu() -> bool:
    if not current_platform.is_cuda():
        return False
    # Hopper is SM90 (compute capability 9.0)
    return current_platform.has_device_capability(90)


def is_sonic_moe_supported() -> bool:
    if not _check_sonicmoe_available():
        return False
    if not _is_hopper_gpu():
        logger.debug("Sonic MoE requires Hopper GPU (H100/H200)")
        return False
    return True


def is_valid_sonic_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_experts: int,
    top_k: int,
) -> bool:
    if not is_sonic_moe_supported():
        return False

    if not hidden_states.is_contiguous():
        logger.debug("Sonic MoE: hidden_states not contiguous")
        return False

    if not w1.is_contiguous() or not w2.is_contiguous():
        logger.debug("Sonic MoE: weights not contiguous")
        return False

    if top_k > 16:
        logger.debug("Sonic MoE: top_k > 16 not optimized")
        return False

    supported_dtypes = {torch.float16, torch.bfloat16}
    if hidden_states.dtype not in supported_dtypes:
        logger.debug("Sonic MoE: unsupported dtype %s", hidden_states.dtype)
        return False

    return True


def permute_weights_for_sonic(w: torch.Tensor) -> torch.Tensor:
    """
    Permute weights from vLLM's silu_and_mul format to Sonic's swiglu format.

    vLLM format: [first_half, second_half] -> silu(first_half) * second_half
    Sonic format: [interleaved] -> even * silu(odd)

    Conversion: rearrange(W, "E (two N) K -> E (N two) K", two=2)

    Reference: https://github.com/Dao-AILab/sonic-moe/issues/12
    """
    E, two_N, K = w.shape
    N = two_N // 2
    w_reshaped = w.view(E, 2, N, K)
    w_permuted = w_reshaped.permute(0, 2, 1, 3)
    return w_permuted.reshape(E, two_N, K).contiguous()


class SonicMoeExperts(mk.FusedMoEPermuteExpertsUnpermute):
    """
    Sonic MoE experts implementation for Hopper GPUs.

    Uses Sonic MoE's optimized kernels for up/down projections.
    Requires weight permutation for swiglu compatibility.
    """

    def __init__(
        self,
        out_dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig = FUSED_MOE_UNQUANTIZED_CONFIG,
        weights_prepermuted: bool = False,
    ):
        super().__init__(quant_config)
        self.out_dtype = out_dtype
        self.weights_prepermuted = weights_prepermuted
        self._w1_sonic: torch.Tensor | None = None
        self._w2_sonic: torch.Tensor | None = None
        self._w1_id: int = -1
        self._w2_id: int = -1

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (
            mk.FusedMoEActivationFormat.Standard,
            mk.FusedMoEActivationFormat.Standard,
        )

    def supports_expert_map(self) -> bool:
        return False  # TODO: Verify Sonic MoE expert mapping support

    def supports_chunking(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (M * topk, max(N, K))
        workspace2 = (M * topk, N // 2)
        output = (M, K)
        return (workspace1, workspace2, output)

    def _ensure_weights_ready(
        self, w1: torch.Tensor, w2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.weights_prepermuted:
            return w1, w2
        if self._w1_id != id(w1) or self._w2_id != id(w2):
            self._w1_sonic = permute_weights_for_sonic(w1)
            self._w2_sonic = w2.contiguous()
            self._w1_id = id(w1)
            self._w2_id = id(w2)
        assert self._w1_sonic is not None and self._w2_sonic is not None
        return self._w1_sonic, self._w2_sonic

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        """
        Apply Sonic MoE computation.

        Orchestrates:
        1. Weight conversion (cached)
        2. Up projection with GLU activation
        3. Down projection
        4. Router-weighted combination
        """
        if activation not in ("silu", "silu_and_mul"):
            raise ValueError(
                f"Sonic MoE only supports silu/silu_and_mul activation, "
                f"got {activation}"
            )

        w1_sonic, w2_sonic = self._ensure_weights_ready(w1, w2)

        try:
            from sonicmoe.enums import ActivationType
            from sonicmoe.functional.forward import (
                _down_projection_forward,
                _router_forward,
                _up_projection_forward,
            )
        except ImportError as e:
            raise RuntimeError(
                "Sonic MoE functional API not available. "
                "Install sonicmoe: pip install sonicmoe"
            ) from e

        M, K = hidden_states.shape
        num_experts, _, N = w1_sonic.shape
        topk = topk_ids.shape[1]

        # TODO(https://github.com/vllm-project/vllm/issues/31578): use router logits
        selected_experts = topk_ids.flatten()
        sorted_expert_idxs, sorted_scattered_idxs = selected_experts.sort()

        expert_frequency = selected_experts.bincount(minlength=num_experts).to(
            torch.int32
        )
        expert_offsets = expert_frequency.cumsum(-1).to(torch.int32)

        x_gather_idx = sorted_scattered_idxs // topk
        s_reverse_scatter_idx = sorted_scattered_idxs

        z = workspace13[: M * topk, :N].view(M * topk, N)
        y1 = workspace2[: M * topk, : N // 2].view(M * topk, N // 2)
        y2 = workspace13[: M * topk, :K].view(M * topk, K)

        act_type = ActivationType.SWIGLU

        _up_projection_forward(
            x=hidden_states,
            w1=w1_sonic,
            z=z,
            y1=y1,
            b1=None,
            expert_frequency_offset=expert_offsets,
            expert_schedule_order=sorted_expert_idxs,
            x_gather_idx=x_gather_idx,
            stream_id=0,
            activation_type=act_type,
            is_glu_activation=True,
            is_inference_mode_enabled=True,
        )

        _down_projection_forward(
            w2=w2_sonic,
            y1=y1,
            y2=y2,
            b2=None,
            expert_frequency_offset=expert_offsets,
            expert_schedule_order=sorted_expert_idxs,
            x_gather_idx=x_gather_idx,
            stream_id=0,
        )

        topk_scores = topk_weights.flatten()[sorted_scattered_idxs]
        _router_forward(
            y2=y2,
            o=output,
            topk_scores=topk_scores,
            s_reverse_scatter_idx=s_reverse_scatter_idx,
            num_activated_expert_per_token_offset=torch.arange(
                0, M * topk + 1, topk, device=output.device, dtype=torch.int32
            ),
            varlen_K_max=topk,
            H=K,
            is_varlen_K=False,
        )


_kernel_cache: dict[torch.dtype, mk.FusedMoEModularKernel] = {}


def sonic_moe_forward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Sonic MoE forward pass using modular kernel infrastructure.
    """
    if not is_sonic_moe_supported():
        raise RuntimeError(
            "Sonic MoE is not supported on this system. "
            "Requires: sonicmoe package + Hopper GPU (H100/H200)"
        )

    dtype = hidden_states.dtype
    if dtype not in _kernel_cache:
        _kernel_cache[dtype] = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            SonicMoeExperts(out_dtype=dtype),
        )
    fused_experts = _kernel_cache[dtype]

    return fused_experts(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
        activation=activation,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
    )
