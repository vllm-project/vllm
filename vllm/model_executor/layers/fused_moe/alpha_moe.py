# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alpha MoE integration for vLLM.

This module provides integration with Alpha MoE, a high-performance fused
Mixture of Experts megakernel optimized for TP servings of MoE models.

Alpha MoE provides FP8 W8A8 quantized MoE kernels that fuse the up projection,
activation, and down projection into a single CUDA kernel.

Usage:
    Set VLLM_USE_ALPHA_MOE=1 to enable Alpha MoE kernels.
    Set VLLM_ALPHA_MOE_CONFIG to the path of your Alpha MoE config file.

The config file can be generated using Alpha MoE's jit_moe.py script:
    python jit_moe.py --E <num_experts> --N <N> --K <K> --out-file config.json
"""

import functools
import json
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)

logger = init_logger(__name__)

# Global flag to track if Alpha MoE is available
_ALPHA_MOE_AVAILABLE: bool | None = None

# Default Alpha MoE kernel configuration
# Constraints from Alpha-MoE torch_interface.cpp:
#   - block_n: must be 64 or 32
#   - warp_n: must be 4 if block_n==64, or 8 if block_n==32
#   - stages: must be > 0 and < 6 (i.e., 1-5)
#   - block_m: must be > 0, <= 128, and divisible by 8
_DEFAULT_ALPHA_MOE_CONFIG: dict[str, int] = {
    "block_m": 128,
    "block_n": 64,
    "warp_n": 4,
    "stages": 3,
}


def is_alpha_moe_available() -> bool:
    """Check if Alpha MoE is installed and available."""
    global _ALPHA_MOE_AVAILABLE
    if _ALPHA_MOE_AVAILABLE is None:
        try:
            import alpha_moe  # noqa: F401

            _ALPHA_MOE_AVAILABLE = True
            logger.info("Alpha MoE is available")
        except ImportError:
            _ALPHA_MOE_AVAILABLE = False
            logger.warning(
                "Alpha MoE is not installed. Install it with: "
                "pip install -e . --no-build-isolation from the Alpha-MoE directory"
            )
    return _ALPHA_MOE_AVAILABLE


def is_alpha_moe_enabled() -> bool:
    """Check if Alpha MoE should be used based on env var and availability."""
    return envs.VLLM_USE_ALPHA_MOE and is_alpha_moe_available()


@functools.lru_cache(maxsize=1)
def _load_alpha_moe_config(path: str) -> dict[str, Any] | None:
    """Load and cache Alpha MoE configuration from JSON file.

    Args:
        path: Path to the Alpha MoE configuration JSON file.

    Returns:
        Dictionary with configuration data, or None if loading fails.
    """
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(
            "Alpha-MoE config file not found at %s. "
            "Please generate a config using Alpha-MoE's jit_moe.py script.",
            path,
        )
        return None
    except json.JSONDecodeError as e:
        logger.warning(
            "Failed to parse Alpha-MoE config file at %s: %s. "
            "Please ensure the config file is valid JSON.",
            path,
            e,
        )
        return None


def get_best_config(config_path: str | None, num_tokens: int) -> dict[str, int]:
    """Get the best Alpha MoE kernel configuration for the given number of tokens.

    Args:
        config_path: Path to the Alpha MoE configuration JSON file.
        num_tokens: Number of tokens to process.

    Returns:
        Dictionary with kernel configuration parameters:
        - block_m: Block size in M dimension (8-128, divisible by 8)
        - block_n: Block size in N dimension (64 or 32)
        - warp_n: Number of warps in N dimension (4 if block_n=64, 8 if block_n=32)
        - stages: Pipeline stages (1-5)
    """
    if config_path is None:
        config_path = envs.VLLM_ALPHA_MOE_CONFIG

    if config_path is None:
        # Return default configuration if no config file specified
        logger.info(
            "No Alpha-MoE config file specified. Using default configuration. "
            "For optimal performance, generate a config using Alpha-MoE's "
            "jit_moe.py script and set VLLM_ALPHA_MOE_CONFIG."
        )
        return _DEFAULT_ALPHA_MOE_CONFIG.copy()

    best_conf = _load_alpha_moe_config(config_path)

    if best_conf is None:
        # Failed to load config, fall back to default
        logger.warning(
            "Failed to load Alpha-MoE config from %s. "
            "Falling back to default configuration.",
            config_path,
        )
        return _DEFAULT_ALPHA_MOE_CONFIG.copy()

    # Find the configuration with the closest number of tokens
    dist = float("inf")
    ret = None
    for nt, val in best_conf.items():
        if abs(int(nt) - num_tokens) < dist:
            dist = abs(int(nt) - num_tokens)
            ret = val

    if ret is None:
        # Config file is empty, fall back to default
        logger.warning(
            "Alpha-MoE config at %s is empty. Falling back to default configuration.",
            config_path,
        )
        return _DEFAULT_ALPHA_MOE_CONFIG.copy()

    return ret


def interleave_tensor(tensor: torch.Tensor, rep: int = 8) -> torch.Tensor:
    """Interleave the up and gate projections in chunks for Alpha MoE.

    Alpha MoE requires weights of Up projection and Gate to be interleaved
    in chunks of 8 (for weights) or 1 (for scales).

    Args:
        tensor: Input tensor of shape [num_experts, N, K] where N is the
            concatenated dimension of gate and up projections.
        rep: Interleave chunk size. Use 8 for weights, 1 for scales.

    Returns:
        Interleaved tensor with same shape.
    """
    M, N, K = tensor.shape

    first_half = tensor[:, : (N // 2), :]
    second_half = tensor[:, (N // 2) :, :]

    first_chunks = first_half.view(M, (N // (2 * rep)), rep, K)
    second_chunks = second_half.view(M, (N // (2 * rep)), rep, K)

    interleaved = torch.stack([first_chunks, second_chunks], dim=2)
    result = interleaved.view(M, N, K)

    return result.contiguous()


def alpha_moe_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    block_shape: list[int] | None = None,
    routed_scaling_factor: float = 1.0,
    global_num_experts: int = -1,
) -> torch.Tensor:
    """Execute MoE using Alpha MoE fused FP8 kernel.

    This function performs a fused Mixture of Experts operation using
    Alpha MoE's optimized CUDA kernel that combines up projection,
    SiLU activation, and down projection in a single kernel call.

    Args:
        hidden_states: Input tensor [M, K] in BF16/FP16/FP32 format.
        w1: First weight matrix [num_experts, N, K] in FP8 format (up+gate projection).
            Must be interleaved using interleave_tensor().
        w2: Second weight matrix [num_experts, K, N//2] in FP8 format (down projection).
        topk_weights: Weights for top-k experts per token [M, top_k].
        topk_ids: Top-k expert indices per token [M, top_k].
        w1_scale: Scale factors for w1 [num_experts, N_groups] or [num_experts].
            Must be interleaved using interleave_tensor(rep=1) if block quantized.
        w2_scale: Scale factors for w2 [num_experts, K_groups] or [num_experts].
        block_shape: Block quantization shape [block_k, block_n] if using block quant.
        routed_scaling_factor: Scaling factor for the output.
        global_num_experts: Total number of experts (for EP support).

    Returns:
        Output tensor [M, K] in BF16 format.
    """
    num_tokens = hidden_states.size(0)
    E = w1.size(0)
    K = hidden_states.size(1)

    if global_num_experts == -1:
        global_num_experts = E

    top_k = topk_ids.size(1)

    # Get the best kernel configuration for this batch size
    config = get_best_config(envs.VLLM_ALPHA_MOE_CONFIG, num_tokens)
    block_m = config["block_m"]
    block_n = config["block_n"]
    warp_n = config["warp_n"]
    stages = config["stages"]

    # Determine group size for FP8 quantization
    group_size = block_shape[1] if block_shape is not None else K

    # Quantize input to FP8
    A, A_scale = per_token_group_quant_fp8(hidden_states, group_size)

    # Allocate output tensor and zero-initialize (required by Alpha MoE)
    out = torch.zeros(num_tokens, K, device=hidden_states.device, dtype=torch.bfloat16)

    # Align tokens to block size for MoE
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_m, global_num_experts
    )

    # Call Alpha MoE kernel
    torch.ops.alpha_moe.fused_moe_w8a8_up_down(
        A,  # x: Input tensor [M, K] in FP8
        A_scale,  # x_scale: Per-token-group scale factors
        w1,  # w: First weight [E, N, K] in FP8 (interleaved)
        w1_scale,  # w_scale: Scale factors for w1
        w2,  # w2: Second weight [E, K, N//2] in FP8
        w2_scale,  # w2_scale: Scale factors for w2
        sorted_token_ids,  # sorted_token_ids
        expert_ids,  # expert_ids
        num_tokens_post_padded,  # num_tokens_post_padded
        topk_weights,  # topk_weights
        out,  # out: Pre-allocated output [M, K] in BF16
        top_k,  # top_k
        block_m,  # block_m
        block_n,  # block_n
        warp_n,  # warp_n
        stages,  # stages
        routed_scaling_factor,  # scaling_factor
    )

    return out


def _valid_alpha_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    use_fp8_w8a8: bool,
    activation: str,
    apply_router_weight_on_input: bool,
    expert_map: torch.Tensor | None,
) -> bool:
    """Check if Alpha MoE can be used for the given configuration.

    Args:
        hidden_states: Input tensor.
        w1: First weight matrix.
        w2: Second weight matrix.
        use_fp8_w8a8: Whether FP8 W8A8 quantization is used.
        activation: Activation function name.
        apply_router_weight_on_input: Whether router weight is applied on input.
        expert_map: Expert mapping for expert parallelism.

    Returns:
        True if Alpha MoE can be used, False otherwise.
    """
    if not is_alpha_moe_enabled():
        return False

    # Alpha MoE only supports FP8 W8A8 quantization
    if not use_fp8_w8a8:
        return False

    # Alpha MoE uses SiLU activation (swiglu)
    if activation not in ("silu", "swiglu"):
        return False

    # Alpha MoE doesn't support router weight on input
    if apply_router_weight_on_input:
        return False

    # Alpha MoE doesn't currently support expert parallelism with expert_map
    if expert_map is not None:
        return False

    # Check data types - Alpha MoE expects BF16 input for hidden states
    if hidden_states.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False

    # Check that weights are FP8
    return w1.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


__all__ = [
    "is_alpha_moe_available",
    "is_alpha_moe_enabled",
    "get_best_config",
    "interleave_tensor",
    "alpha_moe_fused_experts",
    "_valid_alpha_moe",
]
