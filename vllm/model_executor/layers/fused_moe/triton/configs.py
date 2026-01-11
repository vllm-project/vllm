# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import json
import os
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


# Adapted from: https://github.com/sgl-project/sglang/pull/2628
def get_config_file_name(
    E: int, N: int, dtype: str | None, block_shape: list[int] | None = None
) -> str:
    device_name = current_platform.get_device_name().replace(" ", "_")
    # Set device_name to H200 if a device from the H200 family is detected
    if "H200" in device_name.split("_"):
        device_name = "NVIDIA_H200"
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_shape_selector = (
        "" if not block_shape or not all(block_shape) else f",block_shape={block_shape}"
    ).replace(" ", "")
    return f"E={E},N={N},device_name={device_name}{dtype_selector}{block_shape_selector}.json"  # noqa: E501


# Adapted from: https://github.com/sgl-project/sglang/pull/2628
@functools.lru_cache
def get_moe_configs(
    E: int,
    N: int,
    dtype: str | None,
    block_n: int | None = None,
    block_k: int | None = None,
) -> dict[int, Any] | None:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # Avoid optimizing for the batch invariant case. Use default config
    if vllm_is_batch_invariant():
        return None

    # First look up if an optimized configuration is available in the configs
    # directory
    block_shape = [block_n, block_k] if block_n and block_k else None
    json_file_name = get_config_file_name(E, N, dtype, block_shape)

    config_file_paths = []

    # note that we prioritize user defined config
    user_defined_config_folder = envs.VLLM_TUNED_CONFIG_FOLDER
    if user_defined_config_folder is not None:
        user_defined_config_file_path = os.path.join(
            user_defined_config_folder, json_file_name
        )
        config_file_paths.append(user_defined_config_file_path)

    default_config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
    )
    config_file_paths.append(default_config_file_path)

    for config_file_path in config_file_paths:
        if os.path.exists(config_file_path):
            with open(config_file_path) as f:
                logger.info_once(
                    "Using configuration from %s for MoE layer.",
                    config_file_path,
                    scope="global",
                )
                # If a configuration has been found, return it
                tuned_config = json.load(f)
                # Delete triton_version from tuned_config
                tuned_config.pop("triton_version", None)
                return {int(key): val for key, val in tuned_config.items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    logger.warning_once(
        "Using default MoE config. Performance might be sub-optimal! "
        "Config file not found at %s",
        ", ".join(config_file_paths),
        scope="local",
    )
    return None


def should_moe_wna16_use_cuda(
    num_valid_tokens: int, group_size: int, num_experts: int, bit: int
):
    # TODO(rob): once we get all these into the mk objects, this should
    # be part of the kernel selection logic.
    return (
        current_platform.is_cuda()
        and bit == 4
        and group_size in [32, 64, 128]
        and num_valid_tokens / num_experts <= 6
    )


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: str | None,
    block_shape: list[int] | None = None,
) -> dict[str, int]:
    if vllm_is_batch_invariant():
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "SPLIT_K": 1,
        }
        return config

    if dtype == "fp8_w8a8" and block_shape is not None:
        # Block-wise quant: BLOCK_SIZE_N must be divisible by block_shape[0]
        # BLOCK_SIZE_K must be divisible by block_shape[1]
        # num_stages=3 can cause triton.runtime.errors.OutOfResources
        # on ROCm, set it to 2 instead.
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_shape[0],
            "BLOCK_SIZE_K": block_shape[1],
            "GROUP_SIZE_M": 32,
            "SPLIT_K": 1,
            "num_warps": 4,
            "num_stages": 3 if not current_platform.is_rocm() else 2,
        }
    elif dtype in ["int4_w4a16", "int8_w8a16"] and block_shape is not None:
        # moe wna16 kernels
        # only set BLOCK_SIZE_M
        # BLOCK_SIZE_N and BLOCK_SIZE_K would be set later
        bit = 4 if dtype == "int4_w4a16" else 8
        use_moe_wna16_cuda = should_moe_wna16_use_cuda(M * topk, block_shape[1], E, bit)
        if use_moe_wna16_cuda:
            config = {"BLOCK_SIZE_M": min(16, M), "SPLIT_K": 1}
        elif M <= 20:
            config = {"BLOCK_SIZE_M": 16, "GROUP_SIZE_M": 1, "SPLIT_K": 1}
        elif M <= 40:
            config = {"BLOCK_SIZE_M": 32, "GROUP_SIZE_M": 1, "SPLIT_K": 1}
        else:
            config = {"BLOCK_SIZE_M": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1}
    elif M <= E:
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "SPLIT_K": 1,
        }
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "SPLIT_K": 1,
        }
    return config


def try_get_optimal_moe_config(
    w1_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
    top_k: int,
    dtype: str | None,
    M: int,
    block_shape: list[int] | None = None,
) -> dict[str, int]:
    from vllm.model_executor.layers.fused_moe import get_config

    override_config = get_config()
    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        E, _, N = w2_shape
        if dtype == "int4_w4a16":
            N = N * 2
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        configs = get_moe_configs(E, N, dtype, block_n, block_k)

        if configs:
            # If an optimal configuration map has been found, look up the
            # optimal config
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            # Else use the default config
            config = get_default_config(M, E, N, w1_shape[2], top_k, dtype, block_shape)
    return config


def _ensure_block_size_k_divisible(
    size_k: int, block_size_k: int, group_size: int
) -> int:
    """Ensure block_size_k is a divisor of size_k and divisible by group_size.

    This ensures BLOCK_SIZE_K compatibility with MoeWNA16 CUDA kernel which
    requires size_k % BLOCK_SIZE_K == 0 and BLOCK_SIZE_K % group_size == 0.

    Args:
        size_k: The size_k dimension that must be divisible by result.
        block_size_k: Preferred block size (will be adjusted if needed).
        group_size: The result must be divisible by this.

    Returns:
        A valid BLOCK_SIZE_K that divides size_k and is divisible by group_size.
    """
    # Fast path: already valid
    if size_k % block_size_k == 0 and block_size_k % group_size == 0:
        return block_size_k

    # Find the largest value that:
    # 1. Divides size_k (size_k % candidate == 0)
    # 2. Is divisible by group_size (candidate % group_size == 0)
    # 3. Is <= block_size_k (prefer smaller values close to block_size_k)
    #
    # Strategy: Search from min(block_size_k, size_k) down to group_size,
    # stepping by group_size to ensure divisibility by group_size
    max_search = min(block_size_k, size_k)
    start = (max_search // group_size) * group_size
    for candidate in range(start, group_size - 1, -group_size):
        if size_k % candidate == 0:
            return candidate

    # Fallback: if group_size divides size_k, use it
    # This should always be true with correct group_size configuration
    if size_k % group_size == 0:
        return group_size

    # This should not happen with correct group_size, but ensure divisibility
    return size_k


def get_moe_wna16_block_config(
    config: dict[str, int],
    use_moe_wna16_cuda: bool,
    num_valid_tokens: int,
    size_k: int,
    size_n: int,
    num_experts: int,
    group_size: int,
    real_top_k: int,
    block_size_m: int,
):
    if "BLOCK_SIZE_N" in config and "BLOCK_SIZE_K" in config:
        # optimal block config is set
        return {}
    if not use_moe_wna16_cuda:
        # triton moe wna16 kernel
        if num_valid_tokens // real_top_k == 1:
            # if bs=1, use a smaller BLOCK_SIZE_N
            return {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64}
        else:
            return {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}
    else:
        # cuda moe wna16 kernel
        # set default block_size 128, and increase them when num_blocks
        # is too large.
        block_size_n = 128
        block_size_k = 128
        if block_size_k <= group_size:
            block_size_k = group_size

        num_n_blocks = size_k // block_size_k
        num_k_blocks = size_n // block_size_k
        num_m_blocks = (
            num_valid_tokens + block_size_m - 1
        ) / block_size_m + num_experts
        if num_valid_tokens // real_top_k <= block_size_m:
            num_m_blocks = min(num_m_blocks, num_valid_tokens)
        num_blocks = num_m_blocks * num_n_blocks * num_k_blocks

        if size_k % 256 == 0 and num_blocks >= 256 and block_size_k < 256:
            block_size_k = 256
            num_blocks = num_blocks // (256 // block_size_k)

        if (
            num_m_blocks <= 16
            and size_k % (block_size_k * 2) == 0
            and size_k % (block_size_k * 2) == 0
            and block_size_k <= 512
            and num_blocks >= 512
        ):
            block_size_k = block_size_k * 2
            num_blocks = num_blocks // 2

        if num_blocks > 1024:
            block_size_n = 256
            num_n_blocks = num_n_blocks // 2
            num_blocks = num_blocks // 2

        if size_n <= 1024 and num_blocks >= 1024:
            # The kernel performance got much better with BLOCK_SIZE_N=1024
            # when num_blocks is large, event when N is small.
            # Not sure why, maybe it force the CUDA SM process only one block
            # at the same time.
            block_size_n = 1024

        # Ensure BLOCK_SIZE_K is a divisor of size_k for CUDA kernel compatibility
        block_size_k = _ensure_block_size_k_divisible(size_k, block_size_k, group_size)

        return {"BLOCK_SIZE_N": block_size_n, "BLOCK_SIZE_K": block_size_k}
