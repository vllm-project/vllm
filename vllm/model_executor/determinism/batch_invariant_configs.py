# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import NamedTuple

import torch

from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability


class _MatmulMConfig(NamedTuple):
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int


class _MatmulShapeConfig(NamedTuple):
    block_k: int
    m_buckets: tuple[tuple[int, _MatmulMConfig], ...]


_BATCH_INVARIANT_MATMUL_TUNED_CONFIGS: dict[
    str, dict[tuple[int, int], _MatmulShapeConfig]
] = {
    "ada": {
        (12288, 2048): _MatmulShapeConfig(
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 128, 8, 2)),
                (4, _MatmulMConfig(16, 64, 8, 5)),
                (8, _MatmulMConfig(16, 128, 8, 4)),
                (16, _MatmulMConfig(16, 128, 4, 5)),
                (32, _MatmulMConfig(32, 128, 4, 5)),
                (64, _MatmulMConfig(32, 256, 4, 3)),
                (256, _MatmulMConfig(128, 128, 8, 3)),
                (512, _MatmulMConfig(128, 64, 4, 4)),
                (1024, _MatmulMConfig(128, 128, 8, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (2048, 6144): _MatmulShapeConfig(
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 64, 4, 4)),
                (4, _MatmulMConfig(16, 64, 8, 4)),
                (8, _MatmulMConfig(16, 32, 8, 4)),
                (16, _MatmulMConfig(16, 32, 8, 4)),
                (32, _MatmulMConfig(16, 64, 8, 3)),
                (64, _MatmulMConfig(32, 64, 4, 4)),
                (256, _MatmulMConfig(32, 64, 4, 4)),
                (512, _MatmulMConfig(64, 64, 4, 3)),
                (1024, _MatmulMConfig(64, 64, 4, 3)),
                (2048, _MatmulMConfig(64, 64, 4, 3)),
            ),
        ),
        (4096, 2048): _MatmulShapeConfig(
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 256, 8, 3)),
                (4, _MatmulMConfig(16, 64, 4, 3)),
                (8, _MatmulMConfig(16, 128, 8, 2)),
                (16, _MatmulMConfig(16, 32, 4, 5)),
                (32, _MatmulMConfig(16, 64, 4, 4)),
                (64, _MatmulMConfig(64, 64, 8, 3)),
                (256, _MatmulMConfig(64, 64, 4, 5)),
                (512, _MatmulMConfig(64, 64, 4, 5)),
                (1024, _MatmulMConfig(128, 64, 4, 4)),
                (2048, _MatmulMConfig(128, 64, 8, 4)),
            ),
        ),
        (151936, 2048): _MatmulShapeConfig(
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 32, 8, 5)),
                (4, _MatmulMConfig(16, 32, 8, 5)),
                (8, _MatmulMConfig(16, 32, 8, 5)),
                (16, _MatmulMConfig(16, 32, 4, 4)),
                (32, _MatmulMConfig(16, 64, 4, 4)),
                (64, _MatmulMConfig(32, 64, 4, 4)),
                (256, _MatmulMConfig(128, 128, 4, 3)),
                (512, _MatmulMConfig(128, 128, 4, 3)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (2048, 2048): _MatmulShapeConfig(
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(32, 128, 8, 5)),
                (4, _MatmulMConfig(16, 256, 8, 3)),
                (8, _MatmulMConfig(32, 128, 8, 3)),
                (16, _MatmulMConfig(32, 128, 8, 3)),
                (32, _MatmulMConfig(16, 128, 8, 5)),
                (64, _MatmulMConfig(16, 256, 4, 3)),
                (256, _MatmulMConfig(64, 128, 8, 4)),
                (512, _MatmulMConfig(64, 64, 4, 5)),
                (1024, _MatmulMConfig(64, 64, 4, 5)),
                (2048, _MatmulMConfig(64, 64, 4, 5)),
            ),
        ),
    },
    "hopper": {
        (12288, 2048): _MatmulShapeConfig(
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 256, 4, 3)),
                (4, _MatmulMConfig(16, 256, 4, 4)),
                (8, _MatmulMConfig(16, 64, 4, 4)),
                (16, _MatmulMConfig(16, 64, 4, 4)),
                (32, _MatmulMConfig(32, 64, 8, 5)),
                (64, _MatmulMConfig(64, 64, 4, 4)),
                (256, _MatmulMConfig(64, 128, 4, 3)),
                (512, _MatmulMConfig(128, 128, 8, 3)),
                (1024, _MatmulMConfig(128, 256, 8, 2)),
                (2048, _MatmulMConfig(128, 256, 8, 2)),
            ),
        ),
        (2048, 6144): _MatmulShapeConfig(
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 32, 4, 4)),
                (4, _MatmulMConfig(16, 64, 4, 4)),
                (8, _MatmulMConfig(16, 64, 4, 4)),
                (16, _MatmulMConfig(16, 32, 4, 3)),
                (32, _MatmulMConfig(32, 32, 8, 3)),
                (64, _MatmulMConfig(64, 32, 4, 5)),
                (256, _MatmulMConfig(64, 128, 4, 3)),
                (512, _MatmulMConfig(128, 128, 4, 3)),
                (1024, _MatmulMConfig(64, 64, 4, 5)),
                (2048, _MatmulMConfig(64, 128, 4, 4)),
            ),
        ),
        (4096, 2048): _MatmulShapeConfig(
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 64, 8, 5)),
                (4, _MatmulMConfig(16, 64, 4, 5)),
                (8, _MatmulMConfig(16, 128, 4, 3)),
                (16, _MatmulMConfig(16, 64, 4, 5)),
                (32, _MatmulMConfig(16, 64, 4, 4)),
                (64, _MatmulMConfig(64, 32, 4, 3)),
                (256, _MatmulMConfig(64, 256, 4, 2)),
                (512, _MatmulMConfig(64, 64, 4, 3)),
                (1024, _MatmulMConfig(64, 128, 4, 3)),
                (2048, _MatmulMConfig(64, 64, 4, 4)),
            ),
        ),
        (151936, 2048): _MatmulShapeConfig(
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 256, 4, 4)),
                (4, _MatmulMConfig(16, 256, 4, 4)),
                (8, _MatmulMConfig(16, 256, 4, 3)),
                (16, _MatmulMConfig(16, 256, 4, 3)),
                (32, _MatmulMConfig(32, 256, 8, 3)),
                (64, _MatmulMConfig(64, 128, 4, 3)),
                (256, _MatmulMConfig(64, 128, 4, 3)),
                (512, _MatmulMConfig(128, 128, 8, 3)),
                (1024, _MatmulMConfig(128, 256, 8, 2)),
                (2048, _MatmulMConfig(128, 256, 8, 2)),
            ),
        ),
        (2048, 2048): _MatmulShapeConfig(
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 64, 4, 5)),
                (4, _MatmulMConfig(16, 128, 8, 3)),
                (8, _MatmulMConfig(16, 32, 4, 3)),
                (16, _MatmulMConfig(32, 64, 8, 5)),
                (32, _MatmulMConfig(16, 128, 4, 5)),
                (64, _MatmulMConfig(32, 64, 4, 2)),
                (256, _MatmulMConfig(128, 64, 4, 4)),
                (512, _MatmulMConfig(64, 256, 4, 2)),
                (1024, _MatmulMConfig(64, 64, 4, 3)),
                (2048, _MatmulMConfig(64, 128, 4, 3)),
            ),
        ),
    },
    "blackwell": {
        (12288, 2048): _MatmulShapeConfig(
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 128, 8, 5)),
                (4, _MatmulMConfig(16, 128, 4, 5)),
                (8, _MatmulMConfig(16, 128, 4, 5)),
                (16, _MatmulMConfig(16, 128, 4, 5)),
                (32, _MatmulMConfig(32, 128, 4, 5)),
                (64, _MatmulMConfig(64, 128, 8, 5)),
                (256, _MatmulMConfig(128, 256, 4, 4)),
                (512, _MatmulMConfig(128, 256, 4, 4)),
                (1024, _MatmulMConfig(128, 256, 4, 4)),
                (2048, _MatmulMConfig(128, 256, 4, 4)),
            ),
        ),
        (2048, 6144): _MatmulShapeConfig(
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 32, 4, 5)),
                (4, _MatmulMConfig(16, 32, 4, 5)),
                (8, _MatmulMConfig(16, 32, 4, 5)),
                (16, _MatmulMConfig(16, 32, 4, 5)),
                (32, _MatmulMConfig(16, 32, 4, 5)),
                (64, _MatmulMConfig(32, 32, 4, 5)),
                (256, _MatmulMConfig(64, 64, 4, 5)),
                (512, _MatmulMConfig(128, 64, 4, 4)),
                (1024, _MatmulMConfig(128, 128, 8, 3)),
                (2048, _MatmulMConfig(128, 128, 8, 3)),
            ),
        ),
        (4096, 2048): _MatmulShapeConfig(
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 64, 4, 5)),
                (4, _MatmulMConfig(16, 64, 4, 5)),
                (8, _MatmulMConfig(16, 32, 4, 5)),
                (16, _MatmulMConfig(16, 32, 4, 5)),
                (32, _MatmulMConfig(32, 32, 4, 5)),
                (64, _MatmulMConfig(32, 64, 4, 4)),
                (256, _MatmulMConfig(128, 64, 4, 4)),
                (512, _MatmulMConfig(128, 128, 8, 3)),
                (1024, _MatmulMConfig(128, 128, 8, 3)),
                (2048, _MatmulMConfig(128, 128, 8, 3)),
            ),
        ),
        (151936, 2048): _MatmulShapeConfig(
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 256, 4, 5)),
                (4, _MatmulMConfig(16, 256, 4, 5)),
                (8, _MatmulMConfig(16, 256, 4, 5)),
                (16, _MatmulMConfig(16, 256, 4, 5)),
                (32, _MatmulMConfig(64, 256, 4, 5)),
                (64, _MatmulMConfig(64, 256, 4, 5)),
                (256, _MatmulMConfig(128, 256, 4, 4)),
                (512, _MatmulMConfig(128, 256, 4, 4)),
                (1024, _MatmulMConfig(128, 256, 4, 4)),
                (2048, _MatmulMConfig(128, 256, 4, 4)),
            ),
        ),
        (2048, 2048): _MatmulShapeConfig(
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 32, 4, 4)),
                (4, _MatmulMConfig(16, 32, 4, 4)),
                (8, _MatmulMConfig(16, 32, 4, 5)),
                (16, _MatmulMConfig(16, 32, 8, 5)),
                (32, _MatmulMConfig(16, 32, 4, 4)),
                (64, _MatmulMConfig(32, 32, 4, 5)),
                (256, _MatmulMConfig(64, 64, 4, 4)),
                (512, _MatmulMConfig(128, 64, 4, 4)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 8, 3)),
            ),
        ),
    },
}

_TUNED_MATMUL_CONFIGS_FOR_DEVICE: dict[tuple[int, int], _MatmulShapeConfig] | None = (
    None
)
_TUNED_MATMUL_CONFIGS_RESOLVED = False


def _get_tuned_matmul_arch_family(capability: DeviceCapability | None) -> str | None:
    if capability is None:
        return None
    if capability.major == 10:
        return "blackwell"
    if capability.major == 9:
        return "hopper"
    if capability.major == 8 and capability.minor == 9:
        return "ada"
    return None


def resolve_tuned_matmul_configs() -> None:
    global _TUNED_MATMUL_CONFIGS_FOR_DEVICE
    global _TUNED_MATMUL_CONFIGS_RESOLVED

    if _TUNED_MATMUL_CONFIGS_RESOLVED:
        return

    # Normal vLLM initialization resolves this before torch.compile tracing.
    capability = (
        current_platform.get_device_capability() if current_platform.is_cuda() else None
    )
    arch_family = _get_tuned_matmul_arch_family(capability)
    if arch_family is None:
        _TUNED_MATMUL_CONFIGS_FOR_DEVICE = None
    else:
        _TUNED_MATMUL_CONFIGS_FOR_DEVICE = _BATCH_INVARIANT_MATMUL_TUNED_CONFIGS.get(
            arch_family
        )
    _TUNED_MATMUL_CONFIGS_RESOLVED = True


def _get_matmul_config(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    default: dict[str, int],
) -> dict[str, int]:
    if not _TUNED_MATMUL_CONFIGS_RESOLVED:
        resolve_tuned_matmul_configs()
    if dtype != torch.bfloat16:
        return default
    device_configs = _TUNED_MATMUL_CONFIGS_FOR_DEVICE
    if device_configs is None:
        return default
    shape_config = device_configs.get((N, K))
    if shape_config is None:
        return default
    # Values above the tuned range reuse the largest bucket;
    # shape-wide BLOCK_K keeps this batch-invariant.
    m_config = shape_config.m_buckets[-1][1]
    for max_m, bucket_config in shape_config.m_buckets:
        if max_m >= M:
            m_config = bucket_config
            break
    return {
        "BLOCK_SIZE_M": m_config.block_m,
        "BLOCK_SIZE_N": m_config.block_n,
        "BLOCK_SIZE_K": shape_config.block_k,
        "GROUP_SIZE_M": 8,
        "num_warps": m_config.num_warps,
        "num_stages": m_config.num_stages,
    }
