# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Literal, Union

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


class XPUModelRunner(GPUModelRunner):
    """A model runner for XPU devices."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(vllm_config, device)
        # FIXME: To be verified.
        self.cascade_attn_enabled = False

    def _init_device_properties(self) -> None:
        self.num_sms = None

    def _sync_device(self) -> None:
        torch.xpu.synchronize()

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        super().initialize_kv_cache(kv_cache_config)
        if has_kv_transfer_group():
            get_kv_transfer_group().set_host_xfer_buffer_ops(copy_kv_blocks)


def _make_src_and_dst_indices(
    src_block_ids: list[int],
    dst_block_ids: list[int],
    src_device: Union[torch.device, str],
    dst_device: Union[torch.device, str],
) -> tuple[torch.Tensor, torch.Tensor]:
    src_indices = torch.tensor(src_block_ids,
                               device=src_device,
                               dtype=torch.int64)
    dst_indices = torch.tensor(dst_block_ids,
                               device=dst_device,
                               dtype=torch.int64)
    return src_indices, dst_indices


def _insert_blocks_to_xpu(
    cpu_cache: torch.Tensor,
    xpu_cache: torch.Tensor,
    cpu_block_indices: torch.Tensor,
    xpu_block_indices: torch.Tensor,
) -> None:
    """ cpu blocks to xpu blocks"""
    _cpu_cache = cpu_cache[:, cpu_block_indices]
    xpu_cache[:, xpu_block_indices] = _cpu_cache.to(xpu_cache.device)


def _swap_out_xpu_blocks(
    xpu_cache: torch.Tensor,
    cpu_cache: torch.Tensor,
    xpu_block_indices: torch.Tensor,
    cpu_block_indices: torch.Tensor,
) -> None:
    """ xpu blocks to cpu blocks"""
    _xpu_cache = xpu_cache[:, xpu_block_indices]
    cpu_cache[:, cpu_block_indices] = _xpu_cache.cpu()


def copy_kv_blocks(
    src_kv_caches: dict[str, torch.Tensor],
    dst_kv_caches: dict[str, torch.Tensor],
    src_block_ids: list[int],
    dst_block_ids: list[int],
    direction: Literal["h2d", "d2h"],
) -> None:
    """Copy kv blocks between different buffers."""
    if not src_kv_caches or not dst_kv_caches or \
       not src_block_ids or not dst_block_ids or \
       len(src_block_ids) != len(dst_block_ids):
        return

    src_device = next(iter(src_kv_caches.values())).device
    dst_device = next(iter(dst_kv_caches.values())).device

    src_indices, dst_indices = _make_src_and_dst_indices(
        src_block_ids=src_block_ids,
        dst_block_ids=dst_block_ids,
        src_device=src_device,
        dst_device=dst_device)

    _copy_fn = _insert_blocks_to_xpu if direction == "h2d" else \
               _swap_out_xpu_blocks
    for layer_name in src_kv_caches:
        src_tensor = src_kv_caches[layer_name]
        dst_tensor = dst_kv_caches[layer_name]
        _copy_fn(src_tensor, dst_tensor, src_indices, dst_indices)
