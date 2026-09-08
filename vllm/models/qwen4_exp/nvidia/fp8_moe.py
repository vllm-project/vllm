# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen FP8 checkpoint loading for block-aligned tensor parallelism."""

from typing import Literal, overload

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)


def _needs_block_aligned_tp(
    config: FusedMoEConfig, quant_config: QuantizationConfig | None, prefix: str
) -> bool:
    return (
        isinstance(quant_config, Fp8Config)
        and quant_config.is_checkpoint_fp8_serialized
        and quant_config.weight_block_size == [128, 128]
        and quant_config.store_dtype != "mxfp4"
        and config.moe_backend == "flashinfer_trtllm"
        and config.tp_size > 1
        and (config.intermediate_size // config.tp_size) % 128 != 0
        and not is_layer_skipped(
            prefix=prefix,
            ignored_layers=quant_config.ignored_layers,
            fused_mapping=quant_config.packed_modules_mapping,
            match_mode=quant_config.ignored_layers_match_mode,
        )
    )


class Qwen4ExpRoutedExperts(RoutedExperts):
    """Keep checkpoint FP8 blocks intact when Qwen's TP split is unaligned."""

    def __init__(
        self,
        layer_name: str,
        params_dtype: torch.dtype,
        moe_config: FusedMoEConfig,
        quant_config: QuantizationConfig | None,
        **kwargs,
    ):
        if _needs_block_aligned_tp(moe_config, quant_config, layer_name):
            if (
                moe_config.intermediate_size % 128 != 0
                or moe_config.hidden_dim % 128 != 0
                or moe_config.ep_size != 1
                or moe_config.is_lora_enabled
                or moe_config.has_bias
            ):
                raise ValueError(
                    "Block-aligned FP8 TP sharding requires 128-aligned global "
                    "expert dimensions, pure TP, and no LoRA or expert bias."
                )
            original_size = moe_config.intermediate_size_per_partition
            moe_config.intermediate_size_per_partition = round_up(original_size, 128)
            logger.info_once(
                "Qwen FP8 TP checkpoint loading: local intermediate size %d -> %d; "
                "assigning complete quantization blocks without requantization.",
                original_size,
                moe_config.intermediate_size_per_partition,
            )
        super().__init__(layer_name, params_dtype, moe_config, quant_config, **kwargs)

    @overload
    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: Literal[False] = False,
    ) -> None: ...

    @overload
    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: Literal[True],
    ) -> bool: ...

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        if not _needs_block_aligned_tp(
            self.moe_config, self.quant_config, self.layer_name
        ):
            return super().weight_loader(
                param, loaded_weight, weight_name, shard_id, expert_id, return_success
            )

        expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
        if expert_id == -1:
            return False if return_success else None
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"Unexpected expert shard: {shard_id}")
        is_scale = "weight_scale_inv" in weight_name
        block_size = 1 if is_scale else 128
        expected_size = self.moe_config.intermediate_size // (128 if is_scale else 1)
        full_load = loaded_weight.ndim == 3
        shard_dim = int(shard_id == "w2") + int(full_load)
        if loaded_weight.shape[shard_dim] != expected_size:
            raise ValueError(
                "Block-aligned FP8 TP loading expects an unsharded checkpoint "
                f"projection of size {expected_size}, got "
                f"{loaded_weight.shape[shard_dim]}."
            )

        # As in GPT-OSS, weights and scales use the same block-aligned bounds.
        # Distribute remainder blocks over ranks to avoid empty TP4 shards.
        blocks, remainder = divmod(expected_size // block_size, self.moe_config.tp_size)
        rank = self.moe_config.tp_rank
        start = (rank * blocks + min(rank, remainder)) * block_size
        size = (blocks + int(rank < remainder)) * block_size
        destination = param.data if full_load else param.data[expert_id]
        if shard_id in ("w1", "w3"):
            half = destination.shape[shard_dim] // 2
            destination = destination.narrow(
                shard_dim, 0 if shard_id == "w1" else half, half
            )
        destination.fill_(1 if is_scale else 0)
        destination.narrow(shard_dim, 0, size).copy_(
            loaded_weight.narrow(shard_dim, start, size)
        )
        return True if return_success else None


Qwen4ExpRoutedExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
