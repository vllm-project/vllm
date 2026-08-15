# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.quantization import QuantizationConfig


def uses_modelopt_fp8_pb_wo(
    quant_config: QuantizationConfig | None, prefix: str
) -> bool:
    if quant_config is None:
        return False
    resolve = getattr(quant_config, "_resolve_quant_algo", None)
    quant_algo = resolve(prefix) if callable(resolve) else None
    if quant_algo is None:
        quant_algo = getattr(quant_config, "quant_method", None)
    return str(quant_algo).upper() == "FP8_PB_WO"


def pad_merged_output_sizes(
    output_sizes: list[int],
    tp_size: int,
    *,
    disable_tp: bool,
    alignment: int,
    replicated_shard_ids: tuple[int, ...] = (),
) -> tuple[list[int], int]:
    partition_size = 1 if disable_tp else tp_size
    local_output_size = sum(
        size if idx in replicated_shard_ids else size // partition_size
        for idx, size in enumerate(output_sizes)
    )
    padding = -local_output_size % alignment
    if padding == 0:
        return output_sizes, 0
    return [*output_sizes, padding * partition_size], padding
