# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors import CompressionFormat

# Mapping from standard PyTorch/Transformers class names to vLLM class names.
# This is used when quantization configs reference standard class names like "Linear"
# but vLLM uses specialized classes like "LinearBase", "ReplicatedLinear", etc.
#
# Note: The compressed-tensors library already has a built-in special case for
# LinearBase → Linear in its _match_class function. This mapping documents that
# behavior and provides a place to add additional vLLM-specific mappings if needed.
#
# For example, if a config uses "Embedding" as a target, we could add:
#   "Embedding": "VocabParallelEmbedding"
VLLM_CLASS_NAME_MAPPING: dict[str, str] = {
    "Linear": "LinearBase",  # Already handled by compressed-tensors library
}


def is_activation_quantization_format(format: str) -> bool:
    _ACTIVATION_QUANTIZATION_FORMATS = [
        CompressionFormat.naive_quantized.value,
        CompressionFormat.int_quantized.value,
        CompressionFormat.float_quantized.value,
        CompressionFormat.nvfp4_pack_quantized.value,
    ]
    return format in _ACTIVATION_QUANTIZATION_FORMATS
