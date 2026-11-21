# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.quantization.mxfp4 import PackedMxfp4MoEMethod


class CompressedTensorsW4A8Mxfp4MoEMethod(PackedMxfp4MoEMethod):
    """MXFP4 weights with dynamic FP8 activations in compressed-tensors format."""
