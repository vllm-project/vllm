# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CPU preprocessing for Qwen2.5-VL vision models compiled for FlexMLRT.

VitisAI-compiled models partition operations between CPU and NPU. FlexMLRT
requires the CPU path to be implemented explicitly:

- Input: pixel_values [4292, 1176] from the HuggingFace processor
- Output: preprocessed [1073, 4, 1280] ready for NPU
- Postprocessing: apply reverse_index gather to NPU output
"""

from __future__ import annotations

import gc
import logging
import os

import numpy as np
import onnx
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_ONNX_FILENAME = "qwen2_5_vl_vision_stitched_7b.onnx"


def _resolve_onnx_model_path(model_path: str) -> str:
    """Locate the stitched vision ONNX next to the .rai file (same bundle dir)."""
    from vllm.vision_npu.paths import resolve_vision_bundle_dir

    bundle_dir = resolve_vision_bundle_dir(model_path)
    for onnx_path in (
        os.path.join(bundle_dir, _ONNX_FILENAME),
        os.path.join(os.path.dirname(bundle_dir), _ONNX_FILENAME),
    ):
        if os.path.exists(onnx_path):
            return onnx_path
    raise FileNotFoundError(
        f"Cannot find ONNX model {_ONNX_FILENAME} near RAI bundle {bundle_dir}"
    )


class Qwen2_5_VLCpuPreprocessor:
    """CPU preprocessing for Qwen2.5-VL vision models before NPU execution."""

    def __init__(self, model_path: str):
        onnx_model_path = _resolve_onnx_model_path(model_path)
        logger.info(
            "[Qwen2.5-VL CPU Preprocess] Loading ONNX model from %s", onnx_model_path
        )

        model = onnx.load(onnx_model_path)
        graph = model.graph
        initializers = {init.name: init for init in graph.initializer}

        weight_np = onnx.numpy_helper.to_array(initializers["patch_embed.proj.weight"])
        self.conv_weight = torch.from_numpy(weight_np).float()
        self.window_index = onnx.numpy_helper.to_array(
            initializers["blocks.window_index"]
        )
        self.reverse_index = onnx.numpy_helper.to_array(
            initializers["merger.reverse_index"]
        )

        del model, graph, initializers, weight_np
        gc.collect()
        logger.info(
            "[Qwen2.5-VL CPU Preprocess] Initialized (ONNX model released from memory)"
        )

    def preprocess(self, pixel_values: torch.Tensor) -> np.ndarray:
        """Apply CPU preprocessing to pixel_values before NPU execution."""
        pixel_values = pixel_values.cpu().float()
        x = pixel_values.reshape(-1, 3, 2, 14, 14)
        conv_out = F.conv3d(
            x,
            self.conv_weight,
            bias=None,
            stride=(2, 14, 14),
            padding=(0, 0, 0),
        )
        x2 = conv_out.reshape(-1, 1280)
        x3 = x2.reshape(1073, 4, 1280)
        x4_np = x3.numpy()[self.window_index]
        logger.info("[Qwen2.5-VL CPU Preprocess] Output shape: %s", x4_np.shape)
        return x4_np

    def postprocess(self, npu_output: np.ndarray) -> np.ndarray:
        """Apply reverse_index reordering to NPU output."""
        return npu_output[self.reverse_index]
