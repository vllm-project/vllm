# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CPU preprocessing operations for VitisAI-compiled vision models.

This module implements the CPU operations that VitisAI ExecutionProvider
normally handles automatically. When using FlexMLRT directly, we must
manually implement these operations.

For Qwen2.5-VL vision model:
- Input: pixel_values [4292, 1176] from HuggingFace processor
- Output: preprocessed [1073, 4, 1280] ready for NPU
- Postprocessing: Apply reverse_index Gather to NPU output
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class Qwen2_5_VL_CPUPreprocessor:
    """CPU preprocessing for Qwen2.5-VL vision model before NPU execution."""

    def __init__(self, model_cache_dir: str):
        """
        Initialize CPU preprocessor with required parameters.

        Args:
            model_cache_dir: Path to NPU model cache directory containing ONNX model
        """
        import os

        import onnx

        # Load ONNX model to extract parameters
        # model_cache_dir is typically: .../qwen2_5_vl_vision_stitched_7b/vaiml_par_0
        # We need to go up two levels to find the .onnx file
        onnx_model_path = os.path.join(
            os.path.dirname(os.path.dirname(model_cache_dir)),
            "qwen2_5_vl_vision_stitched_7b.onnx",
        )

        if not os.path.exists(onnx_model_path):
            logger.warning(
                "[CPU Preprocess] ONNX not found at %s, trying alternative path",
                onnx_model_path,
            )
            # Alternative: look in parent directory
            alt_path = os.path.join(
                os.path.dirname(model_cache_dir), "qwen2_5_vl_vision_stitched_7b.onnx"
            )
            if os.path.exists(alt_path):
                onnx_model_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Cannot find ONNX model at {onnx_model_path} or {alt_path}"
                )

        logger.info("[CPU Preprocess] Loading ONNX model from %s", onnx_model_path)
        model = onnx.load(onnx_model_path)
        graph = model.graph

        # Extract parameters from ONNX model
        initializers = {init.name: init for init in graph.initializer}

        # Conv weights for patch embedding
        if "patch_embed.proj.weight" in initializers:
            weight_tensor = initializers["patch_embed.proj.weight"]
            self.conv_weight = onnx.numpy_helper.to_array(weight_tensor)
            logger.info(
                "[CPU Preprocess] Loaded conv weight: %s", self.conv_weight.shape
            )
        else:
            raise ValueError("patch_embed.proj.weight not found in ONNX model")

        # Gather indices for window reordering
        if "blocks.window_index" in initializers:
            indices_tensor = initializers["blocks.window_index"]
            self.window_index = onnx.numpy_helper.to_array(indices_tensor)
            logger.info(
                "[CPU Preprocess] Loaded window_index: %s", self.window_index.shape
            )
        else:
            raise ValueError("blocks.window_index not found in ONNX model")

        # Reverse index for final postprocessing
        if "merger.reverse_index" in initializers:
            reverse_tensor = initializers["merger.reverse_index"]
            self.reverse_index = onnx.numpy_helper.to_array(reverse_tensor)
            logger.info(
                "[CPU Preprocess] Loaded reverse_index: %s", self.reverse_index.shape
            )
        else:
            raise ValueError("merger.reverse_index not found in ONNX model")

        logger.info("[CPU Preprocess] Initialized successfully")

    def preprocess(self, pixel_values: torch.Tensor) -> np.ndarray:
        """
        Apply CPU preprocessing operations to pixel_values.

        Args:
            pixel_values: [seq_len, feature_dim] float32 tensor from HF processor
                         Expected shape: [4292, 1176]

        Returns:
            preprocessed: [1073, 4, 1280] float32 numpy array ready for NPU
        """
        # Convert to numpy
        if isinstance(pixel_values, torch.Tensor):
            pixel_values_np = pixel_values.cpu().float().numpy()
        else:
            pixel_values_np = pixel_values.astype(np.float32)

        logger.info("[CPU Preprocess] Input shape: %s", pixel_values_np.shape)

        # Operation 1: Reshape to [batch, 3, 2, 14, 14]
        # pixel_values [4292, 1176] → [4292, 3, 2, 14, 14]
        x = pixel_values_np.reshape(-1, 3, 2, 14, 14)

        # Operation 2: Conv3D for patch embedding
        # Input: [4292, 3, 2, 14, 14]
        # Weight: [1280, 3, 2, 14, 14]
        # Output: [4292, 1280, 1, 1, 1]
        out_channels = self.conv_weight.shape[0]
        batch_size = x.shape[0]
        conv_out = np.zeros((batch_size, out_channels, 1, 1, 1), dtype=np.float32)

        # Naive implementation - can be optimized with torch.nn.functional.conv3d
        for b in range(batch_size):
            for oc in range(out_channels):
                conv_out[b, oc, 0, 0, 0] = np.sum(x[b] * self.conv_weight[oc])

        # Operation 3: Reshape to [4292, 1280]
        x2 = conv_out.reshape(-1, 1280)

        # Operation 4: Reshape to [1073, 4, 1280] - merge patches 4x4
        x3 = x2.reshape(1073, 4, 1280)

        # Operation 5: Gather with window_index (reordering)
        # Note: This maintains shape [1073, 4, 1280]
        x4 = x3[self.window_index]

        logger.info("[CPU Preprocess] Output shape: %s", x4.shape)
        return x4

    def postprocess(self, npu_output: np.ndarray) -> np.ndarray:
        """
        Apply CPU postprocessing to NPU output.

        Args:
            npu_output: [1073, 3584] float32 array from NPU

        Returns:
            final_output: [1073, 3584] float32 array after reverse_index reordering
        """
        # Apply final Gather with reverse_index
        reordered = npu_output[self.reverse_index]
        logger.info(
            "[CPU Postprocess] Applied reverse_index, shape: %s", reordered.shape
        )
        return reordered


class Qwen2_5_VL_CPUPreprocessor_Optimized:
    """Optimized version using torch for Conv3D."""

    def __init__(self, model_cache_dir: str):
        """Initialize with torch-based Conv3D for faster preprocessing."""
        import os

        import onnx

        onnx_model_path = os.path.join(
            os.path.dirname(os.path.dirname(model_cache_dir)),
            "qwen2_5_vl_vision_stitched_7b.onnx",
        )

        if not os.path.exists(onnx_model_path):
            logger.warning(
                "[CPU Preprocess Optimized] ONNX not found at %s, trying alternative",
                onnx_model_path,
            )
            alt_path = os.path.join(
                os.path.dirname(model_cache_dir), "qwen2_5_vl_vision_stitched_7b.onnx"
            )
            if os.path.exists(alt_path):
                onnx_model_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Cannot find ONNX model at {onnx_model_path} or {alt_path}"
                )

        logger.info(
            "[CPU Preprocess Optimized] Loading ONNX model from %s", onnx_model_path
        )
        model = onnx.load(onnx_model_path)
        graph = model.graph
        initializers = {init.name: init for init in graph.initializer}

        # Load parameters and convert to torch
        weight_np = onnx.numpy_helper.to_array(initializers["patch_embed.proj.weight"])
        self.conv_weight = torch.from_numpy(weight_np).float()

        self.window_index = onnx.numpy_helper.to_array(
            initializers["blocks.window_index"]
        )
        self.reverse_index = onnx.numpy_helper.to_array(
            initializers["merger.reverse_index"]
        )

        # Release ONNX model from memory (saves ~600 MB CPU RAM)
        del model, graph, initializers, weight_np
        import gc

        gc.collect()
        logger.info(
            "[CPU Preprocess Optimized] Initialized with torch Conv3D "
            "(ONNX model released from memory)"
        )

    def preprocess(self, pixel_values: torch.Tensor) -> np.ndarray:
        """Optimized preprocessing using torch.nn.functional.conv3d."""
        pixel_values = pixel_values.cpu().float()

        # Reshape to [batch, 3, 2, 14, 14]
        x = pixel_values.reshape(-1, 3, 2, 14, 14)

        # Conv3D using torch (much faster than numpy)
        import torch.nn.functional as F

        # Rearrange to [batch, channels, depth, height, width]
        conv_out = F.conv3d(
            x, self.conv_weight, bias=None, stride=(2, 14, 14), padding=(0, 0, 0)
        )  # Output: [4292, 1280, 1, 1, 1]

        # Reshape to [4292, 1280]
        x2 = conv_out.reshape(-1, 1280)

        # Reshape to [1073, 4, 1280]
        x3 = x2.reshape(1073, 4, 1280)

        # Gather with window_index
        x4_np = x3.numpy()[self.window_index]

        logger.info("[CPU Preprocess Optimized] Output shape: %s", x4_np.shape)
        return x4_np

    def postprocess(self, npu_output: np.ndarray) -> np.ndarray:
        """Apply reverse_index reordering."""
        return npu_output[self.reverse_index]


# Factory function to get appropriate preprocessor
def get_cpu_preprocessor(model_cache_dir: str, optimized: bool = True):
    """
    Get CPU preprocessor for Qwen2.5-VL vision model.

    Args:
        model_cache_dir: Path to NPU model cache
        optimized: Use torch-based optimized version (default: True)

    Returns:
        Preprocessor instance
    """
    if optimized:
        try:
            return Qwen2_5_VL_CPUPreprocessor_Optimized(model_cache_dir)
        except Exception as e:
            logger.warning(
                "Failed to load optimized preprocessor: %s, falling back to numpy",
                e,
            )
            return Qwen2_5_VL_CPUPreprocessor(model_cache_dir)
    else:
        return Qwen2_5_VL_CPUPreprocessor(model_cache_dir)
