# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
FlexMLRT-based vision NPU backend with CPU preprocessing.

VitisAI-compiled models partition operations between CPU and NPU. This backend
implements the CPU preprocessing operations before calling FlexMLRT for NPU
execution, matching the behavior of VitisAI ExecutionProvider.
"""

import contextlib
import logging
import time

import numpy as np
import torch

import vllm.envs as envs

from .backend import NPUVisionBackend
from .cpu_preprocess import get_cpu_preprocessor

logger = logging.getLogger(__name__)

# Cache environment variables for performance (avoids repeated lookups)
VLLM_NPU_TIMING = envs.VLLM_NPU_TIMING


@contextlib.contextmanager
def npu_timing(operation: str, logger_obj=None):
    """Zero-overhead timing for NPU operations when VLLM_NPU_TIMING=1.

    Args:
        operation: Name of the operation being timed
        logger_obj: Optional logger to use (defaults to module logger)
    """
    if not VLLM_NPU_TIMING:
        yield
        return

    start = time.monotonic()
    try:
        yield
    finally:
        elapsed_ms = (time.monotonic() - start) * 1000
        log_func = logger_obj.info if logger_obj else logger.info
        log_func("[NPU Timing] %s: %.2fms", operation, elapsed_ms)


class FlexMLRTVisionBackend(NPUVisionBackend):
    """FlexMLRT implementation of NPU vision backend with CPU preprocessing.

    Uses AMD FlexMLRT library to run vision models on Ryzen AI NPU.
    Implements CPU preprocessing operations that VitisAI EP normally handles.
    """

    def __init__(self, model_cache_path: str, device_name: str = "stx"):
        """Initialize FlexMLRT vision model with CPU preprocessing.

        Args:
            model_cache_path: Path to VAIP model cache (vaiml_par_0 directory)
            device_name: XRT device name ("stx" for Strix, "phx" for Phoenix)
        """
        from vllm.vision_npu._vision_flexmlrt_cpu import VisionFlexMLRTModel

        self.model = VisionFlexMLRTModel(model_cache_path, device_name)

        # Initialize CPU preprocessor
        self.preprocessor = get_cpu_preprocessor(model_cache_path, optimized=True)
        logger.info("[FlexMLRT Backend] Initialized with CPU preprocessing")

    def forward(self, pixel_values: np.ndarray, grid_thw: np.ndarray) -> np.ndarray:
        """Run vision encoding with CPU preprocessing + NPU execution.

        Pipeline:
        1. CPU preprocessing: [4292, 1176] → [1073, 4, 1280]
        2. NPU execution: [1073, 4, 1280] → [1073, 3584]
        3. CPU postprocessing: Apply reverse_index reordering

        Args:
            pixel_values: [seq_len, feature_dim] float32 array from HF processor
            grid_thw: [num_images, 3] int64 array (unused for now)

        Returns:
            embeddings: [merged_seq_len, hidden_dim] float32 array
        """
        total_start = time.monotonic() if VLLM_NPU_TIMING else None

        # Convert numpy to torch for preprocessing
        with npu_timing("NumPy→Torch conversion", logger):
            if isinstance(pixel_values, np.ndarray):
                pixel_values_torch = torch.from_numpy(pixel_values).float()
            else:
                pixel_values_torch = pixel_values.float()

        # Step 1: CPU preprocessing
        logger.debug(
            "[FlexMLRT Backend] Preprocessing input shape: %s", pixel_values.shape
        )
        with npu_timing("CPU preprocessing (total)", logger):
            preprocessed = self.preprocessor.preprocess(pixel_values_torch)

        # Step 2: NPU execution
        logger.debug(
            "[FlexMLRT Backend] Running NPU inference on shape: %s",
            preprocessed.shape,
        )
        with npu_timing("NPU inference", logger):
            npu_output = self.model.forward(preprocessed)

        # Step 3: CPU postprocessing
        logger.debug(
            "[FlexMLRT Backend] Postprocessing NPU output shape: %s", npu_output.shape
        )
        with npu_timing("CPU postprocessing", logger):
            final_output = self.preprocessor.postprocess(npu_output)

        logger.debug("[FlexMLRT Backend] Final output shape: %s", final_output.shape)

        # Log total time and memory stats
        if VLLM_NPU_TIMING and total_start is not None:
            total_ms = (time.monotonic() - total_start) * 1000
            logger.info("[NPU Timing] Total vision pipeline: %.2fms", total_ms)
            logger.info("[NPU Memory] Input: %.2f MB", pixel_values.nbytes / 1024**2)
            logger.info(
                "[NPU Memory] Preprocessed: %.2f MB", preprocessed.nbytes / 1024**2
            )
            logger.info("[NPU Memory] Output: %.2f MB", final_output.nbytes / 1024**2)
            logger.info(
                "[ViT Output] Shape: %s \u2192 %d patches \u00d7 %d embedding_dim",
                final_output.shape,
                final_output.shape[0],
                final_output.shape[1],
            )

        return final_output

    @property
    def output_dim(self) -> int:
        """Get output embedding dimension from FlexMLRT model."""
        return self.model.output_dim()
