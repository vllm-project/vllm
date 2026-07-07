# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Abstract base class for vision NPU backends.
"""

from abc import ABC, abstractmethod

import numpy as np


class NPUVisionBackend(ABC):
    """Base class for vision processing NPU backends.

    This abstract class defines the interface that all NPU vision backends
    must implement. Different NPU implementations (FlexMLRT, ONNX Runtime, etc.)
    can subclass this to provide hardware-accelerated vision processing.
    """

    @abstractmethod
    def __init__(self, model_cache_path: str, device_name: str = "stx"):
        """Load vision model onto NPU.

        Args:
            model_cache_path: Path to compiled .rai cache file
            device_name: NPU device identifier (e.g., "stx" for Strix)
        """
        pass

    @abstractmethod
    def forward(self, pixel_values: np.ndarray, grid_thw: np.ndarray) -> np.ndarray:
        """Run vision encoding on NPU.

        Args:
            pixel_values: Input pixel data [seq_len, feature_dim] float32
            grid_thw: Grid dimensions [num_images, 3] int64 (temporal, height, width)

        Returns:
            embeddings: Vision embeddings [merged_seq_len, hidden_dim] float32
        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Output embedding dimension.

        Returns:
            Hidden dimension of output embeddings (e.g., 3584 for Qwen2.5-VL)
        """
        pass
