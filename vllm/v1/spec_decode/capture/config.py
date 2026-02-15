# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration management for distillation capture in speculative decoding."""

import os
from dataclasses import dataclass
from typing import Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class CaptureConfig:
    """Configuration for distillation capture in speculative decoding.
    
    This configuration controls when and how logits are captured during
    speculative decoding for diagnostic purposes. Capture is based on
    percentile tracking - only the worst X% of acceptance rates are captured.
    """
    
    enabled: bool = False
    """Enable or disable distillation capture feature."""
    
    top_k: int = 10
    """Number of top probabilities to capture per token position."""
    
    output_dir: str = "./capture_data"
    """Directory path where Safetensors files will be written."""
    
    max_capture_percentage: float = 10.0
    """Maximum percentage of drafts to capture (0-100).
    Prevents overwhelming storage with too much data."""
    
    write_queue_size: int = 1000
    """Maximum number of pending write operations in the queue.
    Prevents unbounded memory growth."""


# Backward compatibility alias
LogitsLoggingConfig = CaptureConfig


class ConfigurationManager:
    """Manages and validates distillation capture configuration.
    
    This class validates configuration parameters at initialization and
    provides safe access to configuration values.
    """
    
    def __init__(self, config: Optional[CaptureConfig] = None):
        """Initialize configuration manager with validation.
        
        Args:
            config: CaptureConfig instance. If None, uses default config.
        
        Raises:
            ValueError: If configuration validation fails.
        """
        if config is None:
            config = CaptureConfig()
        
        self.config = self._validate(config)
    
    def _validate(self, config: CaptureConfig) -> CaptureConfig:
        """Validate configuration parameters.
        
        Args:
            config: Configuration to validate.
        
        Returns:
            Validated configuration.
        
        Raises:
            ValueError: If any validation check fails.
        """
        # Validate top_k is positive
        if config.top_k <= 0:
            raise ValueError(
                f"top_k must be > 0, got {config.top_k}"
            )
        
        # Validate max_capture_percentage is in valid range
        if not (0 <= config.max_capture_percentage <= 100):
            raise ValueError(
                f"max_capture_percentage must be in [0, 100], "
                f"got {config.max_capture_percentage}"
            )
        
        # Validate write_queue_size is positive
        if config.write_queue_size <= 0:
            raise ValueError(
                f"write_queue_size must be > 0, got {config.write_queue_size}"
            )
        
        # Check output directory writability if enabled
        if config.enabled:
            self._check_output_dir_writable(config.output_dir)
        
        return config
    
    def _check_output_dir_writable(self, output_dir: str) -> None:
        """Check if output directory is writable.
        
        Creates the directory if it doesn't exist. Logs a warning if
        the directory is not writable but doesn't raise an error.
        
        Args:
            output_dir: Directory path to check.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Test writability by creating a temp file
            test_file = os.path.join(output_dir, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            
        except (OSError, IOError) as e:
            logger.warning(
                f"Output directory {output_dir} is not writable: {e}. "
                "Distillation capture may fail at runtime."
            )
    
    def is_enabled(self) -> bool:
        """Check if distillation capture is enabled.
        
        Returns:
            True if enabled, False otherwise.
        """
        return self.config.enabled
    
    def get_top_k(self) -> int:
        """Get top-k parameter.
        
        Returns:
            Number of top probabilities to capture.
        """
        return self.config.top_k
    
    def get_output_dir(self) -> str:
        """Get output directory path.
        
        Returns:
            Output directory path.
        """
        return self.config.output_dir
    
    def get_max_capture_percentage(self) -> float:
        """Get maximum capture percentage.
        
        Returns:
            Maximum percentage of drafts to capture.
        """
        return self.config.max_capture_percentage
    
    # Backward compatibility alias
    def get_max_logging_percentage(self) -> float:
        """Get maximum capture percentage (backward compat alias).
        
        Returns:
            Maximum percentage of drafts to capture.
        """
        return self.get_max_capture_percentage()
    
    def get_write_queue_size(self) -> int:
        """Get write queue size limit.
        
        Returns:
            Maximum number of pending write operations.
        """
        return self.config.write_queue_size
