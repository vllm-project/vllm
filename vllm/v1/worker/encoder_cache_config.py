"""
Configuration and Feature Flags for Compact Encoder Cache

This module provides configuration options and feature flags for the compact
encoder cache optimization, enabling gradual rollout and A/B testing in
production environments.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EncoderCacheConfig:
    """Configuration for the compact encoder cache optimization."""

    # Core feature flags
    enable_compact_cache: bool = True
    enable_token_aware_scheduling: bool = True
    enable_batch_processing: bool = True
    enable_position_caching: bool = True

    # Rollout configuration
    rollout_percentage: float = (
        0.0  # 0.0 to 1.0, percentage of traffic to use compact cache
    )
    enable_ab_testing: bool = False
    ab_test_group: Optional[str] = None  # 'control' or 'treatment'

    # Performance optimization flags
    enable_special_token_caching: bool = True
    enable_sequence_caching: bool = True
    batch_size_threshold: int = 8
    max_cache_size: int = 1000

    # Memory optimization settings
    enable_memory_monitoring: bool = True
    memory_usage_threshold: float = 0.8  # 80% of available memory
    enable_aggressive_cleanup: bool = False

    # Model-specific settings
    model_specific_optimizations: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize model-specific optimizations if not provided."""
        if self.model_specific_optimizations is None:
            self.model_specific_optimizations = {
                "qwen3-vl": {
                    "max_timestamps_per_video": 12,
                    "timestamp_token_pattern": "sequential",
                    "enable_frame_markers": True,
                },
                "pixtral": {
                    "max_special_tokens_per_image": 10,
                    "image_marker_pattern": "start_end",
                    "enable_region_markers": False,
                },
            }


class FeatureFlagManager:
    """Manages feature flags for the compact encoder cache optimization."""

    def __init__(self, config: Optional[EncoderCacheConfig] = None):
        """
        Initialize the feature flag manager.

        Args:
            config: Configuration object. If None, loads from environment variables.
        """
        self.config = config or self._load_from_environment()
        self._validate_config()

    def _load_from_environment(self) -> EncoderCacheConfig:
        """Load configuration from environment variables."""
        return EncoderCacheConfig(
            enable_compact_cache=self._get_env_bool("VLLM_ENABLE_COMPACT_CACHE", True),
            enable_token_aware_scheduling=self._get_env_bool(
                "VLLM_ENABLE_TOKEN_AWARE_SCHEDULING", True
            ),
            enable_batch_processing=self._get_env_bool(
                "VLLM_ENABLE_BATCH_PROCESSING", True
            ),
            enable_position_caching=self._get_env_bool(
                "VLLM_ENABLE_POSITION_CACHING", True
            ),
            rollout_percentage=float(os.getenv("VLLM_ROLLOUT_PERCENTAGE", "0.0")),
            enable_ab_testing=self._get_env_bool("VLLM_ENABLE_AB_TESTING", False),
            ab_test_group=os.getenv("VLLM_AB_TEST_GROUP"),
            enable_special_token_caching=self._get_env_bool(
                "VLLM_ENABLE_SPECIAL_TOKEN_CACHING", True
            ),
            enable_sequence_caching=self._get_env_bool(
                "VLLM_ENABLE_SEQUENCE_CACHING", True
            ),
            batch_size_threshold=int(os.getenv("VLLM_BATCH_SIZE_THRESHOLD", "8")),
            max_cache_size=int(os.getenv("VLLM_MAX_CACHE_SIZE", "1000")),
            enable_memory_monitoring=self._get_env_bool(
                "VLLM_ENABLE_MEMORY_MONITORING", True
            ),
            memory_usage_threshold=float(
                os.getenv("VLLM_MEMORY_USAGE_THRESHOLD", "0.8")
            ),
            enable_aggressive_cleanup=self._get_env_bool(
                "VLLM_ENABLE_AGGRESSIVE_CLEANUP", False
            ),
        )

    def _get_env_bool(self, key: str, default: bool) -> bool:
        """Get boolean value from environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")

    def _validate_config(self) -> None:
        """Validate the configuration."""
        if not 0.0 <= self.config.rollout_percentage <= 1.0:
            raise ValueError("rollout_percentage must be between 0.0 and 1.0")

        if self.config.enable_ab_testing and self.config.ab_test_group not in (
            "control",
            "treatment",
        ):
            raise ValueError(
                "ab_test_group must be 'control' or 'treatment' when AB testing is enabled"
            )

        if self.config.batch_size_threshold < 1:
            raise ValueError("batch_size_threshold must be at least 1")

        if not 0.0 <= self.config.memory_usage_threshold <= 1.0:
            raise ValueError("memory_usage_threshold must be between 0.0 and 1.0")

    def should_use_compact_cache(self, request_id: str = None) -> bool:
        """
        Determine if compact cache should be used for a request.

        Args:
            request_id: Optional request ID for consistent routing

        Returns:
            True if compact cache should be used
        """
        # Check if compact cache is globally enabled
        if not self.config.enable_compact_cache:
            return False

        # Check rollout percentage
        if self.config.rollout_percentage < 1.0:
            # Use request_id for consistent routing
            if request_id is None:
                import random

                request_id = str(random.random())

            # Simple hash-based routing
            hash_value = hash(request_id) % 100
            if hash_value >= self.config.rollout_percentage * 100:
                return False

        # Check AB testing
        if self.config.enable_ab_testing:
            if self.config.ab_test_group == "control":
                return False
            elif self.config.ab_test_group == "treatment":
                return True
            else:
                # Random assignment for AB testing
                import random

                return random.random() < 0.5

        return True

    def should_use_token_aware_scheduling(self) -> bool:
        """Determine if token-aware scheduling should be used."""
        return (
            self.config.enable_compact_cache
            and self.config.enable_token_aware_scheduling
        )

    def should_use_batch_processing(self) -> bool:
        """Determine if batch processing should be used."""
        return self.config.enable_compact_cache and self.config.enable_batch_processing

    def should_use_position_caching(self) -> bool:
        """Determine if position caching should be used."""
        return self.config.enable_compact_cache and self.config.enable_position_caching

    def get_model_specific_config(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return self.config.model_specific_optimizations.get(model_name, {})

    def update_rollout_percentage(self, percentage: float) -> None:
        """Update the rollout percentage (for dynamic configuration)."""
        if not 0.0 <= percentage <= 1.0:
            raise ValueError("rollout_percentage must be between 0.0 and 1.0")
        self.config.rollout_percentage = percentage
        logger.info(f"Updated rollout percentage to {percentage}")

    def enable_full_rollout(self) -> None:
        """Enable full rollout of compact cache."""
        self.config.rollout_percentage = 1.0
        logger.info("Enabled full rollout of compact cache")

    def disable_rollout(self) -> None:
        """Disable rollout of compact cache."""
        self.config.rollout_percentage = 0.0
        logger.info("Disabled rollout of compact cache")

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        return {
            "enable_compact_cache": self.config.enable_compact_cache,
            "enable_token_aware_scheduling": self.config.enable_token_aware_scheduling,
            "enable_batch_processing": self.config.enable_batch_processing,
            "enable_position_caching": self.config.enable_position_caching,
            "rollout_percentage": self.config.rollout_percentage,
            "enable_ab_testing": self.config.enable_ab_testing,
            "ab_test_group": self.config.ab_test_group,
            "batch_size_threshold": self.config.batch_size_threshold,
            "max_cache_size": self.config.max_cache_size,
            "enable_memory_monitoring": self.config.enable_memory_monitoring,
            "memory_usage_threshold": self.config.memory_usage_threshold,
            "enable_aggressive_cleanup": self.config.enable_aggressive_cleanup,
        }


# Global feature flag manager instance
_feature_flag_manager: Optional[FeatureFlagManager] = None


def get_feature_flag_manager() -> FeatureFlagManager:
    """Get the global feature flag manager instance."""
    global _feature_flag_manager
    if _feature_flag_manager is None:
        _feature_flag_manager = FeatureFlagManager()
    return _feature_flag_manager


def initialize_feature_flags(config: Optional[EncoderCacheConfig] = None) -> None:
    """Initialize the global feature flag manager with custom configuration."""
    global _feature_flag_manager
    _feature_flag_manager = FeatureFlagManager(config)
    logger.info("Initialized feature flag manager with configuration")


def should_use_compact_cache(request_id: str = None) -> bool:
    """Convenience function to check if compact cache should be used."""
    return get_feature_flag_manager().should_use_compact_cache(request_id)


def should_use_token_aware_scheduling() -> bool:
    """Convenience function to check if token-aware scheduling should be used."""
    return get_feature_flag_manager().should_use_token_aware_scheduling()


def should_use_batch_processing() -> bool:
    """Convenience function to check if batch processing should be used."""
    return get_feature_flag_manager().should_use_batch_processing()


def should_use_position_caching() -> bool:
    """Convenience function to check if position caching should be used."""
    return get_feature_flag_manager().should_use_position_caching()
