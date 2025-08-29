# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration-driven GPU detection for attention backend selection."""

import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from functools import cache

import torch
import yaml

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import resolve_obj_by_qualname

logger = init_logger(__name__)


@dataclass
class BackendRequirement:
    """Requirements for a specific attention backend."""
    dependencies: List[str]
    head_sizes: Optional[List[int]] = None
    dtypes: Optional[List[str]] = None
    min_capability: Optional[Tuple[int, int]] = None
    max_capability: Optional[Tuple[int, int]] = None
    description: str = ""


@dataclass
class BackendConfig:
    """Configuration for a specific attention backend."""
    name: str
    priority: int
    requirements: BackendRequirement
    description: str = ""


class ConfigDrivenGPUDetector:
    """Configuration-driven GPU detector for attention backend selection."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the GPU detector with configuration file.
        
        Args:
            config_path: Path to the YAML configuration file. If None, uses default path.
        """
        if config_path is None:
            # Use default config path relative to this file
            config_path = os.path.join(os.path.dirname(__file__), "gpu_detection_config.yaml")
        
        self.config_path = config_path
        self.config = self._load_config()
        self._device_info_cache: Optional[Dict[str, Any]] = None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded GPU detection config from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"GPU detection config not found at {self.config_path}, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load GPU detection config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when config file is not available."""
        return {
            "gpu_architectures": {
                "blackwell": {
                    "min_capability": [10, 0],
                    "max_capability": [10, 9],
                    "backends": [
                        {
                            "name": "flashinfer",
                            "priority": 1,
                            "requirements": {
                                "dependencies": ["flashinfer"],
                                "head_sizes": [64, 128, 256],
                                "dtypes": ["float16", "bfloat16"]
                            }
                        },
                        {
                            "name": "flex_attention",
                            "priority": 2,
                            "requirements": {}
                        }
                    ]
                }
            },
            "fallback_backend": "torch_sdpa"
        }
    
    def get_optimal_backend(self, head_size: int, dtype: torch.dtype,
                           kv_cache_dtype: Optional[str] = None,
                           block_size: int = 16,
                           is_attention_free: bool = False,
                           use_mla: bool = False,
                           has_sink: bool = False) -> str:
        """Get the optimal attention backend based on current hardware and requirements.
        
        Args:
            head_size: Size of attention heads
            dtype: Data type for computations
            kv_cache_dtype: KV cache data type
            block_size: Block size for attention
            is_attention_free: Whether this is an attention-free model
            use_mla: Whether to use MLA (Multi-Level Attention)
            has_sink: Whether the model has sink tokens
            
        Returns:
            Name of the optimal backend to use
        """
        # Handle special cases first
        if is_attention_free:
            return "no_attention"
        
        if use_mla:
            return self._get_mla_backend(head_size, dtype, block_size)
        
        # Check for environment variable overrides
        env_backend = self._check_env_overrides()
        if env_backend:
            return env_backend
        
        # Get current device information
        device_info = self._get_device_info()
        
        # Find matching GPU architecture
        architecture = self._find_matching_architecture(device_info)
        if not architecture:
            logger.warning(f"No matching GPU architecture found for capability {device_info['capability']}")
            return self.config.get('fallback_backend', 'torch_sdpa')
        
        # Check backends in priority order
        backends = architecture.get('backends', [])
        backends.sort(key=lambda x: x.get('priority', 999))
        
        for backend_config in backends:
            if self._check_backend_requirements(backend_config, device_info,
                                              head_size, dtype, kv_cache_dtype, block_size):
                backend_name = backend_config['name']
                logger.info(f"Selected backend: {backend_name} for {device_info['device_name']}")
                return backend_name
        
        # Fallback
        fallback = self.config.get('fallback_backend', 'torch_sdpa')
        logger.warning(f"No suitable backend found, using fallback: {fallback}")
        return fallback
    
    def _get_mla_backend(self, head_size: int, dtype: torch.dtype, block_size: int) -> str:
        """Get MLA-specific backend."""
        # MLA backend selection logic
        if current_platform.has_device_capability(100):  # Blackwell
            return "cutlass_mla"
        elif block_size == 64:
            return "flashmla"
        else:
            return "triton_mla"
    
    def _check_env_overrides(self) -> Optional[str]:
        """Check for environment variable overrides."""
        # Check VLLM_ATTENTION_BACKEND
        backend_env = os.environ.get('VLLM_ATTENTION_BACKEND')
        if backend_env:
            logger.info(f"Using backend from environment variable: {backend_env}")
            return backend_env
        
        return None
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get current device information with caching."""
        if self._device_info_cache is None:
            capability = current_platform.get_device_capability()
            self._device_info_cache = {
                'capability': capability,
                'device_name': current_platform.get_device_name(),
                'has_flash_attn': self._check_dependency('flash_attn'),
                'has_xformers': self._check_dependency('xformers'),
                'has_flashinfer': self._check_dependency('flashinfer'),
                'is_rocm': current_platform.is_rocm(),
            }
        return self._device_info_cache
    
    def _find_matching_architecture(self, device_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find the matching GPU architecture configuration."""
        capability = device_info['capability']
        
        for arch_name, arch_config in self.config.get('gpu_architectures', {}).items():
            min_cap = arch_config.get('min_capability')
            max_cap = arch_config.get('max_capability')
            
            # Check capability range
            if min_cap and capability < tuple(min_cap):
                continue
            if max_cap and capability > tuple(max_cap):
                continue
            
            return arch_config
        
        return None
    
    def _check_backend_requirements(self, backend_config: Dict[str, Any],
                                  device_info: Dict[str, Any],
                                  head_size: int, dtype: torch.dtype,
                                  kv_cache_dtype: Optional[str],
                                  block_size: int) -> bool:
        """Check if a backend meets all requirements."""
        requirements = backend_config.get('requirements', {})
        
        # Check dependencies
        for dep in requirements.get('dependencies', []):
            if not device_info.get(f'has_{dep}'):
                return False
        
        # Check head size
        if 'head_sizes' in requirements:
            if head_size not in requirements['head_sizes']:
                return False
        
        # Check data type
        if 'dtypes' in requirements:
            dtype_str = str(dtype).split('.')[-1]
            if dtype_str not in requirements['dtypes']:
                return False
        
        # Check capability range
        capability = device_info['capability']
        if 'min_capability' in requirements:
            if capability < tuple(requirements['min_capability']):
                return False
        if 'max_capability' in requirements:
            if capability > tuple(requirements['max_capability']):
                return False
        
        # Check special conditions
        if 'rocm_only' in requirements and not device_info['is_rocm']:
            return False
        
        return True
    
    def _check_dependency(self, dep_name: str) -> bool:
        """Check if a dependency is available."""
        try:
            if dep_name == 'flash_attn':
                import vllm.vllm_flash_attn  # noqa: F401
                return True
            elif dep_name == 'xformers':
                from importlib.util import find_spec
                return find_spec("xformers.ops") is not None
            elif dep_name == 'flashinfer':
                # FlashInfer availability check
                try:
                    import flashinfer  # noqa: F401
                    return True
                except ImportError:
                    return False
        except ImportError:
            return False
        return False
    
    def get_backend_class(self, backend_name: str) -> type:
        """Get the backend class by name."""
        backend_map = {
            'flash_attention': 'vllm.attention.backends.flash_attn.FlashAttentionBackend',
            'xformers': 'vllm.attention.backends.xformers.XFormersBackend',
            'flashinfer': 'vllm.attention.backends.flashinfer.FlashInferBackend',
            'flex_attention': 'vllm.attention.backends.flex_attention.FlexAttentionBackend',
            'torch_sdpa': 'vllm.attention.backends.torch_sdpa.TorchSDPABackend',
            'no_attention': 'vllm.attention.backends.placeholder_attn.PlaceholderAttentionBackend',
        }
        
        if backend_name not in backend_map:
            raise ValueError(f"Unknown backend: {backend_name}")
        
        return resolve_obj_by_qualname(backend_map[backend_name])


# Global detector instance
_gpu_detector: Optional[ConfigDrivenGPUDetector] = None


def get_gpu_detector() -> ConfigDrivenGPUDetector:
    """Get the global GPU detector instance."""
    global _gpu_detector
    if _gpu_detector is None:
        _gpu_detector = ConfigDrivenGPUDetector()
    return _gpu_detector


def reset_gpu_detector() -> None:
    """Reset the global GPU detector instance (mainly for testing)."""
    global _gpu_detector
    _gpu_detector = None
