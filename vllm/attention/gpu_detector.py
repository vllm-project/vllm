# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration-driven GPU detection for attention backend selection."""

import os
from typing import Dict, Any, Optional
import torch
import yaml

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import resolve_obj_by_qualname

logger = init_logger(__name__)


class GPUDetector:
    """Simple GPU detector for attention backend selection."""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "gpu_detection_config.yaml")
        
        self.config = self._load_config(config_path)
        self._device_info = None
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "gpu_architectures": {
                "blackwell": {
                    "min_capability": [10, 0],
                    "backends": [
                        {"name": "flashinfer", "priority": 1, "requirements": {"dependencies": ["flashinfer"]}},
                        {"name": "flex_attention", "priority": 2, "requirements": {}}
                    ]
                },
                "ampere": {
                    "min_capability": [8, 0],
                    "backends": [
                        {"name": "flash_attention", "priority": 1, "requirements": {"dependencies": ["flash_attn"]}},
                        {"name": "xformers", "priority": 2, "requirements": {"dependencies": ["xformers"]}}
                    ]
                }
            },
            "fallback_backend": "torch_sdpa"
        }
    
    def get_backend(self, head_size: int, dtype: torch.dtype, 
                   is_attention_free: bool = False, use_mla: bool = False) -> str:
        if is_attention_free:
            return "no_attention"
        
        if use_mla:
            return self._get_mla_backend()
        
        env_backend = os.environ.get('VLLM_ATTENTION_BACKEND')
        if env_backend:
            return env_backend
        
        device_info = self._get_device_info()
        architecture = self._find_architecture(device_info)
        
        if not architecture:
            return self.config.get('fallback_backend', 'torch_sdpa')
        
        backends = architecture.get('backends', [])
        backends.sort(key=lambda x: x.get('priority', 999))
        
        for backend in backends:
            if self._check_requirements(backend, device_info, head_size, dtype):
                return backend['name']
        
        return self.config.get('fallback_backend', 'torch_sdpa')
    
    def _get_mla_backend(self) -> str:
        if current_platform.has_device_capability(100):
            return "cutlass_mla"
        return "triton_mla"
    
    def _get_device_info(self) -> Dict[str, Any]:
        if self._device_info is None:
            self._device_info = {
                'capability': current_platform.get_device_capability(),
                'has_flash_attn': self._check_dep('flash_attn'),
                'has_xformers': self._check_dep('xformers'),
                'has_flashinfer': self._check_dep('flashinfer'),
            }
        return self._device_info
    
    def _find_architecture(self, device_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        capability = device_info['capability']
        
        for arch_config in self.config.get('gpu_architectures', {}).values():
            min_cap = arch_config.get('min_capability')
            if min_cap and capability >= tuple(min_cap):
                return arch_config
        return None
    
    def _check_requirements(self, backend: Dict[str, Any], device_info: Dict[str, Any],
                          head_size: int, dtype: torch.dtype) -> bool:
        requirements = backend.get('requirements', {})
        
        for dep in requirements.get('dependencies', []):
            if not device_info.get(f'has_{dep}'):
                return False
        
        if 'head_sizes' in requirements and head_size not in requirements['head_sizes']:
            return False
        
        if 'dtypes' in requirements:
            dtype_str = str(dtype).split('.')[-1]
            if dtype_str not in requirements['dtypes']:
                return False
        
        return True
    
    def _check_dep(self, dep: str) -> bool:
        try:
            if dep == 'flash_attn':
                import vllm.vllm_flash_attn  # noqa: F401
            elif dep == 'xformers':
                from importlib.util import find_spec
                return find_spec("xformers.ops") is not None
            elif dep == 'flashinfer':
                import flashinfer  # noqa: F401
            return True
        except ImportError:
            return False
    
    def get_backend_class(self, backend_name: str) -> type:
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


# Global instance
_detector = None

def get_detector() -> GPUDetector:
    global _detector
    if _detector is None:
        _detector = GPUDetector()
    return _detector
