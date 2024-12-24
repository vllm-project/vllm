import enum
from typing import Optional

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.wde.core.layers.attention.abstract import AttentionType
from vllm.wde.core.llm_engine import LLMEngine

logger = init_logger(__name__)


class _Backend(enum.Enum):
    FLASH_ATTN = enum.auto()
    XFORMERS = enum.auto()
    ROCM_FLASH = enum.auto()
    TORCH_SDPA = enum.auto()
    OPENVINO = enum.auto()
    FLASHINFER = enum.auto()
    PALLAS = enum.auto()
    IPEX = enum.auto()
    TORCH_NAIVE = enum.auto()

    @staticmethod
    def backend_name_to_enum(backend_name: str) -> "_Backend":
        assert backend_name is not None

        backend_members = _Backend.__members__
        if backend_name not in backend_members:
            raise ValueError(
                f"Invalid attention backend '{backend_name}'. "
                f"Available backends: {', '.join(backend_members)} "
                "(case-sensitive).")

        return _Backend[backend_name]


class AttnBackend:

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        model_config = engine.engine_config.model_config
        num_heads = model_config.get_num_attention_heads()
        head_size = model_config.get_head_size()
        num_kv_heads = model_config.get_num_kv_heads()
        sliding_window = model_config.get_sliding_window()
        dtype = model_config.dtype

        backend = cls.which_attn_to_use(num_heads, head_size, num_kv_heads,
                                        sliding_window, dtype)

        backend_cls = cls.get_backend_cls(backend)

        attn_type = AttentionType.attn_type_name_to_enum(
            engine.workflow.attn_type)

        return backend_cls(attn_type)

    @staticmethod
    def get_backend_cls(backend):
        if backend == _Backend.FLASH_ATTN:
            logger.info("Using FLASH ATTN backend.")
            from vllm.wde.prefill_only.layers.attention.backends.flash_attn import (  # noqa: E501
                PrefillOnlyFlashAttentionBackend)
            return PrefillOnlyFlashAttentionBackend
        if backend == _Backend.XFORMERS:
            logger.info("Using XFormers backend.")
            from vllm.wde.prefill_only.layers.attention.backends.xformers import (  # noqa: E501
                PrefillOnlyXFormersBackend)
            return PrefillOnlyXFormersBackend
        elif backend == _Backend.TORCH_SDPA:
            logger.info("Using Torch SDPA backend.")
            from vllm.wde.prefill_only.layers.attention.backends.torch_sdpa import (  # noqa: E501
                PrefillOnlyTorchSDPABackend)
            return PrefillOnlyTorchSDPABackend
        elif backend == _Backend.FLASHINFER:
            logger.info("Using Flashinfer backend.")
            logger.info("When using Flashinfer backend in encode only models, "
                        "you are actually using FLASH ATTN backend")
            from vllm.wde.prefill_only.layers.attention.backends.flashinfer import (  # noqa: E501
                PrefillOnlyFlashInferBackend)
            return PrefillOnlyFlashInferBackend
        elif backend == _Backend.TORCH_NAIVE:
            logger.info("Using Torch naive backend.")
            from vllm.wde.prefill_only.layers.attention.backends.torch_naive import (  # noqa: E501
                PrefillOnlyTorchNAIVEBackend)
            return PrefillOnlyTorchNAIVEBackend
        else:
            raise ValueError("Invalid attention backend.")

    @classmethod
    def which_attn_to_use(cls, num_heads: int, head_size: int,
                          num_kv_heads: int, sliding_window: Optional[int],
                          dtype: torch.dtype):
        # Default case.
        selected_backend = _Backend.FLASH_ATTN

        # get_env_variable_attn_backend
        # Check the environment variable and override if specified
        backend_by_env_var: Optional[str] = envs.VLLM_ATTENTION_BACKEND
        if backend_by_env_var is not None:
            selected_backend = _Backend.backend_name_to_enum(
                backend_by_env_var)

        # FlashAttn in NVIDIA GPUs.
        if selected_backend == _Backend.FLASH_ATTN:
            if current_platform.get_device_capability()[0] < 8:
                # Volta and Turing NVIDIA GPUs.
                logger.info(
                    "Cannot use FlashAttention-2 backend for Volta and Turing "
                    "GPUs.")
                selected_backend = _Backend.XFORMERS
            elif dtype not in (torch.float16, torch.bfloat16):
                logger.info(
                    "Cannot use FlashAttention-2 backend for dtype other than "
                    "torch.float16 or torch.bfloat16.")
                selected_backend = _Backend.XFORMERS
            elif sliding_window is not None:
                logger.info(
                    "Cannot use FlashAttention-2 backend due to sliding window."
                )
                selected_backend = _Backend.XFORMERS

        return selected_backend
