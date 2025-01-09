from abc import ABC, abstractmethod
from typing import Final, Generic, Optional, Protocol, TypeVar, Union

import torch
from transformers import PretrainedConfig

import vllm.envs as envs
from vllm.attention.selector import (backend_name_to_enum,
                                     get_global_forced_attn_backend)
from vllm.logger import init_logger
from vllm.platforms import _Backend, current_platform

logger = init_logger(__name__)

_C = TypeVar("_C", bound=PretrainedConfig)


class VisionEncoderInfo(ABC, Generic[_C]):

    def __init__(self, vision_config: _C) -> None:
        super().__init__()

        self.vision_config = vision_config

    @abstractmethod
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_max_image_tokens(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_image_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_patch_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_patch_grid_length(self) -> int:
        raise NotImplementedError


class VisionLanguageConfig(Protocol):
    vision_config: Final[PretrainedConfig]


def get_vision_encoder_info(
        hf_config: VisionLanguageConfig) -> VisionEncoderInfo:
    # Avoid circular imports
    from .clip import CLIPEncoderInfo, CLIPVisionConfig
    from .pixtral import PixtralHFEncoderInfo, PixtralVisionConfig
    from .siglip import SiglipEncoderInfo, SiglipVisionConfig

    vision_config = hf_config.vision_config
    if isinstance(vision_config, CLIPVisionConfig):
        return CLIPEncoderInfo(vision_config)
    if isinstance(vision_config, PixtralVisionConfig):
        return PixtralHFEncoderInfo(vision_config)
    if isinstance(vision_config, SiglipVisionConfig):
        return SiglipEncoderInfo(vision_config)

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


def get_vit_attn_backend(support_fa: bool = False) -> _Backend:
    """
    Get the available attention backend for Vision Transformer.
    """
    # TODO(Isotr0py): Remove `support_fa` after support FA for all ViTs attn.
    selected_backend: Optional[_Backend] = get_global_forced_attn_backend()
    if selected_backend is None:
        backend_by_env_var: Optional[str] = envs.VLLM_ATTENTION_BACKEND
        if backend_by_env_var is not None:
            selected_backend = backend_name_to_enum(backend_by_env_var)
    if selected_backend is None:
        # For Volta and Turing GPUs, use xformers instead.
        device_available = current_platform.has_device_capability(80)
        if device_available and support_fa:
            from transformers.utils import is_flash_attn_2_available
            if is_flash_attn_2_available():
                selected_backend = _Backend.FLASH_ATTN
            else:
                logger.warning_once(
                    "Current `vllm-flash-attn` has a bug inside vision module, "
                    "so we use xformers backend instead. You can run "
                    "`pip install flash-attn` to use flash-attention backend.")
                selected_backend = _Backend.XFORMERS
        elif current_platform.is_cpu() or current_platform.is_rocm():
            # ROCM doesn't support xformers
            selected_backend = _Backend.TORCH_SDPA
        else:
            selected_backend = _Backend.XFORMERS
    return selected_backend


def resolve_visual_encoder_outputs(
    encoder_outputs: Union[torch.Tensor, list[torch.Tensor]],
    feature_sample_layers: Optional[list[int]],
    post_layer_norm: Optional[torch.nn.LayerNorm],
    max_possible_layers: int,
) -> torch.Tensor:
    """Given the outputs a visual encoder module that may correspond to the
    output of the last layer, or a list of hidden states to be stacked,
    handle post normalization and resolve it into a single output tensor.

    Args:
        encoder_outputs: Output of encoder's last layer or all hidden states.
        feature_sample_layers: Optional layer indices to grab from the encoder
            outputs; if provided, encoder outputs must be a list.
        post_layer_norm: Post norm to apply to the output of the encoder.
        max_possible_layers: Total layers in the fully loaded visual encoder.

    """
    if feature_sample_layers is None:
        if post_layer_norm is not None:
            return post_layer_norm(encoder_outputs)
        return encoder_outputs

    # Get the hidden states corresponding to the layer indices.
    # Negative values are relative to the full visual encoder,
    # so offset them depending on how many layers were loaded.
    # NOTE: this assumes that encoder_outputs contains a list
    # of hidden states in the same order as the encoder layers
    # that produced them.
    offset = max_possible_layers - len(encoder_outputs)
    hs_pool = [
        encoder_outputs[layer_idx]
        if layer_idx >= 0 else encoder_outputs[layer_idx + offset]
        for layer_idx in feature_sample_layers
    ]

    # Apply post-norm on the final hidden state if we are using it
    uses_last_layer = feature_sample_layers[-1] in (len(hs_pool) - 1, -1)
    if post_layer_norm is not None and uses_last_layer:
        hs_pool[-1] = post_layer_norm(encoder_outputs)
    return torch.cat(hs_pool, dim=-1)
