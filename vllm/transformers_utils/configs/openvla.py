# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from OpenVLA/Prismatic configuration structure

from typing import Any, Dict, List, Optional

from transformers import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING


# === Utilities for Mapping Prismatic names to HF names ===
# fmt: off
VISION_BACKBONE_TO_RESOLUTION: Dict[str, List[int]] = {
    "clip-vit-l": [224], "siglip-vit-so400m": [224], "dinov2-vit-l": [224], "in1k-vit-l": [224],
    "clip-vit-l-336px": [336],
    "siglip-vit-so400m-384px": [384],
    "dinoclip-vit-l-336px": [336, 336],
    "dinosiglip-vit-so-224px": [224, 224],
    "dinosiglip-vit-so-384px": [384, 384],
}
VISION_BACKBONE_TO_TIMM_ID: Dict[str, List[str]] = {
    "clip-vit-l": ["vit_large_patch14_clip_224.openai"],
    "clip-vit-l-336px": ["vit_large_patch14_clip_336.openai"],
    "dinov2-vit-l": ["vit_large_patch14_reg4_dinov2.lvd142m"],
    "in1k-vit-l": ["vit_large_patch16_224.augreg_in21k_ft_in1k"],
    "siglip-vit-so400m": ["vit_so400m_patch14_siglip_224"],
    "siglip-vit-so400m-384px": ["vit_so400m_patch14_siglip_384"],
    "dinoclip-vit-l-336px": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_large_patch14_clip_336.openai"],
    "dinosiglip-vit-so-224px": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224"],
    "dinosiglip-vit-so-384px": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_384"],
}
TIMM_OVERRIDE_ACT_LAYER: Dict[str, List[Optional[str]]] = {
    "clip-vit-l": ["quick_gelu"], "clip-vit-l-336px": ["quick_gelu"],
    "dinov2-vit-l": [None], "in1k-vit-l": [None],
    "siglip-vit-so400m": [None], "siglip-vit-so400m-384px": [None],
    "dinoclip-vit-l-336px": [None, "quick_gelu"],
    "dinosiglip-vit-so-224px": [None, None], "dinosiglip-vit-so-384px": [None, None]
}

LLM_BACKBONE_TO_HF_PATH = {
    "llama2-7b-pure": "meta-llama/Llama-2-7b-hf", "llama2-13b-pure": "meta-llama/Llama-2-13b-hf",
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf", "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "vicuna-v15-7b": "lmsys/vicuna-7b-v1.5", "vicuna-v15-13b": "lmsys/vicuna-13b-v1.5",
    "mistral-v0.1-7b-pure": "mistralai/Mistral-7B-v0.1",
    "mistral-v0.1-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.1",
    "phi-2-3b": "microsoft/phi-2",
}
LLM_BACKBONE_TO_HF_METACLASS = {
    "llama2-7b-pure": "llama", "llama2-13b-pure": "llama", "llama2-7b-chat": "llama", "llama2-13b-chat": "llama",
    "vicuna-v15-7b": "llama", "vicuna-v15-13b": "llama",
    "mistral-v0.1-7b-pure": "mistral", "mistral-v0.1-7b-instruct": "mistral",
    "phi-2-3b": "phi",
}
# fmt: on


class OpenVLAConfig(PretrainedConfig):
    """Configuration for OpenVLA model compatible with vLLM."""
    
    model_type: str = "openvla"
    is_composition: bool = False

    def __init__(
        self,
        vision_backbone_id: str = "siglip-vit-so400m",
        llm_backbone_id: str = "llama2-7b-pure",
        arch_specifier: str = "no-align+fused-gelu-mlp",
        use_fused_vision_backbone: Optional[bool] = None,
        image_resize_strategy: str = "resize-naive",
        image_sizes: Optional[List[int]] = None,
        text_config: Optional[Dict[str, Any]] = None,
        llm_max_length: int = 2048,
        pad_token_id: int = 32000,
        pad_to_multiple_of: int = 64,
        norm_stats: Optional[Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]] = None,
        n_action_bins: int = 256,
        **kwargs: Any,
    ) -> None:
        # Set OpenVLA-specific fields
        self.norm_stats = norm_stats
        self.n_action_bins = n_action_bins
        self.arch_specifier = arch_specifier
        
        # Determine vision backbone ID from image_sizes if not provided
        if vision_backbone_id is None and image_sizes is not None:
            # Try to infer from image_sizes
            if image_sizes == [224, 224]:
                vision_backbone_id = "siglip-vit-so400m"
            elif image_sizes == [384, 384]:
                vision_backbone_id = "siglip-vit-so400m-384px"
            else:
                vision_backbone_id = "siglip-vit-so400m"  # default
        
        # Validate vision backbone
        if vision_backbone_id not in VISION_BACKBONE_TO_RESOLUTION:
            # Use default if not found
            vision_backbone_id = "siglip-vit-so400m"
        
        # Validate LLM backbone
        if llm_backbone_id not in LLM_BACKBONE_TO_HF_PATH:
            llm_backbone_id = "llama2-7b-pure"  # default
        
        self.vision_backbone_id = vision_backbone_id
        self.llm_backbone_id = llm_backbone_id
        
        # Determine if using fused vision backbone
        self.use_fused_vision_backbone = (
            use_fused_vision_backbone
            if use_fused_vision_backbone is not None
            else any(self.vision_backbone_id.startswith(v) for v in ["dinoclip", "dinosiglip"])
        )
        
        # Set vision config fields
        self.timm_model_ids = VISION_BACKBONE_TO_TIMM_ID.get(self.vision_backbone_id, ["vit_so400m_patch14_siglip_224"])
        self.timm_override_act_layers = TIMM_OVERRIDE_ACT_LAYER.get(self.vision_backbone_id, [None])
        if image_sizes is not None:
            self.image_sizes = image_sizes
        else:
            self.image_sizes = VISION_BACKBONE_TO_RESOLUTION.get(self.vision_backbone_id, [224])
        self.image_resize_strategy = image_resize_strategy
        
        # Set LLM config fields
        self.hf_llm_id = LLM_BACKBONE_TO_HF_PATH.get(self.llm_backbone_id, "meta-llama/Llama-2-7b-hf")
        self.llm_max_length = llm_max_length
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        
        # Create text_config (LLM backbone config)
        # [IMPORTANT] HF Utilities actually look for a `text_config` field
        llm_metaclass = LLM_BACKBONE_TO_HF_METACLASS.get(self.llm_backbone_id, "llama")
        self.text_config = (
            CONFIG_MAPPING[llm_metaclass](**text_config)
            if text_config is not None
            else CONFIG_MAPPING[llm_metaclass]()
        )
        
        # Dispatch **kwargs to super()
        super().__init__(pad_token_id=pad_token_id, **kwargs)




