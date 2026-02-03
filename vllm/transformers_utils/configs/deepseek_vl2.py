# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from https://github.com/deepseek-ai/DeepSeek-VL2/blob/faf18023f24b962b32d9f0a2d89e402a8d383a78/deepseek_vl2/models/modeling_deepseek_vl_v2.py#L115-L268

from transformers import DeepseekV2Config, PretrainedConfig


class VisionEncoderConfig(PretrainedConfig):
    model_type: str = "vision"

    model_name: str = "vit_so400m_patch14_siglip_384.webli"
    image_size: int = 384
    patch_size: int = 16
    width: int = 1024
    layers: int = 24
    heads: int = 16
    mlp_ratio: int = 4
    global_pool: str = "map"
    ignore_head: bool = True
    class_token: bool = False
    num_classes: int = 0
    use_checkpoint: bool = False
    weight_init: str = "skip"
    deterministic: bool = False
    num_recomputing_layers: int = 0

    def __init__(
        self,
        model_name: str = "vit_so400m_patch14_siglip_384.webli",
        image_size: int = 384,
        patch_size: int = 16,
        width: int = 1024,
        layers: int = 24,
        heads: int = 16,
        mlp_ratio: int = 4,
        global_pool: str = "map",
        ignore_head: bool = True,
        class_token: bool = False,
        num_classes: int = 0,
        use_checkpoint: bool = False,
        **kwargs,
    ):
        self.model_name = model_name
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.global_pool = global_pool
        self.ignore_head = ignore_head
        self.class_token = class_token
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint

        super().__init__(**kwargs)


class MlpProjectorConfig(PretrainedConfig):
    model_type = "mlp_projector"
    projector_type: str = "downsample_mlp_gelu"
    input_dim: int = 1152
    n_embed: int = 2048
    depth: int = 2
    mlp_ratio: int = 1
    downsample_ratio: int = 2
    token_pooling: bool = False

    def __init__(
        self,
        projector_type: str = "downsample_mlp_gelu",
        input_dim: int = 1152,
        n_embed: int = 2048,
        depth: int = 2,
        mlp_ratio: int = 1,
        downsample_ratio: int = 2,
        **kwargs,
    ):
        self.projector_type = projector_type
        self.input_dim = input_dim
        self.n_embed = n_embed
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.downsample_ratio = downsample_ratio

        super().__init__(**kwargs)


class DeepseekVLV2Config(PretrainedConfig):
    model_type = "deepseek_vl_v2"
    vision_config: VisionEncoderConfig
    projector_config: MlpProjectorConfig

    tile_tag: str = "2D"
    global_view_pos: str = "head"
    candidate_resolutions: tuple[tuple[int, int]] = ((384, 384),)

    def __init__(
        self,
        tile_tag: str = "tile_tag",
        global_view_pos: str = "head",
        candidate_resolutions: tuple[tuple[int, int]] = ((384, 384),),
        **kwargs,
    ):
        super().__init__(**kwargs)

        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionEncoderConfig(**vision_config)

        projector_config = kwargs.get("projector_config", {})
        self.projector_config = MlpProjectorConfig(**projector_config)

        language_config = kwargs.get("language_config", {})
        self.text_config = DeepseekV2Config(**language_config)

        self.tile_tag = tile_tag
        self.global_view_pos = global_view_pos
        self.candidate_resolutions = candidate_resolutions
        self.vocab_size = self.text_config.vocab_size

        # update model_type for OCR models
        architectures = self.architectures or kwargs.get("architectures", [])
        if "DeepseekOCRForCausalLM" in architectures:
            self.model_type = "deepseek_ocr"
        elif "DeepseekOCR2ForCausalLM" in architectures:
            self.model_type = "deepseek_ocr2"
