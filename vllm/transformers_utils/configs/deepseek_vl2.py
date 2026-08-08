# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from https://github.com/deepseek-ai/DeepSeek-VL2/blob/faf18023f24b962b32d9f0a2d89e402a8d383a78/deepseek_vl2/models/modeling_deepseek_vl_v2.py#L115-L268

from huggingface_hub.dataclasses import strict
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


@strict
class DeepseekVLV2TextConfig(DeepseekV2Config):
    kv_lora_rank: int | None = None


class DeepseekVLV2Config(PretrainedConfig):
    model_type = "deepseek_vl_v2"

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
        architectures = kwargs.setdefault("architectures", ["DeepseekVLV2ForCausalLM"])

        self.vision_config = VisionEncoderConfig(**kwargs.pop("vision_config", {}))
        self.projector_config = MlpProjectorConfig(**kwargs.pop("projector_config", {}))

        language_config = kwargs.pop("language_config", {})
        # DeepSeek-VL2 checkpoints (e.g. deepseek-vl2, deepseek-vl2-small) omit
        # several language-model fields in ``language_config`` and rely on the
        # defaults of the *original* DeepSeek-VL2 ``DeepseekV2Config`` shipped
        # with the model. vLLM instead parses ``language_config`` with the
        # built-in Transformers ``DeepseekV2Config``, whose generic defaults
        # differ (e.g. ``vocab_size`` 102400 -> 32000) and which does not treat
        # ``kv_lora_rank`` as optional. As a result the omitted fields silently
        # resolve to wrong values: this disables MLA detection
        # (``ModelConfig.is_deepseek_mla``) -- which crashes
        # ``DeepseekV2Attention`` on ``kv_lora_rank + qk_rope_head_dim`` -- and
        # breaks the vocab-size check when loading ``embed_tokens``.
        #
        # Restore the original DeepSeek-VL2 defaults only for fields the
        # checkpoint does NOT provide. ``setdefault`` intentionally leaves an
        # explicit value untouched, including an explicit ``null`` such as
        # ``kv_lora_rank: null`` used by the MHA-based deepseek-vl2-tiny.
        deepseek_vl2_reference_defaults = {
            "vocab_size": 102400,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
        }
        for key, value in deepseek_vl2_reference_defaults.items():
            language_config.setdefault(key, value)
        self.text_config = DeepseekVLV2TextConfig(**language_config)

        self.tile_tag = tile_tag
        self.global_view_pos = global_view_pos
        self.candidate_resolutions = candidate_resolutions
        self.vocab_size = self.text_config.vocab_size

        # update model_type for OCR models
        if "DeepseekOCRForCausalLM" in architectures:
            kwargs["model_type"] = "deepseek_ocr"
        elif "DeepseekOCR2ForCausalLM" in architectures:
            kwargs["model_type"] = "deepseek_ocr2"
        super().__init__(**kwargs)
