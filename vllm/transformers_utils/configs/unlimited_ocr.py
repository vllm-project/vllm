# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Unlimited-OCR (baidu/Unlimited-OCR) reuses
# the DeepSeek-OCR multimodal layout (DeepEncoder = SAM-ViT-B + CLIP-L, a linear
# MLP projector and a DeepSeek-V2 text backbone). The only architectural
# difference is the language model, which is a DeepSeek-V2 *MoE* with plain
# multi-head attention (``use_mla=False``) instead of the dense MLA backbone.
# We therefore reuse ``DeepseekVLV2Config`` for parsing the nested config.

from vllm.transformers_utils.configs.deepseek_vl2 import DeepseekVLV2Config


class UnlimitedOCRConfig(DeepseekVLV2Config):
    model_type = "unlimited-ocr"

    # An explicit ``__init__`` is required: Transformers v5 processes each
    # concrete config class' ``__init__`` signature to build nested sub-configs,
    # and an empty subclass (only overriding ``model_type``) would skip
    # ``DeepseekVLV2Config.__init__``, leaving ``text_config`` unset.
    def __init__(
        self,
        tile_tag: str = "2D",
        global_view_pos: str = "head",
        candidate_resolutions: tuple[tuple[int, int]] = ((384, 384),),
        rswa_window: int = 128,
        **kwargs,
    ):
        super().__init__(
            tile_tag=tile_tag,
            global_view_pos=global_view_pos,
            candidate_resolutions=candidate_resolutions,
            **kwargs,
        )
        self.rswa_window = rswa_window
