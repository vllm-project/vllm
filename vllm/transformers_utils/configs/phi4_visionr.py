# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import Phi3Config


class Phi4VisionRConfig(Phi3Config):
    """Config for `microsoft/Phi-4-reasoning-vision-15B` (Phi-4 + SigLIP2 NaFlex).

    Vendored so vLLM does not need to import the checkpoint's remote code,
    which reaches into `transformers.models.siglip2.image_processing_siglip2`
    for symbols that moved when that module became a torchvision backend in
    transformers v5.4.
    """

    model_type = "phi4-siglip"

    def __init__(
        self,
        mm_vision_tower: str | None = None,
        mm_projector_type: str = "mlp2x_gelu",
        mm_hidden_size: int = 1152,
        min_num_patches: int = 256,
        max_num_patches: int = 3600,
        vision_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mm_vision_tower = mm_vision_tower
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches
        self.vision_config = vision_config
