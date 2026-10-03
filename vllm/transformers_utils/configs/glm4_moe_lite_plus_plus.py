# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config for ZEDA-GLM-4.7-Flash-Dynamic (Glm4MoeLitePlusPlus).

Extends GLM4 MoE Lite with Zero-Compute Expert (ZCE) support for dynamic MoE.
Bases on PretrainedConfig so all checkpoint fields are stored as attributes.
Reference: ZEDA fork transformers,
models/glm4_moe_lite_plus_plus/configuration_glm4_moe_lite_plus_plus.py
"""

from transformers import PretrainedConfig


class Glm4MoeLitePlusPlusConfig(PretrainedConfig):
    model_type = "glm4_moe_lite_plus_plus"

    def __init__(
        self,
        zce_nums: list = [64],
        zce_types: list = ["copy"],
        use_zce_mask: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.zce_nums = list(zce_nums)
        self.zce_types = list(zce_types)
        self.use_zce_mask = use_zce_mask

    @property
    def total_zce(self) -> int:
        return sum(self.zce_nums)

    @property
    def total_num_experts(self) -> int:
        return self.n_routed_experts + self.total_zce


__all__ = ["Glm4MoeLitePlusPlusConfig"]
