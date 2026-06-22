# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

from vllm.transformers_utils.configs.deepseek_vl2 import DeepseekVLV2Config


class UnlimitedOCRConfig(DeepseekVLV2Config):
    model_type = "unlimited-ocr"

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("architectures", ["UnlimitedOCRForCausalLM"])
        super().__init__(**kwargs)

        sliding_window_size = getattr(self.text_config, "sliding_window_size", None)
        if getattr(self.text_config, "sliding_window", None) is None:
            self.text_config.sliding_window = sliding_window_size

        self.sliding_window_size = sliding_window_size
        self.sliding_window = getattr(self.text_config, "sliding_window", None)
