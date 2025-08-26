# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration for Lite-Whisper models."""

from typing import Any, Dict, List, Optional

from transformers import WhisperConfig


class LiteWhisperConfig(WhisperConfig):
    """Configuration class for Lite-Whisper models.
    
    This configuration extends the standard Whisper configuration to include
    low-rank adaptation settings for efficient inference.
    
    Args:
        low_rank_config (List[Dict[str, int]], optional): 
            List of low-rank configurations for each encoder layer. Each dict
            can contain keys like 'q_proj', 'k_proj', 'v_proj', 'out_proj',
            'fc1', 'fc2' with integer values representing the low-rank dimensions.
            If None, no low-rank adaptations are applied.
    """
    
    model_type = "lite_whisper"
    
    def __init__(
        self,
        low_rank_config: Optional[List[Dict[str, int]]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.low_rank_config = low_rank_config or []
