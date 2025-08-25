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
        
        # # Ensure proper generation config like standard Whisper
        # # These are critical for proper generation behavior
        # if self.begin_suppress_tokens is None:
        #     # Standard Whisper begin_suppress_tokens + additional problematic tokens
        #     self.begin_suppress_tokens = [1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 30, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254, 50258, 50359, 50360, 50361, 50362, 220, 50257]
        # if self.max_length is None:
        #     self.max_length = 448  # Standard Whisper max length
