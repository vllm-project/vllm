# Copyright 2024 Rhymes AI. All rights reserved.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from typing import Mapping

from transformers import PretrainedConfig
from transformers.models.idefics2.configuration_idefics2 import (
    Idefics2VisionConfig)
from transformers.models.llama.configuration_llama import LlamaConfig

from vllm.logger import init_logger

logger = init_logger(__name__)


class AriaVisionConfig(Idefics2VisionConfig):
    model_type = "aria_vision_model"


class AriaMoELMConfig(LlamaConfig):
    """
    Configuration class for AriaMoE language model.

    This class extends the LlamaConfig to include additional parameters specific
    to the Mixture of Experts (MoE) architecture.
    """

    model_type = "aria_moe_lm"

    def __init__(
        self,
        moe_intermediate_size: int = 4096,
        moe_num_experts: int = 8,
        moe_topk: int = 2,
        moe_num_shared_experts: int = 2,
        **kwargs,
    ):
        """
        Initialize the AriaMoELMConfig.

        Args:
            moe_intermediate_size (int): The intermediate size for MoE layers.
                Default is 4096.
            moe_num_experts (int): The number of experts in the MoE layer.
                Default is 8.
            moe_topk (int): The number of top experts to route to for each 
                token. Default is 2.
            moe_num_shared_experts (int): The number of shared experts. Default
                is 2. 
            **kwargs: Additional keyword arguments to be passed to the parent
                LlamaConfig.
        """
        super().__init__(**kwargs)
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_num_experts = moe_num_experts
        self.moe_topk = moe_topk
        self.moe_num_shared_experts = moe_num_shared_experts


class AriaConfig(PretrainedConfig):
    """
    Configuration class for Aria model.
    This class handles the configuration for both vision and text components of
    the Aria model,
    as well as additional parameters for image token handling and projector
    mapping.

    Args:
        vision_config (AriaVisionConfig or dict): Configuration for the vision
            component.
        text_config (AriaMoELMConfig or dict): Configuration for the text
            component.
        projector_patch_to_query_dict (dict): Mapping of patch sizes to query
            dimensions.
        ignore_index (int): Index to ignore in loss calculation.
        image_token_index (int): Index used to represent image tokens.
        **kwargs: Additional keyword arguments passed to the parent class.
    Attributes:
        model_type (str): Type of the model, set to "aria".
        is_composition (bool): Whether the model is a composition of multiple
            components.
        ignore_index (int): Index to ignore in loss calculation.
        image_token_index (int): Index used to represent image tokens.
        projector_patch_to_query_dict (dict): Mapping of patch sizes to query
            dimensions.
        vision_config (AriaVisionConfig): Configuration for the vision
            component.
        text_config (AriaMoELMConfig): Configuration for the text component.
    """

    model_type = "aria"
    is_composition = False

    def __init__(
        self,
        vision_config: AriaVisionConfig = AriaVisionConfig(),  # noqa: B008
        text_config: AriaMoELMConfig = AriaMoELMConfig(),  # noqa: B008
        projector_patch_to_query_dict: Mapping[int, int] = {
            1225: 128,
            4900: 256,
        },
        ignore_index=-100,
        image_token_index=32000,
        tie_word_embeddings=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.tie_word_embeddings = tie_word_embeddings
        attn_implementation = kwargs.pop("attn_implementation", None)

        # Set the default attention implementation to flash_attention_2 if not
        # specified
        self._attn_implementation = ("flash_attention_2"
                                     if attn_implementation is None else
                                     attn_implementation)

        # Convert the keys and values of projector_patch_to_query_dict to
        # integers
        # This ensures consistency even if they were provided as strings
        self.projector_patch_to_query_dict = {
            int(k): int(v)
            for k, v in projector_patch_to_query_dict.items()
        }

        if isinstance(vision_config, dict) and "model_type" in vision_config:
            vision_config = AriaVisionConfig(**vision_config)
            if attn_implementation is None:
                vision_attn_implementation = "flash_attention_2"
            elif attn_implementation == "sdpa":
                logger.warning("SDPA is not supported for vit, using "
                               "flash_attention_2 instead")
                vision_attn_implementation = "flash_attention_2"
            else:
                vision_attn_implementation = attn_implementation
            vision_config._attn_implementation = vision_attn_implementation

        self.vision_config = vision_config

        if isinstance(text_config, dict) and "model_type" in text_config:
            text_attn_implementation = ("sdpa" if attn_implementation is None
                                        else attn_implementation)
            text_config = AriaMoELMConfig(**text_config)
            text_config._attn_implementation = text_attn_implementation

        self.text_config = text_config

        # This is needed for the static kv cache
        self.num_hidden_layers = self.text_config.num_hidden_layers
