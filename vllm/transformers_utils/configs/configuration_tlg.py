# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" OpenAI GPT-2 configuration"""
from collections import OrderedDict
from typing import Any, List, Mapping, Optional

from transformers import PreTrainedTokenizer, TensorType, is_torch_available
from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxConfigWithPast, PatchingSpec
from transformers.utils import logging

""" TNLGv4 model configuration """
logger = logging.get_logger(__name__)

# TLG_PRETRAINED_CONFIG_ARCHIVE_MAP = {
#     "tnlgv4": "https://turingmodelshare.blob.core.windows.net/data/checkpoints/tnlgv4/huggingface/config.json",
#     "tlgv4.6": "https://turingmodelshare.blob.core.windows.net/data/checkpoints/tlgv4.6/huggingface/config.json",
# }


def next_mult(x, y):
    return (x + y - 1) // y * y


class TLGv4Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`GPT2Model`] or a [`TFGPT2Model`]. It is used to
    instantiate a GPT-2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the GPT-2
    [gpt2](https://huggingface.co/gpt2) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GPT2Model`] or [`TFGPT2Model`].
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*, defaults to None):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        summary_type (`string`, *optional*, defaults to `"cls_index"`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].

            Has to be one of the following options:

                - `"last"`: Take the last token hidden state (like XLNet).
                - `"first"`: Take the first token hidden state (like BERT).
                - `"mean"`: Take the mean of all tokens hidden states.
                - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - `"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].

            Whether or not to add a projection after the vector extraction.
        summary_activation (`str`, *optional*):
            Argument used when doing sequence summary. Used in for the multiple choice head in
            [`GPT2DoubleHeadsModel`].

            Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
        summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].

            Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
        summary_first_dropout (`float`, *optional*, defaults to 0.1):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].

            The dropout ratio to be used after the projection and activation.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            Whether to additionally scale attention weights by `1 / layer_idx + 1`.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
            dot-product/softmax to float() when training with mixed precision.

    Example:

    ```python
    >>> from transformers import GPT2Config, GPT2Model

    >>> # Initializing a GPT2 configuration
    >>> configuration = GPT2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = GPT2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "tnlgv4"
    keys_to_ignore_at_inference = ["past_key_values"]
    

    def __init__(
        self,
        # General information about the model
        vocab_size: int =100352,
        max_position_embeddings: int = 8192,
        # RoPE Related Parameters
        rope_embedding_base: float = 10**6,
        rope_position_scale: float = 1.0,
        # General Model Parameters
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        # KV Shared Attention Configurations
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        # GEGELU Related Parameters
        hidden_act: str = "gegelu",
        gegelu_limit: float = 20.0,
        gegelu_pad_to_256: bool = True,
        ff_dim_multiplier: int = 4,
        # Block Sparse Attention
        blocksparse_homo_head_pattern: bool = False,
        blocksparse_block_size: int = 64,
        blocksparse_num_local_blocks: int = 16,
        blocksparse_vert_stride: int = 8,
        blocksparse_triton_kernel_block_size: int = 64,
        # Reegularization parameters
        embedding_dropout_prob: float =0.1,
        attention_dropout_prob: float = 0.0,
        ffn_dropout_prob: float = 0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        # MuP parameters
        mup_use_scaling: bool = True,
        mup_width_multiplier: bool = 8.0,
        mup_embedding_multiplier: bool = 10.0,
        mup_attn_multiplier: bool =1.0,
        use_cache=True,
        # The model does not have a bos token id
        bos_token_id: Optional[int] = None,
        eos_token_id: int = 100257,
        reorder_and_upcast_attn=False,
        dense_attention_every_n_layers: int = 0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_embedding_base = rope_embedding_base
        self.rope_position_scale = rope_position_scale
        self.hidden_size = hidden_size
        # QK Shared Attention
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        # Block Sparse Attention Pattern
        self.blocksparse_homo_head_pattern = blocksparse_homo_head_pattern
        self.blocksparse_block_size = blocksparse_block_size
        self.blocksparse_num_local_blocks = blocksparse_num_local_blocks
        self.blocksparse_vert_stride = blocksparse_vert_stride
        self.blocksparse_triton_kernel_block_size = blocksparse_triton_kernel_block_size

        # Activation function
        self.hidden_act = hidden_act
        self.gegelu_limit = gegelu_limit
        self.gegelu_pad_to_256 = gegelu_pad_to_256
        self.ff_dim_multiplier = ff_dim_multiplier
        # General regularization
        self.embedding_dropout_prob = embedding_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.ffn_dropout_prob = ffn_dropout_prob

        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        
        # MuP parameters
        self.mup_use_scaling = mup_use_scaling
        self.mup_width_multiplier = mup_width_multiplier
        self.mup_embedding_multiplier = mup_embedding_multiplier
        self.mup_attn_multiplier = mup_attn_multiplier
        self.use_cache = use_cache

        self.reorder_and_upcast_attn = reorder_and_upcast_attn

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.dense_attention_every_n_layers = dense_attention_every_n_layers


        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
    
    @property
    def intermediate_size(self) -> int:
        if getattr(self, 'ff_intermediate_size', None):
            return self.ff_intermediate_size
        intermediate_size = (self.ff_dim_multiplier) * (self.hidden_size // 3) * 2
        if self.gegelu_pad_to_256:
            intermediate_size = next_mult(intermediate_size, 256)
        return intermediate_size


class GPT2OnnxConfig(OnnxConfigWithPast):
    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "default",
        patching_specs: List[PatchingSpec] = None,
        use_past: bool = False,
    ):
        super().__init__(config, task=task, patching_specs=patching_specs, use_past=use_past)
        if not getattr(self._config, "pad_token_id", None):
            # TODO: how to do that better?
            self._config.pad_token_id = 0

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = OrderedDict({"input_ids": {0: "batch", 1: "sequence"}})
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")
            common_inputs["attention_mask"] = {0: "batch", 1: "past_sequence + sequence"}
        else:
            common_inputs["attention_mask"] = {0: "batch", 1: "sequence"}

        return common_inputs

    @property
    def num_layers(self) -> int:
        return self._config.n_layer

    @property
    def num_attention_heads(self) -> int:
        return self._config.n_head

    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        common_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # We need to order the input in the way they appears in the forward()
        ordered_inputs = OrderedDict({"input_ids": common_inputs["input_ids"]})

        # Need to add the past_keys
        if self.use_past:
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch

                batch, seqlen = common_inputs["input_ids"].shape
                # Not using the same length for past_key_values
                past_key_values_length = seqlen + 2
                past_shape = (
                    batch,
                    self.num_attention_heads,
                    past_key_values_length,
                    self._config.hidden_size // self.num_attention_heads,
                )
                ordered_inputs["past_key_values"] = [
                    (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(self.num_layers)
                ]

        ordered_inputs["attention_mask"] = common_inputs["attention_mask"]
        if self.use_past:
            mask_dtype = ordered_inputs["attention_mask"].dtype
            ordered_inputs["attention_mask"] = torch.cat(
                [ordered_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )

        return ordered_inputs

    @property
    def default_onnx_opset(self) -> int:
        return 13