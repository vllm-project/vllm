# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for Moondream3 model."""

from transformers import PretrainedConfig


class Moondream3VisionConfig(PretrainedConfig):
    """Vision encoder configuration for Moondream3."""

    model_type = "moondream3_vision"

    def __init__(
        self,
        enc_dim: int = 1152,
        enc_patch_size: int = 14,
        enc_n_layers: int = 27,
        enc_ff_dim: int = 4304,
        enc_n_heads: int = 16,
        proj_inner_dim: int = 8192,
        crop_size: int = 378,
        max_crops: int = 12,
        overlap_margin: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enc_dim = enc_dim
        self.enc_patch_size = enc_patch_size
        self.enc_n_layers = enc_n_layers
        self.enc_ff_dim = enc_ff_dim
        self.enc_n_heads = enc_n_heads
        self.proj_inner_dim = proj_inner_dim
        self.crop_size = crop_size
        self.max_crops = max_crops
        self.overlap_margin = overlap_margin

        # Standard HuggingFace attributes for vision config
        self.hidden_size = enc_dim
        self.num_attention_heads = enc_n_heads
        self.num_hidden_layers = enc_n_layers
        self.intermediate_size = enc_ff_dim
        self.patch_size = enc_patch_size
        self.image_size = crop_size


class Moondream3TextConfig(PretrainedConfig):
    """Text decoder configuration for Moondream3."""

    model_type = "moondream3_text"

    def __init__(
        self,
        dim: int = 2048,
        ff_dim: int = 8192,
        n_layers: int = 24,
        vocab_size: int = 51200,
        max_context: int = 4096,
        n_heads: int = 32,
        n_kv_heads: int = 32,
        prefix_attn: int = 730,
        prefix_lm_left_padding: int = 1,
        rope_theta: float = 1500000.0,
        moe: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Store original moondream3 config names
        self.dim = dim
        self.ff_dim = ff_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.prefix_attn = prefix_attn
        # Include the BOS token in the bidirectional prefix-LM region:
        # prefix range = [BOS] + [729 image tokens].
        self.prefix_lm_left_padding = prefix_lm_left_padding
        self.max_context = max_context
        self.rope_theta = rope_theta

        # MoE config
        moe = moe or {}
        self.moe_start_layer = moe.get("start_layer", 4)
        self.moe_num_experts = moe.get("n_experts", 64)
        self.moe_experts_per_token = moe.get("n_experts_per_tok", 8)
        self.moe_expert_inner_dim = moe.get("expert_inner_dim", 1024)

        # Standard HuggingFace attributes (required by vLLM)
        self.hidden_size = dim
        self.num_attention_heads = n_heads
        self.num_key_value_heads = n_kv_heads
        self.num_hidden_layers = n_layers
        self.intermediate_size = ff_dim
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_context

        # Moondream3 uses token 0 (<|endoftext|>) as both BOS and EOS.
        # Token 3 (<|md_reserved_2|>) is the answer delimiter (answer_id).
        # The HF reference suppresses token 3 during generation so that the
        # model emits token 0 instead, but the practical effect is the same
        # as treating token 3 as an additional stop token.
        self.bos_token_id = 0
        self.eos_token_id = [0, 3]

        # MoE standard attributes
        self.num_local_experts = self.moe_num_experts
        self.num_experts_per_tok = self.moe_experts_per_token


class Moondream3RegionConfig(PretrainedConfig):
    """Region module configuration for Moondream3 (point/detect)."""

    model_type = "moondream3_region"

    def __init__(
        self,
        dim: int = 2048,
        coord_feat_dim: int = 256,
        coord_out_dim: int = 1024,
        size_feat_dim: int = 512,
        size_out_dim: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.coord_feat_dim = coord_feat_dim
        self.coord_out_dim = coord_out_dim
        self.size_feat_dim = size_feat_dim
        self.size_out_dim = size_out_dim


class Moondream3Config(PretrainedConfig):
    """Combined configuration for Moondream3 multimodal model."""

    model_type = "moondream3"
    is_composition = True

    def __init__(
        self,
        config: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        config = config or {}

        # Parse text config
        text_config = config.get("text", {})
        self.text_config = Moondream3TextConfig(**text_config)

        # Parse vision config
        vision_config = config.get("vision", {})
        self.vision_config = Moondream3VisionConfig(**vision_config)

        # Parse region config
        region_config = config.get("region", {})
        self.region_config = Moondream3RegionConfig(**region_config)

        # Token IDs used by the detect/point state machine.
        tokenizer_config = config.get("tokenizer", {})
        self.coord_token_id = tokenizer_config.get("coord_id", 5)
        self.size_token_id = tokenizer_config.get("size_id", 6)
        self.region_eos_token_id = tokenizer_config.get("eos_id", 0)

        # Store the original config dict for model access
        self.config = config

        # Expose key attributes at top level for vLLM compatibility
        self.hidden_size = self.text_config.hidden_size
        self.num_attention_heads = self.text_config.num_attention_heads
        self.num_key_value_heads = self.text_config.num_key_value_heads
        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.vocab_size = self.text_config.vocab_size
        self.intermediate_size = self.text_config.intermediate_size
        self.prefix_lm_left_padding = self.text_config.prefix_lm_left_padding

        # Moondream3 uses token 0 (<|endoftext|>) as both BOS and EOS.
        # Token 3 (<|md_reserved_2|>) is the answer delimiter (answer_id)
        # that the HF reference suppresses during generation; treating it
        # as an additional stop token achieves the same stopping behaviour.
        self.bos_token_id = 0
        self.eos_token_id = [0, 3]

    def get_text_config(self, decoder: bool = False) -> "Moondream3TextConfig":
        """Return the text config for vLLM's text_config detection.

        Args:
            decoder: Ignored. Only used for encoder-decoder models.
        """
        return self.text_config
