from typing import List, Optional

from transformers import PretrainedConfig


#TODO (ywang96): Remove this file and import it from
# transformers once the new release with Chameleon support
# is available.
class ChameleonConfig(PretrainedConfig):
    model_type = "chameleon"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=65536,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        model_parallel_size=1,
        swin_norm=False,
        vq_config=None,
        vocabulary_map=None,
        mlp_bias=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_bias = mlp_bias

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.model_parallel_size = model_parallel_size
        self.swin_norm = swin_norm

        if vq_config is None:
            vq_config = {}

        self.vq_config = ChameleonVQVAEConfig(**vq_config)

        self.vocabulary_map = vocabulary_map

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling,
                          dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, "
                f"`type` and `factor`, got {self.rope_scaling}")
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in [
                "linear", "dynamic"
        ]:
            raise ValueError(
                "`rope_scaling`'s type field must be one of ['linear', "
                f"'dynamic'], got {rope_scaling_type}")
        if rope_scaling_factor is None or not isinstance(
                rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(
                "`rope_scaling`'s factor field must be a float > 1, "
                f"got {rope_scaling_factor}")


class ChameleonVQVAEConfig(PretrainedConfig):

    model_type = "chameleon_vqgan"

    def __init__(
        self,
        embed_dim: int = 256,
        num_embeddings: int = 8192,
        double_latent: bool = False,
        latent_channels: int = 256,
        resolution: int = 512,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_multiplier: List[int] = [1, 1, 2, 2, 4],  #noqa
        num_res_blocks: int = 2,
        attn_resolutions: Optional[List[int]] = None,
        dropout: float = 0.0,
        attn_type: str = "vanilla",
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_embeddings = num_embeddings
        self.double_latent = double_latent
        self.latent_channels = latent_channels
        self.resolution = resolution
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_multiplier = channel_multiplier
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.attn_type = attn_type
        self.initializer_range = initializer_range
