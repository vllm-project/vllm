# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseGlimmer model configuration for vLLM.

Native vLLM copy of the MuseGlimmer HuggingFace configs
(``configuration_muse_glimmer.py``). MuseGlimmer's
``model_type`` is not yet registered in released transformers, so vLLM ships
this config so partners can serve MuseGlimmer checkpoints *without* trust_remote_code.

The field set and defaults are kept byte-for-byte in sync with the HF reference
so a checkpoint's ``config.json`` deserializes identically here. Both the text
and vision configs are consumed by the native multimodal serving path.
"""

from __future__ import annotations

from transformers import Qwen3Config
from transformers.configuration_utils import PretrainedConfig


def _default_no_rope_layers(num_hidden_layers: int) -> list[int]:
    # iRoPE mask: NoPE every 4 layers, counted backward from the last layer.
    stride = 4
    return [
        0 if (num_hidden_layers - 1 - i) % stride == 0 else 1
        for i in range(num_hidden_layers)
    ]


class MuseGlimmerTextConfig(PretrainedConfig):
    model_type = "muse_glimmer_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 202_048,
        hidden_size: int = 6656,
        intermediate_size: int = 19968,
        num_hidden_layers: int = 52,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 2,
        head_dim: int = 128,
        hidden_activation: str = "silu",
        max_position_embeddings: int = 16_384,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        eos_token_id: int | list[int] | None = 200_001,
        bos_token_id: int | None = 200_000,
        tie_word_embeddings: bool = False,
        rope_parameters: dict | None = None,
        rope_theta: float | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        query_pre_attn_scalar: int = 256,
        sliding_window: int | None = 2048,
        layer_types: list[str] | None = None,
        final_logit_softcapping: float | None = 20.0,
        attn_logit_softcapping: float | None = None,
        use_bidirectional_attention: bool | None = None,
        # MuseGlimmer-specific
        qk_scale_factor: float = 43.7840518911,
        use_qk_norm: bool = True,
        use_attn_output_gate: bool = True,
        output_multiplier: float = 0.19611613513818404,
        normalize_tok_embeddings: bool = True,
        post_norm_eps: float = 1e-8,
        no_rope_layers: list[int] | None = None,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_activation = hidden_activation
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping
        self.use_bidirectional_attention = use_bidirectional_attention

        # RoPE: accept either an explicit rope_parameters dict (HF 5.x) or a
        # bare rope_theta; normalize to rope_parameters for vLLM's get_rope.
        if rope_parameters is None:
            theta = rope_theta if rope_theta is not None else 500_000.0
            rope_parameters = {"rope_type": "default", "rope_theta": theta}
        self.rope_parameters = rope_parameters
        # vLLM reads rope_theta off the config in some codepaths.
        self.rope_theta = rope_parameters.get("rope_theta", 500_000.0)

        # MuseGlimmer-specific fields
        self.qk_scale_factor = qk_scale_factor
        self.use_qk_norm = use_qk_norm
        self.use_attn_output_gate = use_attn_output_gate
        self.output_multiplier = output_multiplier
        self.normalize_tok_embeddings = normalize_tok_embeddings
        self.post_norm_eps = post_norm_eps

        self.no_rope_layers = (
            no_rope_layers
            if no_rope_layers is not None
            else _default_no_rope_layers(num_hidden_layers)
        )
        if layer_types is None:
            layer_types = [
                "full_attention" if self.no_rope_layers[i] == 0 else "sliding_attention"
                for i in range(num_hidden_layers)
            ]
        self.layer_types = layer_types

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class MuseGlimmerVisionConfig(PretrainedConfig):
    model_type = "muse_glimmer_vision"

    def __init__(
        self,
        patch_size: int = 14,
        pos_emb_height: int = 32,
        pos_emb_width: int = 32,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 50,
        hidden_size: int = 1536,
        intermediate_size: int = 8960,
        hidden_act: str = "gelu",
        merge_kernel_size: int = 2,
        rope_parameters: dict | None = None,
        max_position_embeddings: int = 32 * 32,
        output_dim: int = 6144,
        patch_temporal: int = 2,
        adapter_dim: int = 4096,
        layer_norm_eps: float = 1e-5,
        layer_types: list[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.pos_emb_height = pos_emb_height
        self.pos_emb_width = pos_emb_width
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.merge_kernel_size = merge_kernel_size
        self.rope_parameters = rope_parameters
        self.max_position_embeddings = max_position_embeddings
        self.output_dim = output_dim
        self.patch_temporal = patch_temporal
        self.adapter_dim = adapter_dim
        self.layer_norm_eps = layer_norm_eps
        if layer_types is None:
            stride = 4
            layer_types = [
                "full_attention"
                if (i + 1) % stride == 0 or i == num_hidden_layers - 1
                else "sliding_attention"
                for i in range(num_hidden_layers)
            ]
        self.layer_types = layer_types


class MuseGlimmerConfig(PretrainedConfig):
    model_type = "muse_glimmer"
    sub_configs = {
        "text_config": MuseGlimmerTextConfig,
        "vision_config": MuseGlimmerVisionConfig,
    }

    # --- Flat (legacy-converter) -> canonical normalization -------------------
    # MuseGlimmer checkpoints exist in two config layouts in the wild:
    #
    #   * CANONICAL (current HF converter, transformers 5.15): nested
    #     ``text_config`` / ``vision_config`` sub-dicts with canonical field
    #     names (``hidden_activation``, ``final_logit_softcapping``,
    #     ``rope_parameters``, ``vision_config.hidden_size`` ...).
    #
    #   * FLAT (older converter, e.g. Ruan's ``rl_v1/hf``, transformers 5.9):
    #     every field is a top-level key, with different names
    #     (``hidden_act``, ``output_soft_cap_temp``, ``rope_theta``,
    #     ``vision_latent_dim`` ...) and NO ``text_config`` nesting.
    #
    # Without normalization a flat config silently deserializes to an
    # ALL-DEFAULT text config (every checkpoint value ignored) — a dangerous
    # correctness bug: a checkpoint whose arch differs from the defaults would
    # load into a wrong-shaped model with no error. So when we detect a flat
    # config we hoist the flat fields into ``text_config`` / ``vision_config``
    # and rename them to canonical names.

    # flat text field name -> canonical MuseGlimmerTextConfig field name
    _FLAT_TEXT_RENAMES = {
        "hidden_act": "hidden_activation",
        "output_soft_cap_temp": "final_logit_softcapping",
    }
    # canonical MuseGlimmerTextConfig constructor params that may appear flat
    _FLAT_TEXT_KEYS = frozenset(
        {
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "hidden_activation",
            "hidden_act",
            "max_position_embeddings",
            "initializer_range",
            "rms_norm_eps",
            "use_cache",
            "tie_word_embeddings",
            "rope_parameters",
            "rope_theta",
            "attention_bias",
            "attention_dropout",
            "query_pre_attn_scalar",
            "sliding_window",
            "layer_types",
            "final_logit_softcapping",
            "output_soft_cap_temp",
            "attn_logit_softcapping",
            "use_bidirectional_attention",
            "qk_scale_factor",
            "use_qk_norm",
            "use_attn_output_gate",
            "output_multiplier",
            "normalize_tok_embeddings",
            "post_norm_eps",
            "no_rope_layers",
        }
    )
    # flat vision field name -> canonical MuseGlimmerVisionConfig field name
    _FLAT_VISION_RENAMES = {
        "vision_latent_dim": "hidden_size",
        "vision_heads": "num_attention_heads",
        "vision_layers": "num_hidden_layers",
        "vision_output_dim": "output_dim",
        "vision_patch_size": "patch_size",
        "vision_patch_temporal": "patch_temporal",
        "vision_adapter_dim": "adapter_dim",
        "vision_pos_emb_grid_h": "pos_emb_height",
        "vision_pos_emb_grid_w": "pos_emb_width",
        "vision_downsample_factor": "merge_kernel_size",
    }

    @classmethod
    def _looks_flat(cls, kwargs: dict) -> bool:
        # Flat if there is no explicit text_config but there ARE text-level
        # fields at the top level (e.g. hidden_size / num_hidden_layers).
        if kwargs.get("text_config") is not None:
            return False
        return any(k in kwargs for k in ("hidden_size", "num_hidden_layers"))

    @classmethod
    def _normalize_flat(cls, kwargs: dict) -> dict:
        kwargs = dict(kwargs)
        text: dict = {}
        for key in list(kwargs.keys()):
            if key in cls._FLAT_TEXT_KEYS:
                canon = cls._FLAT_TEXT_RENAMES.get(key, key)
                text.setdefault(canon, kwargs.pop(key))

        vision_mlp_ratio = kwargs.pop("vision_mlp_ratio", None)
        sparse_factor = kwargs.pop("vision_sparse_attention_factor", None)
        vision: dict = {}
        for key in list(kwargs.keys()):
            if key in cls._FLAT_VISION_RENAMES:
                vision.setdefault(cls._FLAT_VISION_RENAMES[key], kwargs.pop(key))

        if vision_mlp_ratio is not None:
            hidden_size = vision.get("hidden_size", 1536)
            vision["intermediate_size"] = int(vision_mlp_ratio * hidden_size)
        if sparse_factor is not None:
            sparse_factor = int(sparse_factor)
            if sparse_factor <= 0:
                raise ValueError("vision_sparse_attention_factor must be positive")
            num_layers = vision.get("num_hidden_layers", 50)
            vision["layer_types"] = [
                "full_attention"
                if (layer_idx + 1) % sparse_factor == 0 or layer_idx == num_layers - 1
                else "sliding_attention"
                for layer_idx in range(num_layers)
            ]

        if text:
            kwargs["text_config"] = text
        if vision:
            kwargs["vision_config"] = vision
        return kwargs

    def __init__(
        self,
        text_config: dict | MuseGlimmerTextConfig | None = None,
        vision_config: dict | MuseGlimmerVisionConfig | None = None,
        image_token_id: int = 200092,
        video_token_id: int = 200091,
        **kwargs,
    ) -> None:
        # Detect + fold a flat config into nested text/vision before building
        # the sub-configs. (image/video token ids are handled below; a flat
        # ``patch_token_id`` alias is mapped too.)
        if text_config is None and self._looks_flat(kwargs):
            kwargs = self._normalize_flat(kwargs)
            text_config = kwargs.pop("text_config", None)
            vision_config = kwargs.pop("vision_config", vision_config)
            # flat image-token alias
            if "patch_token_id" in kwargs and "image_token_id" not in kwargs:
                image_token_id = kwargs.pop("patch_token_id")

        if text_config is None:
            self.text_config = MuseGlimmerTextConfig()
        elif isinstance(text_config, dict):
            self.text_config = MuseGlimmerTextConfig(**text_config)
        else:
            self.text_config = text_config

        if vision_config is None:
            self.vision_config = MuseGlimmerVisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = MuseGlimmerVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.patch_token_id = image_token_id
        super().__init__(**kwargs)


class MuseGlimmerAssistantConfig(Qwen3Config):
    """Config for the Muse Glimmer DFlash draft head.

    The head is Qwen3-shaped and runs on vLLM's generic ``qwen3_dflash``
    implementation, so this derives from ``Qwen3Config``. It cannot BE
    ``Qwen3Config``, because two of the checkpoint's values do not survive that
    class:

    * ``sliding_window`` is gated behind ``use_sliding_window``, which defaults
      to False -- ``Qwen3Config(sliding_window=2048).sliding_window`` is None.
      The checkpoint declares ``sliding_window: 2048`` and five
      ``sliding_attention`` layers, so the window silently disappears and the
      DFlash path then raises "sliding attention requires a window size".
    * ``vocab_size`` is absent from the checkpoint, so Qwen3's default of
      151936 applies instead of Muse Glimmer's 202048. That one is *silent*: it
      builds an all-zero ``draft_id_to_target_id`` remap, and also puts
      pad/bos/eos/mask token ids out of range.

    Both defaults are set here so an unmodified checkpoint loads correctly.
    """

    model_type = "muse_glimmer_assistant"

    def __init__(
        self,
        vocab_size: int = 202048,
        use_sliding_window: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size, use_sliding_window=use_sliding_window, **kwargs
        )


__all__ = [
    "MuseGlimmerTextConfig",
    "MuseGlimmerVisionConfig",
    "MuseGlimmerConfig",
    "MuseGlimmerAssistantConfig",
]
