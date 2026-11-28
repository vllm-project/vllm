# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 model configs."""

from __future__ import annotations

from typing import Any, ClassVar

from transformers.configuration_utils import PreTrainedConfig

INKLING2_ARCHITECTURE = "Inkling2ForConditionalGeneration"
INKLING2_LOOP_ARCHITECTURE = "Inkling2LoopForConditionalGeneration"
INKLING2_ARCHITECTURES = frozenset({INKLING2_ARCHITECTURE, INKLING2_LOOP_ARCHITECTURE})


class Inkling2ModelConfig(PreTrainedConfig):
    model_type = "inkling2_model"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        *,
        vocab_size: int = 201024,
        unpadded_vocab_size: int | None = None,
        hidden_size: int = 4608,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        swa_num_attention_heads: int | None = None,
        swa_num_key_value_heads: int | None = None,
        swa_head_dim: int | None = None,
        d_rel: int = 16,
        rel_extent: int = 1024,
        local_layer_ids: list[int] | None = None,
        sliding_window_size: int = 1024,
        sconv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-6,
        model_max_length: int = 1048576,
        log_scaling_n_floor: int | None = 128000,
        log_scaling_alpha: float = 0.1,
        pre_norm_pre_affine: bool = False,
        final_norm_pre_affine: bool = False,
        use_value_norm_no_gamma: bool = True,
        qk_l2_clip_threshold: float = 22.627416997969522,
        rel_l2_clip_threshold: float = 8.0,
        n_layers_per_attn_res_block: int = 6,
        attn_res_normalize_values: bool = True,
        final_logit_softcapping: float | None = 30.0,
        n_routed_experts: int = 384,
        num_experts_per_tok: int = 8,
        n_shared_experts: int = 0,
        intermediate_size: int = 2304,
        mlp_gate_softcap: float = 0.0,
        mlp_up_softcap: float = 0.0,
        route_scale: float = 8.0,
        use_gate_bias: bool = True,
        loop_config: dict[str, int | bool] | None = None,
        loop_num_passes: int | None = None,
        loop_share_cache: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        if pre_norm_pre_affine or final_norm_pre_affine:
            raise ValueError("Inkling2 pre-affine norms are not supported")
        if "n_layers_per_attnres_block" in kwargs:
            raise ValueError(
                "n_layers_per_attnres_block was renamed to n_layers_per_attn_res_block"
            )
        self.padded_vocab_size = vocab_size
        self.vocab_size = unpadded_vocab_size or vocab_size
        self.unpadded_vocab_size = self.vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.swa_num_attention_heads = swa_num_attention_heads or num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads or num_key_value_heads
        self.swa_head_dim = swa_head_dim or head_dim
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.local_layer_ids = list(local_layer_ids or [])
        self.sliding_window_size = sliding_window_size
        self.sconv_kernel_size = sconv_kernel_size
        self.rms_norm_eps = rms_norm_eps
        self.model_max_length = model_max_length
        self.max_position_embeddings = model_max_length
        self.log_scaling_n_floor = log_scaling_n_floor
        self.log_scaling_alpha = log_scaling_alpha
        self.use_value_norm_no_gamma = use_value_norm_no_gamma
        self.qk_l2_clip_threshold = qk_l2_clip_threshold
        self.rel_l2_clip_threshold = rel_l2_clip_threshold
        self.n_layers_per_attn_res_block = n_layers_per_attn_res_block
        self.attn_res_normalize_values = attn_res_normalize_values
        self.final_logit_softcapping = final_logit_softcapping
        self.n_routed_experts = n_routed_experts
        self.num_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.intermediate_size = intermediate_size
        self.mlp_gate_softcap = mlp_gate_softcap
        self.mlp_up_softcap = mlp_up_softcap
        self.route_scale = route_scale
        self.use_gate_bias = use_gate_bias
        if loop_config is not None:
            int_fields = {"core_start", "core_end", "num_passes", "gate_rank"}
            if (
                set(loop_config) != int_fields | {"is_varloop"}
                or any(type(loop_config[name]) is not int for name in int_fields)
                or type(loop_config["is_varloop"]) is not bool
            ):
                raise ValueError(
                    f"loop_config requires integer fields {sorted(int_fields)} "
                    "and a boolean is_varloop"
                )
            start, end = loop_config["core_start"], loop_config["core_end"]
            if not 0 < start < end <= num_hidden_layers:
                raise ValueError("loop_config core layers are out of range")
            if loop_config["gate_rank"] < 1:
                raise ValueError("loop_config gate_rank must be >= 1")
            if not attn_res_normalize_values:
                raise ValueError(
                    "Loop K/V messages require attn_res_normalize_values=True"
                )
        self.loop_config = loop_config
        # Serving options of variable-loop checkpoints (set through
        # --hf-overrides '{"text_config": {...}}'): the number of loop passes
        # (None: the checkpoint's), and one KV cache per core layer shared by
        # all passes instead of one per pass (conv states stay per pass).
        self.loop_num_passes = loop_num_passes
        self.loop_share_cache = loop_share_cache
        if loop_config is not None:
            self.check_loop_options()
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def loop_core_layers(self) -> range:
        """The layers that run `num_loop_passes` times (empty without a loop)."""
        loop = self.loop_config
        return range(0) if loop is None else range(loop["core_start"], loop["core_end"])

    def check_loop_options(self) -> None:
        """Validate the loop serving options (again once overrides are set)."""
        loop = self.loop_config
        varloop = loop is not None and loop["is_varloop"]
        if (self.loop_num_passes is not None or self.loop_share_cache) and not varloop:
            raise ValueError(
                "loop_num_passes and loop_share_cache need a checkpoint with "
                "loop_config.is_varloop=true"
            )
        if loop is not None and self.num_loop_passes < (1 if varloop else 2):
            raise ValueError(
                f"This loop checkpoint needs at least {1 if varloop else 2} passes, "
                f"got {self.num_loop_passes}"
            )

    @property
    def num_loop_passes(self) -> int:
        if self.loop_config is None:
            return 1
        if self.loop_num_passes is not None:
            return self.loop_num_passes
        return int(self.loop_config["num_passes"])

    @property
    def num_layer_visits(self) -> int:
        """Decoder layers run per token: each loop pass runs the core again."""
        extra_passes = self.num_loop_passes - 1
        return self.num_hidden_layers + extra_passes * len(self.loop_core_layers)

    def layer_visit_ids(self, layer_id: int) -> list[int]:
        """The indices of a decoder layer's visits in execution order (the
        prelude, every pass of the core, then the coda): one per pass for a
        loop core layer, else one."""
        core, passes = self.loop_core_layers, self.num_loop_passes
        if not core or layer_id < core.start:
            return [layer_id]
        if layer_id >= core.stop:
            return [layer_id + (passes - 1) * len(core)]
        return [layer_id + p * len(core) for p in range(passes)]


class Inkling2MMConfig(PreTrainedConfig):
    model_type = "inkling2"
    architecture = INKLING2_ARCHITECTURE
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs: ClassVar[dict[str, type[PreTrainedConfig]]] = {
        "text_config": Inkling2ModelConfig,
    }

    def __init__(
        self,
        *,
        text_config: dict[str, Any] | Inkling2ModelConfig | None = None,
        vision_config: dict[str, Any] | None = None,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        self.text_config = (
            text_config
            if isinstance(text_config, Inkling2ModelConfig)
            else Inkling2ModelConfig(**(text_config or {}))
        )
        if (self.text_config.loop_config is not None) != (
            self.model_type == "inkling2_loop"
        ):
            raise ValueError(
                "Looped Inkling2 checkpoints use model_type inkling2_loop with a "
                "loop_config; model_type inkling2 has none"
            )
        architectures = kwargs.get("architectures")
        if architectures is not None and architectures != [self.architecture]:
            raise ValueError(
                f"model_type {self.model_type} checkpoints use architectures "
                f"[{self.architecture!r}], got {architectures!r}"
            )
        self.vision_config = vision_config
        # The checkpoint's unembedding runs in BF16 with an FP32 output.
        kwargs.setdefault("head_dtype", "float32")
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def get_text_config(self, *args: Any, **kwargs: Any) -> Inkling2ModelConfig:
        return self.text_config


class Inkling2LoopMMConfig(Inkling2MMConfig):
    model_type = "inkling2_loop"
    architecture = INKLING2_LOOP_ARCHITECTURE
    # A valid loop config requires checkpoint-provided core bounds and passes.
    has_no_defaults_at_init = True

    # Explicit, or transformers generates one that keeps a dict text_config.
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


class Inkling2DSparkConfig(PreTrainedConfig):
    """The DSpark draft of an Inkling2 checkpoint, from its `mtp_config`."""

    model_type = "inkling2_dspark"

    def __init__(
        self,
        *,
        block_size: int = 8,
        target_layer_ids: list[int] | None = None,
        **kwargs,
    ) -> None:
        self.draft_block_size = block_size
        self.dspark_block_size = block_size
        self.sample_from_anchor = False
        self.is_causal = False
        self.mask_token_id = 0
        # Target layer visits (loop passes counted) whose input (the AttnRes
        # partial block) is captured; the last id, the target's
        # num_layer_visits, is its final-norm output.
        self.target_layer_ids = list(target_layer_ids or [])
        self.eagle_aux_hidden_state_layer_ids = self.target_layer_ids
        kwargs.setdefault("architectures", ["Inkling2DSparkModel"])
        super().__init__(tie_word_embeddings=False, **kwargs)

    @classmethod
    def from_target(cls, config: Inkling2MMConfig) -> Inkling2DSparkConfig:
        mtp_config = getattr(config, "mtp_config", None)
        if not mtp_config:
            raise ValueError("The Inkling2 checkpoint has no mtp_config")
        text = config.text_config
        loop = text.loop_config
        if loop is not None and text.num_loop_passes != loop["num_passes"]:
            raise ValueError(
                "Inkling2 DSpark drafts read hidden states at fixed layer visits; "
                f"serve them at the checkpoint's {loop['num_passes']} loop passes"
            )
        draft = {
            k: v
            for k, v in mtp_config.items()
            if k not in ("torch_dtype", "dtype", "architectures")
        }
        features = draft.get("target_layer_ids", [])
        if (
            len(features) < 2
            or features != sorted(set(features))
            or features[-1] != text.num_layer_visits
        ):
            raise ValueError(
                "Inkling2 MTP target_layer_ids must be sorted target layer visits "
                "followed by num_layer_visits (the final-norm output)"
            )
        if (
            draft["hidden_size"] != text.hidden_size
            or draft.get("markov_head_type") != "rnn_hidden"
            or draft.get("q_bias")
            or draft.get("o_bias")
        ):
            raise ValueError("Unsupported Inkling2 mtp_config")
        return cls(
            **draft,
            vocab_size=text.vocab_size,
            padded_vocab_size=text.padded_vocab_size,
            final_logit_softcapping=text.final_logit_softcapping,
            torch_dtype="bfloat16",
        )
