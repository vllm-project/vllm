# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import PreTrainedConfig


class LongSparkConfig(PreTrainedConfig):
    """Configuration for LongSpark DraftFreeKV checkpoints."""

    model_type = "draftfreekv"

    def __init__(
        self,
        target_model_name_or_path: str | None = None,
        target_layer_ids: list[int] | None = None,
        num_target_layers: int = 0,
        block_size: int = 7,
        hidden_size: int = 4096,
        vocab_size: int = 32000,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        parallel_mixer_layers: int = 5,
        parallel_markov_rank: int = 256,
        qwen_intermediate_size: int = 12288,
        qwen_num_attention_heads: int = 32,
        qwen_rope_theta: float = 1_000_000.0,
        qwen_rms_norm_eps: float = 1e-6,
        mask_token_id: int | None = None,
        global16_raw256_enabled: bool = True,
        global_memory_slots: int = 16,
        raw_target_rows: int = 256,
        local_draft_rows: int = 7,
        max_accepted_tokens: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.target_model_name_or_path = target_model_name_or_path
        self.target_layer_ids = list(target_layer_ids or [])
        self.num_target_layers = int(num_target_layers)
        self.block_size = int(block_size)
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.parallel_mixer_layers = int(parallel_mixer_layers)
        self.parallel_markov_rank = int(parallel_markov_rank)
        self.qwen_intermediate_size = int(qwen_intermediate_size)
        self.qwen_num_attention_heads = int(qwen_num_attention_heads)
        self.qwen_rope_theta = float(qwen_rope_theta)
        self.qwen_rms_norm_eps = float(qwen_rms_norm_eps)
        self.mask_token_id = mask_token_id
        self.global16_raw256_enabled = bool(global16_raw256_enabled)
        self.global_memory_slots = int(global_memory_slots)
        self.raw_target_rows = int(raw_target_rows)
        self.local_draft_rows = int(local_draft_rows)
        self.max_accepted_tokens = int(max_accepted_tokens)

        # ModelConfig and generic model helpers consume the canonical names.
        self.num_hidden_layers = self.parallel_mixer_layers
        self.num_attention_heads = self.qwen_num_attention_heads
        self.intermediate_size = self.qwen_intermediate_size
        self.rms_norm_eps = self.qwen_rms_norm_eps
        self.rope_theta = self.qwen_rope_theta
        self.hidden_act = "silu"
        self.max_position_embeddings = int(
            getattr(self, "max_position_embeddings", 131072)
        )
