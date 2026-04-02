# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SarvamMLA model configuration.

This vLLM-local config replaces the trust_remote_code configuration shipped
with sarvamai/sarvam-105b.  The remote configuration calls
``validate_rope(ignore_keys=...)`` which was broken by the transformers v5
API change that removed the ``ignore_keys`` parameter (upstream PR #41250).

By registering ``SarvamMLAConfig`` in vLLM's ``_CLASS_TO_MODULE`` table,
``HFConfigParser.parse()`` loads *this* class instead of the remote one,
bypassing ``trust_remote_code`` and the broken API call entirely.
"""

from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SarvamMLAConfig(PretrainedConfig):
    """Configuration for SarvamMLA models (e.g., ``sarvamai/sarvam-105b``).

    Mirrors the fields present in the model's ``config.json`` and the
    attributes accessed by ``vllm/model_executor/models/sarvam.py``.
    """

    model_type = "sarvam_mla"

    def __init__(
        self,
        vocab_size: int = 262144,
        hidden_size: int = 4096,
        intermediate_size: int = 16384,
        moe_intermediate_size: int | None = None,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 64,
        num_key_value_heads: int | None = None,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        q_lora_rank: int | None = None,
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-6,
        hidden_act: str = "silu",
        num_experts: int | None = None,
        num_experts_per_tok: int | None = None,
        moe_shared_expert_intermediate_size: int | None = None,
        first_k_dense_replace: int = 1,
        rope_theta: float = 10000.0,
        rope_parameters: dict | None = None,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        # Handle rope_scaling → rope_parameters conversion so that both the
        # old-style {"type": ...} field and the v5 {"rope_type": ...} field
        # are accepted.
        rope_scaling = kwargs.pop("rope_scaling", None)
        if rope_scaling is not None and rope_parameters is None:
            rope_parameters = dict(rope_scaling)
            # Normalise "type" → "rope_type" (transformers v5 convention).
            if "type" in rope_parameters and "rope_type" not in rope_parameters:
                rope_parameters["rope_type"] = rope_parameters.pop("type")
        if rope_parameters is None:
            rope_parameters = {"rope_type": "default"}
        if "rope_theta" not in rope_parameters:
            rope_parameters["rope_theta"] = rope_theta
        self.rope_parameters = rope_parameters

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # ``sarvam.py`` accesses ``config.moe_intermediate_size`` directly.
        # The config.json stores a single ``intermediate_size`` field; expose
        # it under both names so the model layer and dense-FFN path both work.
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = (
            moe_intermediate_size
            if moe_intermediate_size is not None
            else intermediate_size
        )
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_key_value_heads
            if num_key_value_heads is not None
            else num_attention_heads
        )

        # MLA-specific attention dimensions.
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank

        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.use_cache = use_cache

        # MoE topology.
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace

        super().__init__(
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
