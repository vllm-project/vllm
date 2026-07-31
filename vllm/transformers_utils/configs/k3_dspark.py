# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import DeepseekV2Config


class K3DSparkConfig(DeepseekV2Config):
    """Configuration for a dense MLA DSpark draft model."""

    model_type = "k3_dspark"
    has_no_defaults_at_init = True

    def __init__(
        self,
        mla_use_nope: bool = False,
        mla_use_output_gate: bool = False,
        mla_use_qk_norm: bool = False,
        rope_theta: float = 50000.0,
        **kwargs,
    ) -> None:
        # DeepseekV2Config defaults to a MoE topology. Zero these fields so
        # generic vLLM config logic also recognizes this draft as dense.
        kwargs.setdefault("n_routed_experts", 0)
        kwargs.setdefault("n_shared_experts", 0)
        kwargs.setdefault("num_experts_per_tok", 0)

        rope_parameters = kwargs.get("rope_parameters")
        if rope_parameters is None:
            kwargs["rope_parameters"] = {
                "rope_type": "default",
                "rope_theta": rope_theta,
            }
        else:
            rope_parameters = dict(rope_parameters)
            rope_parameters.setdefault("rope_type", "default")
            rope_parameters.setdefault("rope_theta", rope_theta)
            kwargs["rope_parameters"] = rope_parameters

        super().__init__(**kwargs)
        self.mla_use_nope = mla_use_nope
        self.mla_use_output_gate = mla_use_output_gate
        self.mla_use_qk_norm = mla_use_qk_norm

        unsupported = [
            name
            for name in (
                "mla_use_nope",
                "mla_use_output_gate",
                "mla_use_qk_norm",
                "dspark_bonus_anchor",
            )
            if getattr(self, name, False)
        ]
        if self.q_lora_rank is None:
            unsupported.append("q_lora_rank=None")
        if unsupported:
            raise ValueError(
                "MLA DSpark does not support " + ", ".join(unsupported) + "."
            )

        self.draft_vocab_size = (
            getattr(self, "draft_vocab_size", None) or self.vocab_size
        )
        if self.draft_vocab_size != self.vocab_size:
            raise ValueError(
                "MLA DSpark requires draft_vocab_size to equal vocab_size when "
                "sharing the target embedding and LM head."
            )

        target_layer_ids = getattr(self, "target_layer_ids", None)
        if not target_layer_ids or getattr(self, "num_target_layers", None) != len(
            target_layer_ids
        ):
            raise ValueError(
                "MLA DSpark requires non-empty target_layer_ids and a matching "
                "num_target_layers."
            )
