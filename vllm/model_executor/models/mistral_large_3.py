# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import regex as re
import torch

from vllm.model_executor.models.deepseek_v2 import DeepseekV3ForCausalLM


class MistralLarge3ForCausalLM(DeepseekV3ForCausalLM):
    # fmt: off
    remapping = {
        r"layers\.(\d+)\.attention_norm\.weight": r"model.layers.\1.input_layernorm.weight",  # noqa: E501
        r"layers\.(\d+)\.attention\.wq_a\.(\w+)": r"model.layers.\1.self_attn.q_a_proj.\2",  # noqa: E501
        r"layers\.(\d+)\.attention\.q_a_norm\.weight": r"model.layers.\1.self_attn.q_a_layernorm.weight",  # noqa: E501
        r"layers\.(\d+)\.attention\.wq_b\.(\w+)": r"model.layers.\1.self_attn.q_b_proj.\2",  # noqa: E501
        r"layers\.(\d+)\.attention\.wkv_a_with_mqa\.(\w+)": r"model.layers.\1.self_attn.kv_a_proj_with_mqa.\2",  # noqa: E501
        r"layers\.(\d+)\.attention\.kv_a_norm\.weight": r"model.layers.\1.self_attn.kv_a_layernorm.weight",  # noqa: E501
        r"layers\.(\d+)\.attention\.wkv_b\.(\w+)": r"model.layers.\1.self_attn.kv_b_proj.\2",  # noqa: E501
        r"layers\.(\d+)\.attention\.wo\.(\w+)": r"model.layers.\1.self_attn.o_proj.\2",  # noqa: E501
        r"layers\.(\d+)\.ffn_norm\.weight": r"model.layers.\1.post_attention_layernorm.weight",  # noqa: E501
        r"layers\.(\d+)\.feed_forward\.w1\.(\w+)": r"model.layers.\1.mlp.gate_proj.\2",  # noqa: E501
        r"layers\.(\d+)\.feed_forward\.w2\.(\w+)": r"model.layers.\1.mlp.down_proj.\2",  # noqa: E501
        r"layers\.(\d+)\.feed_forward\.w3\.(\w+)": r"model.layers.\1.mlp.up_proj.\2",  # noqa: E501
        r"layers\.(\d+)\.gate\.weight": r"model.layers.\1.mlp.gate.weight",  # noqa: E501
        r"layers\.(\d+)\.shared_experts\.w1\.(\w+)": r"model.layers.\1.mlp.shared_experts.gate_proj.\2",  # noqa: E501
        r"layers\.(\d+)\.shared_experts\.w2\.(\w+)": r"model.layers.\1.mlp.shared_experts.down_proj.\2",  # noqa: E501
        r"layers\.(\d+)\.shared_experts\.w3\.(\w+)": r"model.layers.\1.mlp.shared_experts.up_proj.\2",  # noqa: E501
        r"layers\.(\d+)\.experts\.(\d+)\.w1\.(\w+)": r"model.layers.\1.mlp.experts.\2.gate_proj.\3",  # noqa: E501
        r"layers\.(\d+)\.experts\.(\d+)\.w2\.(\w+)": r"model.layers.\1.mlp.experts.\2.down_proj.\3",  # noqa: E501
        r"layers\.(\d+)\.experts\.(\d+)\.w3\.(\w+)": r"model.layers.\1.mlp.experts.\2.up_proj.\3",  # noqa: E501
        r"norm\.weight": "model.norm.weight",  # noqa: E501
        r"tok_embeddings\.weight": "model.embed_tokens.weight",  # noqa: E501
        r"output\.weight": "lm_head.weight",  # noqa: E501
    }
    # fmt: on

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(map(self._remap_mistral_to_ds, weights))

    def _remap_mistral_to_ds(
        self, weight: tuple[str, torch.Tensor]
    ) -> tuple[str, torch.Tensor]:
        """Remap Mistral parameters to DeepseekV2 parameters."""
        name, loaded_weight = weight

        for k, v in self.remapping.items():
            match = re.fullmatch(k, name)
            if match:
                name = re.sub(k, v, name)
                break
        else:
            raise ValueError(f"Cannot remap {name}")

        # Remapping scale names. We could do this in the regex above but it
        # would triple the number of lines for most layers.
        if name.endswith(".qscale_act"):
            name = re.sub(r"\.qscale_act$", ".input_scale", name)
        elif name.endswith(".qscale_weight"):
            name = re.sub(r"\.qscale_weight$", ".weight_scale", name)

        return name, loaded_weight
