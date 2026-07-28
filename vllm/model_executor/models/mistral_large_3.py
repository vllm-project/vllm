# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import regex
import torch

from vllm.model_executor.models.deepseek_v2 import DeepseekV3ForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper


class MistralLarge3ForCausalLM(DeepseekV3ForCausalLM):
    # WeightsMapper applies all matching patterns sequentially (no break on first
    # match). This is safe here because every pattern is anchored at both ends
    # (\A...\Z) and after substitution the resulting key always starts with
    # "model." or "lm_head.", so no later pattern can accidentally match again.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_regex={  # noqa: B950
            regex.compile(
                r"\Alayers\.(\d+)\.attention_norm\.weight\Z"
            ): r"model.layers.\1.input_layernorm.weight",
            regex.compile(
                r"\Alayers\.(\d+)\.attention\.wq_a\.(\w+)\Z"
            ): r"model.layers.\1.self_attn.q_a_proj.\2",
            regex.compile(
                r"\Alayers\.(\d+)\.attention\.q_a_norm\.weight\Z"
            ): r"model.layers.\1.self_attn.q_a_layernorm.weight",
            regex.compile(
                r"\Alayers\.(\d+)\.attention\.wq_b\.(\w+)\Z"
            ): r"model.layers.\1.self_attn.q_b_proj.\2",
            regex.compile(
                r"\Alayers\.(\d+)\.attention\.wkv_a_with_mqa\.(\w+)\Z"
            ): r"model.layers.\1.self_attn.kv_a_proj_with_mqa.\2",
            regex.compile(
                r"\Alayers\.(\d+)\.attention\.kv_a_norm\.weight\Z"
            ): r"model.layers.\1.self_attn.kv_a_layernorm.weight",
            regex.compile(
                r"\Alayers\.(\d+)\.attention\.wkv_b\.(\w+)\Z"
            ): r"model.layers.\1.self_attn.kv_b_proj.\2",
            regex.compile(
                r"\Alayers\.(\d+)\.attention\.wo\.(\w+)\Z"
            ): r"model.layers.\1.self_attn.o_proj.\2",
            regex.compile(
                r"\Alayers\.(\d+)\.ffn_norm\.weight\Z"
            ): r"model.layers.\1.post_attention_layernorm.weight",
            regex.compile(
                r"\Alayers\.(\d+)\.feed_forward\.w1\.(\w+)\Z"
            ): r"model.layers.\1.mlp.gate_proj.\2",
            regex.compile(
                r"\Alayers\.(\d+)\.feed_forward\.w2\.(\w+)\Z"
            ): r"model.layers.\1.mlp.down_proj.\2",
            regex.compile(
                r"\Alayers\.(\d+)\.feed_forward\.w3\.(\w+)\Z"
            ): r"model.layers.\1.mlp.up_proj.\2",
            regex.compile(
                r"\Alayers\.(\d+)\.gate\.weight\Z"
            ): r"model.layers.\1.mlp.gate.weight",
            regex.compile(
                r"\Alayers\.(\d+)\.shared_experts\.w1\.(\w+)\Z"
            ): r"model.layers.\1.mlp.shared_experts.gate_proj.\2",
            regex.compile(
                r"\Alayers\.(\d+)\.shared_experts\.w2\.(\w+)\Z"
            ): r"model.layers.\1.mlp.shared_experts.down_proj.\2",
            regex.compile(
                r"\Alayers\.(\d+)\.shared_experts\.w3\.(\w+)\Z"
            ): r"model.layers.\1.mlp.shared_experts.up_proj.\2",
            regex.compile(
                r"\Alayers\.(\d+)\.experts\.(\d+)\.w1\.(\w+)\Z"
            ): r"model.layers.\1.mlp.experts.\2.gate_proj.\3",
            regex.compile(
                r"\Alayers\.(\d+)\.experts\.(\d+)\.w2\.(\w+)\Z"
            ): r"model.layers.\1.mlp.experts.\2.down_proj.\3",
            regex.compile(
                r"\Alayers\.(\d+)\.experts\.(\d+)\.w3\.(\w+)\Z"
            ): r"model.layers.\1.mlp.experts.\2.up_proj.\3",
            regex.compile(r"\Anorm\.weight\Z"): "model.norm.weight",
            regex.compile(r"\Atok_embeddings\.weight\Z"): "model.embed_tokens.weight",
            regex.compile(r"\Aoutput\.weight\Z"): "lm_head.weight",
        },
        orig_to_new_suffix={
            ".qscale_act": ".input_scale",
            ".qscale_weight": ".weight_scale",
        },
    )

    # Bypass super().load_weights() and construct AutoWeightsLoader(self)
    # directly (same pattern as Qwen2ForCausalLM). Any logic in the parent
    # class's load_weights is a thin wrapper around AutoWeightsLoader, and
    # we must apply hf_to_vllm_mapper before the loader walks the tree.
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
